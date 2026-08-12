#!/usr/bin/env python
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    import accelerate
    import diffusers
    import torch
    import transformers
    from diffusers import FluxControlInpaintPipeline

    call_signature = inspect.signature(FluxControlInpaintPipeline.__call__)
    loader_signature = inspect.signature(FluxControlInpaintPipeline.from_pretrained)
    target = torch.bfloat16

    current_default = torch.get_default_dtype()
    causal_mask_probe = {"status": "not_run"}
    try:
        # This is the exact old-Transformers construction that fails when a
        # third-party loader leaks FP16 as Torch's global default.
        probe = torch.full((2, 2), torch.finfo(target).min)
        causal_mask_probe = {
            "status": "ok",
            "tensor_dtype": str(probe.dtype),
        }
    except Exception as exc:
        causal_mask_probe = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "version": str(torch.version.cuda),
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )

    report = {
        "status": "ok"
        if current_default == torch.float32 and causal_mask_probe["status"] == "ok"
        else "needs_runtime_guard",
        "versions": {
            "torch": _version(torch),
            "diffusers": _version(diffusers),
            "transformers": _version(transformers),
            "accelerate": _version(accelerate),
        },
        "torch_default_dtype": str(current_default),
        "causal_mask_probe": causal_mask_probe,
        "cuda": cuda,
        "flux_api": {
            "prompt_2_supported": "prompt_2" in call_signature.parameters,
            "max_sequence_length_supported": "max_sequence_length" in call_signature.parameters,
            "from_pretrained_signature": str(loader_signature),
        },
        "note": (
            "The pipeline runtime itself resets the global default dtype to float32 and "
            "verifies CLIP/T5/VAE/Transformer precision before every inference call."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
