#!/usr/bin/env python
from __future__ import annotations

"""Small real TRELLIS inference entry point.

This script is executed with TRELLIS_PYTHON. It follows Microsoft's official
example API: TrellisImageTo3DPipeline -> pipeline.run(... formats mesh+gaussian)
-> postprocessing_utils.to_glb(...).  Keeping it as a separate process isolates
TRELLIS CUDA extensions from the main FLUX/pipeline environment.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis_repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sparse_steps", type=int, default=12)
    ap.add_argument("--sparse_cfg", type=float, default=7.5)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--slat_cfg", type=float, default=3.0)
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        repo = Path(args.trellis_repo).resolve()
        if not (repo / "trellis").exists():
            raise FileNotFoundError(f"TRELLIS Python package not found under: {repo}")
        sys.path.insert(0, str(repo))

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
        os.environ.setdefault("SPCONV_ALGO", "native")
        # Blackwell/5080 setups may prefer xformers if flash-attn was not built.
        os.environ.setdefault("ATTN_BACKEND", os.environ.get("TRELLIS_ATTN_BACKEND", "xformers"))

        from PIL import Image
        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        model_source = args.model
        pipeline = TrellisImageTo3DPipeline.from_pretrained(model_source)
        pipeline.cuda()

        image = Image.open(args.image).convert("RGBA")
        outputs = pipeline.run(
            image,
            seed=int(args.seed),
            sparse_structure_sampler_params={
                "steps": int(args.sparse_steps),
                "cfg_strength": float(args.sparse_cfg),
            },
            slat_sampler_params={
                "steps": int(args.slat_steps),
                "cfg_strength": float(args.slat_cfg),
            },
            formats=["mesh", "gaussian"],
        )
        if not outputs.get("mesh") or not outputs.get("gaussian"):
            raise RuntimeError("TRELLIS did not return both mesh and gaussian outputs required for GLB texture extraction.")

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=float(args.simplify),
            texture_size=int(args.texture_size),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output))
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"TRELLIS GLB export produced no file: {output}")

        report = {
            "status": "ok",
            "model_source": model_source,
            "input_image": str(Path(args.image)),
            "output_glb": str(output),
            "seed": int(args.seed),
            "sparse_sampler": {"steps": int(args.sparse_steps), "cfg_strength": float(args.sparse_cfg)},
            "slat_sampler": {"steps": int(args.slat_steps), "cfg_strength": float(args.slat_cfg)},
            "simplify": float(args.simplify),
            "texture_size": int(args.texture_size),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        report_path.write_text(json.dumps({"status": "failed", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
