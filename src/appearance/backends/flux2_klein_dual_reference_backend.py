from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from src.appearance.backends.base import SparseDiffusionBackend
from src.appearance.depth_control_image import load_depth_control_image
from src.appearance.hf_access import get_token, resolve_model_source


def _attach_progress(pipe: Any, call_kwargs: Dict[str, Any], callback, total_steps: int) -> str:
    if not callable(callback):
        return "disabled"
    try:
        params = inspect.signature(pipe.__call__).parameters
    except Exception:
        params = {}
    total = max(int(total_steps), 1)
    if "callback_on_step_end" in params:
        def cb(_pipe, step, _timestep, callback_kwargs):
            callback(min(int(step) + 1, total), total)
            return callback_kwargs
        call_kwargs["callback_on_step_end"] = cb
        return "callback_on_step_end"
    if "callback" in params:
        def cb(step, _timestep, _latents):
            callback(min(int(step) + 1, total), total)
        call_kwargs["callback"] = cb
        if "callback_steps" in params:
            call_kwargs["callback_steps"] = 1
        return "legacy_callback"
    return "unsupported"


class Flux2Klein4BDualReferenceBackend(SparseDiffusionBackend):
    """FLUX.2 [klein] 4B image-edit backend for Stage08.

    Reference contract:
      image 1 = current Stage08 source/condition RGB (including the existing manual warp/fusion)
      image 2 = aligned grayscale camera-Z depth visualization for the same current camera

    FLUX.2 receives both images through its native multi-reference image-editing input.
    The depth image is intentionally *not* presented as a ControlNet tensor; the prompt explicitly
    states its semantics.  This backend is therefore an ablation-friendly bridge between the
    existing FLUX.1 Depth path and a future WorldMesh-style native appearance-reference path.
    """

    capabilities = {
        "text_to_image": True,
        "depth_control": False,
        "depth_as_reference_image": True,
        "inpainting": False,
        "reference_image": True,
        "multi_reference": True,
        "preserve_mask": False,
        "negative_prompt_native": False,
    }

    def __init__(self, config: Dict[str, Any], auth_config=None):
        self.config = dict(config)
        self.auth_config = dict(auth_config or {})
        self.pipe = None
        self.load_metadata: Dict[str, Any] = {}

    def _dtype(self):
        import torch
        name = str(self.config.get("dtype", "bfloat16")).lower()
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16"}:
            return torch.float16
        return torch.float32

    def _load(self, *, quiet_console: bool = False, suppress_progress_bars: bool = False):
        if self.pipe is not None:
            if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
                self.pipe.set_progress_bar_config(disable=True)
            return
        import torch
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.2 Klein backend requires a Diffusers build exposing Flux2KleinPipeline. "
                "Run: pip install -U 'diffusers>=0.38,<0.40' transformers accelerate"
            ) from exc

        model_id = str(self.config.get("model_id", "black-forest-labs/FLUX.2-klein-4B"))
        local_path = str(self.config.get("local_model_path", "models/flux2-klein-4b"))
        source, is_local = resolve_model_source(
            local_path,
            model_id,
            require_local=bool(self.config.get("require_local_models", False)),
        )
        token = None if is_local else get_token(self.auth_config)
        kwargs = {"torch_dtype": self._dtype()}
        if token:
            kwargs["token"] = token
        if is_local and bool(self.config.get("local_files_only_after_download", True)):
            kwargs["local_files_only"] = True
        self.pipe = Flux2KleinPipeline.from_pretrained(source, **kwargs)
        if bool(self.config.get("cpu_offload", True)) and callable(getattr(self.pipe, "enable_model_cpu_offload", None)):
            self.pipe.enable_model_cpu_offload()
            placement = "model_cpu_offload"
        else:
            self.pipe.to(str(self.config.get("device", "cuda")))
            placement = str(self.config.get("device", "cuda"))
        if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
            self.pipe.set_progress_bar_config(disable=True)
        self.load_metadata = {
            "model_id": model_id,
            "resolved_source": source,
            "source_is_local": bool(is_local),
            "dtype": str(self._dtype()),
            "placement": placement,
            "reference_contract": ["source_rgb", "camera_z_depth_visualization"],
        }

    @staticmethod
    def _role_prompt(base_prompt: str) -> str:
        role = (
            "Two aligned reference images are provided. Image 1 is the current source view and is authoritative "
            "for the exact camera, perspective, geometry, silhouettes, object positions, occlusions, and any reliable "
            "appearance already present in it. Image 2 is the camera-Z depth map of Image 1: brighter pixels are nearer "
            "and darker pixels are farther. Use Image 2 only as geometric/depth evidence; never copy its grayscale values "
            "as material, color, texture, or lighting. "
        )
        return role + " ".join(str(base_prompt).split())

    def generate(self, request: Dict[str, Any]):
        import torch
        self._load(
            quiet_console=bool(request.get("quiet_console", False)),
            suppress_progress_bars=bool(request.get("suppress_backend_progress_bars", False)),
        )
        width = int(request.get("width", self.config.get("width", 1376)))
        height = int(request.get("height", self.config.get("height", 768)))
        size = (width, height)
        source = Image.open(request["init_image_path"]).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        depth, depth_meta = load_depth_control_image(request["depth_image_path"], size, mask_image=None)

        preview = Path(request["control_preview_path"])
        preview.parent.mkdir(parents=True, exist_ok=True)
        depth.save(preview)
        preview.with_suffix(".json").write_text(json.dumps(depth_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        prompt = self._role_prompt(str(request["prompt"]))
        seed = int(request.get("seed", self.config.get("seed", 0)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        steps = int(request.get("num_inference_steps", self.config.get("num_inference_steps", 4)))
        guidance = float(self.config.get("guidance_scale", request.get("guidance_scale", 1.0)))

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "image": [source, depth],
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
        }
        try:
            params = inspect.signature(self.pipe.__call__).parameters
        except Exception:
            params = {}
        max_seq = int(self.config.get("max_sequence_length", 512))
        if "max_sequence_length" in params:
            call_kwargs["max_sequence_length"] = max_seq
        progress_api = _attach_progress(self.pipe, call_kwargs, request.get("progress_callback"), steps)

        output = self.pipe(**call_kwargs)
        result = output.images[0]
        out = Path(request["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "backend": "flux2_klein_4b_dual_reference",
            "output_path": str(out),
            "seed": seed,
            "capabilities": self.capabilities,
            "control_preview_path": str(preview),
            "depth_preprocess": depth_meta,
            "model_loading": self.load_metadata,
            "prompt": prompt,
            "prompt_handling": {
                "mode": "backend_specific_role_aware",
                "image_1_role": "current_source_rgb_authoritative_for_camera_geometry_layout_and_existing_reliable_appearance",
                "image_2_role": "aligned_camera_z_depth_geometry_only_near_white_far_black",
            },
            "progress_callback_api": progress_api,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
