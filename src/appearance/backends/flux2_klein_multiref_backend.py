from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from PIL import Image

from src.appearance.backends.base import SparseDiffusionBackend
from src.appearance.backends.flux1_depth_control_inpaint_backend import _attach_pipeline_progress_callback
from src.appearance.backends.flux_runtime_loader import (
    FLUX2_KLEIN_RUNTIME_COMPONENTS,
    _enable_vae_memory_features,
    _from_pretrained_pipeline_with_dtype,
    _prepare_pipeline_runtime,
    _run_pipeline_with_runtime_guards,
    _validate_pipeline_runtime,
)
from src.appearance.hf_access import get_token, resolve_model_source
from src.appearance.prompt_budget import fit_prompt


class Flux2Klein4BMultiReferenceBackend(SparseDiffusionBackend):
    """FLUX.2 [klein] 4B multi-reference image-editing backend.

    Stage08 request contract:
      - prompt
      - reference_image_paths: ordered native FLUX.2 image references
      - reference_roles: same-length diagnostic role labels
      - output_path

    Stage08 native-reference order is fixed to exactly two images:
      Image 1 = exact target-camera RGB condition after manual-warp fusion,
      Image 2 = pixel-aligned Stage07 camera-Z depth visualization.
    Cross-view RGB evidence is already geometry-aligned into Image 1 and is never
    supplied as a separate raw neighbouring-camera reference. Depth is supplied as
    a generic FLUX.2 image reference (not a dedicated ControlNet).
    Depth Anything remains the independent post-generation geometry validation gate.
    """

    capabilities = {
        "text_to_image": True,
        "image_editing": True,
        "multi_reference": True,
        "depth_control": False,
        "inpainting": False,
        "reference_image": True,
        "preserve_mask": False,
        "quantized_nf4": False,
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

    def _resolve_source(self) -> tuple[str, bool]:
        return resolve_model_source(
            self.config.get("local_model_path", "models/flux2-klein-4b"),
            self.config.get("model_id", "black-forest-labs/FLUX.2-klein-4B"),
            require_local=bool(self.config.get("require_local_models", False)),
        )

    def _load(self, *, quiet_console: bool = False, suppress_progress_bars: bool = False):
        if self.pipe is not None:
            if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
                self.pipe.set_progress_bar_config(disable=True)
            return

        from diffusers import Flux2KleinPipeline

        dtype = self._dtype()
        source, is_local = self._resolve_source()
        token = None if is_local else get_token(self.auth_config)
        local_only = bool(self.config.get("local_files_only_after_download", True) and is_local)

        self.pipe = _from_pretrained_pipeline_with_dtype(
            Flux2KleinPipeline,
            source,
            dtype=dtype,
            token=token,
            local_files_only=local_only,
        )
        runtime_preparation = _prepare_pipeline_runtime(
            self.pipe, dtype, required_components=FLUX2_KLEIN_RUNTIME_COMPONENTS
        )
        vae_memory_features = _enable_vae_memory_features(
            self.pipe,
            slicing=bool(self.config.get("vae_slicing", True)),
            tiling=bool(self.config.get("vae_tiling", True)),
        )
        if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
            self.pipe.set_progress_bar_config(disable=True)
        if self.config.get("cpu_offload", True):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.config.get("device", "cuda"))

        runtime_after_offload = _validate_pipeline_runtime(
            self.pipe, dtype, required_components=FLUX2_KLEIN_RUNTIME_COMPONENTS
        )
        self.load_metadata = {
            "source": source,
            "source_is_local": bool(is_local),
            "local_files_only": bool(local_only),
            "dtype": str(dtype).replace("torch.", ""),
            "runtime_preparation": runtime_preparation,
            "runtime_after_offload": runtime_after_offload,
            "vae_memory_features": vae_memory_features,
            "native_reference_contract": "ordered PIL image list",
        }
        if not quiet_console:
            print(
                "[FLUX2 RUNTIME] "
                f"source={source}; dtype={self.load_metadata['dtype']}; "
                "native multi-reference editing enabled",
                flush=True,
            )

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

    @staticmethod
    def _load_references(paths: List[str], size: tuple[int, int]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for value in paths:
            path = Path(value)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"FLUX.2 reference image is missing: {path}")
            image = Image.open(path).convert("RGB")
            if image.size != size:
                raise RuntimeError(
                    f"FLUX.2 native reference {path} has size {image.size}, expected exact Stage07 raster {size}"
                )
            images.append(image)
        if not images:
            raise RuntimeError("FLUX.2 Stage08 requires at least the target-view geometry reference")
        return images

    def generate(self, request: Dict[str, Any]):
        import torch

        quiet_console = bool(request.get("quiet_console", False))
        suppress_progress_bars = bool(request.get("suppress_backend_progress_bars", False))
        self._load(
            quiet_console=quiet_console,
            suppress_progress_bars=suppress_progress_bars,
        )
        if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
            self.pipe.set_progress_bar_config(disable=True)

        width = int(request.get("width", self.config.get("width", 1376)))
        height = int(request.get("height", self.config.get("height", 768)))
        reference_paths = [str(value) for value in request.get("reference_image_paths", []) if str(value)]
        reference_roles = [str(value) for value in request.get("reference_roles", [])]
        if reference_roles and len(reference_roles) != len(reference_paths):
            raise ValueError("reference_roles must match reference_image_paths length")
        if len(reference_paths) != 2:
            raise ValueError(
                "FLUX.2 Stage08 requires exactly two native references: target-view RGB condition and aligned camera-Z depth"
            )
        references = self._load_references(reference_paths, (width, height))

        max_tokens = int(request.get("max_sequence_length", self.config.get("max_sequence_length", 512)))
        prompt_budget = fit_prompt(
            self._role_prompt(str(request.get("prompt", ""))),
            max_tokens=max_tokens,
            tokenizer=getattr(self.pipe, "tokenizer", None),
        )
        seed = int(request.get("seed", self.config.get("seed", 0)))
        generator_device = str(self.config.get("generator_device", "cpu"))
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        steps = int(request.get("num_inference_steps", self.config.get("num_inference_steps", 4)))
        guidance = float(request.get("guidance_scale", self.config.get("guidance_scale", 1.0)))

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt_budget.text,
            "image": references if len(references) > 1 else references[0],
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "max_sequence_length": max_tokens,
        }
        progress_callback_api = _attach_pipeline_progress_callback(
            self.pipe,
            call_kwargs,
            request.get("progress_callback"),
            total_steps=steps,
        )

        output_path = Path(request["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_preflight = _validate_pipeline_runtime(
            self.pipe, self._dtype(), required_components=FLUX2_KLEIN_RUNTIME_COMPONENTS
        )
        runtime_path = output_path.parent / "flux2_runtime_preflight.json"
        runtime_path.write_text(
            __import__("json").dumps(runtime_preflight, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        preserve_reference_raster = bool(self.config.get("preserve_reference_raster", True))
        original_resize_to_target_area = None
        if preserve_reference_raster:
            image_processor = getattr(self.pipe, "image_processor", None)
            original_resize_to_target_area = getattr(image_processor, "_resize_to_target_area", None)
            if not callable(original_resize_to_target_area):
                raise RuntimeError(
                    "FLUX.2 exact-reference-raster mode requires Diffusers image_processor._resize_to_target_area"
                )
            # Diffusers Flux2Klein 0.39 automatically shrinks reference images whose
            # area exceeds 1024^2. Stage07's 1376x768 raster exceeds that threshold
            # by <1%, and both dimensions are already valid VAE packing multiples.
            # Bypass only that area-based shrink so the geometry-authoritative
            # reference keeps its exact pixel grid. The normal preprocess/crop-to-
            # multiple logic remains active and is a no-op at 1376x768.
            image_processor._resize_to_target_area = lambda image, _target_area: image
        try:
            pipeline_output, runtime_call = _run_pipeline_with_runtime_guards(
                self.pipe,
                call_kwargs,
                dtype=self._dtype(),
                required_components=FLUX2_KLEIN_RUNTIME_COMPONENTS,
            )
        finally:
            if preserve_reference_raster and original_resize_to_target_area is not None:
                self.pipe.image_processor._resize_to_target_area = original_resize_to_target_area
        result = pipeline_output.images[0]
        if result.size != (width, height):
            raise RuntimeError(
                f"FLUX.2 returned {result.size}, expected exact Stage07 camera raster {(width, height)}; "
                "refusing to resize because resampling would change the geometry/pixel contract"
            )
        result.save(output_path)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {
            "backend": "flux2_klein_4b_multiref_16gb",
            "output_path": str(output_path),
            "seed": seed,
            "capabilities": self.capabilities,
            "model_loading": self.load_metadata,
            "runtime_preflight_path": str(runtime_path),
            "runtime_call": runtime_call,
            "prompt": prompt_budget.text,
            "prompt_token_count": prompt_budget.token_count,
            "prompt_original_token_count": prompt_budget.original_token_count,
            "prompt_compressed": prompt_budget.compressed,
            "native_reference_image_paths": reference_paths,
            "native_reference_roles": reference_roles,
            "native_reference_count": len(reference_paths),
            "preserve_reference_raster": preserve_reference_raster,
            "reference_raster_size": [width, height],
            "depth_control": "none; Depth Anything is post-generation validation only",
            "semantic_control": "none",
            "denoising_strength": None,
            "strength_requested_for_compatibility_schedule": request.get("strength"),
            "strength_is_native_parameter": False,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "progress_callback_api": progress_callback_api,
            "backend_progress_bar_suppressed": suppress_progress_bars,
            "quiet_console": quiet_console,
            "width": width,
            "height": height,
        }
