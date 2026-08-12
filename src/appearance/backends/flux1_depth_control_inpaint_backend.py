from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from PIL import Image

from src.appearance.backends.base import SparseDiffusionBackend
from src.appearance.depth_control_image import load_depth_control_image
from src.appearance.hf_access import get_token, resolve_model_source, verify_hf_access
from src.appearance.prompt_budget import fit_prompt


from src.appearance.backends.flux_runtime_loader import (
    _enable_vae_memory_features,
    _from_pretrained_pipeline_with_dtype,
    _from_pretrained_with_dtype,
    _prepare_pipeline_runtime,
    _run_pipeline_with_runtime_guards,
    _validate_pipeline_runtime,
)


def _attach_pipeline_progress_callback(
    pipe: Any,
    call_kwargs: Dict[str, Any],
    callback: Callable[[int, int], None] | None,
    *,
    total_steps: int,
) -> str:
    """Attach one project-owned progress callback across Diffusers API generations."""
    if not callable(callback):
        return "disabled"
    try:
        import inspect

        parameters = inspect.signature(pipe.__call__).parameters
    except Exception:
        parameters = {}

    total = max(int(total_steps), 1)
    if "callback_on_step_end" in parameters:
        def callback_on_step_end(_pipe, step, _timestep, callback_kwargs):
            callback(min(int(step) + 1, total), total)
            return callback_kwargs

        call_kwargs["callback_on_step_end"] = callback_on_step_end
        return "callback_on_step_end"

    if "callback" in parameters:
        def legacy_callback(step, _timestep, _latents):
            callback(min(int(step) + 1, total), total)

        call_kwargs["callback"] = legacy_callback
        if "callback_steps" in parameters:
            call_kwargs["callback_steps"] = 1
        return "legacy_callback"

    return "unsupported"


class Flux1DepthControlInpaintNF416GBBackend(SparseDiffusionBackend):
    """FLUX.1 Depth [dev] inpainting backend for pre-physics object/surface generation passes.

    Active request contract:
      - prompt / negative_prompt
      - init_image_path      : current textured render used as RGB reference
      - generation_mask_path : white = active semantic region to repaint
      - depth_image_path     : standard PNG depth condition
      - output_path

    Implementation notes:
      - Uses diffusers.FluxControlInpaintPipeline, which accepts image + mask +
        depth/canny control_image.
      - Uses the official FLUX.1-Depth-dev pipeline as the base, but overrides the
        transformer and T5 text encoder with the NF4 checkpoint so a 16GB GPU can
        run it with CPU offload.
      - FLUX Control Inpaint does not expose a normal negative_prompt argument in
        diffusers; therefore negative terms are logged and optionally converted
        into a short positive "avoid" guard sentence.
    """

    capabilities = {
        "text_to_image": False,
        "depth_control": True,
        "inpainting": True,
        "reference_image": True,
        "preserve_mask": True,
        "quantized_nf4": True,
        "negative_prompt_native": False,
    }

    def __init__(self, config: Dict[str, Any], auth_config=None):
        self.config = config
        self.auth_config = auth_config or {}
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

    def _resolve_sources(self) -> Tuple[str, bool, str, bool]:
        require_local = bool(self.config.get("require_local_models", False))
        base_source, base_is_local = resolve_model_source(
            self.config.get("local_base_path", "models/flux1-depth-dev-base-lite"),
            self.config["base_model_id"],
            require_local=require_local,
        )
        nf4_source, nf4_is_local = resolve_model_source(
            self.config.get("local_nf4_path", "models/flux1-depth-dev-nf4"),
            self.config["quantized_model_id"],
            require_local=require_local,
        )
        return base_source, base_is_local, nf4_source, nf4_is_local

    def _load(self, *, quiet_console: bool = False, suppress_progress_bars: bool = False):
        if self.pipe is not None:
            if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
                self.pipe.set_progress_bar_config(disable=True)
            return

        import torch
        from diffusers import FluxControlInpaintPipeline
        from diffusers.models.transformers import FluxTransformer2DModel
        from transformers import T5EncoderModel

        dtype = self._dtype()
        base_source, base_is_local, nf4_source, nf4_is_local = self._resolve_sources()

        token = None
        if not (base_is_local and nf4_is_local):
            if self.config.get("preflight_check", True):
                verify_hf_access(
                    [self.config["base_model_id"], self.config["quantized_model_id"]],
                    self.auth_config,
                )
            token = get_token(self.auth_config)

        local_only = bool(
            self.config.get("local_files_only_after_download", True)
            and base_is_local
            and nf4_is_local
        )

        transformer = _from_pretrained_with_dtype(
            FluxTransformer2DModel,
            nf4_source,
            dtype=dtype,
            subfolder="transformer",
            token=token,
            local_files_only=local_only,
        )
        text_encoder_2 = _from_pretrained_with_dtype(
            T5EncoderModel,
            nf4_source,
            dtype=dtype,
            subfolder="text_encoder_2",
            token=token,
            local_files_only=local_only,
        )

        # Passing the two large quantized components directly prevents diffusers
        # from first loading the full BF16 transformer/T5 and then replacing them.
        self.pipe = _from_pretrained_pipeline_with_dtype(
            FluxControlInpaintPipeline,
            base_source,
            dtype=dtype,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            token=token,
            local_files_only=local_only,
        )

        runtime_preparation = _prepare_pipeline_runtime(self.pipe, dtype)
        if suppress_progress_bars and callable(getattr(self.pipe, "set_progress_bar_config", None)):
            self.pipe.set_progress_bar_config(disable=True)

        vae_memory_features = _enable_vae_memory_features(
            self.pipe,
            slicing=bool(self.config.get("vae_slicing", True)),
            tiling=bool(self.config.get("vae_tiling", True)),
        )

        if self.config.get("cpu_offload", True):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.config.get("device", "cuda"))

        runtime_after_offload = _validate_pipeline_runtime(self.pipe, dtype)
        component_summary = ", ".join(
            f"{name}={record.get('dtype')}"
            + ("(quantized)" if record.get("quantized") == "true" else "")
            for name, record in runtime_after_offload.get("components", {}).items()
        )
        versions = runtime_after_offload.get("versions", {})
        if not quiet_console:
            print(
                "[FLUX RUNTIME] "
                f"target={str(dtype).replace('torch.', '')}; "
                f"default={runtime_after_offload['default_dtype']['after']}; "
                f"{component_summary}; "
                f"torch={versions.get('torch')} diffusers={versions.get('diffusers')} "
                f"transformers={versions.get('transformers')} accelerate={versions.get('accelerate')}"
            )

        # Optional: keep memory fragmentation under control between object passes.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.load_metadata = {
            "base_source": base_source,
            "nf4_source": nf4_source,
            "base_is_local": base_is_local,
            "nf4_is_local": nf4_is_local,
            "local_files_only": local_only,
            "dtype": str(dtype).replace("torch.", ""),
            "large_components": "transformer and text_encoder_2 loaded from NF4 repo",
            "vae_memory_features": vae_memory_features,
            "runtime_preparation": runtime_preparation,
            "runtime_after_offload": runtime_after_offload,
            "component_dtype_policy": (
                "Torch global default is forced to float32; ordinary CLIP/VAE/T5 components "
                "are aligned to the configured inference dtype before Accelerate hooks; "
                "quantized modules preserve quantizer-managed storage/compute dtype."
            ),
        }

    def _prepare_images(self, request: Dict[str, Any]):
        width = int(request.get("width", self.config.get("width", 1024)))
        height = int(request.get("height", self.config.get("height", 576)))
        size = (width, height)

        init = Image.open(request["init_image_path"]).convert("RGB").resize(
            size, Image.Resampling.LANCZOS
        )
        mask = Image.open(request["generation_mask_path"]).convert("L").resize(
            size, Image.Resampling.NEAREST
        )

        diagnostics_path = None
        if request.get("control_preview_path"):
            diagnostics_path = str(Path(request["control_preview_path"]).with_suffix(".json"))

        if not request.get("depth_image_path"):
            raise RuntimeError(
                "FLUX depth input must be a standard PNG via depth_image_path. "
                "Legacy floating-image depth_path inputs are no longer supported; rerun the producing stage."
            )
        # Blender already generated a normalized 16-bit PNG. Never call
        # convert('RGB') directly on it because Pillow can clip 16-bit values.
        depth, depth_meta = load_depth_control_image(
            request["depth_image_path"],
            size,
            mask_image=mask,
        )

        histogram = depth.convert("L").histogram()
        unique_levels = sum(1 for count in histogram if count > 0)
        foreground_levels = sum(
            1 for value, count in enumerate(histogram) if value > 0 and count > 0
        )
        if foreground_levels == 0:
            raise RuntimeError(
                "Depth control contains no valid foreground pixels before FLUX inference. "
                f"depth_image_path={request.get('depth_image_path')}."
            )

        request["_depth_unique_levels"] = unique_levels
        request["_depth_foreground_unique_levels"] = foreground_levels
        request["_depth_diagnostics_path"] = diagnostics_path
        request["_depth_preprocess"] = depth_meta
        return width, height, init, mask, depth

    def _flux_prompt(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        semantic_class = str(request.get("semantic_class") or "")
        object_name = str(request.get("object_name") or "active object")
        prompt = " ".join(str(request["prompt"]).split()).strip()
        negative = str(request.get("negative_prompt") or "").strip()
        policy = dict(request.get("region_policy", {}))
        clauses = []
        if bool(policy.get("masked_object_only", True)):
            clauses.append(str(self.config.get("target_region_guard", "Edit only the white mask; preserve everything outside it.")))
        if bool(policy.get("preserve_geometry", True)):
            clauses.append("Preserve geometry, placement, silhouette, and camera perspective.")
        if bool(policy.get("continuous_surface", False)):
            clauses.append("Create one continuous surface over the active region without isolated object imagery.")
        avoid_clause = ""
        if self.config.get("append_negative_as_avoid_clause", False) and negative:
            existing = prompt.lower()
            terms = []
            for item in negative.split(","):
                term = item.strip()
                if term and term.lower() not in existing and term.lower() not in {value.lower() for value in terms}:
                    terms.append(term)
                if len(terms) >= 5:
                    break
            if terms:
                avoid_clause = "No " + ", ".join(terms) + "."
        full_prompt = " ".join(value for value in [prompt, *clauses, avoid_clause] if value)
        return full_prompt, {
            "object_name": object_name,
            "semantic_class": semantic_class,
            "region_policy": policy,
            "negative_prompt_was_native": False,
            "negative_prompt_converted_to_avoid_clause": bool(avoid_clause),
        }

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
        width, height, init, mask, depth = self._prepare_images(request)

        preview = Path(request["control_preview_path"])
        preview.parent.mkdir(parents=True, exist_ok=True)
        depth.save(preview)
        preview.with_suffix(".json").write_text(
            __import__("json").dumps(
                request.get("_depth_preprocess", {}),
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        seed = int(request.get("seed", self.config.get("seed", 0)))
        generator = torch.Generator(device="cpu").manual_seed(seed)

        full_prompt, prompt_meta = self._flux_prompt(request)
        clip_tokenizer = getattr(self.pipe, "tokenizer", None)
        t5_tokenizer = getattr(self.pipe, "tokenizer_2", None)
        clip_max_tokens = int(request.get("clip_max_tokens", self.config.get("clip_max_tokens", 75)))
        t5_max_tokens = int(request.get("max_sequence_length", self.config.get("max_sequence_length", 256)))

        # FLUX uses CLIP (77-token hard limit) and T5. The old code budgeted only
        # with T5, so the same long string was silently truncated by CLIP for every
        # object. Give CLIP a concise prompt and T5 the full, separately budgeted one.
        clip_budget = fit_prompt(
            full_prompt,
            max_tokens=clip_max_tokens,
            tokenizer=clip_tokenizer,
        )
        t5_budget = fit_prompt(
            full_prompt,
            max_tokens=t5_max_tokens,
            tokenizer=t5_tokenizer,
        )
        if clip_budget.token_count > clip_max_tokens:
            raise RuntimeError(
                f"CLIP prompt budget failed: {clip_budget.token_count}>{clip_max_tokens}"
            )

        call_kwargs = {
            "prompt": clip_budget.text,
            "image": init,
            "mask_image": mask,
            "control_image": depth,
            "strength": float(request.get("strength", self.config.get("strength", 0.82))),
            "guidance_scale": float(request.get("guidance_scale", self.config.get("guidance_scale", 10.0))),
            "num_inference_steps": int(
                request.get("num_inference_steps", self.config.get("num_inference_steps", 30))
            ),
            "generator": generator,
            "width": width,
            "height": height,
            "max_sequence_length": t5_max_tokens,
        }

        # Modern FluxControlInpaintPipeline accepts prompt_2 for T5. Fall back
        # to the CLIP-safe prompt on older diffusers builds rather than triggering
        # the 77-token CLIP truncation warning.
        try:
            import inspect
            signature = inspect.signature(self.pipe.__call__)
            supports_prompt_2 = "prompt_2" in signature.parameters
        except Exception:
            supports_prompt_2 = False
        if supports_prompt_2:
            call_kwargs["prompt_2"] = t5_budget.text

        if not quiet_console:
            print(
                f"[PROMPT] CLIP {clip_budget.token_count}/{clip_max_tokens} tokens; "
                f"T5 {t5_budget.token_count}/{t5_max_tokens} tokens"
            )
        out = Path(request["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        runtime_preflight = _validate_pipeline_runtime(self.pipe, self._dtype())
        runtime_path = out.parent / "flux_runtime_preflight.json"
        runtime_path.write_text(
            __import__("json").dumps(runtime_preflight, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        progress_callback_api = _attach_pipeline_progress_callback(
            self.pipe,
            call_kwargs,
            request.get("progress_callback"),
            total_steps=int(call_kwargs["num_inference_steps"]),
        )
        pipeline_output, runtime_call = _run_pipeline_with_runtime_guards(
            self.pipe, call_kwargs, dtype=self._dtype()
        )
        result = pipeline_output.images[0]
        result.save(out)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {
            "backend": "flux1_depth_control_inpaint_nf4_16gb",
            "output_path": str(out),
            "seed": seed,
            "capabilities": self.capabilities,
            "control_preview_path": str(preview),
            "depth_diagnostics_path": request.get("_depth_diagnostics_path"),
            "depth_preprocess": request.get("_depth_preprocess", {}),
            "depth_unique_levels": int(request.get("_depth_unique_levels", 0)),
            "depth_foreground_unique_levels": int(request.get("_depth_foreground_unique_levels", 0)),
            "model_loading": self.load_metadata,
            "runtime_preflight_path": str(runtime_path),
            "runtime_call": runtime_call,
            "prompt": clip_budget.text,
            "prompt_2": t5_budget.text if supports_prompt_2 else clip_budget.text,
            "clip_prompt_token_count": clip_budget.token_count,
            "clip_prompt_original_token_count": clip_budget.original_token_count,
            "clip_prompt_compressed": clip_budget.compressed,
            "t5_prompt_token_count": t5_budget.token_count,
            "t5_prompt_original_token_count": t5_budget.original_token_count,
            "t5_prompt_compressed": t5_budget.compressed,
            "prompt_2_supported": supports_prompt_2,
            "negative_prompt": request.get("negative_prompt") or "",
            "prompt_handling": prompt_meta,
            "denoising_strength": float(call_kwargs["strength"]),
            "rgb_reference_strength": float(
                request.get("rgb_reference_strength", 1.0 - float(call_kwargs["strength"]))
            ),
            "depth_control": "FluxControlInpaintPipeline control_image",
            "control_strength_requested": float(request.get("control_strength", 1.0)),
            "control_strength_native_parameter": False,
            "guidance_scale": float(call_kwargs["guidance_scale"]),
            "num_inference_steps": int(call_kwargs["num_inference_steps"]),
            "progress_callback_api": progress_callback_api,
            "backend_progress_bar_suppressed": suppress_progress_bars,
            "quiet_console": quiet_console,
            "width": width,
            "height": height,
        }
