from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import importlib
import warnings


def _is_unexpected_keyword(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    quoted = (
        f"unexpected keyword argument '{keyword}'",
        f'unexpected keyword argument "{keyword}"',
        f"got an unexpected keyword argument '{keyword}'",
        f'got an unexpected keyword argument "{keyword}"',
    )
    return any(fragment in message for fragment in quoted)


def _version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _ensure_safe_torch_default_dtype() -> Dict[str, str]:
    """Keep Torch's process-wide default floating dtype at float32.

    Some model-loading stacks temporarily call ``torch.set_default_dtype`` while
    materialising low-precision weights.  Older Transformers causal-mask helpers
    create ``torch.full`` tensors without an explicit dtype.  If a loader leaks a
    float16 default, a BF16/FP32 ``torch.finfo(...).min`` fill value overflows
    before CLIP even reaches its first attention layer.

    The project never relies on a low-precision *global* default; individual
    modules and tensors are explicitly typed.  Resetting the global default to
    float32 is therefore both safe and required for deterministic runtime
    behaviour.
    """
    import torch

    before = torch.get_default_dtype()
    if before != torch.float32:
        torch.set_default_dtype(torch.float32)
    after = torch.get_default_dtype()
    if after != torch.float32:
        raise RuntimeError(
            f"Unable to restore safe Torch default dtype: before={before}, after={after}."
        )
    return {
        "before": str(before),
        "after": str(after),
        "changed": str(before != after).lower(),
    }


def _from_pretrained_with_dtype(factory, source: str, *, dtype, **kwargs):
    """Load a model component with a narrow API compatibility fallback.

    The current Transformers/Diffusers component loaders prefer ``dtype``.  Some
    older releases explicitly reject it and require ``torch_dtype``.  We only
    fall back for that exact error and reset Torch's global default dtype both
    before and after loading so a third-party loader cannot leak FP16 globally.
    """
    _ensure_safe_torch_default_dtype()
    try:
        try:
            return factory.from_pretrained(source, dtype=dtype, **kwargs)
        except TypeError as exc:
            if not _is_unexpected_keyword(exc, "dtype"):
                raise
            return factory.from_pretrained(source, torch_dtype=dtype, **kwargs)
    finally:
        _ensure_safe_torch_default_dtype()


def _from_pretrained_pipeline_with_dtype(factory, source: str, *, dtype, **kwargs):
    """Load a Diffusers pipeline through the keyword this compatibility range consumes.

    The user's installed FluxControlInpaintPipeline accepts arbitrary ``dtype``
    kwargs but ignores them, while still consuming ``torch_dtype``.  Silent
    ignore is worse than a deprecation warning because base CLIP/VAE components
    then keep checkpoint precision.  Use ``torch_dtype`` first, verify actual
    component precision afterwards, and only fall back when the keyword is
    explicitly rejected.
    """
    _ensure_safe_torch_default_dtype()
    try:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
                )
                return factory.from_pretrained(source, torch_dtype=dtype, **kwargs)
        except TypeError as exc:
            if not _is_unexpected_keyword(exc, "torch_dtype"):
                raise
            return factory.from_pretrained(source, dtype=dtype, **kwargs)
    finally:
        _ensure_safe_torch_default_dtype()


def _floating_dtype(module) -> Any:
    if module is None:
        return None
    try:
        for parameter in module.parameters():
            if getattr(parameter, "is_floating_point", lambda: False)():
                return parameter.dtype
    except Exception:
        pass
    try:
        for buffer in module.buffers():
            if getattr(buffer, "is_floating_point", lambda: False)():
                return buffer.dtype
    except Exception:
        pass
    return getattr(module, "dtype", None)


def _is_quantized_module(module) -> bool:
    if module is None:
        return False
    for attr in (
        "is_loaded_in_4bit",
        "is_loaded_in_8bit",
        "is_quantized",
    ):
        try:
            if bool(getattr(module, attr, False)):
                return True
        except Exception:
            pass
    method = str(getattr(module, "quantization_method", "") or "").lower()
    if method and method not in {"none", "false"}:
        return True
    config = getattr(module, "config", None)
    quantization_config = getattr(config, "quantization_config", None)
    return quantization_config is not None


def _align_component_dtype(pipe, component_name: str, dtype, *, required: bool) -> Dict[str, str]:
    module = getattr(pipe, component_name, None)
    if module is None:
        if required:
            raise RuntimeError(
                f"Flux pipeline has no required component {component_name!r}; cannot validate precision."
            )
        return {
            "component": component_name,
            "status": "missing_optional",
            "before": "None",
            "requested": str(dtype),
            "after": "None",
            "quantized": "false",
        }

    quantized = _is_quantized_module(module)
    before = _floating_dtype(module)

    # Quantized transformer/T5 modules may reject .to(dtype=...) because their
    # storage format is controlled by the quantizer.  Their floating compute
    # dtype is inspected and reported, but only ordinary base components are
    # forcibly cast.
    if not quantized and before != dtype:
        try:
            moved = module.to(dtype=dtype)
            if moved is not None and moved is not module:
                setattr(pipe, component_name, moved)
                module = moved
        except Exception as exc:
            raise RuntimeError(
                f"Unable to align Flux component {component_name} from {before} to {dtype}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    after = _floating_dtype(module)
    if required and not quantized and after is not None and after != dtype:
        raise RuntimeError(
            f"Flux component dtype alignment failed for {component_name}: "
            f"requested={dtype}, actual={after}."
        )

    return {
        "component": component_name,
        "before": str(before),
        "requested": str(dtype),
        "after": str(after),
        "quantized": str(quantized).lower(),
        "status": (
            "quantized_compute_dtype_preserved"
            if quantized
            else ("aligned" if before != after else "already_aligned")
        ),
    }


def _align_pipeline_vae_dtype(pipe, dtype) -> Dict[str, str]:
    """Backward-compatible VAE-only wrapper retained for tests and callers."""
    report = _align_component_dtype(pipe, "vae", dtype, required=True)
    return {
        "before": report["before"],
        "requested": report["requested"],
        "after": report["after"],
        "status": report["status"],
    }


FLUX1_RUNTIME_COMPONENTS: Tuple[str, ...] = (
    "text_encoder",
    "text_encoder_2",
    "vae",
    "transformer",
)

FLUX2_KLEIN_RUNTIME_COMPONENTS: Tuple[str, ...] = (
    "text_encoder",
    "vae",
    "transformer",
)


def _normalise_required_components(required_components: Iterable[str] | None) -> Tuple[str, ...]:
    components = tuple(required_components or FLUX1_RUNTIME_COMPONENTS)
    if not components:
        raise ValueError("Flux runtime component profile cannot be empty.")
    if len(set(components)) != len(components):
        raise ValueError(f"Flux runtime component profile contains duplicates: {components!r}")
    return components


def _prepare_pipeline_runtime(
    pipe,
    dtype,
    *,
    required_components: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Align every required non-quantized Flux component before Accelerate hooks.

    FLUX.1 uses two text encoders, while FLUX.2 Klein uses a single Qwen text
    encoder.  Callers must select the profile that belongs to their concrete
    pipeline instead of weakening validation globally.
    """
    component_names = _normalise_required_components(required_components)
    default_dtype = _ensure_safe_torch_default_dtype()
    components = {
        name: _align_component_dtype(pipe, name, dtype, required=True)
        for name in component_names
    }
    return {
        "default_dtype": default_dtype,
        "required_components": list(component_names),
        "components": components,
        "versions": {
            "torch": _version_of("torch"),
            "diffusers": _version_of("diffusers"),
            "transformers": _version_of("transformers"),
            "accelerate": _version_of("accelerate"),
        },
    }


def _validate_pipeline_runtime(
    pipe,
    dtype,
    *,
    required_components: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Fail before inference when the active pipeline is internally inconsistent."""
    component_names = _normalise_required_components(required_components)
    default_dtype = _ensure_safe_torch_default_dtype()
    components: Dict[str, Dict[str, str]] = {}
    for name in component_names:
        module = getattr(pipe, name, None)
        if module is None:
            raise RuntimeError(f"Flux pipeline is missing required component {name!r}.")
        actual = _floating_dtype(module)
        quantized = _is_quantized_module(module)
        if not quantized and actual is not None and actual != dtype:
            raise RuntimeError(
                f"Flux runtime precision mismatch before inference: component={name}, "
                f"expected={dtype}, actual={actual}."
            )
        components[name] = {
            "dtype": str(actual),
            "quantized": str(quantized).lower(),
            "device": str(getattr(module, "device", "managed_by_accelerate")),
        }
    return {
        "default_dtype": default_dtype,
        "required_components": list(component_names),
        "components": components,
        "versions": {
            "torch": _version_of("torch"),
            "diffusers": _version_of("diffusers"),
            "transformers": _version_of("transformers"),
            "accelerate": _version_of("accelerate"),
        },
        "status": "ready",
    }


def _run_pipeline_with_runtime_guards(
    pipe,
    call_kwargs: Dict[str, Any],
    *,
    dtype,
    required_components: Iterable[str] | None = None,
):
    """Execute one Flux call with stable global/default and component precision.

    This wrapper intentionally lives in the runtime-only module so future
    compatibility fixes do not invalidate expensive Stage06/08 output hashes.
    """
    report = _validate_pipeline_runtime(
        pipe, dtype, required_components=required_components
    )
    _ensure_safe_torch_default_dtype()
    try:
        return pipe(**call_kwargs), report
    except RuntimeError as exc:
        message = str(exc)
        known_precision_fragments = (
            "cannot be converted to type c10::Half without overflow",
            "Input type (c10::BFloat16) and bias type (c10::Half)",
            "Input type (torch.cuda.BFloat16Tensor) and weight type",
        )
        if any(fragment in message for fragment in known_precision_fragments):
            raise RuntimeError(
                "Flux inference failed inside a mixed/default precision path even after runtime "
                f"hardening. Runtime report={report}. Original error: {message}"
            ) from exc
        raise
    finally:
        # A third-party forward hook must not leak a low-precision global default
        # into the next object/camera pass.
        _ensure_safe_torch_default_dtype()


def _enable_vae_memory_features(pipe, *, slicing: bool, tiling: bool) -> Dict[str, str]:
    """Enable VAE memory controls through the non-deprecated API when available."""
    result: Dict[str, str] = {}
    vae = getattr(pipe, "vae", None)
    for enabled, feature, vae_method, legacy_method in (
        (slicing, "vae_slicing", "enable_slicing", "enable_vae_slicing"),
        (tiling, "vae_tiling", "enable_tiling", "enable_vae_tiling"),
    ):
        if not enabled:
            result[feature] = "disabled"
            continue
        try:
            if vae is not None and callable(getattr(vae, vae_method, None)):
                getattr(vae, vae_method)()
                result[feature] = f"pipe.vae.{vae_method}"
            elif callable(getattr(pipe, legacy_method, None)):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    getattr(pipe, legacy_method)()
                result[feature] = f"pipe.{legacy_method}_legacy_fallback"
            else:
                result[feature] = "unsupported"
        except Exception as exc:
            result[feature] = f"failed:{type(exc).__name__}:{exc}"
    return result
