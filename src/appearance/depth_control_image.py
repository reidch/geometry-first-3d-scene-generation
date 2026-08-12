from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _as_single_channel(array: np.ndarray) -> np.ndarray:
    """Return one scalar depth channel without silently using alpha.

    Blender writes the control image as grayscale PNG, but PIL may expose it as
    I;16, I, L, RGB, or RGBA depending on PNG metadata and Pillow version.
    """
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[2] < 1:
        raise ValueError(f"Unsupported depth-control array shape: {array.shape}")

    rgb = array[..., : min(3, array.shape[2])]
    if rgb.shape[2] == 1:
        return rgb[..., 0]

    # Blender's grayscale output should have equal RGB channels. Averaging is
    # robust to tiny codec/rounding differences and deliberately ignores alpha.
    return rgb.astype(np.float32).mean(axis=2)


def _normalise_integer_depth(channel: np.ndarray, source_mode: str) -> Tuple[np.ndarray, str]:
    max_value = float(np.max(channel)) if channel.size else 0.0
    dtype = channel.dtype

    if dtype == np.uint8 and max_value <= 255.0:
        return channel.astype(np.float32) / 255.0, "uint8_div_255"

    if dtype == np.uint16 or "16" in source_mode or max_value > 255.0:
        # Blender compositor output is already normalised to [0, 1] before PNG
        # encoding. Preserve that absolute mapping instead of min/max stretching.
        return channel.astype(np.float32) / 65535.0, "uint16_div_65535"

    info = np.iinfo(dtype)
    return channel.astype(np.float32) / float(info.max), f"integer_div_{info.max}"


def _normalise_float_depth(channel: np.ndarray) -> Tuple[np.ndarray, str]:
    values = channel.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Depth-control image contains no finite values.")

    finite_values = values[finite]
    lo = float(finite_values.min())
    hi = float(finite_values.max())

    if lo >= 0.0 and hi <= 1.0 + 1e-6:
        out = values
        mode = "float_already_0_1"
    elif lo >= 0.0 and hi <= 255.0 + 1e-6:
        out = values / 255.0
        mode = "float_div_255"
    elif lo >= 0.0 and hi <= 65535.0 + 1e-6:
        out = values / 65535.0
        mode = "float_div_65535"
    elif hi > lo:
        out = (values - lo) / (hi - lo)
        mode = "float_minmax_fallback"
    else:
        out = np.zeros_like(values, dtype=np.float32)
        mode = "float_constant"

    out[~finite] = 0.0
    return out, mode


def load_depth_control_image(
    path: str | Path,
    size: Tuple[int, int],
    *,
    mask_image: Image.Image | None = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load Blender's depth-control PNG without destroying 16-bit gradients.

    The previous implementation called ``Image.open(...).convert('RGB')`` on a
    16-bit PNG. Pillow clips values above 255 during that conversion, turning
    almost the entire object into white and making the result look like a mask.
    This function explicitly maps 16-bit values to 8-bit before RGB conversion.
    """
    path = Path(path)
    source = Image.open(path)
    source_mode = source.mode
    raw = np.asarray(source)
    channel = _as_single_channel(raw)

    if np.issubdtype(channel.dtype, np.integer):
        normalised, conversion = _normalise_integer_depth(channel, source_mode)
    elif np.issubdtype(channel.dtype, np.floating):
        normalised, conversion = _normalise_float_depth(channel)
    else:
        raise ValueError(
            f"Unsupported depth-control dtype {channel.dtype} for {path}"
        )

    normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)
    normalised = np.clip(normalised, 0.0, 1.0)
    gray8 = np.rint(normalised * 255.0).astype(np.uint8)
    source_gray = Image.fromarray(gray8, mode="L")
    resized_gray = source_gray.resize(size, Image.Resampling.BILINEAR)

    used = np.asarray(resized_gray, dtype=np.uint8)
    if mask_image is not None:
        mask = np.asarray(
            mask_image.convert("L").resize(size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        ) > 0
    else:
        mask = used > 0

    foreground = used[mask]
    background = used[~mask]
    source_nonzero = gray8[gray8 > 0]

    source_unique = int(np.unique(source_nonzero).size) if source_nonzero.size else 0
    used_unique = int(np.unique(foreground).size) if foreground.size else 0
    foreground_std = float(foreground.std()) if foreground.size else 0.0
    background_max = int(background.max()) if background.size else 0

    if foreground.size == 0 or int(foreground.max()) == 0:
        raise RuntimeError(
            f"Depth control has no nonzero foreground after preprocessing: {path}"
        )

    # Detect the exact 16-bit clipping regression: a rich source becoming an
    # almost-binary used image. Constant/planar depth remains legal.
    if source_unique >= 16 and used_unique <= 4:
        raise RuntimeError(
            "Depth-control gradients collapsed during preprocessing. "
            f"source_unique_levels={source_unique}, used_unique_levels={used_unique}, "
            f"source_mode={source_mode}, conversion={conversion}, path={path}"
        )

    metadata: Dict[str, Any] = {
        "source_path": str(path),
        "source_mode": source_mode,
        "source_dtype": str(raw.dtype),
        "source_shape": list(raw.shape),
        "source_min": float(np.min(channel)) if channel.size else 0.0,
        "source_max": float(np.max(channel)) if channel.size else 0.0,
        "conversion": conversion,
        "source_foreground_unique_levels_8bit": source_unique,
        "used_foreground_unique_levels": used_unique,
        "used_foreground_std": foreground_std,
        "used_background_max": background_max,
        "output_size": [int(size[0]), int(size[1])],
        "output_mode": "RGB",
        "resampling": "bilinear",
    }
    return resized_gray.convert("RGB"), metadata
