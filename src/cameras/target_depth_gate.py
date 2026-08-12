from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from src.cameras.depth_geometry import depth_convention


def bounding_sphere_radius_from_aabb(aabb: Mapping[str, Any]) -> float:
    extent = np.asarray(aabb.get("extent"), dtype=np.float64)
    if extent.shape != (3,) or not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
        minimum = np.asarray(aabb.get("minimum"), dtype=np.float64)
        maximum = np.asarray(aabb.get("maximum"), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("Target AABB must provide extent or finite minimum/maximum vectors")
        extent = maximum - minimum
    radius = 0.5 * float(np.linalg.norm(extent))
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError(f"Target AABB produced an invalid bounding-sphere radius: {radius}")
    return radius


def _normalised_depth_png(path: str | Path) -> np.ndarray:
    source = Image.open(path)
    array = np.asarray(source)
    if array.ndim == 3:
        array = array[..., : min(array.shape[2], 3)].astype(np.float32).mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Unsupported target-depth image shape: {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        maximum = 65535.0 if array.dtype == np.uint16 or float(array.max(initial=0)) > 255.0 else 255.0
        values = array.astype(np.float32) / maximum
    elif np.issubdtype(array.dtype, np.floating):
        values = array.astype(np.float32)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max()) > 1.0 + 1e-6:
            maximum = 65535.0 if float(finite.max()) > 255.0 else 255.0
            values = values / maximum
    else:
        raise ValueError(f"Unsupported target-depth image dtype: {array.dtype}")
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def decode_linear_target_depth(path: str | Path, encoding: Mapping[str, Any]) -> np.ndarray:
    """Decode Stage07's near-bright 16-bit PNG back to positive camera-Z depth."""
    if depth_convention(encoding) != "camera_z":
        raise ValueError("Stage07 target-depth gate requires camera-Z depth")
    near = float(encoding["near"])
    far = float(encoding["far"])
    valid_min_gray = float(encoding.get("valid_min_gray", 24))
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        raise ValueError(f"Invalid target-depth encoding bounds: near={near}, far={far}")
    floor = min(max(valid_min_gray / 255.0, 0.0), 1.0 - 1e-8)
    encoded = _normalised_depth_png(path)
    valid = encoded > 0.0
    inverted = np.clip((encoded - floor) / max(1.0 - floor, 1e-8), 0.0, 1.0)
    depth = near + (1.0 - inverted) * (far - near)
    depth = depth.astype(np.float32)
    depth[~valid] = np.nan
    return depth


def normalise_target_depth_config(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(raw or {})
    result = {
        "enabled": bool(cfg.get("enabled", True)),
        "normalization": str(cfg.get("normalization", "target_bounding_sphere_radius")),
        "minimum_good_depth": float(cfg.get("minimum_good_depth", 0.35)),
        "maximum_good_depth": float(cfg.get("maximum_good_depth", 12.0)),
        "maximum_bad_depth_fraction": float(cfg.get("maximum_bad_depth_fraction", 0.20)),
    }
    if result["normalization"] != "target_bounding_sphere_radius":
        raise ValueError(
            "camera_candidate_validation.target_depth_distribution.normalization "
            "must be 'target_bounding_sphere_radius'"
        )
    if not 0.0 <= result["maximum_bad_depth_fraction"] <= 1.0:
        raise ValueError("maximum_bad_depth_fraction must be within [0, 1]")
    if not 0.0 < result["minimum_good_depth"] < result["maximum_good_depth"]:
        raise ValueError("Target depth interval must satisfy 0 < minimum_good_depth < maximum_good_depth")
    return result


def evaluate_target_depth_distribution(
    *,
    linear_depth: np.ndarray,
    target_mask: np.ndarray,
    target_bounding_sphere_radius: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = normalise_target_depth_config(config)
    mask = np.asarray(target_mask, dtype=bool)
    depth = np.asarray(linear_depth, dtype=np.float32)
    if depth.shape != mask.shape:
        raise ValueError(f"Target depth/mask shape mismatch: {depth.shape} != {mask.shape}")

    target_pixel_count = int(np.count_nonzero(mask))
    radius = max(float(target_bounding_sphere_radius), 1e-8)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_values = depth[valid]
    normalized = valid_values / radius if valid_values.size else np.empty((0,), dtype=np.float32)

    too_near = normalized < float(cfg["minimum_good_depth"])
    too_far = normalized > float(cfg["maximum_good_depth"])
    invalid_count = int(target_pixel_count - valid_values.size)
    near_count = int(np.count_nonzero(too_near))
    far_count = int(np.count_nonzero(too_far))
    bad_count = invalid_count + near_count + far_count
    bad_fraction = float(bad_count / max(target_pixel_count, 1))
    accepted = bool(
        cfg["enabled"]
        and target_pixel_count > 0
        and bad_fraction <= float(cfg["maximum_bad_depth_fraction"])
    )
    if not cfg["enabled"]:
        accepted = target_pixel_count > 0

    percentiles = {}
    if normalized.size:
        p05, p50, p95 = np.percentile(normalized, [5.0, 50.0, 95.0]).tolist()
        percentiles = {
            "normalized_depth_p05": float(p05),
            "normalized_depth_p50": float(p50),
            "normalized_depth_p95": float(p95),
            "normalized_depth_min": float(normalized.min()),
            "normalized_depth_max": float(normalized.max()),
        }

    if target_pixel_count <= 0:
        reason = "target_has_no_isolated_pixels"
    elif bad_fraction > float(cfg["maximum_bad_depth_fraction"]):
        reason = "target_depth_out_of_good_range_fraction_above_threshold"
    else:
        reason = "target_depth_distribution_passed"

    return {
        "accepted": accepted,
        "reason": reason,
        "enabled": bool(cfg["enabled"]),
        "normalization": cfg["normalization"],
        "target_bounding_sphere_radius": radius,
        "minimum_good_depth": float(cfg["minimum_good_depth"]),
        "maximum_good_depth": float(cfg["maximum_good_depth"]),
        "maximum_bad_depth_fraction": float(cfg["maximum_bad_depth_fraction"]),
        "target_pixel_count": target_pixel_count,
        "valid_depth_pixel_count": int(valid_values.size),
        "invalid_depth_pixel_count": invalid_count,
        "too_near_pixel_count": near_count,
        "too_far_pixel_count": far_count,
        "bad_depth_pixel_count": bad_count,
        "bad_depth_fraction": bad_fraction,
        **percentiles,
    }
