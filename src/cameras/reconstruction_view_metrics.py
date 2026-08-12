from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from src.appearance.triangle_id_map import load_triangle_id_map
from src.cameras.depth_geometry import depth_convention


def decode_metric_depth(path: str | Path, encoding: Mapping[str, Any]) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    maximum = 65535.0 if image.max(initial=0.0) > 255.0 else 255.0
    encoded = image / maximum
    near = float(encoding["near"])
    far = float(encoding["far"])
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        raise ValueError(f"Invalid Stage07 depth encoding: near={near}, far={far}")
    floor = float(np.clip(float(encoding.get("valid_min_gray", 24)) / 255.0, 0.0, 1.0 - 1e-8))
    valid = encoded > 0.0
    near_bright = np.clip((encoded - floor) / max(1.0 - floor, 1e-8), 0.0, 1.0)
    depth = np.zeros_like(encoded, dtype=np.float32)
    depth[valid] = near + (1.0 - near_bright[valid]) * (far - near)
    return depth


def robust_log_depth_range(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> Dict[str, float]:
    values = np.asarray(depth, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0.0)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    values = values[mask]
    if values.size < 16:
        return {
            "valid_depth_pixel_count": int(values.size),
            "q10_m": 0.0,
            "q50_m": 0.0,
            "q90_m": 0.0,
            "robust_log_depth_range": 0.0,
            "depth_ratio_q90_q10": 1.0,
        }
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    ratio = max(float(q90 / max(q10, 1e-8)), 1.0)
    return {
        "valid_depth_pixel_count": int(values.size),
        "q10_m": float(q10),
        "q50_m": float(q50),
        "q90_m": float(q90),
        "robust_log_depth_range": float(math.log(ratio)),
        "depth_ratio_q90_q10": ratio,
    }


def _range_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [dict(item) for item in manifest.get("scene_triangle_ranges", [])]
    if records:
        return sorted(records, key=lambda item: int(item["scene_triangle_start"]))
    return [
        {
            "scene_triangle_start": int(item["scene_triangle_id"]),
            "scene_triangle_end_exclusive": int(item["scene_triangle_id"]) + 1,
            "semantic_owner_id": str(item["semantic_owner_id"]),
        }
        for item in manifest.get("triangles", [])
    ]


def semantic_pixel_counts(triangle_map: np.ndarray, manifest: Mapping[str, Any]) -> Dict[str, int]:
    valid_ids, counts = np.unique(triangle_map[triangle_map >= 0], return_counts=True)
    records = _range_records(manifest)
    starts = np.asarray([int(item["scene_triangle_start"]) for item in records], dtype=np.int64)
    ends = np.asarray([int(item["scene_triangle_end_exclusive"]) for item in records], dtype=np.int64)
    result: Dict[str, int] = {}
    if not records:
        return result
    positions = np.searchsorted(starts, valid_ids, side="right") - 1
    for scene_id, count, position in zip(valid_ids.tolist(), counts.tolist(), positions.tolist()):
        if position < 0 or int(scene_id) >= int(ends[position]):
            continue
        owner = str(records[position]["semantic_owner_id"])
        result[owner] = result.get(owner, 0) + int(count)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return float(numerator / denominator)


def _unit_threshold(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"Stage07 candidate_scoring.{key} must be within [0, 1], got {value}")
    return value


def _nonnegative_threshold(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"Stage07 candidate_scoring.{key} must be finite and non-negative, got {value}")
    return value


def _largest_fraction(
    owners: list[str], semantic_screen_fractions: Mapping[str, float]
) -> tuple[str | None, float]:
    visible = [
        (owner, float(semantic_screen_fractions.get(owner, 0.0)))
        for owner in owners
        if float(semantic_screen_fractions.get(owner, 0.0)) > 0.0
    ]
    if not visible:
        return None, 0.0
    return max(visible, key=lambda item: (item[1], item[0]))


def _camera_plane_projected_bbox_scale(
    corners_world: Any,
    camera: Mapping[str, Any],
) -> Dict[str, float]:
    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError(
            "Stage07 projected owner scale requires exactly eight finite world-space bbox corners"
        )
    position = np.asarray(camera.get("position"), dtype=np.float64)
    target = np.asarray(camera.get("target"), dtype=np.float64)
    up_hint = np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
    if position.shape != (3,) or target.shape != (3,) or up_hint.shape != (3,):
        raise ValueError("Stage07 camera position/target/up must each contain three finite values")
    if not np.isfinite(position).all() or not np.isfinite(target).all() or not np.isfinite(up_hint).all():
        raise ValueError("Stage07 camera position/target/up must be finite")
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("Stage07 camera target must differ from camera position")
    forward /= forward_norm
    right = np.cross(forward, up_hint)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        fallback_up = np.asarray([0.0, 1.0, 0.0] if abs(forward[1]) < 0.9 else [1.0, 0.0, 0.0])
        right = np.cross(forward, fallback_up)
        right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("Stage07 could not construct a camera-plane basis")
    right /= right_norm
    camera_up = np.cross(right, forward)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1e-12)
    relative = corners - position[None, :]
    camera_x = relative @ right
    camera_y = relative @ camera_up
    span_x = float(np.max(camera_x) - np.min(camera_x))
    span_y = float(np.max(camera_y) - np.min(camera_y))
    scale = max(span_x, span_y)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Stage07 projected owner scale must be positive, got {scale}")
    return {
        "camera_x_span_m": span_x,
        "camera_y_span_m": span_y,
        "view_projected_scale_m": scale,
    }


def _owner_aware_near_depth_metrics(
    *,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    triangle_map: np.ndarray,
    manifest: Mapping[str, Any],
    semantic_pixel_counts_by_owner: Mapping[str, int],
    owner_generation_modes: Mapping[str, str],
    owner_bbox_corners_world: Mapping[str, Any],
    camera: Mapping[str, Any],
    room_diagonal_m: float,
    view_projected_bbox_scale_ratio: float,
    room_surface_room_diagonal_ratio: float,
) -> Dict[str, Any]:
    """Classify near pixels using each semantic owner's own physical scale.

    Non-room owners use the maximum physical span of their world-space bounding
    box after projection onto the current camera x-y plane. The camera-forward
    depth extent is deliberately ignored because it does not directly fill the
    image. Room surfaces use room diagonal. Triangle ranges keep lookup vectorized.
    """

    if depth.shape != triangle_map.shape or depth.shape != valid_mask.shape:
        raise ValueError(
            "Stage07 owner-aware near-depth inputs must share one image shape: "
            f"depth={depth.shape}, triangle={triangle_map.shape}, valid={valid_mask.shape}"
        )
    room_diagonal = float(room_diagonal_m)
    if not np.isfinite(room_diagonal) or room_diagonal <= 0.0:
        raise ValueError(
            f"Stage07 room_diagonal_m must be finite and positive, got {room_diagonal_m}"
        )

    records = _range_records(manifest)
    maximum_record_end = max(
        [int(record["scene_triangle_end_exclusive"]) for record in records] + [0]
    )
    lookup_size = max(int(manifest.get("triangle_count", 0)), maximum_record_end)
    cutoff_lookup = np.zeros((lookup_size,), dtype=np.float32)
    resolved_lookup = np.zeros((lookup_size,), dtype=bool)
    owner_cutoffs: Dict[str, Dict[str, Any]] = {}

    visible_owners = {
        str(owner)
        for owner, count in semantic_pixel_counts_by_owner.items()
        if int(count) > 0
    }
    for record in records:
        owner = str(record["semantic_owner_id"])
        if owner not in visible_owners:
            continue
        mode = str(owner_generation_modes.get(owner, ""))
        if mode in {"", "group"}:
            continue
        if mode == "surface_texture":
            scale_source = "room_diagonal"
            scale_m = room_diagonal
            ratio = float(room_surface_room_diagonal_ratio)
            owner_kind = "room_surface"
        else:
            scale_source = "camera_plane_projected_bbox"
            ratio = float(view_projected_bbox_scale_ratio)
            owner_kind = "non_room"
            projected: Dict[str, float] = {}
            if ratio <= 0.0:
                scale_m = 0.0
                scale_source = "disabled_zero_ratio"
            else:
                corners = owner_bbox_corners_world.get(owner)
                if corners is None:
                    raise RuntimeError(
                        "Stage07 owner-aware near-depth requires world-space bbox corners for "
                        f"visible non-room owner {owner!r}"
                    )
                try:
                    projected = _camera_plane_projected_bbox_scale(corners, camera)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Stage07 owner-aware near-depth received invalid projected bbox metadata "
                        f"for visible non-room owner {owner!r}: {exc}"
                    ) from exc
                scale_m = float(projected["view_projected_scale_m"])

        cutoff_m = float(scale_m * ratio)
        start = max(int(record["scene_triangle_start"]), 0)
        end = min(int(record["scene_triangle_end_exclusive"]), lookup_size)
        if end <= start:
            continue
        cutoff_lookup[start:end] = cutoff_m
        resolved_lookup[start:end] = True
        owner_cutoffs[owner] = {
            "generation_mode": mode,
            "owner_kind": owner_kind,
            "scale_source": scale_source,
            "scale_m": float(scale_m),
            "ratio": float(ratio),
            "cutoff_m": cutoff_m,
            **(projected if owner_kind == "non_room" else {}),
        }

    triangle_ids = np.asarray(triangle_map, dtype=np.int64)
    in_lookup = (triangle_ids >= 0) & (triangle_ids < lookup_size)
    clipped_ids = np.clip(triangle_ids, 0, max(lookup_size - 1, 0)) if lookup_size else None
    if lookup_size:
        owner_resolved = in_lookup & resolved_lookup[clipped_ids]
        cutoff_map = np.zeros(depth.shape, dtype=np.float32)
        cutoff_map[in_lookup] = cutoff_lookup[clipped_ids[in_lookup]]
    else:
        owner_resolved = np.zeros(depth.shape, dtype=bool)
        cutoff_map = np.zeros(depth.shape, dtype=np.float32)

    # ``depth`` is camera-Z by project contract at this point.
    camera_z_depth = np.asarray(depth, dtype=np.float32)
    resolved_valid = np.asarray(valid_mask, dtype=bool) & owner_resolved
    near_mask = resolved_valid & (camera_z_depth < cutoff_map)
    near_pixels = int(np.count_nonzero(near_mask))
    valid_pixels = int(np.count_nonzero(valid_mask))
    near_counts = semantic_pixel_counts(np.where(near_mask, triangle_map, -1), manifest)
    room_near_pixels = int(
        sum(
            int(count)
            for owner, count in near_counts.items()
            if str(owner_generation_modes.get(owner, "")) == "surface_texture"
        )
    )
    non_room_near_pixels = int(near_pixels - room_near_pixels)
    unresolved_valid_pixels = int(valid_pixels - np.count_nonzero(resolved_valid))

    return {
        "computed": True,
        "pixel_count": near_pixels,
        "fraction": float(near_pixels / max(valid_pixels, 1)),
        "resolved_valid_pixel_count": int(np.count_nonzero(resolved_valid)),
        "unresolved_valid_pixel_count": unresolved_valid_pixels,
        "non_room_near_pixel_count": non_room_near_pixels,
        "room_surface_near_pixel_count": room_near_pixels,
        "near_pixel_counts_by_owner": near_counts,
        "owner_cutoffs": owner_cutoffs,
        "view_projected_bbox_scale_ratio": float(view_projected_bbox_scale_ratio),
        "room_surface_room_diagonal_ratio": float(room_surface_room_diagonal_ratio),
        "threshold_policy": (
            "camera-z compared with per-pixel semantic-owner scale: non-room uses max "
            "camera-plane bbox x/y span; surface_texture uses room diagonal"
        ),
        "comparison_depth_convention": "camera_z",
    }


def evaluate_reconstruction_candidate(
    *,
    triangle_id_path: str | Path,
    depth_path: str | Path,
    depth_encoding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_owner_id: str,
    target_profile: str,
    owner_generation_modes: Mapping[str, str],
    owner_bbox_corners_world: Mapping[str, Any] | None = None,
    camera: Mapping[str, Any] | None = None,
    camera_calibration: Mapping[str, Any] | None = None,
    interior_object_count: int,
    room_diagonal_m: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    del interior_object_count  # Direct bad-view gates do not use scene-complexity heuristics.

    # Decode depth first because every later gate is defined over valid scene pixels.
    depth = decode_metric_depth(depth_path, depth_encoding)
    calibration = dict(camera_calibration or {})
    source_convention = depth_convention(depth_encoding)
    if source_convention != "camera_z":
        raise RuntimeError(f"Stage07 requires camera-Z depth, got {source_convention}")
    if calibration:
        width = int(calibration.get("width", depth.shape[1]))
        height = int(calibration.get("height", depth.shape[0]))
        if width != depth.shape[1] or height != depth.shape[0]:
            raise RuntimeError("Stage07 depth resolution does not match render camera calibration")
    camera_z_depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(camera_z_depth) & (camera_z_depth > 0.0)
    frame_pixels = int(depth.shape[0] * depth.shape[1])
    valid_pixels = int(valid.sum())
    denominator = max(valid_pixels, 1)
    valid_fraction = float(valid_pixels / max(frame_pixels, 1))
    minimum_valid_fraction = _unit_threshold(config, "minimum_valid_scene_fraction", 0.65)

    triangle_map = load_triangle_id_map(
        triangle_id_path,
        valid_triangle_count=int(manifest.get("triangle_count", 0)),
    )
    if triangle_map.shape != depth.shape:
        raise ValueError(
            "Stage07 candidate depth and triangle-ID images must share one shape: "
            f"depth={depth.shape}, triangle={triangle_map.shape}"
        )

    # Fractions are defined over valid scene pixels only. Masking the triangle IDs
    # with valid metric depth prevents transport/background mismatches from making
    # semantic fractions exceed one.
    masked_triangle_map = np.where(valid, triangle_map, -1)
    counts = semantic_pixel_counts(masked_triangle_map, manifest)
    fractions = {owner: float(pixels / denominator) for owner, pixels in counts.items()}

    significant_fraction = _unit_threshold(config, "significant_semantic_fraction", 0.01)

    # The semantic universe comes only from JSON-derived owner identities. Empty
    # and group records are structural rather than renderable semantic bodies.
    all_semantic_owners = sorted(
        str(owner)
        for owner, mode in owner_generation_modes.items()
        if str(mode) not in {"", "group"}
    )
    all_semantic_set = set(all_semantic_owners)
    room_surface_owners = sorted(
        owner
        for owner in all_semantic_owners
        if str(owner_generation_modes.get(owner, "")) == "surface_texture"
    )
    room_surface_set = set(room_surface_owners)
    non_room_semantic_owners = sorted(
        owner for owner in all_semantic_owners if owner not in room_surface_set
    )
    non_room_semantic_set = set(non_room_semantic_owners)

    significant = sorted(
        owner
        for owner, fraction in fractions.items()
        if owner in all_semantic_set and fraction >= significant_fraction
    )
    significant_non_room = sorted(owner for owner in significant if owner in non_room_semantic_set)
    target_pixels = int(counts.get(str(target_owner_id), 0))

    non_room_count = len(non_room_semantic_owners)
    all_semantic_count = len(all_semantic_owners)
    significant_non_room_count = len(significant_non_room)
    significant_count = len(significant)

    non_room_ratio = _ratio(significant_non_room_count, non_room_count)
    all_significant_ratio = _ratio(significant_count, all_semantic_count)
    minimum_non_room_ratio = _unit_threshold(
        config,
        "minimum_non_room_significant_count_ratio",
        0.25,
    )

    additional_significant_count_raw = config.get("additional_significant_semantic_count", 3)
    if isinstance(additional_significant_count_raw, bool):
        raise ValueError(
            "Stage07 candidate_scoring.additional_significant_semantic_count "
            "must be a non-negative integer, not bool"
        )
    additional_significant_count = int(additional_significant_count_raw)
    if (
        additional_significant_count < 0
        or float(additional_significant_count_raw) != float(additional_significant_count)
    ):
        raise ValueError(
            "Stage07 candidate_scoring.additional_significant_semantic_count "
            f"must be a non-negative integer, got {additional_significant_count_raw}"
        )

    required_non_room_significant_count = min(
        non_room_count,
        int(math.ceil(minimum_non_room_ratio * non_room_count)),
    )
    required_all_significant_count = min(
        all_semantic_count,
        required_non_room_significant_count + additional_significant_count,
    )
    derived_minimum_all_ratio = _ratio(
        required_all_significant_count,
        all_semantic_count,
    )

    # Bad-view mode 1: too little non-room semantic content.
    total_non_room_pixels = int(
        sum(int(counts.get(owner, 0)) for owner in non_room_semantic_owners)
    )
    total_non_room_fraction = float(total_non_room_pixels / denominator)
    minimum_total_non_room_fraction = _unit_threshold(
        config,
        "minimum_total_non_room_pixel_fraction",
        0.12,
    )
    non_room_pixel_gate_applicable = non_room_count > 0

    # Bad-view mode 2: one room surface dominates the image.
    largest_room_owner, largest_room_fraction = _largest_fraction(
        room_surface_owners,
        fractions,
    )
    maximum_single_room_surface_fraction = _unit_threshold(
        config,
        "maximum_single_room_surface_pixel_fraction",
        0.60,
    )
    room_surface_gate_applicable = len(room_surface_owners) > 0

    # Diagnostic only. A large non-room object is not itself evidence of face-close framing.
    largest_non_room_owner, largest_non_room_fraction = _largest_fraction(
        non_room_semantic_owners,
        fractions,
    )

    depth_cfg_raw = config.get("depth_diversity", {})
    if not isinstance(depth_cfg_raw, Mapping):
        raise ValueError("Stage07 candidate_scoring.depth_diversity must be an object")
    depth_cfg = dict(depth_cfg_raw)
    minimum_depth_range = _nonnegative_threshold(
        depth_cfg,
        "minimum_robust_log_depth_range",
        0.32,
    )
    depth_threshold_meta = {
        "threshold": minimum_depth_range,
        "threshold_policy": "fixed_external_for_all_target_profiles",
        "equivalent_minimum_q90_q10_ratio": float(math.exp(minimum_depth_range)),
    }

    near_cfg_raw = config.get("near_depth", {})
    if not isinstance(near_cfg_raw, Mapping):
        raise ValueError("Stage07 candidate_scoring.near_depth must be an object")
    near_cfg = dict(near_cfg_raw)
    object_near_ratio = _unit_threshold(near_cfg, "view_projected_bbox_scale_ratio", 0.75)
    room_surface_near_ratio = _unit_threshold(
        near_cfg,
        "room_surface_room_diagonal_ratio",
        0.04,
    )
    maximum_near_fraction = _unit_threshold(near_cfg, "maximum_pixel_fraction", 0.10)

    generation_mode = str(owner_generation_modes.get(str(target_owner_id), ""))
    surface_significant = [owner for owner in significant if owner in room_surface_set]
    object_significant = [owner for owner in significant if owner in non_room_semantic_set]

    # Run inexpensive histogram-only hard gates before quantiles; the owner-aware
    # per-pixel near-depth gate runs only after depth-diversity checks pass.
    hard_gate_order = [
        "valid_scene_fraction",
        "center_target_visible",
        "total_non_room_pixel_fraction",
        "largest_room_surface_pixel_fraction",
        "non_room_significant_count_ratio",
        "derived_all_significant_count",
        "robust_log_depth_range",
        "owner_aware_near_depth_fraction",
    ]
    reasons: list[str] = []
    if valid_fraction < minimum_valid_fraction:
        reasons.append("valid_scene_fraction_below_threshold")
    if target_pixels <= 0:
        reasons.append("center_target_not_visible")
    if (
        non_room_pixel_gate_applicable
        and total_non_room_fraction < minimum_total_non_room_fraction
    ):
        reasons.append("total_non_room_pixel_fraction_below_threshold")
    if (
        room_surface_gate_applicable
        and largest_room_fraction > maximum_single_room_surface_fraction
    ):
        reasons.append("single_room_surface_pixel_fraction_above_threshold")
    if significant_non_room_count < required_non_room_significant_count:
        reasons.append("non_room_significant_count_ratio_below_threshold")
    if significant_count < required_all_significant_count:
        reasons.append("all_significant_count_ratio_below_derived_threshold")

    short_circuit_stage: str | None = None
    if reasons:
        short_circuit_stage = "semantic_histogram_gates"
        depth_metrics: Dict[str, Any] = {
            "computed": False,
            "skipped": True,
            "skip_reason": "earlier_semantic_or_validity_gate_failed",
            "valid_depth_pixel_count": valid_pixels,
            "q10_m": None,
            "q50_m": None,
            "q90_m": None,
            "robust_log_depth_range": None,
            "depth_ratio_q90_q10": None,
        }
    else:
        depth_metrics = {
            "computed": True,
            "skipped": False,
            "source_depth_convention": depth_convention(depth_encoding),
            "evaluation_depth_convention": "camera_z",
            **robust_log_depth_range(camera_z_depth, valid),
        }
        if float(depth_metrics["robust_log_depth_range"]) < minimum_depth_range:
            reasons.append("depth_diversity_below_threshold")
            short_circuit_stage = "depth_diversity"

    if reasons:
        near_metrics: Dict[str, Any] = {
            "computed": False,
            "skipped": True,
            "skip_reason": (
                "earlier_semantic_or_validity_gate_failed"
                if short_circuit_stage == "semantic_histogram_gates"
                else "depth_diversity_gate_failed"
            ),
            "pixel_count": None,
            "fraction": None,
            "resolved_valid_pixel_count": None,
            "unresolved_valid_pixel_count": None,
            "non_room_near_pixel_count": None,
            "room_surface_near_pixel_count": None,
            "near_pixel_counts_by_owner": {},
            "owner_cutoffs": {},
            "view_projected_bbox_scale_ratio": object_near_ratio,
            "room_surface_room_diagonal_ratio": room_surface_near_ratio,
            "threshold_policy": (
                "per-pixel semantic-owner scale: non-room uses max camera-plane bbox x/y "
                "span; surface_texture uses room diagonal"
            ),
        }
    else:
        near_metrics = _owner_aware_near_depth_metrics(
            depth=depth,
            valid_mask=valid,
            triangle_map=triangle_map,
            manifest=manifest,
            semantic_pixel_counts_by_owner=counts,
            owner_generation_modes=owner_generation_modes,
            owner_bbox_corners_world=dict(owner_bbox_corners_world or {}),
            camera=dict(camera or {}),
            room_diagonal_m=room_diagonal_m,
            view_projected_bbox_scale_ratio=object_near_ratio,
            room_surface_room_diagonal_ratio=room_surface_near_ratio,
        )
        if float(near_metrics["fraction"]) > maximum_near_fraction:
            reasons.append("near_depth_fraction_above_threshold")
            short_circuit_stage = "owner_aware_near_depth"

    depth_value_for_score = float(depth_metrics.get("robust_log_depth_range") or 0.0)
    near_value_for_score = float(near_metrics.get("fraction") or 0.0)
    score = [
        float(non_room_ratio),
        float(total_non_room_fraction),
        float(all_significant_ratio),
        float(-largest_room_fraction),
        depth_value_for_score,
        float(-near_value_for_score),
    ]
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "hard_gate_execution": {
            "order": hard_gate_order,
            "short_circuit_stage": short_circuit_stage,
            "semantic_histogram_gates_passed": short_circuit_stage != "semantic_histogram_gates",
            "depth_diversity_computed": bool(depth_metrics.get("computed", False)),
            "owner_aware_near_depth_computed": bool(near_metrics.get("computed", False)),
            "policy": "cheap high-rejection gates first; skip quantiles and per-pixel owner lookup after an earlier failure",
        },
        "target_owner_id": str(target_owner_id),
        "target_generation_mode": generation_mode,
        "target_profile": target_profile,
        "center_target_may_be_non_significant": True,
        "target_pixel_count": target_pixels,
        "target_screen_fraction": float(fractions.get(str(target_owner_id), 0.0)),
        "frame_pixels": frame_pixels,
        "valid_scene_pixel_count": valid_pixels,
        "valid_scene_fraction": valid_fraction,
        "minimum_valid_scene_fraction": minimum_valid_fraction,
        "semantic_pixel_counts": counts,
        "semantic_screen_fractions": fractions,
        "significant_semantic_fraction_threshold": significant_fraction,
        "significant_semantics": significant,
        "significant_semantic_count": len(significant),
        "significant_surface_semantics": surface_significant,
        "significant_object_semantics": object_significant,
        "all_scene_semantics": all_semantic_owners,
        "all_scene_semantic_count": all_semantic_count,
        "all_significant_count_ratio": {
            "numerator": significant_count,
            "denominator": all_semantic_count,
            "value": all_significant_ratio,
            "minimum": derived_minimum_all_ratio,
            "required_count": required_all_significant_count,
            "additional_significant_semantic_count": additional_significant_count,
            "minimum_source": (
                "derived_from_ceil(minimum_non_room_significant_count_ratio * "
                "all_non_room_semantic_count) + additional_significant_semantic_count"
            ),
        },
        "all_non_room_semantics": non_room_semantic_owners,
        "all_non_room_semantic_count": non_room_count,
        "significant_non_room_semantics": significant_non_room,
        "significant_non_room_semantic_count": significant_non_room_count,
        "non_room_significant_count_ratio": {
            "numerator": significant_non_room_count,
            "denominator": non_room_count,
            "value": non_room_ratio,
            "minimum": minimum_non_room_ratio,
            "required_count": required_non_room_significant_count,
        },
        "total_non_room_pixel_fraction": {
            "applicable": non_room_pixel_gate_applicable,
            "pixel_count": total_non_room_pixels,
            "denominator_valid_scene_pixels": valid_pixels,
            "value": total_non_room_fraction,
            "minimum": minimum_total_non_room_fraction,
            "policy": "hard_gate_against_non_room_semantics_too_sparse_in_screen_space",
        },
        "largest_room_surface_pixel_fraction": {
            "applicable": room_surface_gate_applicable,
            "semantic_id": largest_room_owner,
            "value": float(largest_room_fraction),
            "maximum": maximum_single_room_surface_fraction,
            "policy": "hard_gate_against_one_room_surface_dominating_the_view",
        },
        "largest_non_room_fraction": {
            "semantic_id": largest_non_room_owner,
            "value": float(largest_non_room_fraction),
            "policy": "diagnostic_only_not_an_acceptance_gate",
        },
        "semantic_count_ratio_denominator_policy": (
            "JSON semantic owners with generation.mode not empty/group; "
            "non-room additionally excludes surface_texture"
        ),
        "depth_diversity": {**depth_metrics, **depth_threshold_meta},
        "near_depth": {
            **near_metrics,
            "maximum_fraction": maximum_near_fraction,
        },
        "score_lexicographic": score,
        "score_policy": (
            "diagnostic_only: non_room_significant_count_ratio_desc, "
            "total_non_room_pixel_fraction_desc, all_significant_count_ratio_desc, "
            "largest_room_surface_pixel_fraction_asc, robust_log_depth_range_desc, "
            "owner_aware_near_depth_fraction_asc; no diagnostic is added to sampling probability"
        ),
    }
