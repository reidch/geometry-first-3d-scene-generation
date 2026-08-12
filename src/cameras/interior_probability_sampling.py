from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from src.cameras.scene_geometry import scaffold_points
from src.io.json_io import load_json
from src.scene_ir.json_scene import flat_objects


def _as_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {vector.shape}")
    return vector


def _aabb(minimum: Iterable[float], maximum: Iterable[float]) -> Dict[str, list[float]]:
    lower = _as_vector(minimum)
    upper = _as_vector(maximum)
    if np.any(upper <= lower):
        raise ValueError(f"Invalid AABB: minimum={lower.tolist()} maximum={upper.tolist()}")
    return {
        "minimum": lower.tolist(),
        "maximum": upper.tolist(),
        "center": (0.5 * (lower + upper)).tolist(),
        "extent": (upper - lower).tolist(),
    }


def point_inside_aabb(point: Iterable[float], box: Mapping[str, Any], margin: float = 0.0) -> bool:
    p = _as_vector(point)
    minimum = _as_vector(box["minimum"]) - float(margin)
    maximum = _as_vector(box["maximum"]) + float(margin)
    return bool(np.all(p >= minimum) and np.all(p <= maximum))


def distance_to_aabb(point: Iterable[float], box: Mapping[str, Any]) -> float:
    """Euclidean distance to an axis-aligned box; zero inside the box."""
    p = _as_vector(point)
    minimum = _as_vector(box["minimum"])
    maximum = _as_vector(box["maximum"])
    delta = np.maximum(np.maximum(minimum - p, 0.0), p - maximum)
    return float(np.linalg.norm(delta))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return float(value >= edge1)
    t = min(1.0, max(0.0, (float(value) - edge0) / (edge1 - edge0)))
    return float(t * t * (3.0 - 2.0 * t))


def distance_probability_weight(distance: float, config: Mapping[str, Any]) -> float:
    """Band-pass free-space weight based on distance to the nearest object AABB.

    The weight is zero inside/very near an object, rises into a preferred camera
    distance band, then falls again so sampling does not collapse into room corners.
    """
    zero_below = float(config.get("zero_inside_or_below_m", 0.12))
    peak_start = float(config.get("peak_start_m", 0.50))
    peak_end = float(config.get("peak_end_m", 1.50))
    zero_after = float(config.get("zero_after_m", 2.80))
    if not (0.0 <= zero_below < peak_start <= peak_end < zero_after):
        raise ValueError(
            "distance_probability must satisfy 0 <= zero_inside_or_below < "
            "peak_start <= peak_end < zero_after"
        )
    rising = _smoothstep(zero_below, peak_start, float(distance))
    falling = 1.0 - _smoothstep(peak_end, zero_after, float(distance))
    return float(max(0.0, min(1.0, rising * falling)))


def _room_surface_bounds(scene: Mapping[str, Any], margin_m: float | Iterable[float]) -> Dict[str, list[float]]:
    points_by_id = scaffold_points(scene)
    surface_ids = {
        str(record["object_id"])
        for record in flat_objects(scene)
        if str(dict(record.get("generation", {})).get("mode", "")) == "surface_texture"
    }
    arrays = [points_by_id[object_id] for object_id in sorted(surface_ids) if object_id in points_by_id]
    if not arrays:
        raise RuntimeError(
            "Stage07 cannot infer a room interior: no surface_texture scaffold geometry was found"
        )
    points = np.concatenate(arrays, axis=0)
    if isinstance(margin_m, (list, tuple, np.ndarray)):
        margin = _as_vector(margin_m)
    else:
        margin = np.full(3, float(margin_m), dtype=np.float64)
    if np.any(margin < 0.0):
        raise ValueError("Room boundary margins must be non-negative")
    minimum = points.min(axis=0) + margin
    maximum = points.max(axis=0) - margin
    if np.any(maximum <= minimum):
        raise RuntimeError(
            f"Room interior collapsed after boundary margin {margin.tolist()}: "
            f"minimum={minimum.tolist()} maximum={maximum.tolist()}"
        )
    return _aabb(minimum, maximum)


def _stage05_aabbs(out: Path) -> Dict[str, Dict[str, Any]]:
    report_path = out / "05_scene_assets" / "blender_import_report.json"
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    result: Dict[str, Dict[str, Any]] = {}
    for record in report.get("records", []):
        object_id = str(record.get("object_id", ""))
        box = record.get("visual_world_aabb")
        if object_id and isinstance(box, Mapping) and box.get("minimum") and box.get("maximum"):
            result[object_id] = _aabb(box["minimum"], box["maximum"])
    return result


def build_sampling_context(
    out: str | Path,
    scene: Mapping[str, Any],
    camera_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build final-scene target AABBs, quotas, and the room-interior AABB."""
    out = Path(out)
    cfg = dict(camera_config.get("interior_probability_sampling", {}))
    target_modes = {
        str(value)
        for value in cfg.get("target_generation_modes", ["asset_3d", "external_asset", "scaffold_only"])
    }
    default_count = max(1, int(cfg.get("default_target_camera_count", 6)))
    points_by_id = scaffold_points(scene)
    final_aabbs = _stage05_aabbs(out)
    targets: Dict[str, Dict[str, Any]] = {}
    for record in flat_objects(scene):
        object_id = str(record["object_id"])
        mode = str(dict(record.get("generation", {})).get("mode", ""))
        if mode not in target_modes:
            continue
        refinement = dict(record.get("refinement", {}))
        if not bool(refinement.get("camera_target", True)):
            continue
        quota = int(refinement.get("target_camera_count", default_count))
        if quota <= 0:
            continue
        box = final_aabbs.get(object_id)
        source = "stage05_final_visual"
        if box is None:
            points = points_by_id.get(object_id)
            if points is None or len(points) == 0:
                raise RuntimeError(f"No final or scaffold AABB is available for camera target {object_id}")
            box = _aabb(points.min(axis=0), points.max(axis=0))
            source = "json_scaffold_fallback"
        targets[object_id] = {
            "object_id": object_id,
            "generation_mode": mode,
            "quota": quota,
            "aabb": box,
            "center": list(box["center"]),
            "aabb_source": source,
        }
    if not targets:
        raise RuntimeError("No non-room semantic object is enabled as a Stage07 camera target")
    room_margin = cfg.get("room_boundary_margin_xyz_m", cfg.get("room_boundary_margin_m", 0.12))
    room = _room_surface_bounds(scene, room_margin)
    return {
        "room_interior_aabb": room,
        "targets": targets,
        "all_target_aabbs": [targets[key]["aabb"] for key in sorted(targets)],
        "config": cfg,
    }


def _view_direction(position: Iterable[float], target: Iterable[float]) -> np.ndarray:
    vector = _as_vector(position) - _as_vector(target)
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        raise ValueError("Camera position coincides with target center")
    return vector / length


def angular_separation_degrees(first: Iterable[float], second: Iterable[float]) -> float:
    a = _as_vector(first)
    b = _as_vector(second)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _minimum_separation(cfg: Mapping[str, Any], attempt_fraction: float) -> float:
    strict = float(cfg.get("minimum_view_separation_deg", 28.0))
    relaxed = float(cfg.get("relaxed_view_separation_deg", 18.0))
    relax_after = float(cfg.get("relax_after_attempt_fraction", 0.70))
    return relaxed if float(attempt_fraction) >= relax_after else strict


def sample_candidate_batch(
    context: Mapping[str, Any],
    rng: np.random.Generator,
    accepted_directions: Mapping[str, list[list[float]]],
    attempts_by_target: Dict[str, int],
    *,
    round_index: int,
) -> Dict[str, Any]:
    """Sample one deterministic batch for currently incomplete targets.

    This stage performs only cheap spatial rejection. Render-based target visibility
    and occlusion checks are intentionally deferred to Blender batch evaluation.
    """
    cfg = dict(context["config"])
    targets = dict(context["targets"])
    room = dict(context["room_interior_aabb"])
    room_minimum = _as_vector(room["minimum"])
    room_maximum = _as_vector(room["maximum"])
    boxes = list(context["all_target_aabbs"])
    bbox_margin = float(cfg.get("object_bbox_margin_m", 0.12))
    probability_cfg = dict(cfg.get("distance_probability", {}))
    batch_size = max(1, int(cfg.get("candidate_batch_size", 64)))
    max_attempts = max(1, int(cfg.get("max_candidate_attempts_per_target", 320)))
    focal_length = float(cfg.get("focal_length", 32.0))

    proposed_directions: Dict[str, list[list[float]]] = {key: [] for key in targets}
    rejection_counts: Dict[str, int] = {
        "inside_or_too_near_object_bbox": 0,
        "distance_probability_rejected": 0,
        "view_direction_too_similar": 0,
        "camera_target_coincident": 0,
        "no_incomplete_target": 0,
    }
    cameras = []
    draw_limit = batch_size * 250
    draws = 0
    while len(cameras) < batch_size and draws < draw_limit:
        draws += 1
        incomplete = [
            object_id
            for object_id, target in targets.items()
            if len(accepted_directions.get(object_id, [])) + len(proposed_directions[object_id]) < int(target["quota"])
            and int(attempts_by_target.get(object_id, 0)) < max_attempts
        ]
        if not incomplete:
            rejection_counts["no_incomplete_target"] += 1
            break

        position = rng.uniform(room_minimum, room_maximum)
        if any(point_inside_aabb(position, box, bbox_margin) for box in boxes):
            rejection_counts["inside_or_too_near_object_bbox"] += 1
            continue
        nearest_distance = min(distance_to_aabb(position, box) for box in boxes)
        probability = distance_probability_weight(nearest_distance, probability_cfg)
        if probability <= 0.0 or float(rng.random()) > probability:
            rejection_counts["distance_probability_rejected"] += 1
            continue

        remaining = np.asarray(
            [max(1, int(targets[object_id]["quota"]) - len(accepted_directions.get(object_id, []))) for object_id in incomplete],
            dtype=np.float64,
        )
        remaining /= remaining.sum()
        target_id = str(rng.choice(incomplete, p=remaining))
        attempts_by_target[target_id] = int(attempts_by_target.get(target_id, 0)) + 1
        target = targets[target_id]
        try:
            direction = _view_direction(position, target["center"])
        except ValueError:
            rejection_counts["camera_target_coincident"] += 1
            continue
        attempt_fraction = attempts_by_target[target_id] / max_attempts
        minimum_separation = _minimum_separation(cfg, attempt_fraction)
        existing = list(accepted_directions.get(target_id, [])) + proposed_directions[target_id]
        if existing and min(angular_separation_degrees(direction, item) for item in existing) < minimum_separation:
            rejection_counts["view_direction_too_similar"] += 1
            continue

        index = len(cameras)
        camera_id = f"interior_r{int(round_index):02d}_{index:03d}_{target_id}"
        camera = {
            "camera_id": camera_id,
            "group": "refinement_candidate",
            "camera_type": "perspective",
            "position": [float(value) for value in position],
            "target": [float(value) for value in target["center"]],
            "focal_length": focal_length,
            "up": [0.0, 0.0, 1.0],
            "target_object_id": target_id,
            "target_source": target_id,
            "target_valid": True,
            "source": "interior_probability_sampling",
            "sampling_round": int(round_index),
            "nearest_object_aabb_distance_m": float(nearest_distance),
            "position_probability_weight": float(probability),
            "view_direction_from_target": [float(value) for value in direction],
            "minimum_view_separation_deg": float(minimum_separation),
        }
        cameras.append(camera)
        proposed_directions[target_id].append(camera["view_direction_from_target"])

    return {
        "cameras": cameras,
        "draw_count": draws,
        "rejection_counts": rejection_counts,
        "attempts_by_target": dict(attempts_by_target),
    }
