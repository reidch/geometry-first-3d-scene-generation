from __future__ import annotations

import math

from src.cameras.refinement_candidates import generate_refinement_candidates
from src.cameras.scene_geometry import scene_bounds
from src.cameras.smart_selection import select_sparse_candidates


def _camera(camera_id, group, position, target, focal_length=28, camera_type="perspective", ortho_scale=None):
    result = {"camera_id": camera_id, "group": group, "position": list(position), "target": list(target), "focal_length": float(focal_length), "camera_type": camera_type, "up": [0.0, 0.0, 1.0]}
    if ortho_scale is not None:
        result["ortho_scale"] = float(ortho_scale)
    return result


def generate_sparse_generation_candidates(config, scene=None):
    if scene is None:
        return []
    count = int(config.get("candidate_pool", {}).get("count", 32))
    return generate_refinement_candidates(config, scene, count=count, seed=int(config.get("seed", 20260710)))


def generate_sparse_candidate_cameras(config, scene=None):
    return generate_sparse_generation_candidates(config, scene)


def generate_room_surface_probe_cameras(config, scene=None):
    return []


def generate_sparse_cameras(config, scene=None):
    candidates = generate_sparse_generation_candidates(config, scene)
    selected, _, _ = select_sparse_candidates(candidates, scene or {"scene": {"objects": []}}, count=int(config.get("sparse", {}).get("count", 4)), config=config.get("selection", {}))
    return selected


def generate_dense_perimeter_cameras(config, scene=None):
    if scene is None:
        return []
    count = int(config.get("dense", {}).get("count", 16))
    return generate_refinement_candidates(config, scene, count=count, seed=int(config.get("seed", 20260710)) + 1)


def generate_overhead_cameras(config, scene=None):
    if scene is None or not bool(config.get("overhead", {}).get("enabled", True)):
        return []
    minimum, maximum = scene_bounds(scene)
    center = 0.5 * (minimum + maximum)
    size = maximum - minimum
    return [_camera("overhead_000", "overhead", [center[0], center[1], maximum[2] + max(size[2] * 0.15, 0.25)], center, camera_type="orthographic", ortho_scale=max(size[0], size[1]) * 1.08)]


def generate_camera_set(config, scene=None):
    return {
        "sparse": generate_sparse_cameras(config, scene),
        "dense": generate_dense_perimeter_cameras(config, scene),
        "overhead": generate_overhead_cameras(config, scene),
    }
