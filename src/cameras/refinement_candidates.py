from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from src.cameras.scene_geometry import camera_targets, scaffold_points, scene_bounds
from src.scene_ir.json_scene import flat_objects


def _camera(camera_id, position, target, focal_length=30.0, *, target_source=None, target_valid=True, **metadata):
    result = {
        "camera_id": camera_id,
        "group": "refinement_candidate",
        "position": [float(value) for value in position],
        "target": [float(value) for value in target],
        "focal_length": float(focal_length),
        "camera_type": "perspective",
        "up": [0.0, 0.0, 1.0],
        "target_source": target_source,
        "target_valid": bool(target_valid),
    }
    result.update(metadata)
    return result


def generate_refinement_candidates(camera_config: Dict, scene: Dict, count: int = 96, seed: int = 20260710) -> List[Dict]:
    """Legacy scene-perimeter sampler retained for non-Stage07 camera users."""
    pool = dict(camera_config.get("candidate_pool", {}))
    minimum, maximum = scene_bounds(scene, float(pool.get("boundary_margin_ratio", 0.08)))
    center = 0.5 * (minimum + maximum)
    size = np.maximum(maximum - minimum, 1e-3)
    target_margin = float(pool.get("target_margin_ratio", 0.03)) * size
    target_minimum = minimum + target_margin
    target_maximum = maximum - target_margin
    targets = camera_targets(scene)
    if not targets:
        targets = [{"object_id": "scene", "position": center.tolist(), "importance": 1.0}]
    weights = np.asarray([max(float(item["importance"]), 0.0) for item in targets], dtype=np.float64)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(targets), 1.0 / len(targets))
    heights_fraction = [float(value) for value in pool.get("heights_fraction", [0.35, 0.5, 0.65])]
    heights = [minimum[2] + fraction * size[2] for fraction in heights_fraction]
    focal_choices = [float(value) for value in pool.get("focal_lengths", [28.0, 32.0, 36.0])]
    rng = np.random.default_rng(int(seed))
    perimeter_count = max(8, min(int(count), int(math.ceil(count * 0.60))))
    radius_x = max(0.42 * size[0], 0.5)
    radius_y = max(0.42 * size[1], 0.5)
    candidates: List[Dict] = []
    for index in range(perimeter_count):
        angle = 2.0 * math.pi * index / perimeter_count
        position = [
            center[0] + radius_x * math.cos(angle),
            center[1] + radius_y * math.sin(angle),
            heights[index % len(heights)],
        ]
        target_record = targets[index % len(targets)]
        target = np.asarray(target_record["position"], dtype=np.float64)
        target = 0.75 * target + 0.25 * center
        target = np.clip(target, target_minimum, target_maximum)
        candidates.append(_camera(
            f"refine_candidate_{len(candidates):03d}", position, target,
            focal_choices[index % len(focal_choices)],
            target_source=str(target_record.get("object_id", "scene")),
            target_valid=bool(np.all(target >= target_minimum - 1e-8) and np.all(target <= target_maximum + 1e-8)),
        ))
    jitter = float(pool.get("target_jitter_ratio", 0.04)) * float(np.linalg.norm(size))
    while len(candidates) < int(count):
        side = int(rng.integers(0, 4))
        x = float(rng.uniform(minimum[0], maximum[0]))
        y = float(rng.uniform(minimum[1], maximum[1]))
        if side == 0:
            y = float(minimum[1])
        elif side == 1:
            x = float(maximum[0])
        elif side == 2:
            y = float(maximum[1])
        else:
            x = float(minimum[0])
        position = [x, y, float(rng.choice(heights))]
        target_record = targets[int(rng.choice(len(targets), p=weights))]
        target = np.asarray(target_record["position"], dtype=np.float64)
        target += rng.normal(0.0, jitter, size=3)
        target = np.clip(target, target_minimum, target_maximum)
        candidates.append(_camera(
            f"refine_candidate_{len(candidates):03d}", position, target,
            float(rng.choice(focal_choices)),
            target_source=str(target_record.get("object_id", "scene")),
            target_valid=bool(np.all(target >= target_minimum - 1e-8) and np.all(target <= target_maximum + 1e-8)),
        ))
    return candidates[: int(count)]


def generate_per_object_refinement_candidates(
    camera_config: Dict,
    scene: Dict,
    *,
    views_per_object: int = 9,
) -> List[Dict]:
    """Generate exactly three evenly sampled views on each of three rings per JSON object.

    The object list, centers, and extents come only from the input JSON scaffold. No
    semantic name/class is interpreted. Target generation modes are explicit config
    values, and ``camera.importance <= 0`` can still disable an object in JSON.
    """
    orbit = dict(camera_config.get("per_object_refinement_orbit", {}))
    elevations = [float(value) for value in orbit.get("elevation_degrees", [-45.0, 0.0, 45.0])]
    azimuths_per_ring = int(orbit.get("azimuths_per_ring", 3))
    if len(elevations) != 3:
        raise ValueError("per_object_refinement_orbit.elevation_degrees must contain exactly three rings")
    expected = len(elevations) * azimuths_per_ring
    if int(views_per_object) != expected or expected != 9:
        raise ValueError("Stage07 per-object refinement orbit requires exactly 9 views: 3 rings x 3 azimuths")

    points_by_object = scaffold_points(scene)
    targets = camera_targets(scene)
    target_modes = set(
        str(value)
        for value in orbit.get(
            "target_generation_modes",
            ["asset_3d", "external_asset", "scaffold_only"],
        )
    )
    target_ids = {
        str(record["object_id"])
        for record in flat_objects(scene)
        if str(dict(record.get("generation", {})).get("mode", "")) in target_modes
    }
    targets = [record for record in targets if str(record.get("object_id")) in target_ids]
    raw_minimum, raw_maximum = scene_bounds(scene, 0.0)
    scene_size = np.maximum(raw_maximum - raw_minimum, 1e-3)
    scene_diagonal = float(np.linalg.norm(scene_size))
    horizontal_margin = float(orbit.get("horizontal_position_margin_ratio", orbit.get("position_margin_ratio", 0.02)))
    vertical_margin = float(orbit.get("vertical_position_margin_ratio", 0.06))
    inset = np.asarray(
        [scene_size[0] * horizontal_margin, scene_size[1] * horizontal_margin, scene_size[2] * vertical_margin],
        dtype=np.float64,
    )
    position_minimum = raw_minimum + inset
    position_maximum = raw_maximum - inset
    invalid_axes = position_minimum > position_maximum
    position_minimum[invalid_axes] = raw_minimum[invalid_axes]
    position_maximum[invalid_axes] = raw_maximum[invalid_axes]

    radius_scale = float(orbit.get("radius_scale", 1.8))
    minimum_radius = float(orbit.get("minimum_radius", 0.75))
    maximum_radius = float(orbit.get("maximum_radius_scene_diagonal_ratio", 0.55)) * scene_diagonal
    focal_length = float(orbit.get("focal_length", 32.0))
    phase = float(orbit.get("azimuth_phase_degrees", 0.0))
    clamp_positions = bool(orbit.get("clamp_positions_to_scene_bounds", True))
    ring_names = ["lower", "middle", "upper"]

    candidates: List[Dict] = []
    for target_index, target_record in enumerate(targets):
        object_id = str(target_record["object_id"])
        points = points_by_object.get(object_id)
        if points is None or len(points) == 0:
            continue
        object_minimum = np.asarray(points, dtype=np.float64).min(axis=0)
        object_maximum = np.asarray(points, dtype=np.float64).max(axis=0)
        center = 0.5 * (object_minimum + object_maximum)
        extent = np.maximum(object_maximum - object_minimum, 1e-3)
        half_diagonal = 0.5 * float(np.linalg.norm(extent))
        radius = min(max(minimum_radius, radius_scale * half_diagonal), max(maximum_radius, minimum_radius))
        target_valid = bool(np.all(center >= raw_minimum - 1e-8) and np.all(center <= raw_maximum + 1e-8))

        for ring_index, (ring_name, elevation_deg) in enumerate(zip(ring_names, elevations)):
            elevation = math.radians(elevation_deg)
            horizontal_radius = radius * math.cos(elevation)
            z_offset = radius * math.sin(elevation)
            for azimuth_index in range(azimuths_per_ring):
                azimuth_deg = phase + 360.0 * azimuth_index / azimuths_per_ring
                azimuth = math.radians(azimuth_deg)
                position = np.asarray([
                    center[0] + horizontal_radius * math.cos(azimuth),
                    center[1] + horizontal_radius * math.sin(azimuth),
                    center[2] + z_offset,
                ], dtype=np.float64)
                unclamped = position.copy()
                if clamp_positions:
                    position = np.clip(position, position_minimum, position_maximum)
                camera_id = f"refine_object_{target_index:04d}_{ring_name}_{azimuth_index:02d}"
                candidates.append(_camera(
                    camera_id,
                    position,
                    center,
                    focal_length,
                    target_source=object_id,
                    target_valid=target_valid,
                    target_object_id=object_id,
                    target_object_index=int(target_index),
                    ring=ring_name,
                    ring_index=int(ring_index),
                    elevation_deg=float(elevation_deg),
                    azimuth_deg=float(azimuth_deg % 360.0),
                    azimuth_index=int(azimuth_index),
                    views_per_object=9,
                    orbit_radius=float(radius),
                    position_was_clamped=bool(np.linalg.norm(position - unclamped) > 1e-9),
                ))

    if not candidates:
        raise RuntimeError("No explicit JSON scaffold object is eligible for per-object refinement camera generation")
    return candidates


