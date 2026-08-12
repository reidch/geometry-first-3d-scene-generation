from __future__ import annotations

import itertools
from typing import Dict, Iterable, Mapping

import numpy as np

from src.scene_ir.json_scene import flat_objects
from src.scene_ir.transforms import matrix_from_transform


def _apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def scaffold_points(document: Mapping) -> Dict[str, np.ndarray]:
    """Return conservative world-space samples for every explicit JSON scaffold."""
    result: Dict[str, np.ndarray] = {}
    cube = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=np.float64)
    for record in flat_objects(document):
        world = matrix_from_transform(record.get("world_transform", {}))
        points = []
        for part in record.get("scaffold", {}).get("parts", []):
            part_matrix = matrix_from_transform(part.get("transform", {}))
            # Primitive-specific exact geometry is unnecessary for camera bounds;
            # every supported primitive is contained by its transformed unit cube.
            points.append(_apply(_apply(cube, part_matrix), world))
        if points:
            result[str(record["object_id"])] = np.concatenate(points, axis=0)
    return result


def scene_bounds(document: Mapping, padding_ratio: float = 0.04):
    points_by_object = scaffold_points(document)
    if not points_by_object:
        minimum = np.asarray([-1.0, -1.0, 0.0], dtype=np.float64)
        maximum = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)
    else:
        points = np.concatenate(list(points_by_object.values()), axis=0)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
    size = np.maximum(maximum - minimum, 1e-3)
    padding = size * float(padding_ratio)
    return minimum - padding, maximum + padding


def camera_targets(document: Mapping):
    points_by_object = scaffold_points(document)
    records = {str(item["object_id"]): item for item in flat_objects(document)}
    targets = []
    for object_id, points in points_by_object.items():
        camera = dict(records[object_id].get("camera", {}))
        importance = float(camera.get("importance", 1.0))
        if importance <= 0.0:
            continue
        targets.append(
            {
                "object_id": object_id,
                "position": points.mean(axis=0).tolist(),
                "importance": importance,
            }
        )
    return targets


def scaffold_collision_bodies(document: Mapping) -> list[dict]:
    """Return scaffold-derived world-space primitive collision bodies.

    Stage07 camera selection should use the JSON scaffold structure rather than
    coarse final-mesh object AABBs.  Each explicit scaffold part becomes one
    independent world-space oriented bounding box (OBB), which is much tighter
    than a whole-object AABB while still remaining cheap and deterministic.
    """
    result: list[dict] = []
    for record in flat_objects(document):
        mode = str(dict(record.get("generation", {})).get("mode", ""))
        if mode in {"", "group", "surface_texture"}:
            continue
        world_transform = dict(record.get("world_transform", {}))
        if isinstance(world_transform.get("matrix"), list):
            world = np.asarray(world_transform["matrix"], dtype=np.float64)
        else:
            world = matrix_from_transform(world_transform)
        for part in record.get("scaffold", {}).get("parts", []):
            local = matrix_from_transform(part.get("transform", {}))
            composed = np.asarray(world, dtype=np.float64) @ np.asarray(local, dtype=np.float64)
            linear = composed[:3, :3].astype(np.float64)
            scale = np.linalg.norm(linear, axis=0)
            safe = np.where(scale > 1e-12, scale, 1.0)
            axes = linear / safe[None, :]
            half_extents = 0.5 * scale
            result.append({
                "collider_type": "scaffold_primitive_obb",
                "owner_id": str(record.get("object_id", "object")),
                "part_id": str(part.get("id", "part")),
                "primitive": str(part.get("primitive", "box")),
                "center": composed[:3, 3].astype(float).tolist(),
                "axes": axes.astype(float).tolist(),
                "half_extents": half_extents.astype(float).tolist(),
            })
    return result
