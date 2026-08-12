from __future__ import annotations

import math

from src.cameras.camera_math import approx_overlap, direction_similarity


def _characteristic_scale(cameras):
    points = [list(map(float, camera["position"])) for camera in cameras]
    points += [list(map(float, camera["target"])) for camera in cameras]
    if not points:
        return 1.0
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return max(math.sqrt(sum((maximum[i] - minimum[i]) ** 2 for i in range(3))), 1e-6)


def build_camera_graph(cameras):
    nodes = [{"camera_id": camera["camera_id"], "group": camera["group"]} for camera in cameras]
    edges = []
    scale = _characteristic_scale(cameras)
    for index, first in enumerate(cameras):
        for other_index, second in enumerate(cameras):
            if index >= other_index:
                continue
            overlap = approx_overlap(first, second, scale)
            rotation_similarity = direction_similarity(first, second)
            if overlap > 0.15:
                edges.append({
                    "source": first["camera_id"],
                    "target": second["camera_id"],
                    "approx_overlap": overlap,
                    "rotation_similarity": rotation_similarity,
                })
    return {"nodes": nodes, "edges": edges, "characteristic_scale": scale}
