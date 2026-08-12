from __future__ import annotations

import math


def _distance(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _direction(camera):
    value = [float(camera["target"][i]) - float(camera["position"][i]) for i in range(3)]
    length = max(math.sqrt(sum(component * component for component in value)), 1e-12)
    return [component / length for component in value]


def _rotation_cost(a, b):
    first, second = _direction(a), _direction(b)
    cosine = max(-1.0, min(1.0, sum(first[i] * second[i] for i in range(3))))
    return 0.5 * (1.0 - cosine)


def _camera_scale(cameras):
    positions = [camera["position"] for camera in cameras]
    if not positions:
        return 1.0
    minimum = [min(float(position[axis]) for position in positions) for axis in range(3)]
    maximum = [max(float(position[axis]) for position in positions) for axis in range(3)]
    return max(_distance(minimum, maximum), 1e-6)


def plan_dense_generation_order(sparse_cameras, dense_cameras, overhead_cameras=None, weights=None):
    weights = dict(weights or {})
    rotation_weight = float(weights.get("rotation_weight", 0.55))
    distance_weight = float(weights.get("overlap_weight", 0.30))
    remaining = list(dense_cameras)
    ordered = []
    scale = _camera_scale(list(sparse_cameras) + list(dense_cameras) + list(overhead_cameras or []))
    current = sparse_cameras[-1] if sparse_cameras else (remaining[0] if remaining else None)
    while remaining:
        if current is None:
            chosen = remaining[0]
        else:
            chosen = min(
                remaining,
                key=lambda camera: rotation_weight * _rotation_cost(current, camera)
                + distance_weight * (_distance(current["position"], camera["position"]) / scale),
            )
        ordered.append(chosen)
        remaining.remove(chosen)
        current = chosen
    ordered.extend(list(overhead_cameras or []))
    return ordered
