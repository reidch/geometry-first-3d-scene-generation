from __future__ import annotations

import math


def sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def norm(v):
    return math.sqrt(max(dot(v, v), 1e-12))


def normalize(v):
    n = norm(v)
    return [float(x) / n for x in v]


def distance(a, b):
    return norm(sub(a, b))


def look_direction(position, target):
    return normalize(sub(target, position))


def direction_similarity(cam_a, cam_b):
    da = look_direction(cam_a["position"], cam_a["target"])
    db = look_direction(cam_b["position"], cam_b["target"])
    return max(0.0, min(1.0, (dot(da, db) + 1.0) * 0.5))


def approx_overlap(cam_a, cam_b, characteristic_scale=1.0):
    """Approximate overlap using a scale derived from the actual camera set."""
    scale = max(float(characteristic_scale), 1e-6)
    target_dist = distance(cam_a["target"], cam_b["target"])
    target_score = max(0.0, 1.0 - target_dist / (0.65 * scale))
    dir_score = direction_similarity(cam_a, cam_b)
    pos_dist = distance(cam_a["position"], cam_b["position"])
    pos_score = max(0.0, 1.0 - pos_dist / scale)
    return max(0.0, min(1.0, 0.50 * dir_score + 0.35 * target_score + 0.15 * pos_score))
