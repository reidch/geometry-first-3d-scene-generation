from __future__ import annotations

import math

import numpy as np

from src.cameras.scene_geometry import camera_targets, scene_bounds


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-12)


def estimate_view_metrics(camera, scene):
    position = np.asarray(camera["position"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    view = _normalize(target - position)
    targets = camera_targets(scene)
    minimum, maximum = scene_bounds(scene)
    diagonal = max(float(np.linalg.norm(maximum - minimum)), 1e-6)
    visible_weight = 0.0
    angular_scores = []
    distance_scores = []
    for item in targets:
        point = np.asarray(item["position"], dtype=np.float64)
        direction = _normalize(point - position)
        angular = max(0.0, float(np.dot(view, direction)))
        distance = float(np.linalg.norm(point - position))
        distance_score = math.exp(-distance / diagonal)
        weight = float(item["importance"])
        visible_weight += weight * angular * distance_score
        angular_scores.append(angular)
        distance_scores.append(distance_score)
    total_weight = max(sum(float(item["importance"]) for item in targets), 1e-9)
    return {
        "visibility_score": visible_weight / total_weight,
        "mean_target_alignment": float(np.mean(angular_scores)) if angular_scores else 0.0,
        "mean_distance_score": float(np.mean(distance_scores)) if distance_scores else 0.0,
        "target_count": len(targets),
    }


def _novelty(candidate, selected, characteristic_scale):
    if not selected:
        return 1.0
    position = np.asarray(candidate["position"], dtype=np.float64)
    direction = _normalize(np.asarray(candidate["target"]) - position)
    values = []
    for other in selected:
        other_position = np.asarray(other["position"], dtype=np.float64)
        other_direction = _normalize(np.asarray(other["target"]) - other_position)
        positional = min(1.0, float(np.linalg.norm(position - other_position)) / max(float(characteristic_scale), 1e-6))
        angular = 0.5 * (1.0 - float(np.dot(direction, other_direction)))
        values.append(0.5 * positional + 0.5 * angular)
    return min(values)


def select_sparse_candidates(candidates, scene, count=4, config=None):
    config = dict(config or {})
    metrics = {item["camera_id"]: estimate_view_metrics(item, scene) for item in candidates}
    minimum, maximum = scene_bounds(scene)
    characteristic_scale = max(float(np.linalg.norm(maximum - minimum)), 1e-6)
    selected = []
    decisions = []
    remaining = list(candidates)
    while remaining and len(selected) < int(count):
        scored = []
        for candidate in remaining:
            visibility = metrics[candidate["camera_id"]]["visibility_score"]
            novelty = _novelty(candidate, selected, characteristic_scale)
            score = float(config.get("visibility_weight", 0.65)) * visibility + float(config.get("novelty_weight", 0.35)) * novelty
            scored.append((score, candidate, visibility, novelty))
        scored.sort(key=lambda item: item[0], reverse=True)
        score, chosen, visibility, novelty = scored[0]
        selected.append(chosen)
        remaining.remove(chosen)
        decisions.append({"camera_id": chosen["camera_id"], "score": score, "visibility": visibility, "novelty": novelty})
    return selected, metrics, decisions
