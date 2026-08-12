from __future__ import annotations

from typing import Dict, Mapping


def occlusion_priority(object_id: str, visible_fraction: float, unseen_fraction: float, camera_importance: float = 1.0) -> Dict:
    """Generic priority based on measured visibility, not semantic category names."""
    score = float(camera_importance) * max(0.0, float(visible_fraction)) * max(0.0, float(unseen_fraction))
    return {"object_id": str(object_id), "score": score}


def rank_occluded_objects(measurements: Mapping[str, Mapping]) -> list[Dict]:
    records = [
        occlusion_priority(
            object_id,
            float(values.get("visible_fraction", 0.0)),
            float(values.get("unseen_fraction", 0.0)),
            float(values.get("importance", 1.0)),
        )
        for object_id, values in measurements.items()
    ]
    return sorted(records, key=lambda record: record["score"], reverse=True)
