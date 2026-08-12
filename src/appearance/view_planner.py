from __future__ import annotations

from typing import Dict, Iterable, Mapping


def score_view(visible_fractions: Mapping[str, float], object_importance: Mapping[str, float] | None = None) -> Dict:
    importance = dict(object_importance or {})
    weighted = 0.0
    total = 0.0
    for object_id, fraction in visible_fractions.items():
        weight = float(importance.get(object_id, 1.0))
        weighted += weight * float(fraction)
        total += weight
    return {
        "score": weighted / max(total, 1e-9),
        "visible_objects": sorted(object_id for object_id, fraction in visible_fractions.items() if float(fraction) > 0.0),
    }


def select_views(candidates: Iterable[Mapping], count: int):
    return sorted((dict(candidate) for candidate in candidates), key=lambda value: float(value.get("score", 0.0)), reverse=True)[: int(count)]
