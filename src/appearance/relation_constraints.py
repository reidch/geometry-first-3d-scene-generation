from __future__ import annotations

from typing import Dict, Mapping


def explicit_relations(record: Mapping) -> Dict:
    placement = dict(record.get("placement", {}))
    generation = dict(record.get("generation", {}))
    dependencies = [str(value) for value in generation.get("depends_on", [])]
    support_target = placement.get("support_target")
    return {
        "support_target": str(support_target) if support_target else None,
        "depends_on": dependencies,
        "clearance_m": float(placement.get("clearance_m", 0.0)),
        "support_axis_world": list(placement.get("support_axis_world", [0.0, 0.0, 1.0])),
    }


def build_relation_map(plan: Mapping) -> Dict[str, Dict]:
    return {str(record["object_id"]): explicit_relations(record) for record in plan.get("objects", [])}
