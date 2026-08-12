from __future__ import annotations

from typing import Dict, List, Mapping


def plan_object_passes(plan: Mapping) -> List[Dict]:
    """Topologically order objects using only explicit JSON relationships."""
    records = {str(record["object_id"]): dict(record) for record in plan.get("objects", [])}
    dependencies = {}
    for object_id, record in records.items():
        generation = dict(record.get("generation", {}))
        placement = dict(record.get("placement", {}))
        values = {str(value) for value in generation.get("depends_on", []) if str(value) in records}
        if placement.get("support_target") in records:
            values.add(str(placement["support_target"]))
        dependencies[object_id] = values
    ordered = []
    remaining = set(records)
    while remaining:
        ready = sorted(object_id for object_id in remaining if not (dependencies[object_id] & remaining))
        if not ready:
            cycle = sorted(remaining)
            raise ValueError(f"Explicit generation/support dependency cycle: {cycle}")
        for object_id in ready:
            ordered.append(records[object_id])
            remaining.remove(object_id)
    return ordered


def plan_semantic_passes(plan: Mapping, config=None):
    return plan_object_passes(plan)
