from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping

from src.scene_ir.json_scene import flat_objects, scene_payload


def build_generation_plan(scene: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Build a routing plan by copying explicit JSON generation fields.

    The semantic label is opaque metadata. This function never classifies an object
    from its name, label, prompt, physics type, or scaffold shape.
    """
    objects: List[Dict[str, Any]] = []
    for record in flat_objects(scene, include_groups=True):
        generation = deepcopy(dict(record.get("generation", {})))
        mode = str(generation.get("mode", ""))
        objects.append({
            "object_id": record["object_id"],
            "name": record.get("name", record["object_id"]),
            "semantic_class": record.get("semantic_class", ""),
            "parent_id": record.get("parent_id"),
            "children_ids": list(record.get("children_ids", [])),
            "generation_mode": mode,
            "generation": generation,
            "appearance": deepcopy(dict(record.get("appearance", {}))),
            "placement": deepcopy(dict(record.get("placement", {}))),
            "physics": deepcopy(dict(record.get("physics", {}))),
            "camera": deepcopy(dict(record.get("camera", {}))),
            "refinement": deepcopy(dict(record.get("refinement", {}))),
            "world_transform": deepcopy(dict(record.get("world_transform", {}))),
            "scaffold": deepcopy(dict(record.get("scaffold", {}))),
        })
    counts = {
        mode: sum(1 for record in objects if record["generation_mode"] == mode)
        for mode in sorted({record["generation_mode"] for record in objects})
    }
    payload = scene_payload(scene)
    return {
        "schema_version": 3,
        "routing_source": "explicit_json_generation_mode",
        "semantic_labels_are_opaque": True,
        "scene_defaults": deepcopy(dict(payload.get("defaults", {}))),
        "coordinate_system": deepcopy(dict(payload.get("coordinate_system", {}))),
        "objects": objects,
        "counts": counts,
    }


def objects_by_mode(plan: Mapping[str, Any], mode: str) -> List[Dict[str, Any]]:
    return [deepcopy(dict(record)) for record in plan.get("objects", []) if record.get("generation_mode") == mode]
