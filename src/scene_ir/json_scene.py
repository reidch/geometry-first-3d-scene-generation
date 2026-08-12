from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.scene_ir.transforms import compose_matrices, decompose_matrix, matrix_from_transform

SUPPORTED_GENERATION_MODES = {
    "asset_3d",
    "surface_texture",
    "scaffold_only",
    "external_asset",
    "group",
}
SUPPORTED_PHYSICS_MODES = {"static", "dynamic", "kinematic", "visual_only"}
SUPPORTED_BODY_TYPES = {"none", "rigid", "elastic", "fluid"}
SUPPORTED_PRIMITIVES = {"box", "sphere", "cylinder", "cone", "capsule"}



def deep_merge(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Dict[str, Any]:
    result = deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def scene_payload(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical v2 scene payload; runtime inference from legacy files is forbidden."""
    if not isinstance(document, Mapping) or not isinstance(document.get("scene"), Mapping):
        raise ValueError("Scene JSON must use schema v2 with a top-level scene object; run the migration tool first")
    return dict(document["scene"])

def scene_id(document: Mapping[str, Any]) -> str:
    payload = scene_payload(document)
    return str(payload.get("id", ""))


def scene_prompt(document: Mapping[str, Any]) -> str:
    return str(scene_payload(document).get("prompt", ""))


def object_id(record: Mapping[str, Any]) -> str:
    return str(record.get("id", ""))


def object_semantic(record: Mapping[str, Any]) -> str:
    return str(record.get("semantic", ""))


def object_name(record: Mapping[str, Any]) -> str:
    return str(record.get("name", ""))


def local_transform(record: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(record.get("transform", {}))
    return {
        "position": [float(v) for v in source.get("position", [0.0, 0.0, 0.0])],
        "rotation_deg": [float(v) for v in source.get("rotation_deg", [0.0, 0.0, 0.0])],
        "scale": [float(v) for v in source.get("scale", [1.0, 1.0, 1.0])],
    }


def generation_spec(record: Mapping[str, Any]) -> Dict[str, Any]:
    value = record.get("generation")
    if not isinstance(value, Mapping):
        raise ValueError(f"Object {object_id(record)!r} requires an explicit generation object")
    return deepcopy(dict(value))

def physics_spec(record: Mapping[str, Any]) -> Dict[str, Any]:
    value = record.get("physics")
    if not isinstance(value, Mapping):
        raise ValueError(f"Object {object_id(record)!r} requires an explicit physics object")
    return deepcopy(dict(value))

def scaffold_parts(record: Mapping[str, Any]) -> list[Dict[str, Any]]:
    scaffold = record.get("scaffold")
    if isinstance(scaffold, Mapping) and isinstance(scaffold.get("parts"), list):
        return deepcopy(scaffold["parts"])
    return []


def iter_object_tree(
    document: Mapping[str, Any],
    *,
    include_groups: bool = True,
) -> Iterator[Dict[str, Any]]:
    payload = scene_payload(document)
    defaults = dict(payload.get("defaults", {}))
    roots = payload.get("objects", [])
    if not isinstance(roots, list):
        return

    def visit(record: Mapping[str, Any], parent_id: str | None, parent_matrix: np.ndarray, path: list[str]):
        oid = object_id(record)
        local = local_transform(record)
        world_matrix = compose_matrices(parent_matrix, matrix_from_transform(local))
        generation = deep_merge(defaults.get("generation", {}), generation_spec(record))
        physics = deep_merge(defaults.get("physics", {}), physics_spec(record))
        appearance = deep_merge(defaults.get("appearance", {}), record.get("appearance", {}))
        placement = deep_merge(defaults.get("placement", {}), record.get("placement", {}))
        camera = deep_merge(defaults.get("camera", {}), record.get("camera", {}))
        flat = deepcopy(dict(record))
        flat["object_id"] = oid
        flat["name"] = object_name(record)
        flat["semantic_class"] = object_semantic(record)
        flat["parent_id"] = parent_id
        flat["path"] = path + [oid]
        flat["transform"] = local
        flat["world_transform"] = decompose_matrix(world_matrix)
        flat["generation"] = generation
        flat["physics"] = physics
        flat["appearance"] = appearance
        flat["placement"] = placement
        flat["camera"] = camera
        flat["scaffold"] = {"parts": scaffold_parts(record)}
        flat["children_ids"] = [object_id(child) for child in record.get("children", []) if isinstance(child, Mapping)]
        flat.pop("children", None)
        if include_groups or generation.get("mode") != "group":
            yield flat
        for child in record.get("children", []):
            if isinstance(child, Mapping):
                yield from visit(child, oid, world_matrix, path + [oid])

    identity = np.eye(4, dtype=np.float64)
    for root in roots:
        if isinstance(root, Mapping):
            yield from visit(root, None, identity, [])


def flat_objects(document: Mapping[str, Any], *, include_groups: bool = False) -> list[Dict[str, Any]]:
    return list(iter_object_tree(document, include_groups=include_groups))


def object_lookup(document: Mapping[str, Any], *, include_groups: bool = True) -> Dict[str, Dict[str, Any]]:
    return {record["object_id"]: record for record in iter_object_tree(document, include_groups=include_groups)}


def objects_for_generation_mode(document_or_plan: Mapping[str, Any], mode: str) -> list[Dict[str, Any]]:
    if isinstance(document_or_plan.get("objects"), list) and all(
        isinstance(item, Mapping) and "generation_mode" in item for item in document_or_plan.get("objects", [])
    ):
        return [deepcopy(dict(item)) for item in document_or_plan["objects"] if item.get("generation_mode") == mode]
    return [record for record in flat_objects(document_or_plan) if generation_spec(record).get("mode") == mode]


def normalize_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    payload = scene_payload(document)
    if "scene" not in document:
        raise ValueError(
            "Legacy scene JSON cannot be normalized without explicit v2 generation/scaffold/physics fields. "
            "Use tools/migrate_scene_v1_to_v2.py and review its output."
        )
    normalized = deepcopy(dict(document))
    normalized["schema_version"] = "2.0"
    normalized["scene"]["id"] = scene_id(document)
    normalized["scene"]["name"] = str(payload.get("name", scene_id(document)))
    normalized["scene"].setdefault("prompt", "")
    normalized["scene"].setdefault("coordinate_system", {"up_axis": "Z", "forward_axis": "Y", "unit_scale_m": 1.0})
    normalized["scene"].setdefault("defaults", {})
    return normalized
