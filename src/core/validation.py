from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping

from src.scene_ir.json_scene import (
    SUPPORTED_BODY_TYPES,
    SUPPORTED_GENERATION_MODES,
    SUPPORTED_PHYSICS_MODES,
    SUPPORTED_PRIMITIVES,
    flat_objects,
    generation_spec,
    normalize_document,
    object_id,
    object_lookup,
    physics_spec,
    scaffold_parts,
    scene_payload,
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_vec3(name: str, value: Any, *, positive: bool = False, nonzero: bool = False) -> list[str]:
    issues: list[str] = []
    if not isinstance(value, list) or len(value) != 3:
        return [f"{name} must be a list of exactly 3 numbers"]
    norm2 = 0.0
    for index, item in enumerate(value):
        if not _is_number(item):
            issues.append(f"{name}[{index}] must be numeric")
        else:
            number = float(item)
            norm2 += number * number
            if positive and number <= 0:
                issues.append(f"{name}[{index}] must be > 0")
    if nonzero and not issues and norm2 <= 1e-16:
        issues.append(f"{name} must be non-zero")
    return issues


def _validate_transform(prefix: str, transform: Any) -> list[str]:
    if not isinstance(transform, Mapping):
        return [f"{prefix} must be an object"]
    issues = []
    issues.extend(validate_vec3(f"{prefix}.position", transform.get("position", [0, 0, 0])))
    issues.extend(validate_vec3(f"{prefix}.rotation_deg", transform.get("rotation_deg", [0, 0, 0])))
    issues.extend(validate_vec3(f"{prefix}.scale", transform.get("scale", [1, 1, 1]), positive=True))
    return issues


def _walk_raw_objects(objects: Any, parent_id: str | None = None, path: str = "scene.objects"):
    if not isinstance(objects, list):
        return
    for index, record in enumerate(objects):
        prefix = f"{path}[{index}]"
        if isinstance(record, Mapping):
            yield prefix, record, parent_id
            yield from _walk_raw_objects(record.get("children", []), object_id(record), prefix + ".children")
        else:
            yield prefix, record, parent_id



def validate_json_schema(document: Mapping[str, Any], schema_path: str | None = None) -> list[str]:
    """Validate the canonical document against the checked-in JSON Schema."""
    from pathlib import Path

    import jsonschema

    from src.io.json_io import load_json

    path = Path(schema_path) if schema_path else Path(__file__).resolve().parents[2] / "schemas/scene_v2.schema.json"
    schema = load_json(path)
    validator = jsonschema.Draft202012Validator(schema)
    issues = []
    for error in sorted(validator.iter_errors(document), key=lambda item: list(item.absolute_path)):
        location = ".".join(str(value) for value in error.absolute_path) or "<root>"
        issues.append(f"schema {location}: {error.message}")
    return issues

def validate_scene_dict(scene_dict: Mapping[str, Any], registry=None):
    """Validate the explicit v2 scene contract.

    ``semantic`` is deliberately treated as opaque data. No routing, physical rule,
    prompt, camera behaviour, or registration setting is inferred from its text.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if not isinstance(scene_dict, Mapping):
        return ["scene root must be an object"], []
    if str(scene_dict.get("schema_version", "")) != "2.0":
        issues.append("schema_version must be exactly '2.0'")
    if not isinstance(scene_dict.get("scene"), Mapping):
        issues.append("scene must be an object")
        return issues, warnings

    payload = scene_payload(scene_dict)
    if not isinstance(payload.get("id"), str) or not payload.get("id", "").strip():
        issues.append("scene.id must be a non-empty string")
    if not isinstance(payload.get("name"), str) or not payload.get("name", "").strip():
        issues.append("scene.name must be a non-empty string")
    if not isinstance(payload.get("objects"), list):
        issues.append("scene.objects must be a list")
        return issues, warnings

    raw_entries = list(_walk_raw_objects(payload.get("objects", [])))
    seen: set[str] = set()
    parent_of: Dict[str, str | None] = {}
    for prefix, record, parent_id in raw_entries:
        if not isinstance(record, Mapping):
            issues.append(f"{prefix} must be an object")
            continue
        oid = object_id(record)
        if not oid:
            issues.append(f"{prefix}.id must be a non-empty string")
            continue
        if oid in seen:
            issues.append(f"duplicate object id: {oid}")
        seen.add(oid)
        parent_of[oid] = parent_id
        for field in ("name", "semantic"):
            if not isinstance(record.get(field), str) or not record.get(field, "").strip():
                issues.append(f"{oid}: {field} must be a non-empty string")
        issues.extend(_validate_transform(f"{oid}.transform", record.get("transform", {})))

        try:
            generation = generation_spec(record)
        except Exception as exc:
            issues.append(f"{oid}: {exc}")
            generation = {}
        mode = generation.get("mode")
        if mode not in SUPPORTED_GENERATION_MODES:
            issues.append(f"{oid}: unsupported generation.mode {mode!r}")
        if mode in {"asset_3d", "surface_texture"}:
            prompt = generation.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                issues.append(f"{oid}: generation.prompt is required for mode {mode!r}")
        if mode == "external_asset" and not str(generation.get("external_asset_path", "")).strip():
            issues.append(f"{oid}: generation.external_asset_path is required")

        try:
            physics = physics_spec(record)
        except Exception as exc:
            issues.append(f"{oid}: invalid physics block: {exc}")
            physics = {}
        if physics.get("mode") not in SUPPORTED_PHYSICS_MODES:
            issues.append(f"{oid}: unsupported physics.mode {physics.get('mode')!r}")
        if physics.get("body") not in SUPPORTED_BODY_TYPES:
            issues.append(f"{oid}: unsupported physics.body {physics.get('body')!r}")
        if physics.get("mode") == "visual_only" and physics.get("body") != "none":
            issues.append(f"{oid}: visual_only physics requires body='none'")
        if physics.get("body") == "elastic":
            topology = physics.get("topology")
            if not isinstance(topology, Mapping):
                issues.append(f"{oid}: elastic physics requires physics.topology")
            elif topology.get("generator") == "grid":
                resolution = topology.get("resolution")
                if not isinstance(resolution, list) or len(resolution) != 2 or any(not isinstance(v, int) or v < 2 for v in resolution):
                    issues.append(f"{oid}: grid topology resolution must contain two integers >= 2")
                size = topology.get("size")
                if not isinstance(size, list) or len(size) != 2 or any(not _is_number(v) or float(v) <= 0 for v in size):
                    issues.append(f"{oid}: grid topology size must contain two positive numbers")
            elif not (isinstance(topology.get("particles"), list) and isinstance(topology.get("springs"), list)):
                issues.append(f"{oid}: elastic topology must use generator='grid' or explicit particles and springs")

        parts = scaffold_parts(record)
        if mode != "group" and not parts:
            issues.append(f"{oid}: scaffold.parts is required for every non-group object")
        part_ids: set[str] = set()
        for part_index, part in enumerate(parts):
            part_prefix = f"{oid}.scaffold.parts[{part_index}]"
            pid = str(part.get("id", ""))
            if not pid:
                issues.append(f"{part_prefix}.id must be a non-empty string")
            elif pid in part_ids:
                issues.append(f"{oid}: duplicate scaffold part id {pid!r}")
            part_ids.add(pid)
            primitive = part.get("primitive")
            if primitive not in SUPPORTED_PRIMITIVES:
                issues.append(f"{part_prefix}.primitive must be one of {sorted(SUPPORTED_PRIMITIVES)}")
            issues.extend(_validate_transform(f"{part_prefix}.transform", part.get("transform", {})))

        placement = record.get("placement", {})
        if placement is not None and not isinstance(placement, Mapping):
            issues.append(f"{oid}: placement must be an object")
        elif isinstance(placement, Mapping):
            support = placement.get("support_target")
            if support is not None and (not isinstance(support, str) or not support.strip()):
                issues.append(f"{oid}: placement.support_target must be a non-empty object id")
            if "support_axis_world" in placement:
                issues.extend(validate_vec3(f"{oid}.placement.support_axis_world", placement.get("support_axis_world"), nonzero=True))
            if "clearance_m" in placement and (not _is_number(placement["clearance_m"]) or float(placement["clearance_m"]) < 0):
                issues.append(f"{oid}: placement.clearance_m must be >= 0")
            # Stage05 intentionally uses one stable scalar support-snap path.
            # Legacy contact-mode/clustering/conformation fields are no longer
            # part of the supported scene contract.

        refinement = record.get("refinement", {})
        if refinement is not None and not isinstance(refinement, Mapping):
            issues.append(f"{oid}: refinement must be an object")
        elif isinstance(refinement, Mapping):
            if "camera_target" in refinement and not isinstance(refinement.get("camera_target"), bool):
                issues.append(f"{oid}: refinement.camera_target must be boolean")
            if "target_camera_count" in refinement:
                value = refinement.get("target_camera_count")
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 64:
                    issues.append(f"{oid}: refinement.target_camera_count must be an integer in [0,64]")
            if "update_weight" in refinement:
                value = refinement.get("update_weight")
                if not _is_number(value) or float(value) <= 0.0:
                    issues.append(f"{oid}: refinement.update_weight must be > 0")
            for strength_field in ("generation_strength", "fusion_strength", "max_periodicity_score"):
                if strength_field in refinement:
                    value = refinement.get(strength_field)
                    if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                        issues.append(f"{oid}: refinement.{strength_field} must be in [0,1]")

    all_ids = set(seen)
    for _, record, _ in raw_entries:
        if not isinstance(record, Mapping):
            continue
        oid = object_id(record)
        placement = record.get("placement", {})
        if isinstance(placement, Mapping) and placement.get("support_target") not in (None, ""):
            target = str(placement["support_target"])
            if target not in all_ids:
                issues.append(f"{oid}: placement.support_target {target!r} does not exist")
            if target == oid:
                issues.append(f"{oid}: placement.support_target cannot reference itself")
        dependencies = generation_spec(record).get("depends_on", []) if oid else []
        if dependencies is not None:
            if not isinstance(dependencies, list):
                issues.append(f"{oid}: generation.depends_on must be a list")
            else:
                for dep in dependencies:
                    if str(dep) not in all_ids:
                        issues.append(f"{oid}: generation.depends_on target {dep!r} does not exist")

    # Detect support cycles independently of hierarchy.
    support_parent: Dict[str, str] = {}
    for _, record, _ in raw_entries:
        if isinstance(record, Mapping):
            placement = record.get("placement", {})
            if isinstance(placement, Mapping) and placement.get("support_target"):
                support_parent[object_id(record)] = str(placement["support_target"])
    for start in support_parent:
        visited: set[str] = set()
        node = start
        while node in support_parent:
            if node in visited:
                issues.append(f"support relation cycle detected from {start!r}")
                break
            visited.add(node)
            node = support_parent[node]

    return issues, warnings


def normalize_scene(scene_dict: Mapping[str, Any]):
    return normalize_document(scene_dict)
