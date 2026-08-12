from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from src.blender.object_identity import get_semantic_owner_id
from src.io.json_io import load_json


def _load_contract(path: str | Path | None):
    if path in (None, ""):
        return None
    payload = load_json(path)
    values = payload.get("owner_ids", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"Active-owner manifest must contain owner_ids list: {path}")
    active = {str(value) for value in values if str(value)}
    if not active:
        raise ValueError(f"Active-owner manifest is empty: {path}")
    records = {}
    if isinstance(payload, dict):
        for record in payload.get("owners", []):
            if isinstance(record, dict) and record.get("owner_id"):
                records[str(record["owner_id"])] = dict(record)
    return active, records


def load_active_owner_ids(path: str | Path | None) -> set[str] | None:
    contract = _load_contract(path)
    return None if contract is None else contract[0]


def _matrix_from_transform(transform):
    from mathutils import Euler, Matrix, Vector

    transform = dict(transform or {})
    if isinstance(transform.get("matrix"), list):
        rows = transform["matrix"]
        if len(rows) == 4 and all(isinstance(row, list) and len(row) == 4 for row in rows):
            return Matrix([[float(value) for value in row] for row in rows])
    position = Vector([float(value) for value in transform.get("position", [0.0, 0.0, 0.0])])
    rotation = Euler(
        [math.radians(float(value)) for value in transform.get("rotation_deg", [0.0, 0.0, 0.0])],
        "XYZ",
    )
    scale = [float(value) for value in transform.get("scale", [1.0, 1.0, 1.0])]
    return Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((*scale, 1.0))


def _generated_visuals_by_owner(bpy, active: set[str]):
    result = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if bool(obj.get("pgw_physics_proxy", False)):
            continue
        if not bool(obj.get("pgw_generated_asset", False)) and str(obj.get("pgw_visual_role", "")) != "render_asset":
            continue
        owner = str(get_semantic_owner_id(obj) or "")
        if owner in active:
            result.setdefault(owner, []).append(obj)
    for values in result.values():
        values.sort(key=lambda obj: obj.name)
    return result


def _hierarchy_order(records: Dict[str, Dict[str, Any]], owners: set[str]):
    state = {}
    order = []

    def visit(owner):
        mark = state.get(owner, 0)
        if mark == 2:
            return
        if mark == 1:
            raise RuntimeError(f"Active-owner hierarchy cycle detected at {owner}")
        state[owner] = 1
        parent = str(records.get(owner, {}).get("parent_id") or "")
        if parent in owners:
            visit(parent)
        state[owner] = 2
        order.append(owner)

    for owner in sorted(owners):
        visit(owner)
    return order


def apply_active_owner_filter(bpy, path: str | Path | None) -> Dict[str, Any]:
    """Apply the current Scene JSON contract to an older cached downstream blend.

    Stale owners are hidden. Generated roots are reset to current JSON world transforms,
    and generated hierarchy children keep current JSON local transforms relative to the
    generated parent. This makes Stage07/09 reuse of an older Stage06 both visually and
    hierarchically consistent without mutating or regenerating Stage06 on disk.
    """
    contract = _load_contract(path)
    if contract is None:
        return {
            "enabled": False,
            "active_owner_count": None,
            "hidden_owner_ids": [],
            "hidden_object_names": [],
            "hierarchy_overlay_applied": False,
        }
    active, records = contract
    hidden_owners = set()
    hidden_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        owner = str(get_semantic_owner_id(obj) or "")
        if owner and owner not in active:
            obj.hide_render = True
            obj.hide_viewport = True
            obj["pgw_downstream_excluded_by_active_scene"] = True
            hidden_owners.add(owner)
            hidden_objects.append(obj.name)

    visual_map = _generated_visuals_by_owner(bpy, active)
    primary = {owner: values[0] for owner, values in visual_map.items() if values}
    hierarchy_records = {owner: records[owner] for owner in records if owner in active}
    repositioned = []
    from mathutils import Matrix
    for owner in _hierarchy_order(hierarchy_records, set(hierarchy_records)):
        visual = primary.get(owner)
        if visual is None:
            continue
        record = hierarchy_records[owner]
        parent_id = str(record.get("parent_id") or "")
        if parent_id in primary:
            visual.parent = primary[parent_id]
            visual.matrix_parent_inverse = Matrix.Identity(4)
            visual.matrix_basis = _matrix_from_transform(record.get("transform", {}))
            method = "generated_parent_plus_current_scene_local_matrix"
        else:
            visual.parent = None
            visual.matrix_world = _matrix_from_transform(record.get("world_transform", {}))
            method = "current_scene_root_world_matrix"
        visual["pgw_downstream_hierarchy_overlay"] = method
        repositioned.append({"owner_id": owner, "object_name": visual.name, "method": method})

    bpy.context.view_layer.update()
    return {
        "enabled": True,
        "active_owner_count": len(active),
        "active_owner_ids": sorted(active),
        "hidden_owner_ids": sorted(hidden_owners),
        "hidden_object_names": sorted(hidden_objects),
        "hierarchy_overlay_applied": bool(records),
        "repositioned_generated_visuals": repositioned,
    }
