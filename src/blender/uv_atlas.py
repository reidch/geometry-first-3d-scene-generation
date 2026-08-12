from __future__ import annotations

import math
from collections import defaultdict

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import get_semantic_owner_id


def _activate_only(obj):
    bpy = require_bpy()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _smart_unwrap(obj):
    bpy = require_bpy()
    _activate_only(obj)
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=1.15192,
            island_margin=0.025,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True,
        )
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    obj.data.uv_layers.active_index = 0


def _deterministic_box_unwrap(obj):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active_index = 0
    uv = mesh.uv_layers.active.data
    coordinates = [[float(vertex.co[i]) for vertex in mesh.vertices] for i in range(3)]
    mins = [min(values) for values in coordinates]
    maxs = [max(values) for values in coordinates]

    def norm(value, lower, upper):
        return 0.5 if abs(upper - lower) < 1e-9 else (value - lower) / (upper - lower)

    tiles = {
        "px": (0.02, 0.52, 0.31, 0.98),
        "nx": (0.35, 0.52, 0.64, 0.98),
        "py": (0.68, 0.52, 0.82, 0.98),
        "ny": (0.84, 0.52, 0.98, 0.98),
        "pz": (0.02, 0.02, 0.48, 0.48),
        "nz": (0.52, 0.02, 0.98, 0.48),
    }
    for polygon in mesh.polygons:
        normal = polygon.normal
        components = (normal.x, normal.y, normal.z)
        axis = max(range(3), key=lambda index: abs(components[index]))
        sign = components[axis] >= 0
        key = ("p" if sign else "n") + ("x", "y", "z")[axis]
        u0, v0, u1, v1 = tiles[key]
        for loop_index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if axis == 2:
                a, b = norm(co.x, mins[0], maxs[0]), norm(co.y, mins[1], maxs[1])
            elif axis == 0:
                a, b = norm(co.y, mins[1], maxs[1]), norm(co.z, mins[2], maxs[2])
            else:
                a, b = norm(co.x, mins[0], maxs[0]), norm(co.z, mins[2], maxs[2])
            uv[loop_index].uv = (u0 + a * (u1 - u0), v0 + b * (v1 - v0))


def _unwrap_object(obj):
    mode = str(obj.get("uv_mode", "auto"))
    if mode == "smart":
        _smart_unwrap(obj)
    elif mode == "box" or (mode == "auto" and str(obj.get("primitive", "")) == "box"):
        _deterministic_box_unwrap(obj)
    else:
        _smart_unwrap(obj)


def assign_stable_object_atlas_uvs():
    """Give every semantic object one atlas, using only primitive/UV metadata."""
    bpy = require_bpy()
    groups = defaultdict(list)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            groups[get_semantic_owner_id(obj)].append(obj)
    manifest = {}
    for semantic_owner_id, parts in sorted(groups.items()):
        parts = sorted(parts, key=lambda item: str(item.get("part_id", item.name)))
        grid = max(1, int(math.ceil(math.sqrt(len(parts)))))
        margin = 0.035
        tiles = []
        for index, obj in enumerate(parts):
            _unwrap_object(obj)
            column, row = index % grid, index // grid
            scale = (1.0 - 2.0 * margin) / grid
            offset_u = column / grid + margin / grid
            offset_v = row / grid + margin / grid
            for loop in obj.data.uv_layers.active.data:
                loop.uv.x = offset_u + loop.uv.x * scale
                loop.uv.y = offset_v + loop.uv.y * scale
            obj["atlas_object_name"] = semantic_owner_id
            obj["atlas_owner_id"] = semantic_owner_id
            obj["atlas_tile_index"] = index
            obj["atlas_grid"] = grid
            tiles.append({"part_id": str(obj.get("part_id", obj.name)), "tile_index": index, "grid": grid})
        manifest[semantic_owner_id] = {
            "semantic_owner_id": semantic_owner_id,
            "part_count": len(parts),
            "grid": grid,
            "tiles": tiles,
            "uv_strategy": "json_uv_mode_or_primitive_auto",
        }
    return manifest
