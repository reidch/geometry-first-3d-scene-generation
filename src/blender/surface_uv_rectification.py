from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import get_semantic_owner_id
from src.io.json_io import save_json


def _projection_direction(record: Dict, obj):
    from mathutils import Vector

    spatial = dict(record.get("spatial_category", {}))
    surface = dict(dict(record.get("generation", {})).get("surface", {}))
    inward = spatial.get("wall_inward_normal_world")
    if isinstance(inward, list) and len(inward) == 3:
        vec = Vector([float(v) for v in inward])
    else:
        projection = str(surface.get("projection", "front"))
        local_map = {
            "top": (0.0, 0.0, 1.0),
            "bottom": (0.0, 0.0, -1.0),
            "front": (0.0, -1.0, 0.0),
            "back": (0.0, 1.0, 0.0),
            "side": (1.0, 0.0, 0.0),
            "opposite_side": (-1.0, 0.0, 0.0),
        }
        local = Vector(local_map.get(projection, (0.0, -1.0, 0.0)))
        vec = obj.matrix_world.to_3x3() @ local
    if vec.length <= 1e-10:
        raise RuntimeError(f"Surface projection direction is zero for {record.get('object_id')}")
    vec.normalize()
    return vec


def _basis_from_normal(normal):
    from mathutils import Vector

    up = Vector((0.0, 0.0, 1.0))
    if abs(float(normal.dot(up))) > 0.98:
        # horizontal plane: keep texture 'up' aligned with world +Y for predictable layout
        v_axis = Vector((0.0, 1.0, 0.0))
        if abs(float(normal.dot(v_axis))) > 0.98:
            v_axis = Vector((1.0, 0.0, 0.0))
    else:
        v_axis = up - normal * float(up.dot(normal))
    if v_axis.length <= 1e-10:
        v_axis = Vector((1.0, 0.0, 0.0))
    v_axis.normalize()
    u_axis = v_axis.cross(normal)
    if u_axis.length <= 1e-10:
        u_axis = Vector((1.0, 0.0, 0.0))
    u_axis.normalize()
    # re-orthogonalise v against computed u_axis
    v_axis = normal.cross(u_axis)
    v_axis.normalize()
    return u_axis, v_axis


def _face_world_normal(obj, polygon):
    return (obj.matrix_world.to_3x3() @ polygon.normal).normalized()


def _face_world_center(obj, polygon):
    return obj.matrix_world @ polygon.center


def _loop_world_point(obj, loop_index: int):
    vertex_index = obj.data.loops[loop_index].vertex_index
    return obj.matrix_world @ obj.data.vertices[vertex_index].co


def _select_target_polygons(obj, target_direction):
    polys = list(obj.data.polygons)
    if not polys:
        return []
    alignment = []
    for poly in polys:
        n = _face_world_normal(obj, poly)
        d = float(n.dot(target_direction))
        alignment.append((poly.index, d, float(poly.area), _face_world_center(obj, poly)))
    best_dot = max(item[1] for item in alignment)
    if best_dot < 0.25:
        # fall back to the most aligned polygon(s) even when the mesh is not a perfect box
        dot_threshold = best_dot - 1e-6
    else:
        dot_threshold = max(0.95, best_dot - 0.02)
    aligned = [item for item in alignment if item[1] >= dot_threshold]
    if not aligned:
        aligned = [max(alignment, key=lambda item: item[1])]
    max_plane = max(float(item[3].dot(target_direction)) for item in aligned)
    extent = max(
        max((float((obj.matrix_world @ v.co).dot(target_direction)) for v in obj.data.vertices), default=max_plane) -
        min((float((obj.matrix_world @ v.co).dot(target_direction)) for v in obj.data.vertices), default=max_plane),
        1e-4,
    )
    plane_eps = max(1e-4, extent * 1e-4)
    selected = [item[0] for item in aligned if float(item[3].dot(target_direction)) >= max_plane - plane_eps]
    if not selected:
        selected = [max(alignment, key=lambda item: item[1])[0]]
    return selected


def rectify_surface_owner_uvs(record: Dict, uv_name: str = "PGW_SURFACE_RECTIFIED_UV") -> Dict:
    bpy = require_bpy()
    object_id = str(record.get("object_id") or record.get("id"))
    report = {
        "object_id": object_id,
        "uv_layer": uv_name,
        "meshes": [],
        "status": "ok",
    }
    meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not bool(obj.get("pgw_physics_proxy", False)) and str(get_semantic_owner_id(obj)) == object_id
    ]
    if not meshes:
        report["status"] = "missing_mesh"
        return report
    for obj in meshes:
        direction = _projection_direction(record, obj)
        u_axis, v_axis = _basis_from_normal(direction)
        selected_polygons = _select_target_polygons(obj, direction)
        if not selected_polygons:
            report["meshes"].append({"mesh_object_name": obj.name, "status": "no_polygons"})
            report["status"] = "failed"
            continue
        uv_layer = obj.data.uv_layers.get(uv_name) or obj.data.uv_layers.new(name=uv_name)

        target_loop_indices = [
            loop_index
            for poly in obj.data.polygons
            if poly.index in selected_polygons
            for loop_index in poly.loop_indices
        ]
        coords_u = []
        coords_v = []
        for loop_index in target_loop_indices:
            point = _loop_world_point(obj, loop_index)
            coords_u.append(float(point.dot(u_axis)))
            coords_v.append(float(point.dot(v_axis)))
        min_u, max_u = min(coords_u), max(coords_u)
        min_v, max_v = min(coords_v), max(coords_v)
        span_u = max(max_u - min_u, 1e-6)
        span_v = max(max_v - min_v, 1e-6)
        selected_polygon_set = set(selected_polygons)
        polygon_by_loop = {}
        for poly in obj.data.polygons:
            for loop_index in poly.loop_indices:
                polygon_by_loop[loop_index] = poly.index
        for loop_index, uv in enumerate(uv_layer.data):
            point = _loop_world_point(obj, loop_index)
            uu = (float(point.dot(u_axis)) - min_u) / span_u
            vv = (float(point.dot(v_axis)) - min_v) / span_v
            if polygon_by_loop.get(loop_index) not in selected_polygon_set:
                uu = min(1.0, max(0.0, uu))
                vv = min(1.0, max(0.0, vv))
            uv.uv = (float(uu), float(vv))
        obj.data.uv_layers.active = uv_layer
        try:
            obj.data.uv_layers.active_render = uv_layer
        except Exception:
            pass
        report["meshes"].append(
            {
                "mesh_object_name": obj.name,
                "status": "ok",
                "selected_polygon_indices": list(selected_polygons),
                "selected_polygon_count": int(len(selected_polygons)),
                "target_direction_world": [float(v) for v in direction],
                "u_axis_world": [float(v) for v in u_axis],
                "v_axis_world": [float(v) for v in v_axis],
                "target_face_u_range": [float(min_u), float(max_u)],
                "target_face_v_range": [float(min_v), float(max_v)],
            }
        )
    if any(entry.get("status") != "ok" for entry in report["meshes"]):
        report["status"] = "failed"
    return report


def rectify_surface_uvs(records: Iterable[Dict], uv_name: str = "PGW_SURFACE_RECTIFIED_UV", report_path: str | Path | None = None) -> Dict:
    records = list(records)
    report = {
        "status": "ok",
        "uv_layer": uv_name,
        "surface_count": len(records),
        "records": [rectify_surface_owner_uvs(record, uv_name=uv_name) for record in records],
    }
    if any(item.get("status") not in {"ok"} for item in report["records"]):
        report["status"] = "failed"
    if report_path is not None:
        save_json(report, report_path)
    return report
