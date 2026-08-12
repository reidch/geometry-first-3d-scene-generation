#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.assets.representative_anchor_selection import DEFAULT_CANDIDATE_ORBIT, build_candidate_views
from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import _look_at
from src.blender.condition_renderer import configure_condition_render, render_depth_control_png
from src.blender.object_identity import get_semantic_owner_id
from src.io.json_io import load_json, save_json


def _owner_id(obj):
    return get_semantic_owner_id(obj)


def _target_meshes(bpy, object_id):
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and _owner_id(obj) == str(object_id) and not bool(obj.get("pgw_physics_proxy", False))]


def _world_bbox(objects):
    from mathutils import Vector
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("Target object has no world-space bounds")
    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    size = [max(maxs[axis] - mins[axis], 1e-4) for axis in range(3)]
    return center, size, mins, maxs


def _camera_depth_bounds(camera, objects):
    from mathutils import Vector
    inverse = camera.matrix_world.inverted()
    depths = []
    for obj in objects:
        for corner in obj.bound_box:
            camera_point = inverse @ (obj.matrix_world @ Vector(corner))
            depth = -float(camera_point.z)
            if depth > 0.0 and math.isfinite(depth):
                depths.append(depth)
    if not depths:
        raise RuntimeError("Could not compute positive camera-space depth bounds")
    near, far = min(depths), max(depths)
    padding = max((far - near) * 0.01, 1e-4)
    return max(camera.data.clip_start, near - padding), min(camera.data.clip_end, far + padding)


def _isolate(bpy, object_id):
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            visible = _owner_id(obj) == str(object_id) and not bool(obj.get("pgw_physics_proxy", False))
            obj.hide_render = not visible
            obj.hide_viewport = not visible


def _object_rotation(out: Path, object_id: str):
    from mathutils import Matrix
    flat_path = out / "00_validated" / "objects.flat.json"
    if not flat_path.exists():
        return Matrix.Identity(3)
    payload = load_json(flat_path)
    record = next((item for item in payload.get("objects", []) if str(item.get("object_id")) == str(object_id)), None)
    if record is None:
        return Matrix.Identity(3)
    matrix_values = record.get("world_transform", {}).get("matrix")
    if not matrix_values:
        return Matrix.Identity(3)
    matrix = Matrix(matrix_values).to_3x3()
    columns = []
    for index in range(3):
        column = matrix.col[index].copy()
        if column.length > 1e-12:
            column.normalize()
        columns.append(column)
    rotation = Matrix.Identity(3)
    for index, column in enumerate(columns):
        rotation.col[index] = column
    return rotation


def _object_views(out: Path, object_id: str, global_config):
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    record = next((item for item in plan.get("objects", []) if str(item.get("object_id")) == str(object_id)), None)
    generation = dict(record.get("generation", {})) if record else {}
    spatial = dict(record.get("spatial_category", {})) if record else {}
    explicit = generation.get("representative_views")
    if explicit:
        return build_candidate_views({"views": explicit})
    orbit_cfg = dict(DEFAULT_CANDIDATE_ORBIT)
    orbit_cfg.update(dict(global_config.get("representative_image_generation", {}).get("candidate_orbit", {})))
    orbit_cfg.update(dict(generation.get("representative_candidate_orbit", {})))
    orbit_cfg["spatial_category"] = str(spatial.get("type", "free"))
    if spatial.get("wall_inward_normal_world") is not None:
        orbit_cfg["wall_inward_normal_world"] = spatial["wall_inward_normal_world"]
    return build_candidate_views(orbit_cfg)


def _fit_perspective_distance(radius: float, fov_deg: float, margin_ratio: float = 0.80) -> float:
    half_angle = math.radians(max(1e-3, float(fov_deg)) * 0.5)
    safe_angle = max(1e-3, min(half_angle * float(margin_ratio), half_angle - 1e-3))
    return float(radius / max(math.sin(safe_angle), 1e-4))


def _create_camera(bpy, object_id, center, size, view_cfg, object_rotation):
    from mathutils import Vector
    extent = max(size)
    radius = 0.5 * math.sqrt(sum(float(value) * float(value) for value in size))
    azimuth = math.radians(float(view_cfg.get("azimuth_deg", 45.0)))
    elevation = math.radians(float(view_cfg.get("elevation_deg", 18.0)))
    projection = str(view_cfg.get("camera_projection", "ORTHO")).upper()
    horizontal = math.cos(elevation)
    if projection == "PERSP":
        distance = max(float(view_cfg.get("distance_scale", 1.0)) * radius, _fit_perspective_distance(radius, float(view_cfg.get("fov_deg", 40.0))))
    else:
        distance = float(view_cfg.get("distance_scale", 2.0)) * extent
    if view_cfg.get("world_direction") is not None:
        direction = Vector(view_cfg["world_direction"]).normalized()
        location = Vector(center) + distance * direction
    else:
        local_offset = Vector((
            distance * math.cos(azimuth) * horizontal,
            distance * math.sin(azimuth) * horizontal,
            distance * math.sin(elevation),
        ))
        location = Vector(center) + object_rotation @ local_offset
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = f"PGW_REP_CAMERA__{object_id}__{view_cfg['name']}"
    camera.data.type = projection
    if projection == "PERSP":
        camera.data.angle = math.radians(float(view_cfg.get("fov_deg", 40.0)))
    else:
        camera.data.ortho_scale = float(view_cfg.get("ortho_scale_mult", 1.75)) * extent
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    _look_at(camera, center)
    bpy.context.scene.camera = camera
    return camera


def _configure_transparent(scene):
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def _render_rgba(bpy, camera, output):
    scene = bpy.context.scene
    scene.camera = camera
    scene.use_nodes = False
    _configure_transparent(scene)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def _white_emission_material(bpy):
    name = "__PGW_MASK_WHITE__"
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    try:
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1, 1, 1, 1)
        emission.inputs["Strength"].default_value = 1
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
    except Exception:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        shader.inputs["Emission Strength"].default_value = 1
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _render_mask_rgba(bpy, camera, target_meshes, output):
    originals = {obj.name: list(obj.data.materials) for obj in target_meshes}
    material = _white_emission_material(bpy)
    try:
        for obj in target_meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        _render_rgba(bpy, camera, output)
    finally:
        for obj in target_meshes:
            obj.data.materials.clear()
            for old in originals[obj.name]:
                obj.data.materials.append(old)


def _camera_matrix(camera, scene, depsgraph):
    rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    proj = camera.calc_matrix_camera(
        depsgraph, x=rx, y=ry,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    return proj @ camera.matrix_world.inverted(), rx, ry


def _project_point(pv, point):
    clip = pv @ point.to_4d()
    if abs(float(clip.w)) < 1e-9:
        return None
    x = float(clip.x / clip.w)
    y = float(clip.y / clip.w)
    z = float(clip.z / clip.w)
    return x, y, z


def _normal_bin_name(normal):
    values = [abs(float(normal.x)), abs(float(normal.y)), abs(float(normal.z))]
    axis = values.index(max(values))
    sign = "+" if float((normal.x, normal.y, normal.z)[axis]) >= 0.0 else "-"
    axis_name = ("x", "y", "z")[axis]
    return f"{axis_name}{sign}"


def _collect_triangle_catalog(bpy, meshes):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    catalog = []
    total_area = 0.0
    part_ids = []
    triangle_areas = {}
    for obj in meshes:
        part_ids.append(obj.name)
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            normal_matrix = eval_obj.matrix_world.to_3x3().inverted().transposed()
            for local_index, tri in enumerate(mesh.loop_triangles):
                worlds = [eval_obj.matrix_world @ mesh.vertices[vi].co for vi in tri.vertices]
                center = (worlds[0] + worlds[1] + worlds[2]) / 3.0
                world_area = 0.5 * float(((worlds[1] - worlds[0]).cross(worlds[2] - worlds[0])).length)
                key = f"{obj.name}:{local_index}"
                triangle_areas[key] = world_area
                total_area += world_area
                normal = (normal_matrix @ tri.normal).normalized()
                catalog.append({
                    "triangle_id": key,
                    "part_id": obj.name,
                    "center": center,
                    "normal": normal,
                    "area": world_area,
                })
        finally:
            eval_obj.to_mesh_clear()
    return {
        "triangles": catalog,
        "areas": triangle_areas,
        "total_area": total_area,
        "part_ids": sorted(set(part_ids)),
        "part_count": len(set(part_ids)),
    }


def _measure_view(bpy, camera, meshes, catalog, bbox_points):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pv, _, _ = _camera_matrix(camera, scene, depsgraph)
    cam_pos = camera.matrix_world.translation
    visible_area = 0.0
    visible_triangle_ids = []
    visible_parts = set()
    normal_bins = defaultdict(float)
    total_area = max(float(catalog.get("total_area", 0.0)), 1e-9)
    ray_epsilon = max(1e-4, math.sqrt(total_area) * 1e-4)

    for triangle in catalog.get("triangles", []):
        center = triangle["center"]
        view_dir = (cam_pos - center)
        distance = float(view_dir.length)
        if distance <= 1e-9:
            continue
        direction = view_dir.normalized()
        frontality = max(0.0, float(triangle["normal"].dot(direction)))
        if frontality <= 1e-6:
            continue
        hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, cam_pos, (center - cam_pos).normalized(), distance=distance - 1e-5)
        if hit and hit_obj is not None:
            if hit_obj.name != triangle["part_id"]:
                continue
            if (location - center).length > ray_epsilon:
                continue
        area = float(triangle["area"])
        visible_area += area
        visible_triangle_ids.append(str(triangle["triangle_id"]))
        visible_parts.add(str(triangle["part_id"]))
        normal_bins[_normal_bin_name(triangle["normal"])] += area

    projected = [item for item in (_project_point(pv, point) for point in bbox_points) if item is not None]
    if not projected:
        completeness = 0.0
        area_ratio = 0.0
        center_offset = 1.0
    else:
        xs = [item[0] for item in projected]
        ys = [item[1] for item in projected]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin = 0.86
        inside = [(-margin <= item[0] <= margin and -margin <= item[1] <= margin and -1.0 <= item[2] <= 1.0) for item in projected]
        completeness = 1.0 if all(inside) else max(0.0, sum(1.0 for flag in inside if flag) / max(len(inside), 1))
        width = max(0.0, min(1.0, (max_x - min_x) * 0.5))
        height = max(0.0, min(1.0, (max_y - min_y) * 0.5))
        area_ratio = max(0.0, min(1.0, width * height))
        bbox_center_x = (min_x + max_x) * 0.5
        bbox_center_y = (min_y + max_y) * 0.5
        center_offset = min(1.0, math.sqrt(bbox_center_x * bbox_center_x + bbox_center_y * bbox_center_y) / math.sqrt(2.0))
    target_area = 0.42
    area_term = max(0.0, 1.0 - abs(area_ratio - target_area) / target_area)
    framing_score = max(0.0, min(1.0, 0.55 * completeness + 0.35 * area_term + 0.10 * (1.0 - center_offset)))

    total_normal = sum(normal_bins.values())
    if total_normal > 1e-9:
        entropy = 0.0
        active = 0
        for value in normal_bins.values():
            probability = float(value) / total_normal
            if probability > 1e-12:
                entropy -= probability * math.log(probability)
                active += 1
        normal_diversity = float(entropy / math.log(active)) if active > 1 else 0.0
    else:
        normal_diversity = 0.0

    return {
        "visible_area_fraction": visible_area / total_area,
        "part_coverage_fraction": len(visible_parts) / max(int(catalog.get("part_count", 1)), 1),
        "normal_diversity": normal_diversity,
        "framing_score": framing_score,
        "bbox_completeness": completeness,
        "projected_area_ratio": area_ratio,
        "center_offset_normalized": center_offset,
        "fully_inside_frustum": bool(completeness >= 1.0),
        "visible_triangle_ids": visible_triangle_ids,
        "visible_part_ids": sorted(visible_parts),
        "normal_bins": {key: float(value) for key, value in sorted(normal_bins.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--asset_config", default="configs/asset_pipeline.json")
    args = parser.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blend = out / "02_blender_scaffold" / "scaffold.blend"
    if not blend.exists():
        raise FileNotFoundError(f"Missing scaffold blend: {blend}")
    bpy.ops.wm.open_mainfile(filepath=str(blend))

    meshes = _target_meshes(bpy, args.object_id)
    if not meshes:
        raise RuntimeError(f"No scaffold mesh found for object owner: {args.object_id}")
    center, size, mins, maxs = _world_bbox(meshes)
    configure_condition_render((1024, 1024))
    config = load_json(args.asset_config) if Path(args.asset_config).exists() else {}
    view_configs = _object_views(out, args.object_id, config)
    object_rotation = _object_rotation(out, args.object_id)
    from mathutils import Vector
    bbox_points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    triangle_catalog = _collect_triangle_catalog(bpy, meshes)

    records = []
    for view_cfg in view_configs:
        camera = _create_camera(bpy, args.object_id, center, size, view_cfg, object_rotation)
        view_dir = output_dir / "views" / str(view_cfg["name"])
        view_dir.mkdir(parents=True, exist_ok=True)
        # Stage03 serves asset generation only. Stage07 independently samples
        # full-scene cameras after final assets and placements exist.
        _isolate(bpy, args.object_id)
        _render_rgba(bpy, camera, view_dir / "scaffold_rgba.png")
        depth_near, depth_far = _camera_depth_bounds(camera, meshes)
        render_depth_control_png(camera, view_dir / "depth_control.png", depth_near, depth_far, valid_min_gray=24)
        _render_mask_rgba(bpy, camera, meshes, view_dir / "mask_rgba.png")
        metrics = _measure_view(bpy, camera, meshes, triangle_catalog, bbox_points)
        records.append({
            "name": str(view_cfg["name"]),
            "ring": view_cfg.get("ring"),
            "text_hint": view_cfg.get("text_hint"),
            "azimuth_deg": float(view_cfg.get("azimuth_deg", 0.0)),
            "elevation_deg": float(view_cfg.get("elevation_deg", 0.0)),
            "output_dir": str(view_dir),
            "camera": {
                "name": camera.name,
                "location": [float(value) for value in camera.location],
                "rotation_euler": [float(value) for value in camera.rotation_euler],
                "type": camera.data.type,
                "ortho_scale": float(getattr(camera.data, "ortho_scale", 0.0)),
                "fov_deg": float(math.degrees(getattr(camera.data, "angle", 0.0))) if camera.data.type == "PERSP" else None,
                "focal_length_mm": float(getattr(camera.data, "lens", 0.0)) if camera.data.type == "PERSP" else None,
                "target": [float(value) for value in center],
            },
            "outputs": {
                "scaffold_rgba": str(view_dir / "scaffold_rgba.png"),
                "depth_control": str(view_dir / "depth_control.png"),
                "depth_encoding": "uint16_normalized_camera_z_near_bright_background_zero",
                "depth_convention": "camera_z",
                "depth_near": float(depth_near),
                "depth_far": float(depth_far),
                "mask_rgba": str(view_dir / "mask_rgba.png"),
            },
            "metrics": metrics,
            "visible_triangle_ids": metrics["visible_triangle_ids"],
            "visible_part_ids": metrics["visible_part_ids"],
        })
        bpy.data.objects.remove(camera, do_unlink=True)

    save_json({
        "status": "ok",
        "object_id": args.object_id,
        "bbox_center": center,
        "bbox_size": size,
        "bbox_min": mins,
        "bbox_max": maxs,
        "triangle_catalog": {
            "total_area": float(triangle_catalog["total_area"]),
            "part_count": int(triangle_catalog["part_count"]),
            "part_ids": triangle_catalog["part_ids"],
            "areas": triangle_catalog["areas"],
        },
        "views": records,
    }, output_dir / "capture_report.json")


if __name__ == "__main__":
    main()
