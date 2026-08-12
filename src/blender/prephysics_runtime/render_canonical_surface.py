#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.atlas_condition_renderer import render_uv_png_bundle, uv_png_bundle_paths
from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import _look_at
from src.blender.condition_renderer import configure_condition_render, render_depth_control_png, render_still_png
from src.blender.object_identity import get_semantic_owner_id
from src.blender.semantic_render import collect_original_materials, restore_materials
from src.blender.texture_materials import apply_object_texture_materials
from src.io.json_io import load_json, save_json
from src.blender.triangle_id_render import render_triangle_id_png


def _find_source_object(document, target_object_id):
    scene = dict(document.get("scene", {}))
    stack = list(scene.get("objects", []))
    target = str(target_object_id)
    while stack:
        record = stack.pop()
        if not isinstance(record, dict):
            continue
        if str(record.get("id", "")) == target:
            return record
        stack.extend(record.get("children", []))
    return {}


def _camera_depth_bounds(camera, objects):
    """Return finite camera-space depth bounds for visible target geometry."""
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
    return max(float(camera.data.clip_start), near - padding), min(float(camera.data.clip_end), far + padding)

def _semantic_owner_id(obj):
    return get_semantic_owner_id(obj)


def _visual_target_meshes(bpy, object_id):
    return [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and _semantic_owner_id(obj) == str(object_id)
        and not bool(obj.get("pgw_physics_proxy", False))
    ]


def _isolate_target(bpy, object_id):
    target_mesh_count = 0
    hidden_mesh_count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        show = _semantic_owner_id(obj) == str(object_id) and not bool(obj.get("pgw_physics_proxy", False))
        obj.hide_render = not show
        obj.hide_viewport = not show
        if show:
            target_mesh_count += 1
        else:
            hidden_mesh_count += 1
    return {
        "policy": "target_surface_only",
        "target_mesh_count": int(target_mesh_count),
        "hidden_non_target_mesh_count": int(hidden_mesh_count),
        "non_target_meshes_hidden": True,
            }


def _bbox_world(objects):
    from mathutils import Vector
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = [min(point[index] for point in points) for index in range(3)]
    maximum = [max(point[index] for point in points) for index in range(3)]
    center = [(minimum[index] + maximum[index]) * 0.5 for index in range(3)]
    size = [max(maximum[index] - minimum[index], 1e-4) for index in range(3)]
    return center, size


def _projection_vector(surface):
    mapping = {
        "top": [0.0, 0.0, 1.0],
        "bottom": [0.0, 0.0, -1.0],
        "front": [0.0, -1.0, 0.0],
        "back": [0.0, 1.0, 0.0],
        "side": [1.0, 0.0, 0.0],
        "opposite_side": [-1.0, 0.0, 0.0],
    }
    if isinstance(surface.get("view_from_local"), list):
        return [float(value) for value in surface["view_from_local"]]
    return mapping.get(str(surface.get("projection", "front")), [0.0, -1.0, 0.0])


def _resolve_world_view_direction(record, surface, target_objects):
    from mathutils import Vector

    explicit_world = surface.get("view_from_world")
    if isinstance(explicit_world, list) and len(explicit_world) == 3:
        direction = Vector([float(value) for value in explicit_world])
        source = "generation.surface.view_from_world"
    else:
        spatial = dict(record.get("spatial_category", {}))
        inward = spatial.get("wall_inward_normal_world")
        if isinstance(inward, list) and len(inward) == 3:
            # This is explicit JSON geometry metadata, not a hard-coded object or
            # semantic name.  It guarantees that a wall is captured from the room
            # interior rather than accidentally writing the exterior slab face.
            direction = Vector([float(value) for value in inward])
            source = "spatial_category.wall_inward_normal_world"
        else:
            local_direction = Vector(_projection_vector(surface))
            frame = target_objects[0].matrix_world.to_3x3()
            direction = frame @ local_direction
            source = "generation.surface.projection_in_object_frame"
    if direction.length <= 1e-10:
        raise ValueError(f"Canonical view direction is zero for {record.get('object_id')}")
    direction.normalize()
    return direction, source


def _round_resolution(value, multiple):
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _camera_plane_span(camera, objects):
    from mathutils import Vector

    inverse = camera.matrix_world.inverted()
    points = [inverse @ (obj.matrix_world @ Vector(corner)) for obj in objects for corner in obj.bound_box]
    x_values = [float(point.x) for point in points]
    y_values = [float(point.y) for point in points]
    return max(max(x_values) - min(x_values), 1e-4), max(max(y_values) - min(y_values), 1e-4)


def _capture_resolution(surface, horizontal_span, vertical_span):
    long_edge = max(256, int(surface.get("resolution", 1024)))
    minimum_short = max(256, int(surface.get("minimum_short_edge_resolution", 512)))
    multiple = max(8, int(surface.get("resolution_multiple", 64)))
    aspect = float(horizontal_span / max(vertical_span, 1e-8))
    if aspect >= 1.0:
        width = long_edge
        height = max(minimum_short, int(round(long_edge / aspect)))
    else:
        height = long_edge
        width = max(minimum_short, int(round(long_edge * aspect)))
    return _round_resolution(width, multiple), _round_resolution(height, multiple)


def _create_camera(bpy, object_id, center, size, record, surface, target_objects):
    from mathutils import Vector
    target = Vector(center)
    world_direction, direction_source = _resolve_world_view_direction(record, surface, target_objects)
    distance = max(max(size) * float(surface.get("distance_scale", 1.2)), 1.0)
    location = target + world_direction * distance
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = f"PGW_CANONICAL__{object_id}"
    camera.data.type = "ORTHO"
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    _look_at(camera, target)
    bpy.context.view_layer.update()
    horizontal_span, vertical_span = _camera_plane_span(camera, target_objects)
    width, height = _capture_resolution(surface, horizontal_span, vertical_span)
    aspect = float(width / max(height, 1))
    framing_margin = float(surface.get("framing_margin", 1.055))
    legacy_multiplier = float(surface.get("ortho_scale_multiplier", 1.0))
    camera.data.ortho_scale = max(
        vertical_span * framing_margin,
        horizontal_span * framing_margin / max(aspect, 1e-8),
    ) * legacy_multiplier
    bpy.context.scene.camera = camera
    return (
        camera,
        direction_source,
        [float(value) for value in world_direction],
        [int(width), int(height)],
        {
            "horizontal_span_world": float(horizontal_span),
            "vertical_span_world": float(vertical_span),
            "frame_aspect": aspect,
            "framing_margin": framing_margin,
        },
    )

def _mask_material(bpy):
    mat = bpy.data.materials.get("__PGW_SURFACE_MASK__") or bpy.data.materials.new("__PGW_SURFACE_MASK__")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    try:
        em = nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1, 1, 1, 1)
        em.inputs["Strength"].default_value = 1
        links.new(em.outputs["Emission"], out.inputs["Surface"])
    except Exception:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = 1
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _render_mask(bpy, camera, objects, output):
    originals = collect_original_materials()
    scene = bpy.context.scene
    old_transparent = scene.render.film_transparent
    try:
        mat = _mask_material(bpy)
        for obj in objects:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.camera = camera
        scene.use_nodes = False
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.film_transparent = old_transparent
        restore_materials(originals)


def _triangle_manifest(bpy, camera, target_object, output_path):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    proj = camera.calc_matrix_camera(
        depsgraph, x=rx, y=ry,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    view = camera.matrix_world.inverted()
    pv = proj @ view
    cam_pos = camera.matrix_world.translation
    triangles = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if _semantic_owner_id(obj) != str(target_object) or bool(obj.get("pgw_physics_proxy", False)):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
            if uv_layer is None:
                raise RuntimeError(f"No UV layer for {obj.name}")
            normal_matrix = eval_obj.matrix_world.to_3x3().inverted().transposed()
            base = int(obj.get("pgw_triangle_base", 0))
            for local_index, tri in enumerate(mesh.polygons):
                if len(tri.vertices) != 3:
                    raise RuntimeError(f"Visible mesh is not triangulated: {obj.name}")
                clips, uvs, worlds = [], [], []
                for loop_index in tri.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    world = eval_obj.matrix_world @ mesh.vertices[vertex_index].co
                    clip = pv @ world.to_4d()
                    uv = uv_layer[loop_index].uv
                    worlds.append(world)
                    clips.append([float(clip.x), float(clip.y), float(clip.z), float(clip.w)])
                    uvs.append([float(uv.x), float(uv.y)])
                center = (worlds[0] + worlds[1] + worlds[2]) / 3.0
                view_dir = (cam_pos - center).normalized()
                normal = (normal_matrix @ tri.normal).normalized()
                world_area = 0.5 * (worlds[1] - worlds[0]).cross(worlds[2] - worlds[0]).length
                triangles.append({
                    "global_triangle_id": int(base + local_index),
                    "mesh_object_name": obj.name,
                    "local_triangle_id": int(local_index),
                    "uv": uvs,
                    "clip": clips,
                    "world_area": float(world_area),
                    "frontality": max(0.0, float(normal.dot(view_dir))),
                })
        finally:
            eval_obj.to_mesh_clear()
    save_json({
        "target_object": str(target_object),
        "camera_id": camera.name,
        "resolution": [rx, ry],
        "triangles": triangles,
    }, output_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_json", required=True)
    args = parser.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(out / "05_scene_assets" / "scene_assets.blend"))
    apply_object_texture_materials(out / "05_texture_state", render_mode="albedo", interpolation="Linear")
    visibility = _isolate_target(bpy, args.object_id)
    objects = _visual_target_meshes(bpy, args.object_id)
    if not objects:
        raise RuntimeError(f"No visible target meshes for canonical surface object {args.object_id}")
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    record = next((item for item in plan.get("objects", []) if str(item["object_id"]) == str(args.object_id)), None)
    if record is None:
        raise KeyError(f"Object is absent from generation plan: {args.object_id}")
    source_record = _find_source_object(load_json(args.scene_json), args.object_id)
    view_record = dict(record)
    view_record["spatial_category"] = dict(source_record.get("spatial_category", {}))
    surface = dict(record.get("generation", {})).get("surface", {})
    center, size = _bbox_world(objects)
    camera, view_direction_source, view_direction_world, capture_resolution, framing = _create_camera(
        bpy, args.object_id, center, size, view_record, surface, objects
    )
    configure_condition_render(tuple(capture_resolution))
    render_still_png(camera, directory / "rgb.png")
    depth_near, depth_far = _camera_depth_bounds(camera, objects)
    render_depth_control_png(
        camera,
        directory / "depth_control.png",
        depth_near,
        depth_far,
        valid_min_gray=24,
    )
    _render_mask(bpy, camera, objects, directory / "mask_rgba.png")
    render_uv_png_bundle(camera, directory / "uv_map.json")
    render_triangle_id_png(camera, directory / "triangle_id.png")
    _triangle_manifest(bpy, camera, args.object_id, directory / "triangles.json")
    save_json(
        {
            "status": "ok",
            "object_id": args.object_id,
            "surface_spec": surface,
            "camera": {
                "camera_id": camera.name,
                "position": list(camera.location),
                "rotation_euler": list(camera.rotation_euler),
                "camera_type": "orthographic",
                "ortho_scale": float(camera.data.ortho_scale),
                "target": center,
                "view_direction_source": view_direction_source,
                "view_direction_world": view_direction_world,
                "capture_resolution": capture_resolution,
                "framing": framing,
            },
            "visibility": visibility,
            "depth_control": {
                "path": str(directory / "depth_control.png"),
                "encoding": "normalized_camera_z_near_bright_background_zero",
                "depth_convention": "camera_z",
                "depth_near": float(depth_near),
                "depth_far": float(depth_far),
                "valid_min_gray": 24,
                "bit_depth": 16,
            },
            "uv_map": str(directory / "uv_map.json"),
            "buffer_transport": "standard_images_and_json",
            "triangle_id": str(directory / "triangle_id.png"),
        },
        directory / "capture_report.json",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--output_dir" in argv:
            directory = Path(argv[argv.index("--output_dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise
