from __future__ import annotations

import json
from pathlib import Path

from src.blender.atlas_condition_renderer import render_uv_png_bundle, uv_png_bundle_paths
from src.blender.blender_runtime import require_bpy
from src.blender.camera_calibration import export_camera_calibration
from src.blender.camera_utils import set_active_camera
from src.blender.object_identity import (
    get_runtime_object_id,
    get_semantic_owner_id,
    get_world_object_id,
)
from src.blender.semantic_render import (
    apply_object_semantic_materials,
    collect_original_materials,
    restore_materials,
)


def configure_condition_render(resolution=(1024, 576)):
    bpy = require_bpy()
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = -1.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    try:
        scene.eevee.taa_render_samples = 64
    except Exception:
        pass


def render_still_png(camera, out_path):
    bpy = require_bpy()
    scene = bpy.context.scene
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    scene.use_nodes = False
    scene.render.filepath = str(out_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)


def _clear_compositor_nodes(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    return tree


def _rename_latest_png(directory, prefix, final_path):
    directory = Path(directory)
    final_path = Path(final_path)
    candidates = list(directory.glob(prefix + "*.png"))
    if not candidates:
        raise RuntimeError(
            "Blender did not write the expected PNG output: "
            f"directory={directory}, prefix={prefix}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    if final_path.exists():
        final_path.unlink()
    latest.rename(final_path)
    return final_path


def camera_scene_depth_bounds(camera, objects=None, margin_fraction=0.03):
    """Compute useful positive camera-Z bounds from visible mesh AABBs."""
    bpy = require_bpy()
    from mathutils import Vector

    if objects is None:
        objects = [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH"
            and not bool(getattr(obj, "hide_render", False))
            and not bool(obj.hide_get())
        ]
    inverse = camera.matrix_world.inverted()
    depths = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            camera_point = inverse @ world
            depth = -float(camera_point.z)
            if depth > 1e-6:
                depths.append(depth)
    if not depths:
        near = max(float(camera.data.clip_start), 0.01)
        far = max(near + 1.0, min(float(camera.data.clip_end), near + 10.0))
        return near, far

    near = min(depths)
    far = max(depths)
    span = max(far - near, 1e-3)
    margin = max(0.01, span * float(margin_fraction))
    near = max(float(camera.data.clip_start), near - margin)
    far = min(float(camera.data.clip_end), far + margin)
    if far <= near:
        far = near + max(0.01, span)
    return float(near), float(far)


def render_depth_control_png(camera, depth_path, depth_near, depth_far, valid_min_gray=24):
    """Render a normalized 16-bit depth image directly in Blender.

    Zero is background. Valid geometry occupies [valid_min_gray/255, 1], with
    nearer geometry brighter. No floating image format or Python image plugin is
    required downstream.
    """
    bpy = require_bpy()
    scene = bpy.context.scene
    scene.camera = camera
    depth_path = Path(depth_path)
    depth_path.parent.mkdir(parents=True, exist_ok=True)

    near = float(depth_near)
    far = float(depth_far)
    if not (far > near):
        far = near + 1e-4
    floor_value = max(0.0, min(1.0, float(valid_min_gray) / 255.0))

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True

    old_film_transparent = bool(scene.render.film_transparent)
    scene.render.film_transparent = True
    tree = _clear_compositor_nodes(scene)
    nodes = tree.nodes
    links = tree.links

    render_layers = nodes.new(type="CompositorNodeRLayers")
    if "Depth" not in render_layers.outputs or "Alpha" not in render_layers.outputs:
        raise RuntimeError("Render Layers must expose Depth and Alpha outputs.")

    subtract_near = nodes.new(type="CompositorNodeMath")
    subtract_near.operation = "SUBTRACT"
    subtract_near.inputs[1].default_value = near

    divide_range = nodes.new(type="CompositorNodeMath")
    divide_range.operation = "DIVIDE"
    divide_range.inputs[1].default_value = far - near
    divide_range.use_clamp = True

    invert = nodes.new(type="CompositorNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    invert.use_clamp = True

    scale = nodes.new(type="CompositorNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 1.0 - floor_value

    add_floor = nodes.new(type="CompositorNodeMath")
    add_floor.operation = "ADD"
    add_floor.inputs[1].default_value = floor_value
    add_floor.use_clamp = True

    apply_alpha = nodes.new(type="CompositorNodeMath")
    apply_alpha.operation = "MULTIPLY"
    apply_alpha.use_clamp = True

    output = nodes.new(type="CompositorNodeOutputFile")
    output.base_path = str(depth_path.parent)
    output.file_slots[0].path = depth_path.stem + "_"
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "16"
    output.format.compression = 15

    links.new(render_layers.outputs["Depth"], subtract_near.inputs[0])
    links.new(subtract_near.outputs[0], divide_range.inputs[0])
    links.new(divide_range.outputs[0], invert.inputs[1])
    links.new(invert.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], add_floor.inputs[0])
    links.new(add_floor.outputs[0], apply_alpha.inputs[0])
    links.new(render_layers.outputs["Alpha"], apply_alpha.inputs[1])
    links.new(apply_alpha.outputs[0], output.inputs[0])

    try:
        bpy.ops.render.render(write_still=False)
        _rename_latest_png(depth_path.parent, depth_path.stem + "_", depth_path)
    finally:
        scene.use_nodes = False
        scene.render.film_transparent = old_film_transparent

    return depth_path


def render_camera_conditions(cam_data, out_dirs, resolution=(1024, 576)):
    """Render the active condition set using only PNG images and JSON text."""
    bpy = require_bpy()
    scene = bpy.context.scene
    configure_condition_render(resolution)
    camera = set_active_camera(cam_data)
    camera_id = cam_data["camera_id"]

    rgb_path = Path(out_dirs["rgb_scaffold"]) / (camera_id + ".png")
    depth_path = Path(out_dirs["depth"]) / (camera_id + ".png")
    semantic_path = Path(out_dirs["semantic_object"]) / (camera_id + ".png")
    uv_path = Path(out_dirs["uv_map"]) / (camera_id + ".uv_map.json")
    calibration_path = Path(out_dirs["camera_calibration"]) / (camera_id + ".json")

    depth_near, depth_far = camera_scene_depth_bounds(camera)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration = export_camera_calibration(camera, cam_data, scene)
    calibration["condition_buffers"] = {
        "transport": "standard_images_and_json",
        "depth": {
            "path": str(depth_path),
            "encoding": "uint16_normalized_camera_z_near_bright_background_zero",
            "depth_convention": "camera_z",
            "near": depth_near,
            "far": depth_far,
        },
        "uv": {
            "path": str(uv_path),
            "encoding": "json_manifest_with_uint16_u_v_png_and_valid_png",
        },
    }
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    render_still_png(camera, rgb_path)
    render_depth_control_png(camera, depth_path, depth_near, depth_far)
    render_uv_png_bundle(camera, uv_path)

    originals = collect_original_materials()
    palette = apply_object_semantic_materials()
    old_transform = scene.view_settings.view_transform
    old_look = scene.view_settings.look
    old_exposure = scene.view_settings.exposure
    old_gamma = scene.view_settings.gamma
    old_transparent = scene.render.film_transparent
    old_dither = float(scene.render.dither_intensity)
    old_world_color = tuple(scene.world.color) if scene.world is not None else None
    try:
        try:
            scene.view_settings.view_transform = "Raw"
            scene.view_settings.look = "None"
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0
        except Exception:
            pass
        scene.render.film_transparent = False
        scene.render.dither_intensity = 0.0
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)
        render_still_png(camera, semantic_path)
    finally:
        restore_materials(originals)
        try:
            scene.view_settings.view_transform = old_transform
            scene.view_settings.look = old_look
            scene.view_settings.exposure = old_exposure
            scene.view_settings.gamma = old_gamma
        except Exception:
            pass
        scene.render.film_transparent = old_transparent
        scene.render.dither_intensity = old_dither
        if scene.world is not None and old_world_color is not None:
            scene.world.color = old_world_color

    expected_outputs = [rgb_path, depth_path, semantic_path, uv_path, *uv_png_bundle_paths(uv_path), calibration_path]
    missing_outputs = [
        str(path) for path in expected_outputs if not Path(path).exists() or Path(path).stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            "Condition rendering did not produce all required outputs for "
            + camera_id
            + ":\n"
            + "\n".join("- " + item for item in missing_outputs)
        )

    return {
        "camera_id": camera_id,
        "rgb_scaffold": str(rgb_path),
        "depth": str(depth_path),
        "semantic_object": str(semantic_path),
        "uv_map": str(uv_path),
        "camera_calibration": str(calibration_path),
        "semantic_palette": palette,
        "transport": "standard_images_and_json",
    }


def collect_visibility_metadata(cam_data):
    bpy = require_bpy()
    visible = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            visible.append(
                {
                    "blender_object_name": obj.name,
                    "world_object_id": get_world_object_id(obj),
                    "runtime_object_id": get_runtime_object_id(obj),
                    "semantic_owner_id": get_semantic_owner_id(obj),
                    "object_name": get_world_object_id(obj),
                    "object_id": get_runtime_object_id(obj),
                    "semantic_class": obj.get("semantic_class", "unknown"),
                    "part_id": obj.get("part_id", None),
                }
            )
    return {
        "camera": cam_data,
        "visible_candidates": visible,
        "note": "Candidate mesh list. Pixel visibility is determined by PNG semantic buffers.",
    }
