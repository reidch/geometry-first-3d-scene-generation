#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import _look_at
from src.blender.condition_renderer import (
    camera_scene_depth_bounds,
    configure_condition_render,
    render_depth_control_png,
    render_still_png,
)
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.blender.active_owner_filter import apply_active_owner_filter
from src.blender.object_identity import get_semantic_owner_id
from src.blender.semantic_render import (
    build_semantic_palette,
    collect_original_materials,
    restore_materials,
)
from src.blender.triangle_id_render import assign_triangle_id_attribute, render_triangle_id_png
from src.blender.texture_materials import (
    apply_object_texture_materials,
    configure_worldmesh_flat_render,
)
from src.io.json_io import load_json, save_json


def _visible_meshes(bpy):
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and not obj.hide_render
        and not bool(obj.get("pgw_physics_proxy", False))
    ]


def _prepare_scene_geometry(bpy, output_path: Path):
    """Assign scene-global IDs and write a compact owner/range manifest.

    v56 no longer projects every accepted camera against every scene triangle.
    The Triangle-ID image is the authoritative per-pixel correspondence, so one
    contiguous range per mesh is enough to recover owner identity on the host.
    """
    meshes = sorted(_visible_meshes(bpy), key=lambda obj: (get_semantic_owner_id(obj), obj.name))
    owners = {}
    ranges = []
    scene_base = 0
    for obj in meshes:
        if any(len(poly.vertices) != 3 for poly in obj.data.polygons):
            raise RuntimeError(f"Visible mesh is not triangulated: {obj.name}")
        owner = str(get_semantic_owner_id(obj))
        owner_base = int(obj.get("pgw_triangle_base", 0))
        count = len(obj.data.polygons)
        record = {
            "scene_triangle_start": int(scene_base),
            "scene_triangle_end_exclusive": int(scene_base + count),
            "semantic_owner_id": owner,
            "owner_triangle_start": owner_base,
            "mesh_object_name": obj.name,
        }
        ranges.append(record)
        state = owners.setdefault(
            owner,
            {"mesh_object_names": [], "triangle_count": 0, "scene_triangle_ranges": []},
        )
        state["mesh_object_names"].append(obj.name)
        state["triangle_count"] = max(int(state["triangle_count"]), owner_base + count)
        state["scene_triangle_ranges"].append(dict(record))
        assign_triangle_id_attribute(obj, scene_base)
        scene_base += count
    save_json(
        {
            "schema_version": 3,
            "id_space": "scene_global_for_stage07_reconstruction_buffers",
            "triangle_count": int(scene_base),
            "owners": owners,
            "scene_triangle_ranges": ranges,
            "per_camera_triangle_projection": False,
            "lookup": "owner_triangle_id = owner_triangle_start + scene_id - scene_triangle_start",
        },
        output_path,
    )
    return meshes, owners, int(scene_base)

def _create_reusable_camera(bpy):
    data = bpy.data.cameras.new("PGW_REFINEMENT_SHARED_CAMERA_DATA")
    camera = bpy.data.objects.new("PGW_REFINEMENT_SHARED_CAMERA", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def _update_camera(camera, record):
    camera.location = tuple(float(value) for value in record["position"])
    _look_at(camera, record["target"])
    if str(record.get("camera_type", "perspective")) == "orthographic":
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = float(record.get("ortho_scale") or 5.0)
    else:
        camera.data.type = "PERSP"
        camera.data.lens = float(record.get("focal_length", 28.0))
        camera.data.sensor_width = float(record.get("sensor_width_mm", 36.0))
        camera.data.sensor_fit = str(record.get("sensor_fit", "HORIZONTAL"))
    camera.data.clip_start = 0.03
    camera.data.clip_end = 50.0


def _matrix_rows(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _camera_calibration(camera, width: int, height: int):
    import math
    from mathutils import Matrix

    if camera.data.type != "PERSP":
        raise RuntimeError("Stage07 reconstruction buffers currently require perspective cameras")
    fx = 0.5 * float(width) / math.tan(0.5 * float(camera.data.angle_x))
    fy = 0.5 * float(height) / math.tan(0.5 * float(camera.data.angle_y))
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    c2w_blender = camera.matrix_world.copy()
    w2c_blender = c2w_blender.inverted()
    blender_camera_to_opencv = Matrix((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    w2c_opencv = blender_camera_to_opencv @ w2c_blender
    c2w_opencv = c2w_blender @ blender_camera_to_opencv
    return {
        "schema_version": 1,
        "width": int(width),
        "height": int(height),
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "camera_to_world_blender": _matrix_rows(c2w_blender),
        "world_to_camera_blender": _matrix_rows(w2c_blender),
        "camera_to_world_opencv": _matrix_rows(c2w_opencv),
        "world_to_camera_opencv": _matrix_rows(w2c_opencv),
        "blender_camera_axis": "x_right_y_up_minus_z_forward",
        "opencv_camera_axis": "x_right_y_down_z_forward",
        "pixel_center_convention": "principal point at width/2,height/2",
        "near": float(camera.data.clip_start),
        "far": float(camera.data.clip_end),
    }


def _render_world_normals(bpy, camera, output_path: Path):
    scene = bpy.context.scene
    originals = collect_original_materials()
    old = (
        scene.view_settings.view_transform,
        scene.view_settings.look,
        scene.view_settings.exposure,
        scene.view_settings.gamma,
        scene.render.film_transparent,
        scene.render.image_settings.color_depth,
        scene.render.dither_intensity,
    )
    material = bpy.data.materials.get("PGW_WORLD_NORMAL_MATERIAL")
    if material is None:
        material = bpy.data.materials.new("PGW_WORLD_NORMAL_MATERIAL")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    multiply = nodes.new("ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry.outputs["Normal"], multiply.inputs[0])
    links.new(multiply.outputs[0], add.inputs[0])
    links.new(add.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs["Surface"])
    try:
        for obj in _visible_meshes(bpy):
            obj.data.materials.clear()
            obj.data.materials.append(material)
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.render.film_transparent = False
        scene.render.image_settings.color_depth = "16"
        scene.render.dither_intensity = 0.0
        render_still_png(camera, output_path)
    finally:
        restore_materials(originals)
        scene.view_settings.view_transform = old[0]
        scene.view_settings.look = old[1]
        scene.view_settings.exposure = old[2]
        scene.view_settings.gamma = old[3]
        scene.render.film_transparent = old[4]
        scene.render.image_settings.color_depth = old[5]
        scene.render.dither_intensity = old[6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--active_owner_manifest")
    parser.add_argument("--lighting_config")
    args = parser.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_source = resolve_scene_for_textured_downstream(out)
    bpy.ops.wm.open_mainfile(filepath=str(scene_source))
    active_owner_filter = apply_active_owner_filter(bpy, args.active_owner_manifest)
    save_json(active_owner_filter, output_dir / "active_owner_filter_report.json")
    configure_condition_render((int(args.width), int(args.height)))
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    surface_records = [
        record for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) == "surface_texture"
    ]
    required_surface_owners = [str(record["object_id"]) for record in surface_records]
    lighting_config = load_json(args.lighting_config) if args.lighting_config else {}
    surface_uv_policy = {
        "status": "disabled",
        "policy": "preserve_original_surface_uvs",
        "surface_count": len(surface_records),
        "reason": "stage06_commits_generated_surface_into_exact_original_uv_bbox",
    }
    texture_binding = output_dir / "surface_texture_binding.json"
    apply_object_texture_materials(
        out / "05_texture_state",
        render_mode="albedo",
        interpolation="Linear",
        required_owner_ids=required_surface_owners,
        binding_report_path=texture_binding,
        strict=True,
        lighting_config=lighting_config,
    )
    preview_lighting = configure_worldmesh_flat_render(lighting_config)
    binding_payload = load_json(texture_binding)
    binding_payload.update({
        "scene_source": str(scene_source),
        "published_surface_scene_used": scene_source.name == "scene_surface_textured.blend",
        "stage07_rgb_mode": "worldmesh_final_flat_unlit_base_color",
        "stage07_selected_view_rgb_rendered": True,
        "stage07_selected_view_rgb_reused_by_stage08": True,
        "preview_lighting": preview_lighting,
        "surface_uv_policy": surface_uv_policy,
    })
    save_json(binding_payload, texture_binding)
    try:
        bpy.context.scene.eevee.taa_render_samples = int(args.samples)
    except Exception:
        pass
    cameras = load_json(args.camera_file)["cameras"]
    render_fingerprint = str(os.environ.get("PGW_STAGE07B_RENDER_FINGERPRINT", ""))
    manifest_path = output_dir / "triangle_owner_manifest.json"
    _meshes, owners, triangle_count = _prepare_scene_geometry(bpy, manifest_path)
    palette_path = output_dir / "semantic.palette.json"
    save_json(build_semantic_palette(_visible_meshes(bpy)), palette_path)
    camera = _create_reusable_camera(bpy)

    results = []
    total_cameras = len(cameras)
    for camera_index, record in enumerate(cameras, start=1):
        camera_id = str(record["camera_id"])
        print(
            f"[PGW_STAGE07_SHARED_PROGRESS] camera={camera_id} "
            f"index={camera_index} total={total_cameras}",
            flush=True,
        )
        directory = output_dir / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        failed = directory / ".failed"
        if failed.exists():
            failed.unlink()
        try:
            _update_camera(camera, record)
            per_camera_lighting = {
                "policy": "worldmesh_final_flat_unlit",
                "worldmesh_equivalent": "pyrender.RenderFlags.FLAT",
                "dynamic_lights": False,
            }
            rgb = directory / "rgb.png"
            depth = directory / "depth_control.png"
            normal_world = directory / "normal_world.png"
            camera_json = directory / "camera.json"
            triangle_id = directory / "triangle_id.png"
            selected_marker = directory / "selected.txt"
            required = [rgb, depth, normal_world, camera_json, palette_path, triangle_id]
            if all(path.exists() and path.stat().st_size > 0 for path in required):
                try:
                    existing_report = load_json(directory / "camera_report.json")
                except Exception:
                    existing_report = None
                if (
                    existing_report is not None
                    and str(existing_report.get("status", "ok")) == "ok"
                    and str(existing_report.get("render_fingerprint", "")) == render_fingerprint
                ):
                    selected_marker.touch(exist_ok=True)
                    existing_report["selected_marker"] = str(selected_marker)
                    save_json(existing_report, directory / "camera_report.json")
                    results.append(existing_report)
                    continue
            # WorldMesh final pipeline uses --flat-lighting, which skips all lights
            # and renders base color with RenderFlags.FLAT.  The Blender equivalent
            # here is an emission/albedo material pass with every light disabled.
            render_still_png(camera, rgb)
            depth_near, depth_far = camera_scene_depth_bounds(camera)
            render_depth_control_png(camera, depth, depth_near, depth_far, valid_min_gray=24)
            _render_world_normals(bpy, camera, normal_world)
            save_json(_camera_calibration(camera, int(args.width), int(args.height)), camera_json)
            render_triangle_id_png(camera, triangle_id)
            missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
            if missing:
                raise RuntimeError("Shared candidate buffer render incomplete: " + ", ".join(missing))
            # Empty marker created only after every selected-view output is complete.
            selected_marker.touch(exist_ok=True)
            camera_report = {
                "camera_id": camera_id,
                "status": "ok",
                "render_fingerprint": render_fingerprint,
                "resolution": [int(args.width), int(args.height)],
                "buffer_transport": "standard_images_and_json",
                "depth_encoding": {
                    "type": "uint16_normalized_camera_z_near_bright_background_zero",
                    "depth_convention": "camera_z",
                    "near": float(depth_near),
                    "far": float(depth_far),
                    "valid_min_gray": 24,
                },
                "rgb": str(rgb),
                "rgb_source": "stage07_selected_view_beauty_render",
                "rgb_render_mode": "worldmesh_final_flat_unlit_base_color",
                "per_camera_lighting": per_camera_lighting,
                "depth": str(depth),
                "normal_world": str(normal_world),
                "normal_encoding": {
                    "type": "uint16_world_xyz_mapped_minus1_plus1_to_0_1",
                    "decode": "normal = rgb * 2 - 1; renormalize",
                },
                "camera": str(camera_json),
                "semantic": None,
                "palette": str(palette_path),
                "semantic_synthesized_on_host_from_triangle_id": True,
                "triangle_id": str(triangle_id),
                "selected_marker": str(selected_marker),
                "per_camera_triangle_projection": False,
                "uv_rendered": False,
            }
            save_json(camera_report, directory / "camera_report.json")
            results.append(camera_report)
        except Exception:
            detail = traceback.format_exc()
            failed.write_text(detail, encoding="utf-8")
            results.append({"camera_id": camera_id, "status": "failed", "error": detail[-4000:]})

    report = {
        "status": "ok" if any(item.get("status") == "ok" for item in results) else "failed",
        "camera_count": len(cameras),
        "success_count": sum(item.get("status") == "ok" for item in results),
        "failure_count": sum(item.get("status") != "ok" for item in results),
        "resolution": [int(args.width), int(args.height)],
        "samples": int(args.samples),
        "selected_view_rgb_rendered": True,
        "selected_view_rgb_reused_by_stage08": True,
        "triangle_count": triangle_count,
        "triangle_owner_manifest": str(manifest_path),
        "semantic_palette": str(palette_path),
        "surface_texture_binding": str(texture_binding),
        "active_owner_filter": active_owner_filter,
        "render_fingerprint": render_fingerprint,
        "lighting_config": lighting_config,
        "lighting_config_path": str(args.lighting_config or ""),
        "preview_lighting": preview_lighting,
        "results": results,
    }
    save_json(report, output_dir / "batch_report.json")
    if report["success_count"] == 0:
        raise RuntimeError("All shared refinement buffer renders failed")


if __name__ == "__main__":
    main()
