#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import _look_at
from src.blender.condition_renderer import configure_condition_render
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.blender.object_identity import get_semantic_owner_id
from src.blender.active_owner_filter import apply_active_owner_filter
from src.blender.triangle_id_render import ATTRIBUTE_NAME, _triangle_id_material, assign_triangle_id_attribute
from src.io.json_io import load_json, save_json



def _save_json_atomic(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)

def _visible_meshes(bpy):
    return [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render and not bool(obj.get("pgw_physics_proxy", False))
    ]


def _prepare_scene_triangle_ids(bpy, output_path: Path):
    """Assign scene-global Triangle IDs and publish a compact range manifest.

    Candidate acceptance no longer needs per-triangle world coordinates, normals,
    clip projection, or a million-triangle frustum cache.  Each mesh contributes
    one contiguous scene-ID range, which is sufficient to recover semantic owner
    and owner-local triangle IDs from the low-resolution Triangle ID images.
    """
    meshes = sorted(_visible_meshes(bpy), key=lambda obj: (get_semantic_owner_id(obj), obj.name))
    owner_ids = sorted({str(get_semantic_owner_id(obj)) for obj in meshes})
    owners = {
        owner: {
            "mesh_object_names": [],
            "triangle_count": 0,
            "actual_triangle_count": 0,
            "scene_triangle_ranges": [],
        }
        for owner in owner_ids
    }
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
            "owner_triangle_start": int(owner_base),
            "mesh_object_name": obj.name,
        }
        ranges.append(record)
        owners[owner]["scene_triangle_ranges"].append(dict(record))
        owners[owner]["mesh_object_names"].append(obj.name)
        owners[owner]["triangle_count"] = max(
            int(owners[owner]["triangle_count"]),
            owner_base + count,
        )
        owners[owner]["actual_triangle_count"] += count
        assign_triangle_id_attribute(obj, scene_base)
        scene_base += count

    save_json({
        "schema_version": 3,
        "id_space": "scene_global_for_stage07_candidate_render",
        "triangle_count": int(scene_base),
        "owners": owners,
        "scene_triangle_ranges": ranges,
        "candidate_lookup_policy": {
            "storage": "one contiguous scene-ID range per mesh object",
            "owner_triangle_id": "owner_triangle_start + scene_triangle_id - scene_triangle_start",
            "per_triangle_world_geometry_cached": False,
            "purpose": "candidate target masks, visible semantic grouping, and occlusion only",
        },
    }, output_path)
    return meshes


def _configure_id_materials(bpy, meshes, resolution, samples):
    scene = bpy.context.scene
    configure_condition_render(tuple(int(value) for value in resolution))
    try:
        scene.eevee.taa_render_samples = int(samples)
    except Exception:
        pass
    mat = _triangle_id_material()
    for obj in meshes:
        if obj.data.color_attributes.get(ATTRIBUTE_NAME) is None:
            raise RuntimeError(f"Visible mesh lacks {ATTRIBUTE_NAME}: {obj.name}")
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
    scene.use_nodes = False
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.film_transparent = False
    scene.render.filter_size = 0.01
    scene.render.dither_intensity = 0.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    if scene.world is not None:
        scene.world.color = (0.0, 0.0, 0.0)


def _create_reusable_camera(bpy):
    data = bpy.data.cameras.new("PGW_REFINEMENT_BATCH_CAMERA_DATA")
    camera = bpy.data.objects.new("PGW_REFINEMENT_BATCH_CAMERA", data)
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


def _render_one(bpy, camera, output_path: Path):
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.filepath = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Blender did not write triangle ID image: {output_path}")
    # Deliberately do not decode the PNG in Blender's private Python runtime.
    # Blender installations frequently do not include NumPy/Pillow, while the
    # host project environment does. Exact ID validation is performed by the
    # parent process in ``evaluate_candidate`` after Blender exits.
    return {
        "validation_runtime": "host_project_python",
        "validation_deferred": True,
        "file_size_bytes": int(output_path.stat().st_size),
    }



def _rename_latest_png(directory: Path, prefix: str, final_path: Path):
    candidates = list(Path(directory).glob(prefix + "*.png"))
    if not candidates:
        raise RuntimeError(
            f"Blender did not write the expected PNG: directory={directory}, prefix={prefix}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    latest = candidates[0]
    if final_path.exists():
        final_path.unlink()
    latest.rename(final_path)


def _render_triangle_id_with_depth(
    bpy,
    camera,
    triangle_id_path: Path,
    depth_path: Path,
    *,
    depth_near: float,
    depth_far: float,
    valid_min_gray: int,
):
    """Write Triangle ID and linear-depth PNG from one Blender render."""
    scene = bpy.context.scene
    triangle_id_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    scene.render.filepath = str(triangle_id_path)
    scene.render.filepath = str(triangle_id_path)
    near = float(depth_near)
    far = float(depth_far)
    if far <= near:
        raise ValueError(f"Invalid Stage07 target-depth bounds: near={near}, far={far}")
    floor_value = max(0.0, min(1.0, float(valid_min_gray) / 255.0))

    scene.view_layers[0].use_pass_z = True
    old_film_transparent = bool(scene.render.film_transparent)
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    nodes = tree.nodes
    links = tree.links

    render_layers = nodes.new(type="CompositorNodeRLayers")
    composite = nodes.new(type="CompositorNodeComposite")
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])

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
    links.new(render_layers.outputs["Alpha"], apply_alpha.inputs[1])
    links.new(add_floor.outputs[0], apply_alpha.inputs[0])
    links.new(apply_alpha.outputs[0], output.inputs[0])

    try:
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        _rename_latest_png(depth_path.parent, depth_path.stem + "_", depth_path)
    finally:
        scene.use_nodes = False
        scene.render.film_transparent = old_film_transparent

    for path in (triangle_id_path, depth_path):
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Blender did not write Stage07 isolated buffer: {path}")
    return {
        "validation_runtime": "host_project_python",
        "validation_deferred": True,
        "triangle_id_file_size_bytes": int(triangle_id_path.stat().st_size),
        "depth_file_size_bytes": int(depth_path.stat().st_size),
        "single_render_multi_output": True,
    }

def _render_target_isolated_buffers(
    bpy,
    meshes,
    camera,
    target_owner_id: str,
    triangle_id_path: Path,
    depth_path: Path,
    *,
    valid_min_gray: int = 24,
):
    """Render target-isolated Triangle ID and depth in one pass."""
    previous_hide_render = {obj.name: bool(obj.hide_render) for obj in meshes}
    try:
        for obj in meshes:
            owner = str(get_semantic_owner_id(obj))
            generation_mode = str(obj.get("generation_mode", ""))
            obj.hide_render = not (
                owner == str(target_owner_id)
                or generation_mode == "surface_texture"
            )
        return _render_triangle_id_with_depth(
            bpy,
            camera,
            triangle_id_path,
            depth_path,
            depth_near=float(camera.data.clip_start),
            depth_far=float(camera.data.clip_end),
            valid_min_gray=int(valid_min_gray),
        )
    finally:
        for obj in meshes:
            obj.hide_render = previous_hide_render[obj.name]
        bpy.context.view_layer.update()


def _render_isolated_record(bpy, meshes, camera, record, directory: Path):
    camera_id = str(record["camera_id"])
    directory.mkdir(parents=True, exist_ok=True)
    _update_camera(camera, record)
    isolated_id = directory / "target_isolated_triangle_id.png"
    isolated_depth = directory / "target_isolated_depth.png"
    valid_min_gray = int(record.get("target_depth_valid_min_gray", 24))
    diagnostics = _render_target_isolated_buffers(
        bpy,
        meshes,
        camera,
        str(record.get("target_object_id") or record.get("target_source") or ""),
        isolated_id,
        isolated_depth,
        valid_min_gray=valid_min_gray,
    )
    report = {
        "camera_id": camera_id,
        "status": "ok",
        "target_isolated_triangle_id": str(isolated_id),
        "target_isolated_depth": str(isolated_depth),
        "target_isolated_depth_encoding": {
            "type": "uint16_normalized_camera_z_near_bright_background_zero",
            "depth_convention": "camera_z",
            "near": float(camera.data.clip_start),
            "far": float(camera.data.clip_end),
            "valid_min_gray": valid_min_gray,
        },
        "target_isolated_render_diagnostics": diagnostics,
    }
    save_json(report, directory / "isolated_render_report.json")
    return report



def _matrix_rows(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _camera_calibration(camera, width: int, height: int):
    import math
    from mathutils import Matrix

    fx = 0.5 * float(width) / math.tan(0.5 * float(camera.data.angle_x))
    fy = 0.5 * float(height) / math.tan(0.5 * float(camera.data.angle_y))
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    c2w_blender = camera.matrix_world.copy()
    w2c_blender = c2w_blender.inverted()
    blender_camera_to_opencv = Matrix(((1.0,0.0,0.0,0.0),(0.0,-1.0,0.0,0.0),(0.0,0.0,-1.0,0.0),(0.0,0.0,0.0,1.0)))
    return {
        "width": int(width), "height": int(height),
        "K": [[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]],
        "camera_to_world_opencv": _matrix_rows(c2w_blender @ blender_camera_to_opencv),
        "world_to_camera_opencv": _matrix_rows(blender_camera_to_opencv @ w2c_blender),
        "pixel_center_offset": 0.5,
    }

def _render_full_record(bpy, camera, record, directory: Path):
    """Render the full-scene ID and metric depth in one compositor pass.

    Stage07 candidate acceptance uses JSON-relative significant-semantic
    count ratios together with robust depth diversity and near-depth occupancy.
    The target must be visible but can remain below the significance threshold.
    """
    camera_id = str(record["camera_id"])
    directory.mkdir(parents=True, exist_ok=True)
    _update_camera(camera, record)
    image_path = directory / "triangle_id.png"
    depth_path = directory / "depth_control.png"
    valid_min_gray = int(record.get("target_depth_valid_min_gray", 24))
    diagnostics = _render_triangle_id_with_depth(
        bpy,
        camera,
        image_path,
        depth_path,
        depth_near=float(camera.data.clip_start),
        depth_far=float(camera.data.clip_end),
        valid_min_gray=valid_min_gray,
    )
    report = {
        "camera_id": camera_id,
        "status": "ok",
        "triangle_id": str(image_path),
        "depth": str(depth_path),
        "depth_encoding": {
            "type": "uint16_normalized_camera_z_near_bright_background_zero",
            "depth_convention": "camera_z",
            "near": float(camera.data.clip_start),
            "far": float(camera.data.clip_end),
            "valid_min_gray": valid_min_gray,
        },
        "camera_calibration": _camera_calibration(
            camera, int(bpy.context.scene.render.resolution_x), int(bpy.context.scene.render.resolution_y)
        ),
        "render_diagnostics": diagnostics,
    }
    save_json(report, directory / "full_render_report.json")
    return report


def _render_record(bpy, meshes, camera, record, directory: Path):
    """Compatibility batch path: isolated buffers first, then full-scene ID."""
    isolated = _render_isolated_record(bpy, meshes, camera, record, directory)
    full = _render_full_record(bpy, camera, record, directory)
    report = {**isolated, **full, "status": "ok"}
    save_json(report, directory / "render_report.json")
    return report


def _run_worker_loop(bpy, meshes, camera, manifest_path: Path, output_dir: Path, ready_file: Path, resolution, samples):
    _save_json_atomic({
        "status": "ready",
        "triangle_owner_manifest": str(manifest_path),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "samples": int(samples),
    }, ready_file)
    processed = 0
    succeeded = 0
    failed_count = 0
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except Exception:
            print("[PGW_STAGE07_WORKER] ignored invalid JSON command", flush=True)
            continue
        action = str(command.get("action", ""))
        if action == "shutdown":
            break
        if action not in {"render", "render_isolated", "render_full"}:
            continue
        record = dict(command.get("camera") or {})
        camera_id = str(record.get("camera_id") or f"worker_camera_{processed:05d}")
        directory = Path(command.get("output_dir") or (output_dir / camera_id))
        default_response = {"render_isolated": "isolated_response.json", "render_full": "full_response.json"}.get(action, "worker_response.json")
        response_file = Path(command.get("response_file") or (directory / default_response))
        processed += 1
        try:
            if action == "render_isolated":
                report = _render_isolated_record(bpy, meshes, camera, record, directory)
            elif action == "render_full":
                report = _render_full_record(bpy, camera, record, directory)
            else:
                report = _render_record(bpy, meshes, camera, record, directory)
            response = {**report, "action": action, "triangle_owner_manifest": str(manifest_path)}
            succeeded += 1
        except Exception:
            detail = traceback.format_exc()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f".{action}_failed").write_text(detail, encoding="utf-8")
            response = {
                "camera_id": camera_id,
                "action": action,
                "status": "failed",
                "error": detail[-4000:],
                "triangle_owner_manifest": str(manifest_path),
            }
            failed_count += 1
        _save_json_atomic(response, response_file)
        print(
            f"[PGW_STAGE07_WORKER] action={action} {camera_id} status={response['status']} "
            f"requests={processed} success={succeeded} failed={failed_count}",
            flush=True,
        )
    _save_json_atomic({
        "status": "stopped",
        "processed_count": processed,
        "success_count": succeeded,
        "failure_count": failed_count,
        "triangle_owner_manifest": str(manifest_path),
    }, output_dir / "worker_summary.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max_invalid_id_ratio", type=float, default=0.0005)
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--ready_file")
    parser.add_argument("--active_owner_manifest")
    args = parser.parse_args(argv)

    if not args.worker_mode and not args.camera_file:
        parser.error("--camera_file is required unless --worker_mode is used")

    bpy = require_bpy()
    out = Path(args.out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(resolve_scene_for_textured_downstream(out)))
    active_owner_filter = apply_active_owner_filter(bpy, args.active_owner_manifest)
    save_json(active_owner_filter, output_dir / "active_owner_filter_report.json")
    manifest_path = output_dir / "triangle_owner_manifest.json"
    meshes = _prepare_scene_triangle_ids(bpy, manifest_path)
    _configure_id_materials(bpy, meshes, (args.width, args.height), args.samples)
    camera = _create_reusable_camera(bpy)

    if args.worker_mode:
        ready_file = Path(args.ready_file) if args.ready_file else output_dir / "worker_ready.json"
        _run_worker_loop(
            bpy,
            meshes,
            camera,
            manifest_path,
            output_dir,
            ready_file,
            (args.width, args.height),
            args.samples,
        )
        return

    cameras = load_json(args.camera_file)["cameras"]
    results = []
    for record in cameras:
        camera_id = str(record["camera_id"])
        directory = output_dir / camera_id
        try:
            results.append(_render_record(bpy, meshes, camera, record, directory))
        except Exception:
            detail = traceback.format_exc()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".failed").write_text(detail, encoding="utf-8")
            results.append({"camera_id": camera_id, "status": "failed", "error": detail[-4000:]})

    batch_report = {
        "status": "ok" if any(item["status"] == "ok" for item in results) else "failed",
        "camera_count": len(cameras),
        "success_count": sum(item["status"] == "ok" for item in results),
        "failure_count": sum(item["status"] != "ok" for item in results),
        "resolution": [int(args.width), int(args.height)],
        "samples": int(args.samples),
        "dither_intensity": 0.0,
        "max_invalid_id_ratio": float(args.max_invalid_id_ratio),
        "id_validation_runtime": "host_project_python",
        "triangle_owner_manifest": str(manifest_path),
        "active_owner_filter": active_owner_filter,
        "results": results,
    }
    save_json(batch_report, output_dir / "batch_report.json")
    if batch_report["success_count"] == 0:
        raise RuntimeError("All refinement candidate renders failed")


if __name__ == "__main__":
    main()
