#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.io.json_io import load_json
from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import set_active_camera
from src.blender.condition_renderer import (camera_scene_depth_bounds, configure_condition_render, render_depth_control_png, render_still_png)
from src.blender.atlas_condition_renderer import render_uv_png_bundle
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.blender.object_identity import get_semantic_owner_id
from src.blender.semantic_render import collect_original_materials, restore_materials, apply_object_semantic_materials
from src.blender.texture_materials import apply_object_texture_materials
from src.blender.triangle_id_render import render_triangle_id_png


def _semantic_name(obj):
    return get_semantic_owner_id(obj)


def _apply_hide_list(bpy, hide_names):
    hidden_names = {str(x) for x in hide_names}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        hidden = bool(obj.get('pgw_physics_proxy', False)) or _semantic_name(obj) in hidden_names
        obj.hide_render = hidden
        obj.hide_viewport = hidden


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
    pv = proj @ camera.matrix_world.inverted()
    cam_pos = camera.matrix_world.translation
    triangles = []
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render or bool(obj.get('pgw_physics_proxy', False)):
            continue
        if _semantic_name(obj) != str(target_object):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
            if uv_layer is None:
                raise RuntimeError(f'No UV layer for target mesh {obj.name}')
            normal_matrix = eval_obj.matrix_world.to_3x3().inverted().transposed()
            base = int(obj.get('pgw_triangle_base', 0))
            for local_index, tri in enumerate(mesh.polygons):
                if len(tri.vertices) != 3:
                    raise RuntimeError(f"Target mesh is not triangulated: {obj.name}")
                clips, uvs, worlds = [], [], []
                for loop_index in tri.loop_indices:
                    vi = mesh.loops[loop_index].vertex_index
                    world = eval_obj.matrix_world @ mesh.vertices[vi].co
                    clip = pv @ world.to_4d()
                    uv = uv_layer[loop_index].uv
                    worlds.append(world)
                    clips.append([float(clip.x), float(clip.y), float(clip.z), float(clip.w)])
                    uvs.append([float(uv.x), float(uv.y)])
                center = (worlds[0] + worlds[1] + worlds[2]) / 3.0
                view_dir = (cam_pos - center).normalized()
                normal = (normal_matrix @ tri.normal).normalized()
                frontality = max(0.0, float(normal.dot(view_dir)))
                if frontality <= 1e-6:
                    continue
                world_area = 0.5 * float(((worlds[1] - worlds[0]).cross(worlds[2] - worlds[0])).length)
                triangles.append({
                    'global_triangle_id': int(base + local_index),
                    'part_name': obj.name,
                    'local_triangle_id': int(local_index),
                    'uv': uvs,
                    'clip': clips,
                    'world_area': world_area,
                    'frontality': frontality,
                })
        finally:
            eval_obj.to_mesh_clear()
    Path(output_path).write_text(json.dumps({
        'target_object': str(target_object),
        'image_size': [rx, ry],
        'triangle_count': len(triangles),
        'triangles': triangles,
    }), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera_file', required=True)
    ap.add_argument('--camera_id', required=True)
    ap.add_argument('--rgb_output', required=True)
    ap.add_argument('--semantic_output', required=True)
    ap.add_argument('--uv_output', required=True)
    ap.add_argument('--depth_output', required=True)
    ap.add_argument('--texture_root')
    ap.add_argument('--target_object', required=True)
    ap.add_argument('--triangles_output', required=True)
    ap.add_argument('--triangle_id_output', required=True)
    ap.add_argument('--hide_object', action='append', default=[])
    args = ap.parse_args(argv)

    outputs = [Path(args.rgb_output), Path(args.semantic_output), Path(args.uv_output),
               Path(args.depth_output), Path(args.triangles_output), Path(args.triangle_id_output)]
    root = outputs[0].parent
    root.mkdir(parents=True, exist_ok=True)
    done, failed = root / '.done', root / '.failed'
    for marker in (done, failed):
        if marker.exists(): marker.unlink()
    try:
        bpy = require_bpy()
        out = Path(args.out)
        bpy.ops.wm.open_mainfile(filepath=str(resolve_scene_for_textured_downstream(out)))
        cameras = load_json(args.camera_file)['cameras']
        camera_data = next(c for c in cameras if c['camera_id'] == args.camera_id)
        _apply_hide_list(bpy, args.hide_object)
        configure_condition_render((1024, 576))
        camera = set_active_camera(camera_data)
        apply_object_texture_materials(args.texture_root or out / '05_texture_state', render_mode='albedo', interpolation='Linear')
        _triangle_manifest(bpy, camera, args.target_object, args.triangles_output)
        render_still_png(camera, args.rgb_output)
        depth_near, depth_far = camera_scene_depth_bounds(camera)
        render_depth_control_png(camera, args.depth_output, depth_near, depth_far, valid_min_gray=24)
        render_uv_png_bundle(camera, args.uv_output)
        render_triangle_id_png(camera, args.triangle_id_output)

        scene = bpy.context.scene
        originals = collect_original_materials()
        palette = apply_object_semantic_materials()
        old = (scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure,
               scene.view_settings.gamma, scene.render.film_transparent,
               tuple(scene.world.color) if scene.world is not None else None)
        try:
            scene.view_settings.view_transform = 'Raw'
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0
            scene.render.film_transparent = False
            if scene.world is not None: scene.world.color = (0.0, 0.0, 0.0)
            render_still_png(camera, args.semantic_output)
        finally:
            restore_materials(originals)
            scene.view_settings.view_transform, scene.view_settings.look = old[0], old[1]
            scene.view_settings.exposure, scene.view_settings.gamma = old[2], old[3]
            scene.render.film_transparent = old[4]
            if scene.world is not None and old[5] is not None: scene.world.color = old[5]
        palette_path = Path(args.semantic_output).with_suffix('.palette.json')
        palette_path.write_text(json.dumps(palette, indent=2), encoding='utf-8')
        required = outputs + [palette_path]
        missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError('Subpass render incomplete: ' + ', '.join(missing))
        done.write_text('ok\n', encoding='utf-8')
    except Exception:
        failed.write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
