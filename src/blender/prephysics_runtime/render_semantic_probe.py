#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.io.json_io import load_json
from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import set_active_camera
from src.blender.condition_renderer import configure_condition_render, render_still_png
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.blender.object_identity import get_semantic_owner_id
from src.blender.semantic_render import collect_original_materials, restore_materials, apply_object_semantic_materials


def _apply_hide_list(bpy, hide_names):
    hidden_names = {str(x) for x in hide_names}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        name = get_semantic_owner_id(obj)
        hidden = bool(obj.get('pgw_physics_proxy', False)) or name in hidden_names
        obj.hide_render = hidden
        obj.hide_viewport = hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera_file', required=True)
    ap.add_argument('--camera_id', required=True)
    ap.add_argument('--semantic_output', required=True)
    ap.add_argument('--hide_object', action='append', default=[])
    args = ap.parse_args(argv)

    semantic_path = Path(args.semantic_output)
    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    palette_path = semantic_path.with_suffix('.palette.json')
    done = semantic_path.with_suffix('.done')
    failed = semantic_path.with_suffix('.failed')
    for marker in (done, failed):
        if marker.exists():
            marker.unlink()
    try:
        bpy = require_bpy()
        out = Path(args.out)
        bpy.ops.wm.open_mainfile(filepath=str(resolve_scene_for_textured_downstream(out)))
        cameras = load_json(args.camera_file)['cameras']
        camera_data = next(c for c in cameras if c['camera_id'] == args.camera_id)
        _apply_hide_list(bpy, args.hide_object)
        configure_condition_render((1024, 576))
        camera = set_active_camera(camera_data)
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
            if scene.world is not None:
                scene.world.color = (0.0, 0.0, 0.0)
            render_still_png(camera, semantic_path)
        finally:
            restore_materials(originals)
            scene.view_settings.view_transform, scene.view_settings.look = old[0], old[1]
            scene.view_settings.exposure, scene.view_settings.gamma = old[2], old[3]
            scene.render.film_transparent = old[4]
            if scene.world is not None and old[5] is not None:
                scene.world.color = old[5]
        palette_path.write_text(json.dumps(palette, indent=2), encoding='utf-8')
        missing = [str(p) for p in (semantic_path, palette_path) if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError('Semantic probe incomplete: ' + ', '.join(missing))
        done.write_text('ok\n', encoding='utf-8')
    except Exception:
        failed.write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
