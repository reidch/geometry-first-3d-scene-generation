#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.io.json_io import load_json
from src.blender.blender_runtime import require_bpy
from src.blender.camera_utils import set_active_camera
from src.blender.condition_renderer import configure_condition_render, render_still_png
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.blender.object_identity import get_semantic_owner_id
from src.blender.texture_materials import apply_object_texture_materials


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
    ap.add_argument('--output', required=True)
    ap.add_argument('--texture_root')
    ap.add_argument('--hide_object', action='append', default=[])
    ap.add_argument('--render_mode', choices=['albedo', 'beauty'], default='albedo')
    ap.add_argument('--texture_interpolation', default='Linear')
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=576)
    ap.add_argument('--binding_report')
    args = ap.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    bpy.ops.wm.open_mainfile(filepath=str(resolve_scene_for_textured_downstream(out)))
    cameras = load_json(args.camera_file)['cameras']
    camera_data = next(c for c in cameras if c['camera_id'] == args.camera_id)
    plan = load_json(out / '01_world_ir' / 'generation_plan.json')
    required_surface_owners = [
        str(record['object_id'])
        for record in plan.get('objects', [])
        if str(record.get('generation_mode', '')) == 'surface_texture'
    ]
    apply_object_texture_materials(
        args.texture_root or out / '05_texture_state',
        render_mode=args.render_mode,
        interpolation=args.texture_interpolation,
        required_owner_ids=required_surface_owners,
        binding_report_path=args.binding_report,
        strict=True,
    )
    _apply_hide_list(bpy, args.hide_object)
    configure_condition_render((int(args.width), int(args.height)))
    render_still_png(set_active_camera(camera_data), args.output)


if __name__ == '__main__':
    main()
