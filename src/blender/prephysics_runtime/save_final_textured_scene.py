#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.blender_runtime import require_bpy
from src.blender.scene_input import resolve_scene_for_textured_downstream, stage05_scene_path
from src.blender.texture_materials import apply_object_texture_materials, verify_texture_material_bindings
from src.io.json_io import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source")
    ap.add_argument("--texture_root")
    ap.add_argument("--render_mode", choices=["albedo", "beauty"], default="beauty")
    ap.add_argument("--binding_report")
    args = ap.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    if args.source:
        source = Path(args.source)
    elif args.render_mode == "albedo":
        # Stage06 publication must start from the immutable Stage05 geometry,
        # otherwise rerunning Stage06 could recursively reopen its own output.
        source = stage05_scene_path(out)
    else:
        source = resolve_scene_for_textured_downstream(out)
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Source Blender scene is missing: {source}")

    bpy.ops.wm.open_mainfile(filepath=str(source))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    texture_root = Path(args.texture_root) if args.texture_root else out / "05_texture_state"
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    required_surface_owners = [
        str(record["object_id"])
        for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) == "surface_texture"
    ]
    surface_records = [
        record for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) == "surface_texture"
    ]
    # Keep the original authored UVs for room surfaces. Stage06 writes generated
    # surface appearance into the occupied atlas region instead of rewriting mesh
    # UVs to a synthetic [0,1]^2 layout.
    surface_uv_policy = {
        "status": "disabled",
        "policy": "preserve_original_surface_uvs",
        "surface_count": len(surface_records),
        "reason": "stage06_commits_generated_surface_into_exact_original_uv_bbox",
    }

    apply_object_texture_materials(
        texture_root,
        render_mode=args.render_mode,
        interpolation="Linear",
        required_owner_ids=required_surface_owners,
        binding_report_path=args.binding_report,
        strict=True,
    )
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and bool(obj.get("pgw_physics_proxy", False)):
            obj.hide_render = True
            obj.hide_viewport = True

    # Keep external images live.  Stage08 updates these same atlas files, and
    # every downstream renderer explicitly reloads/rebinds them on open.
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            try:
                image.filepath = str(Path(bpy.path.abspath(image.filepath)).resolve())
            except Exception:
                pass

    bpy.ops.wm.save_as_mainfile(filepath=str(output))

    # Reopen the exact published file and verify the saved material graph, rather
    # than trusting only the in-memory state that existed before save_as_mainfile.
    bpy.ops.wm.open_mainfile(filepath=str(output))
    saved_binding = verify_texture_material_bindings(
        texture_root, required_owner_ids=required_surface_owners
    )
    saved_binding["saved_scene_reopen_verified"] = True
    saved_binding["published_scene"] = str(output)
    saved_binding["surface_uv_policy"] = surface_uv_policy
    if args.binding_report:
        from src.io.json_io import save_json
        save_json(saved_binding, args.binding_report)
    if saved_binding["status"] != "ok":
        raise RuntimeError(
            f"Published Blender scene lost required surface material bindings: {saved_binding}"
        )


if __name__ == "__main__":
    main()
