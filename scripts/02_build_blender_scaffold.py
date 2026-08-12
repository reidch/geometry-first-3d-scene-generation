#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.blender.scaffold_from_json import build_scaffold_from_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default=None, help="Defaults to <out>/00_validated/scene.normalized.json")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    scene_id = out.name
    step_dir = out / "02_blender_scaffold"
    log_path = Path("logs") / scene_id / "02_build_blender_scaffold.log"
    logger = setup_step_logger(log_path)

    status("[02] Building Blender scaffold...")

    scene_path = Path(args.scene) if args.scene else out / "00_validated" / "scene.normalized.json"
    stage01_dir = out / "01_world_ir"
    object_registry_path = stage01_dir / "object_registry.json"
    render_manifest_path = stage01_dir / "render_manifest.json"
    physics_manifest_path = stage01_dir / "physics_manifest.json"
    binding_records_path = stage01_dir / "binding_records.json"
    material_manifest_path = stage01_dir / "material_manifest.json"

    required = [
        scene_path,
        object_registry_path,
        render_manifest_path,
        physics_manifest_path,
        binding_records_path,
        material_manifest_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 00/01 artifacts: " + "; ".join(missing))

    step_dir.mkdir(parents=True, exist_ok=True)

    blend_path = step_dir / "scaffold.blend"
    preview_path = step_dir / "scaffold_preview.png" if args.preview else None

    manifest = build_scaffold_from_json(
        scene_path=scene_path,
        object_registry_path=object_registry_path,
        render_manifest_path=render_manifest_path,
        physics_manifest_path=physics_manifest_path,
        binding_records_path=binding_records_path,
        material_manifest_path=material_manifest_path,
        out_blend_path=blend_path,
        preview_path=preview_path,
    )

    manifest_path = step_dir / "blender_object_manifest.json"
    stage_report_path = step_dir / "stage_report.json"
    artifact_index_path = step_dir / "artifact_index.json"

    save_json(manifest, manifest_path)
    save_json({
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "blend_path": str(blend_path),
        "preview_path": str(preview_path) if preview_path else None,
        "blender_object_count": len(manifest.get("objects", [])),
        "uv_atlas_object_count": len(manifest.get("uv_atlas", {})),
        "runtime_rule": "Blender script uses JSON only; no external Python packages and no binary Python object loading.",
        "status": "ok",
    }, stage_report_path)

    artifact_index = ArtifactIndex(scene_id=scene_id, step="02_build_blender_scaffold")
    artifact_index.add("blend", blend_path)
    artifact_index.add("blender_object_manifest", manifest_path)
    artifact_index.add("stage_report", stage_report_path)
    if preview_path:
        artifact_index.add("preview", preview_path)
    artifact_index.save(artifact_index_path)

    logger.info("Created Blender scaffold with %d objects", len(manifest.get("objects", [])))
    mark_done(step_dir)
    status("[02] Done. Blender scaffold saved.")

if __name__ == "__main__":
    main()
