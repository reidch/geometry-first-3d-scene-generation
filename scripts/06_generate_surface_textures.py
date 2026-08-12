#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.logging_utils import setup_step_logger, status
from src.appearance.atlas_state import remove_legacy_texture_observation_state
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.pipeline.texture_state_snapshots import create_snapshot, replace_tree
from src.room_surfaces.surface_publish import publish_surface_textured_scene
from src.room_surfaces.surface_pipeline import generate_surface_textures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface_config", default="configs/room_surface_pipeline.json")
    parser.add_argument("--scene_json", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    step = out / "06_surface_textures"
    logger = setup_step_logger(Path("logs") / out.name / "06_generate_surface_textures.log")
    status("[06] Generating textures for JSON-routed surface objects...")
    try:
        # Stage06 always starts from the immutable Stage05 atlas snapshot.
        # This makes an interrupted or explicitly rebuilt Stage06 deterministic
        # instead of blending on top of a partially written previous run.
        replace_tree(step.parent / "05_scene_assets" / "texture_state_snapshot", out / "05_texture_state")
        remove_legacy_texture_observation_state(out / "05_texture_state")
        report = generate_surface_textures(out, load_json(args.surface_config), args.scene_json)
        published_scene = step / "scene_surface_textured.blend"
        scene_publish = publish_surface_textured_scene(out, published_scene)
        snapshot = create_snapshot(out / "05_texture_state", step / "texture_state_snapshot")
        report["scene_publish"] = scene_publish
        from src.io.json_io import save_json
        save_json(report, step / "stage_report.json")
        artifact = ArtifactIndex(scene_id=out.name, step="06_generate_surface_textures")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("texture_state_snapshot", snapshot)
        artifact.add("surface_textured_scene", published_scene)
        artifact.add("texture_publish_report", step / "texture_publish_report.json")
        artifact.add("scene_publish_report", step / "scene_publish_report.json")
        artifact.add("surface_material_binding", step / "surface_material_binding.json")
        artifact.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage06 report: %s", report)
        status("[06] Done. Surface atlases committed, materials bound, and textured scene published.")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
