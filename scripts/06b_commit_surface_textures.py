#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.appearance.atlas_state import remove_legacy_texture_observation_state
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.pipeline.texture_state_snapshots import create_snapshot, replace_tree
from src.room_surfaces.surface_pipeline import commit_generated_surface_textures
from src.room_surfaces.surface_publish import publish_surface_textured_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface_config", default="configs/room_surface_pipeline.json")
    parser.add_argument("--scene_json", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    step = out / "06_surface_textures"
    state = step / "06b_state"
    logger = setup_step_logger(Path("logs") / out.name / "06b_commit_surface_textures.log")
    status("[06b] Committing existing Stage06a images by exact surface-triangle UV mapping...")
    try:
        # Always restart atlas publication from immutable Stage05 texture state.
        # Generated images/captures under 06_surface_textures are preserved.
        replace_tree(step.parent / "05_scene_assets" / "texture_state_snapshot", out / "05_texture_state")
        remove_legacy_texture_observation_state(out / "05_texture_state")

        commit_report = commit_generated_surface_textures(
            out,
            load_json(args.surface_config),
            args.scene_json,
        )
        published_scene = step / "scene_surface_textured.blend"
        scene_publish = publish_surface_textured_scene(out, published_scene)
        snapshot = create_snapshot(out / "05_texture_state", step / "texture_state_snapshot")
        generation_report_path = step / "stage06a_report.json"
        generation_report = load_json(generation_report_path) if generation_report_path.exists() else {
            "status": "migrated_existing_stage06_outputs",
            "records": [],
        }
        stage_report = {
            "status": "ok",
            "stage": "06_surface_textures_split_pipeline",
            "stage06a_generation": generation_report,
            "stage06b_commit": commit_report,
            "scene_publish": scene_publish,
            "generated_images_reused_without_diffusion": True,
        }
        save_json(stage_report, step / "stage_report.json")

        artifact = ArtifactIndex(scene_id=out.name, step="06b_commit_surface_textures")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("stage06b_report", step / "stage06b_report.json")
        artifact.add("texture_state_snapshot", snapshot)
        artifact.add("surface_textured_scene", published_scene)
        artifact.add("texture_publish_report", step / "texture_publish_report.json")
        artifact.add("scene_publish_report", step / "scene_publish_report.json")
        artifact.add("surface_material_binding", step / "surface_material_binding.json")
        artifact.save(step / "artifact_index.json")
        # Stage06b has an independent cache state so it can be rerun without
        # invalidating Stage06a.  The private state must own its own artifact
        # index before pgw_cache_mark writes the output manifest.
        artifact.save(state / "artifact_index.json")
        mark_done(state)
        mark_done(step)
        logger.info("Stage06b committed %s surface atlases", len(commit_report.get("records", [])))
        status("[06b] Done. Exact triangle UV writeback completed and textured scene republished.")
    except Exception:
        state.mkdir(parents=True, exist_ok=True)
        (state / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
