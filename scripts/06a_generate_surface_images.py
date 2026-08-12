#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.room_surfaces.surface_pipeline import generate_surface_textures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface_config", default="configs/room_surface_pipeline.json")
    parser.add_argument("--scene_json", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    step = out / "06_surface_textures"
    state = step / "06a_state"
    logger = setup_step_logger(Path("logs") / out.name / "06a_generate_surface_images.log")
    status("[06a] Generating room-surface images only; atlas writeback is deferred to Stage06b...")
    try:
        report = generate_surface_textures(
            out,
            load_json(args.surface_config),
            args.scene_json,
            generation_only=True,
        )
        artifact = ArtifactIndex(scene_id=out.name, step="06a_generate_surface_images")
        artifact.add("generation_report", step / "stage06a_report.json")
        for item in report.get("records", []):
            object_id = str(item["object_id"])
            directory = step / object_id
            artifact.add(f"{object_id}_generated_image", directory / "generated_locked.png")
            artifact.add(f"{object_id}_generation_manifest", directory / "generation_manifest.json")
            artifact.add(f"{object_id}_capture", directory / "capture")
        artifact.save(step / "artifact_index_06a.json")
        # Stage06a is cached through its private 06a_state directory.  The cache
        # manifest writer requires an artifact_index.json inside that exact state
        # directory, so publish the same authoritative artifact set there too.
        artifact.save(state / "artifact_index.json")
        mark_done(state)
        logger.info("Stage06a generated %s surface images", len(report.get("records", [])))
        status("[06a] Done. Generated images and exact geometry/UV captures are ready for Stage06b.")
    except Exception:
        state.mkdir(parents=True, exist_ok=True)
        (state / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
