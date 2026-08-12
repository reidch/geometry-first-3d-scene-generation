#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.appearance.viewwise_refinement import run_viewwise_refinement
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.room_surfaces.surface_commit import validate_stage06_surface_publication
from src.pipeline.texture_state_snapshots import create_snapshot, replace_tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refinement_config", default="configs/refinement_pipeline.json")
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--prompts_json", default="configs/prompts.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    step = out / "08_viewwise_refinement"
    logger = setup_step_logger(Path("logs") / out.name / "08_run_viewwise_refinement.log")
    status("[08] Running real all-object low-strength viewwise refinement...")
    try:
        # Rebuilding an incomplete Stage08 must start from the fully published
        # Stage06 texture state, not from a partially refined previous attempt.
        replace_tree(out / "06_surface_textures" / "texture_state_snapshot", out / "05_texture_state")
        surface_preflight = validate_stage06_surface_publication(out)
        report = run_viewwise_refinement(out, load_json(args.refinement_config), args.scene_json, args.prompts_json)
        report["stage06_surface_preflight"] = surface_preflight
        from src.io.json_io import save_json
        save_json(report, step / "stage_report.json")
        final_snapshot = create_snapshot(out / "05_texture_state", step / "texture_state_snapshot")
        artifact = ArtifactIndex(scene_id=out.name, step="08_run_viewwise_refinement")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("final_texture_state_snapshot", final_snapshot)
        artifact.add("final_textured_scene", Path(report["final_textured_scene"]))
        artifact.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage08 status: %s; completed views: %s", report["status"], report["completed_views"])
        status(f"[08] Done. {report['completed_views']} real refinement views completed ({report['status']}).")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
