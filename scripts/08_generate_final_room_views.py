#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.appearance.whole_view_generation import generate_final_room_views
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--refinement_config", default="configs/refinement_pipeline.json")
    parser.add_argument("--scene_json", required=True)
    parser.add_argument("--prompts_json", default="configs/prompts.json")
    parser.add_argument("--parameters_json", default="configs/parameters.json")
    parser.add_argument("--phase", choices=("all", "forward", "repair"), default="all")
    parser.add_argument("--reset_repair_to_forward_snapshot", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    step = out / "08_viewwise_refinement"
    log_name = {"forward": "08a_generate_final_room_views.log", "repair": "08b_repair_final_room_views.log"}.get(args.phase, "08_generate_final_room_views.log")
    logger = setup_step_logger(Path("logs") / out.name / log_name)
    if args.phase == "forward":
        status("[08A] Running Stage08 forward generation only; repair will not run in this command...")
    elif args.phase == "repair":
        status("[08B] Running Stage08 repair pass only from completed Stage08A outputs...")
    else:
        status("[08] Reusing Stage07 RGB and correlation graph; forward generation followed by repair pass...")
    try:
        report = generate_final_room_views(
            out,
            load_json(args.refinement_config),
            args.scene_json,
            args.prompts_json,
            args.parameters_json,
            execution_phase=args.phase,
            reset_repair_to_forward_snapshot=bool(args.reset_repair_to_forward_snapshot),
        )
        if args.phase == "forward":
            substep = step / "08a_forward"
            substep.mkdir(parents=True, exist_ok=True)
            save_json(report, substep / "stage_report.json")
            artifact = ArtifactIndex(scene_id=out.name, step="08a_forward")
            artifact.add("stage_report", substep / "stage_report.json")
            artifact.add("forward_pass_summary", Path(report["forward_pass_summary"]))
            artifact.add("final_views", Path(report["final_views_root"]))
            artifact.add("generated_neighbor_registry", Path(report["generated_neighbor_registry"]))
            artifact.add("weighted_frontier_generation_order", Path(report["weighted_frontier_generation_order"]))
            artifact.save(substep / "artifact_index.json")
            mark_done(substep)
            logger.info("Stage08A completed=%s strict=%s", report["completed_views"], report["strict_accepted_views"])
            status(f"[08A] Done. {report['completed_views']} forward views completed. Run bash run_stage08b.sh to repair them.")
        else:
            save_json(report, step / "stage_report.json")
            artifact = ArtifactIndex(scene_id=out.name, step="08_generate_final_room_views")
            artifact.add("stage_report", step / "stage_report.json")
            artifact.add("stage09_training_manifest", Path(report["stage09_training_manifest"]))
            artifact.add("final_views", Path(report["final_views_root"]))
            artifact.add("mesh_scene", Path(report["mesh_scene"]))
            artifact.add("generated_neighbor_registry", Path(report["generated_neighbor_registry"]))
            artifact.add("weighted_frontier_generation_order", Path(report["weighted_frontier_generation_order"]))
            artifact.save(step / "artifact_index.json")
            mark_done(step)
            if args.phase == "repair":
                substep = step / "08b_repair"
                substep.mkdir(parents=True, exist_ok=True)
                save_json(report, substep / "stage_report.json")
                subartifact = ArtifactIndex(scene_id=out.name, step="08b_repair")
                subartifact.add("stage_report", substep / "stage_report.json")
                subartifact.add("stage09_training_manifest", Path(report["stage09_training_manifest"]))
                subartifact.add("repair_pass_summary", Path(report["repair_pass"]["summary_path"]))
                subartifact.add("final_views", Path(report["final_views_root"]))
                subartifact.save(substep / "artifact_index.json")
                mark_done(substep)
                status(f"[08B] Done. Repair completed for {len(report['repair_pass']['processed_camera_ids'])} views; Stage09 manifest is final.")
            else:
                status(f"[08] Done. {report['completed_views']} processed; repair complete and Stage09 manifest finalized.")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
