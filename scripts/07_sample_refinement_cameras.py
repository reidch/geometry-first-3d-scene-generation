#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cameras.reconstruction_sampler import prepare_reconstruction_cameras
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.room_surfaces.surface_commit import validate_stage06_surface_publication


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refinement_config", default="configs/refinement_pipeline.json")
    ap.add_argument("--camera_config", default="configs/cameras.json")
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    step = out / "07_refinement_cameras"
    logger = setup_step_logger(Path("logs") / out.name / "07_sample_refinement_cameras.log")
    status("[07] Building WorldMesh-style base cameras and repairing uncovered room-shell regions...")
    try:
        surface_preflight = validate_stage06_surface_publication(out)
        refinement_config = load_json(args.refinement_config)
        camera_config = load_json(args.camera_config)
        scene = load_json(args.scene_json)
        report = prepare_reconstruction_cameras(
            out,
            scene,
            camera_config,
            refinement_config,
        )
        report["stage06_surface_preflight"] = surface_preflight
        save_json(report, step / "stage_report.json")
        artifact = ArtifactIndex(scene_id=out.name, step="07_sample_reconstruction_cameras")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("accepted_cameras", step / "cameras.accepted.json")
        artifact.add("reconstruction_dataset_manifest", step / "reconstruction_dataset_manifest.json")
        artifact.add("active_owner_manifest", step / "active_owner_ids.json")
        artifact.add("shared_buffers", step / "shared_buffers")
        artifact.add("room_coverage_graph", step / "room_coverage_graph.json")
        artifact.add("surface_texture_binding", step / "shared_buffers" / "surface_texture_binding.json")
        artifact.save(step / "artifact_index.json")
        mark_done(step)
        logger.info(
            "Stage07 report: base=%s repair=%s accepted=%s selected_rgb=%s",
            report["base_camera_count"],
            report["repair_camera_count"],
            report["accepted_camera_count"],
            report["selected_view_rgb_rendered"],
        )
        status(
            f"[07] Done. {report['accepted_camera_count']} cameras accepted; "
            "RGB/depth/normal/semantic/triangle-ID buffers exported for Stage08 reuse."
        )
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
