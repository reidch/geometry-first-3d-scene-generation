#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cameras.reconstruction_sampler import select_reconstruction_cameras
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.pipeline.resume import mark_done
from src.room_surfaces.surface_commit import validate_stage06_surface_publication


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refinement_config", default="configs/refinement_pipeline.json")
    ap.add_argument("--camera_config", default="configs/cameras.json")
    ap.add_argument("--scene_json", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    step = out / "07_refinement_cameras"
    logger = setup_step_logger(Path("logs") / out.name / "07a_select_refinement_cameras.log")
    status("[07a] Constructing WorldMesh-style cameras and deterministic coverage repair...")
    try:
        surface_preflight = validate_stage06_surface_publication(out)
        refinement_config = load_json(args.refinement_config)
        camera_config = load_json(args.camera_config)
        scene = load_json(args.scene_json)
        report = select_reconstruction_cameras(out, scene, camera_config, refinement_config)
        payload = {
            "status": "ok",
            "base_camera_count": int(report.get("base_camera_count", 0)),
            "repair_camera_count": int(report.get("repair_camera_count", 0)),
            "generated_camera_count": int(report.get("generated_camera_count", 0)),
            "accepted_camera_count": len(report.get("accepted_cameras", [])),
            "selection_report": report.get("selection_report", {}),
            "active_owner_manifest": report.get("active_owner_manifest"),
            "active_owner_ids": report.get("active_owner_ids", []),
            "stage06_surface_preflight": surface_preflight,
        }
        save_json(payload, step / "stage07a_report.json")
        mark_done(step)
        logger.info("Stage07a report: base=%s repair=%s accepted=%s", payload["base_camera_count"], payload["repair_camera_count"], payload["accepted_camera_count"])
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
