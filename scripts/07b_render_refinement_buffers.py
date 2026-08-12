#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cameras.reconstruction_sampler import render_reconstruction_buffers
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
    logger = setup_step_logger(Path("logs") / out.name / "07b_render_refinement_buffers.log")
    status("[07b] Rendering refinement shared buffers from accepted cameras only...")
    try:
        surface_preflight = validate_stage06_surface_publication(out)
        refinement_config = load_json(args.refinement_config)
        camera_config = load_json(args.camera_config)
        scene = load_json(args.scene_json)
        report = render_reconstruction_buffers(out, scene, camera_config, refinement_config)
        report["stage06_surface_preflight"] = surface_preflight
        save_json(report, step / "stage07b_report.json")
        mark_done(step / "shared_buffers")
        logger.info("Stage07b frame_count=%s", report.get("frame_count"))
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
