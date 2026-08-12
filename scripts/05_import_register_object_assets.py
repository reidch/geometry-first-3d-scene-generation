#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assets.asset_importer import import_align_and_initialize
from src.core.logging_utils import setup_step_logger, status
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.pipeline.texture_state_snapshots import create_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset_config", default="configs/asset_pipeline.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    step = out / "05_scene_assets"
    logger = setup_step_logger(Path("logs") / out.name / "05_import_register_object_assets.log")
    status("[05] Registering generated assets to their JSON scaffolds and initializing editable atlases...")
    try:
        report = import_align_and_initialize(out, args.asset_config)
        snapshot = create_snapshot(out / "05_texture_state", step / "texture_state_snapshot")
        artifact = ArtifactIndex(scene_id=out.name, step="05_import_register_object_assets")
        artifact.add("scene_assets", Path(report["scene_assets_blend"]))
        artifact.add("registration_plan", step / "registration_plan.json")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("texture_state_snapshot", snapshot)
        artifact.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage05 report: %s", report)
        status("[05] Done. JSON-guided object registration and texture-state initialization completed.")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
