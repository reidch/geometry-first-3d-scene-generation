#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assets.multiview_representative_builder import generate_multiview_representative_images
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset_config", default="configs/asset_pipeline.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    step = out / "03_object_representative_images"
    logger = setup_step_logger(Path("logs") / out.name / "03_generate_object_representative_images.log")
    status("[03] Generating representative views for every JSON object with generation.mode='asset_3d'...")
    try:
        config = load_json(args.asset_config)
        config["__path__"] = args.asset_config
        report = generate_multiview_representative_images(out, config)
        index = ArtifactIndex(scene_id=out.name, step="03_generate_object_representative_images")
        index.add("stage_report", step / "stage_report.json")
        for record in report.get("objects", []):
            index.add(f"{record['object_id']}_representative", Path(record["representative_image"]))
        index.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage03 report: %s", report)
        status("[03] Done. Object representative images generated.")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
