#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.logging_utils import setup_step_logger, status
from src.gaussian.nerfstudio_stage09 import run_worldmesh_nerfstudio_stage09
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/gaussian_pipeline.json")
    args = parser.parse_args()

    out = Path(args.out)
    step = out / "09_gaussian_splat"
    step.mkdir(parents=True, exist_ok=True)
    logger = setup_step_logger(Path("logs") / out.name / "09_train_gaussian_scene.log")
    status("[09] Exporting Stage08/Stage07 data and training WorldMesh-compatible Nerfstudio depth-splatfacto...")
    try:
        config = load_json(args.config)
        manifest_path = out / str(dict(config.get("dataset", {})).get(
            "manifest", "08_viewwise_refinement/stage09_training_manifest.json"
        ))
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Stage09 input is missing: {manifest_path}. run_stage09.sh never runs Stage08 automatically."
            )
        report = run_worldmesh_nerfstudio_stage09(manifest_path, config, step)
        artifact = ArtifactIndex(scene_id=out.name, step="09_train_gaussian_scene")
        artifact.add("stage_report", step / "stage_report.json")
        artifact.add("training_manifest", manifest_path)
        artifact.add("dataset_export_report", step / "dataset" / "dataset_export_report.json")
        artifact.add("nerfstudio_config_path", step / "nerfstudio_config_path.txt")
        artifact.add("viewer_command", step / "view_scene.txt")
        evaluation = Path(report["evaluation_json"])
        if evaluation.is_file():
            artifact.add("final_evaluation", evaluation)
        artifact.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage09 Nerfstudio config: %s", report["nerfstudio_config"])
        status(f"[09] Done. Nerfstudio config: {report['nerfstudio_config']}")
    except Exception:
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
