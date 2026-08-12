#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.logging_utils import setup_step_logger, status
from src.core.validation import normalize_scene, validate_json_schema, validate_scene_dict
from src.io.json_io import load_json, save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.scene_ir.json_scene import flat_objects, scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    step_dir = out / "00_validated"
    logger = setup_step_logger(Path("logs") / out.name / "00_validate_scene.log")
    status("[00] Validating explicit JSON scene contract...")
    step_dir.mkdir(parents=True, exist_ok=True)

    document = load_json(args.scene)
    schema_issues = validate_json_schema(document)
    semantic_issues, warnings = validate_scene_dict(document)
    issues = schema_issues + semantic_issues
    normalized = normalize_scene(document) if not issues else document
    flat = flat_objects(normalized, include_groups=True) if not issues else []

    normalized_path = step_dir / "scene.normalized.json"
    flat_path = step_dir / "objects.flat.json"
    summary_path = step_dir / "scene_summary.json"
    report_path = step_dir / "validation_report.json"
    save_json(normalized, normalized_path)
    save_json({"objects": flat}, flat_path)

    summary = {
        "scene_id": scene_id(normalized) if not issues else str(out.name),
        "object_count_including_groups": len(flat),
        "semantic_object_count": sum(1 for item in flat if item.get("generation", {}).get("mode") != "group"),
        "valid": not issues,
        "issue_count": len(issues),
        "warning_count": len(warnings),
    }
    save_json(summary, summary_path)
    save_json({"valid": not issues, "issues": issues, "warnings": warnings}, report_path)

    artifact_index = ArtifactIndex(scene_id=out.name, step="00_validate_scene")
    artifact_index.add("normalized_scene", normalized_path)
    artifact_index.add("flat_objects", flat_path)
    artifact_index.add("scene_summary", summary_path)
    artifact_index.add("validation_report", report_path)
    artifact_index.save(step_dir / "artifact_index.json")
    logger.info("summary=%s", summary)
    logger.info("issues=%s", issues)
    logger.info("warnings=%s", warnings)
    if issues:
        status(f"[00] Validation failed with {len(issues)} issue(s).")
        raise SystemExit(1)
    mark_done(step_dir)
    status("[00] Done. JSON scene is valid and hierarchy was flattened for downstream stages.")


if __name__ == "__main__":
    main()
