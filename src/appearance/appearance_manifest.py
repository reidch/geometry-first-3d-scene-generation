from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from src.io.json_io import load_json, save_json


def build_appearance_manifest(generation_plan: Mapping, output_path: str | Path | None = None) -> Dict:
    """Compile appearance data without interpreting semantic labels."""
    objects = {}
    for record in generation_plan.get("objects", []):
        object_id = str(record["object_id"])
        objects[object_id] = {
            "name": record.get("name", object_id),
            "semantic_class": record.get("semantic_class", ""),
            "generation_mode": record.get("generation_mode"),
            "prompt": dict(record.get("generation", {})).get("prompt", ""),
            "negative_prompt": dict(record.get("generation", {})).get("negative_prompt", ""),
            "appearance": dict(record.get("appearance", {})),
        }
    result = {"schema_version": 2, "source": "generation_plan", "objects": objects}
    if output_path is not None:
        save_json(result, output_path)
    return result


def load_or_build(step_root, config=None):
    path = Path(step_root) / "scene_appearance_manifest.json"
    if path.exists():
        return load_json(path)
    plan_path = Path(step_root).parent / "01_world_ir" / "generation_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing generation plan: {plan_path}")
    return build_appearance_manifest(load_json(plan_path), path)
