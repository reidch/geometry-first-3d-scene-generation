#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = (
    "src.blender.scaffold_from_json",
    "src.blender.prephysics_runtime.capture_object_multiview_inputs",
    "src.blender.prephysics_runtime.import_align_prepare_assets",
    "src.blender.prephysics_runtime.render_canonical_surface",
    "src.blender.prephysics_runtime.render_refinement_candidates_batch",
    "src.blender.prephysics_runtime.render_refinement_shared_buffers_batch",
    "src.blender.prephysics_runtime.render_textured_view",
    "src.blender.prephysics_runtime.render_object_subpass_buffers",
    "src.blender.prephysics_runtime.save_final_textured_scene",
    "src.blender.prephysics_runtime.render_semantic_probe",
)


def _load_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    results = []
    for module_name in MODULES:
        try:
            __import__(module_name)
            results.append({"module": module_name, "status": "ok"})
        except Exception:
            results.append({"module": module_name, "status": "failed", "error": traceback.format_exc()})

    stage02 = ROOT / "scripts" / "02_build_blender_scaffold.py"
    try:
        _load_file(stage02, "pgw_stage02_blender_probe")
        results.append({"module": str(stage02.relative_to(ROOT)), "status": "ok"})
    except Exception:
        results.append({
            "module": str(stage02.relative_to(ROOT)),
            "status": "failed",
            "error": traceback.format_exc(),
        })

    report = {
        "status": "ok" if all(item["status"] == "ok" for item in results) else "failed",
        "python": sys.executable,
        "results": results,
    }
    print("PGW_BLENDER_IMPORT_PROBE=" + json.dumps(report))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
