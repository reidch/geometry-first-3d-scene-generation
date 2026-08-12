from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict

from src.io.json_io import load_json, save_json
from src.room_surfaces.surface_commit import (
    surface_object_ids_from_plan,
    validate_surface_atlas_commits,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_surface_textured_scene(out: str | Path, output_scene: str | Path) -> Dict:
    out = Path(out)
    output_scene = Path(output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    surface_ids = surface_object_ids_from_plan(plan)
    atlas_validation = validate_surface_atlas_commits(out / "05_texture_state", surface_ids)
    if atlas_validation["status"] != "ok":
        raise RuntimeError(
            "Refusing to publish Stage06 because generated surface images were not committed "
            f"to their object-owned atlases: {atlas_validation['problems']}"
        )

    blender = os.environ.get("BLENDER_BIN", "blender")
    script = Path("src/blender/prephysics_runtime/save_final_textured_scene.py")
    binding_report_path = output_scene.parent / "surface_material_binding.json"
    command = [
        blender,
        "--background",
        "--python",
        str(script),
        "--",
        "--out",
        str(out),
        "--source",
        str(out / "05_scene_assets" / "scene_assets.blend"),
        "--texture_root",
        str(out / "05_texture_state"),
        "--render_mode",
        "albedo",
        "--binding_report",
        str(binding_report_path),
        "--output",
        str(output_scene),
    ]
    subprocess.run(command, check=True)
    if not output_scene.exists() or output_scene.stat().st_size == 0:
        raise RuntimeError(f"Blender did not publish the Stage06 textured scene: {output_scene}")
    if not binding_report_path.exists() or binding_report_path.stat().st_size == 0:
        raise RuntimeError(f"Blender did not write the Stage06 material-binding report: {binding_report_path}")
    material_binding = load_json(binding_report_path)
    if material_binding.get("status") != "ok":
        raise RuntimeError(
            "Stage06 generated atlases, but Blender did not bind all required surface materials: "
            f"{material_binding}"
        )

    report = {
        "status": "ok",
        "source_scene": str(out / "05_scene_assets" / "scene_assets.blend"),
        "published_scene": str(output_scene),
        "surface_uv_policy": {
            "status": "disabled",
            "policy": "preserve_original_surface_uvs",
            "reason": "stage06_commits_generated_surface_into_exact_original_uv_bbox",
        },
        "published_scene_sha256": _sha256(output_scene),
        "texture_root": str(out / "05_texture_state"),
        "material_mode": "albedo",
        "surface_object_ids": surface_ids,
        "atlas_validation": atlas_validation,
        "material_binding": material_binding,
        "material_binding_report": str(binding_report_path),
        "external_atlas_files_remain_live": True,
        "stage08_rebinds_live_atlases_each_render": True,
    }
    # Canonical report name used by stage caching and downstream preflight.
    save_json(report, output_scene.parent / "texture_publish_report.json")
    # Compatibility copy for older inspection scripts.
    save_json(report, output_scene.parent / "scene_publish_report.json")
    return report
