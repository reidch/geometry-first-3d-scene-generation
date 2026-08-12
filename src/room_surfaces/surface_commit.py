from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image

from src.io.json_io import load_json


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def surface_object_ids_from_plan(plan: Dict) -> List[str]:
    return [
        str(record["object_id"])
        for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) == "surface_texture"
    ]


def inspect_surface_atlas(texture_root: str | Path, object_id: str) -> Dict:
    root = Path(texture_root)
    directory = root / str(object_id)
    color_path = directory / "base_color.png"
    metadata_path = directory / "metadata.json"
    result = {
        "object_id": str(object_id),
        "directory": str(directory),
        "base_color": str(color_path),
        "metadata": str(metadata_path),
        "valid": False,
        "problems": [],
    }
    if not color_path.exists() or color_path.stat().st_size == 0:
        result["problems"].append("missing_base_color")
        return result
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        result["problems"].append("missing_metadata")
        return result
    metadata = load_json(metadata_path)
    image = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
    result.update(
        {
            "sha256": sha256_file(color_path),
            "shape": list(image.shape),
            "luminance_std_u8": float(
                (
                    0.2126 * image[..., 0].astype(np.float32)
                    + 0.7152 * image[..., 1].astype(np.float32)
                    + 0.0722 * image[..., 2].astype(np.float32)
                ).std()
            ),
            "metadata_commit": dict(metadata.get("stage06_surface_commit", {})),
        }
    )
    commit = result["metadata_commit"]
    if not bool(commit.get("committed", False)):
        result["problems"].append("stage06_commit_missing")
    if int(commit.get("unique_observed_texels", 0)) <= 0:
        result["problems"].append("no_observed_texels")
    if int(commit.get("changed_texels", 0)) <= 0:
        result["problems"].append("atlas_unchanged")
    writeback_core = str(commit.get("writeback_core", ""))
    accepted_cores = {
        "src.appearance.atlas_fusion.fuse_view",
        "src.room_surfaces.surface_pipeline._commit_rectified_surface_direct",
        "src.room_surfaces.exact_surface_uv_commit.commit_rectified_surface_by_exact_triangles",
    }
    if writeback_core not in accepted_cores:
        result["problems"].append("unexpected_surface_writeback_core")
    if bool(commit.get("direct_rectified_full_surface_commit", False)):
        resolution = list(commit.get("texture_resolution", []))
        if len(resolution) != 2 or any(int(v) <= 0 for v in resolution):
            result["problems"].append("direct_surface_resolution_missing")
    if not bool(commit.get("non_target_meshes_hidden", False)):
        result["problems"].append("surface_capture_visibility_not_isolated")
    if float(commit.get("target_bbox_width_fraction", 0.0)) <= 0.0 or float(
        commit.get("target_bbox_height_fraction", 0.0)
    ) <= 0.0:
        result["problems"].append("surface_framing_metrics_missing")
    if commit.get("after_sha256") and str(commit.get("after_sha256")) != result["sha256"]:
        result["problems"].append("atlas_hash_mismatch")
    result["valid"] = not result["problems"]
    return result


def validate_surface_atlas_commits(
    texture_root: str | Path,
    surface_object_ids: Iterable[str],
) -> Dict:
    records = [inspect_surface_atlas(texture_root, object_id) for object_id in surface_object_ids]
    problems = [
        {"object_id": record["object_id"], "problems": list(record["problems"])}
        for record in records
        if not record["valid"]
    ]
    return {
        "status": "ok" if not problems else "failed",
        "texture_root": str(Path(texture_root)),
        "surface_object_count": len(records),
        "records": records,
        "problems": problems,
    }


def validate_stage06_surface_publication(out: str | Path) -> Dict:
    out = Path(out)
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    surface_ids = surface_object_ids_from_plan(plan)
    texture_root = out / "05_texture_state"
    atlas_validation = validate_surface_atlas_commits(texture_root, surface_ids)
    publish_report_path = out / "06_surface_textures" / "texture_publish_report.json"
    if not publish_report_path.exists():
        raise RuntimeError(
            f"Stage06 texture publication report is missing: {publish_report_path}. "
            "Re-run Stage06 with the current pipeline."
        )
    publish_report = load_json(publish_report_path)
    binding = dict(publish_report.get("material_binding", {}))
    binding_records = {str(record.get("object_id")): record for record in binding.get("records", [])}
    missing_bindings = [
        object_id
        for object_id in surface_ids
        if not bool(binding_records.get(object_id, {}).get("valid", False))
    ]
    published_scene = Path(str(publish_report.get("published_scene", "")))
    problems = []
    if atlas_validation["status"] != "ok":
        problems.append({"atlas_validation": atlas_validation["problems"]})
    if str(binding.get("status", "")) != "ok":
        problems.append({"material_binding_status": binding.get("status")})
    if missing_bindings:
        problems.append({"missing_or_invalid_surface_bindings": missing_bindings})
    if not published_scene.exists() or published_scene.stat().st_size == 0:
        problems.append({"missing_published_scene": str(published_scene)})
    if problems:
        raise RuntimeError(
            "Stage06 surface textures were not fully committed to atlases and Blender materials: "
            f"{problems}. Re-run Stage06 before Stage07/08."
        )
    return {
        "status": "ok",
        "surface_object_ids": surface_ids,
        "atlas_validation": atlas_validation,
        "material_binding": binding,
        "published_scene": str(published_scene),
    }
