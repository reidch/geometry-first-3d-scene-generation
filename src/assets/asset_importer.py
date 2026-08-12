from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw

from src.appearance.atlas_state import ObjectAtlas, stable_debug_color
from src.assets.registration_planner import build_registration_plan
from src.io.json_io import load_json, save_json


def _uv_surface_labels(triangles, resolution: int, gutter_px: int = 2):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "Stage05 texture-state initialization requires SciPy. Install it with: python -m pip install scipy"
        ) from exc
    image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(image)
    for triangle in triangles:
        points = [
            (float(u) * (resolution - 1), (1.0 - float(v)) * (resolution - 1))
            for u, v in triangle["uv"]
        ]
        draw.polygon(points, fill=255)
    surface = np.asarray(image, dtype=np.uint8) > 0
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, island_count = ndimage.label(surface, structure=structure)
    labels = labels.astype(np.int32, copy=False)
    if gutter_px > 0 and island_count > 0:
        background = labels == 0
        distances, nearest_indices = ndimage.distance_transform_edt(background, return_indices=True)
        nearest_labels = labels[tuple(nearest_indices)]
        grow = background & (distances <= float(gutter_px)) & (nearest_labels > 0)
        labels[grow] = nearest_labels[grow]
    return labels, int(island_count)


def _base_color(record: Dict) -> tuple[int, int, int]:
    value = dict(record.get("appearance", {})).get("base_color")
    if isinstance(value, list) and len(value) >= 3:
        numbers = [float(v) for v in value[:3]]
        if max(numbers) <= 1.0:
            numbers = [v * 255.0 for v in numbers]
        return tuple(max(0, min(255, int(round(v)))) for v in numbers)
    return stable_debug_color(str(record["object_id"]))




def _inspect_texture(path: Path, expected_resolution: int) -> Dict:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {"valid": False, "reason": "missing_or_empty", "path": str(path)}
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        return {"valid": False, "reason": f"decode_failed:{type(exc).__name__}", "path": str(path)}
    if rgb.shape[:2] != (int(expected_resolution), int(expected_resolution)):
        return {
            "valid": False,
            "reason": "unexpected_resolution",
            "path": str(path),
            "shape": list(rgb.shape),
        }
    luminance = 0.2126 * rgb[..., 0].astype(np.float32) + 0.7152 * rgb[..., 1].astype(np.float32) + 0.0722 * rgb[..., 2].astype(np.float32)
    return {
        "valid": True,
        "reason": "decoded",
        "path": str(path),
        "rgb_min": int(rgb.min()),
        "rgb_max": int(rgb.max()),
        "mean_luminance_u8": float(luminance.mean()),
        "luminance_std_u8": float(luminance.std()),
        "nonzero_rgb_fraction": float(np.any(rgb > 0, axis=2).mean()),
        "has_detail": bool(float(luminance.std()) >= 0.75),
    }

def _initialize_texture_state(out: Path, config: Dict, blender_report: Dict) -> Dict:
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    manifest = load_json(out / "05_scene_assets" / "uv_triangle_manifest.json")["objects"]
    blender_records = {str(record["object_id"]): record for record in blender_report.get("records", [])}
    stage04_path = out / "04_object_assets" / "stage_report.json"
    stage04 = load_json(stage04_path) if stage04_path.exists() else {"objects": []}
    generation_by_object = {str(item.get("object_id")): item for item in stage04.get("objects", [])}
    texture_root = out / "05_texture_state"
    if texture_root.exists():
        shutil.rmtree(texture_root)
    texture_root.mkdir(parents=True, exist_ok=True)
    alignment = dict(config.get("alignment", {}))
    resolution = int(alignment.get("atlas_resolution", 1024))
    objects = [record for record in plan.get("objects", []) if str(record["object_id"]) in manifest]
    print(f"[05] Initializing object-owned texture state for {len(objects)} objects at {resolution}x{resolution}...", flush=True)
    records = []
    for index, record in enumerate(objects, start=1):
        object_id = str(record["object_id"])
        semantic = str(record.get("semantic_class", ""))
        color = _base_color(record)
        atlas = ObjectAtlas(texture_root, object_id, semantic, resolution, color)
        atlas.initialize()
        generation_result = generation_by_object.get(object_id, {})
        fallback_used = bool(generation_result.get("fallback_used", False))
        blender_record = blender_records.get(object_id, {})
        transferred_path = Path(blender_record.get("baked_texture", "")) if blender_record.get("baked_texture") else None
        transfer_diagnostics = dict(blender_record.get("atlas_transfer", {}))
        texture_inspection = _inspect_texture(transferred_path, resolution) if transferred_path else {
            "valid": False, "reason": "no_registered_asset_texture"
        }
        transfer_valid = bool(texture_inspection.get("valid")) and str(transfer_diagnostics.get("status", "ok")) == "ok"
        generated_or_external = record.get("generation_mode") in {"asset_3d", "external_asset"}

        if transfer_valid:
            shutil.copy2(transferred_path, atlas.color_path)
            initial_source = "registered_asset_material_atlas"
        else:
            Image.new("RGB", (resolution, resolution), color).save(atlas.color_path)
            initial_source = "json_base_color"

        triangle_count = int(manifest[object_id]["triangle_count"])
        print(f"[05] Texture state {index}/{len(objects)}: {object_id} ({triangle_count} triangles)", flush=True)
        labels, island_count = _uv_surface_labels(
            manifest[object_id]["triangles"],
            resolution,
            gutter_px=int(alignment.get("atlas_gutter_px", 2)),
        )
        surface = labels > 0
        np.save(atlas.island_path, labels)
        Image.fromarray((surface * 255).astype(np.uint8), "L").save(atlas.reachable_path)

        # Triangle visibility is retained for camera/refinement scheduling only.
        # The texture state itself is just base_color.png; no texel coverage or
        # visit-count files are created.
        np.save(atlas.dir / "triangle_seen.npy", np.zeros((triangle_count,), dtype=np.bool_))
        np.save(atlas.dir / "triangle_texture_valid.npy", np.full((triangle_count,), True, dtype=np.bool_))

        metadata = load_json(atlas.meta_path)
        metadata.update(
            {
                "visual_source": (
                    "registered_asset" if generated_or_external and not fallback_used
                    else "json_scaffold_fallback" if fallback_used
                    else "json_scaffold"
                ),
                "generation_mode": record.get("generation_mode"),
                "fallback_used": fallback_used,
                "triangle_count": triangle_count,
                "triangle_seen_path": str(atlas.dir / "triangle_seen.npy"),
                "triangle_texture_valid_path": str(atlas.dir / "triangle_texture_valid.npy"),
                "initial_texture_source": initial_source,
                "initial_texture_valid": True,
                "initial_texture_transfer_valid": bool(transfer_valid),
                "initial_texture_has_detail": bool(texture_inspection.get("has_detail", False)),
                "initial_texture_inspection": texture_inspection,
                "atlas_transfer": transfer_diagnostics,
                "state_contract": {
                    "base_color": "the complete persistent appearance texture",
                    "triangle_seen": "camera/refinement scheduling state only",
                },
                "uv_surface_texels": int(surface.sum()),
                "uv_island_count": island_count,
            }
        )
        save_json(metadata, atlas.meta_path)
        records.append(
            {
                "object_id": object_id,
                "generation_mode": record.get("generation_mode"),
                "triangle_count": triangle_count,
                "initial_triangle_seen": 0,
                "initial_texture_source": initial_source,
                "initial_texture_transfer_valid": bool(transfer_valid),
                "fallback_used": fallback_used,
                "uv_surface_texels": int(surface.sum()),
            }
        )
    return {"texture_root": str(texture_root), "objects": records}


def import_align_and_initialize(out: str | Path, asset_config_path: str | Path) -> Dict:
    out = Path(out)
    config = load_json(asset_config_path)
    step = out / "05_scene_assets"
    step.mkdir(parents=True, exist_ok=True)
    registration = build_registration_plan(out, config)
    blender = os.environ.get("BLENDER_BIN", "blender")
    script = Path("src/blender/prephysics_runtime/import_align_prepare_assets.py")
    runtime_root = step / "blender_runtime"
    user_config = runtime_root / "user_config"
    user_scripts = runtime_root / "user_scripts"
    temp_root = runtime_root / "tmp"
    for directory in (user_config, user_scripts, temp_root):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        str((out / "02_blender_scaffold" / "scaffold.blend").resolve()),
        "--python",
        str(script),
        "--",
        "--out",
        str(out),
        "--asset_config",
        str(asset_config_path),
    ]
    env = os.environ.copy()
    env["BLENDER_USER_CONFIG"] = str(user_config.resolve())
    env["BLENDER_USER_SCRIPTS"] = str(user_scripts.resolve())
    env["TMPDIR"] = str(temp_root.resolve())
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        checkpoint_path = step / "blender_checkpoint.json"
        checkpoint = load_json(checkpoint_path) if checkpoint_path.exists() else {}
        crash_files = sorted(temp_root.glob("*.crash.txt"), key=lambda path: path.stat().st_mtime)
        crash_tail = ""
        crash_path = None
        if crash_files:
            crash_path = crash_files[-1]
            crash_tail = crash_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        detail = [
            f"Blender Stage05 exited with return code {exc.returncode}.",
            f"Last checkpoint: {checkpoint}",
        ]
        if crash_path is not None:
            detail.append(f"Native crash report: {crash_path}")
            detail.append(crash_tail)
        raise RuntimeError("\n".join(detail)) from exc
    report_path = step / "blender_import_report.json"
    if not report_path.exists():
        detail = (step / ".blender_failed").read_text(encoding="utf-8") if (step / ".blender_failed").exists() else ""
        raise RuntimeError("Blender asset import/registration did not produce its report.\n" + detail[-8000:])
    blender_report = load_json(report_path)
    scene_path = Path(blender_report["scene_blend"])
    if not scene_path.exists() or scene_path.stat().st_size == 0:
        raise RuntimeError(f"Registered scene blend missing: {scene_path}")
    texture_report = _initialize_texture_state(out, config, blender_report)
    report = {
        "status": "ok",
        "stage": "05_import_register_object_assets",
        "scene_assets_blend": str(scene_path),
        "registration_plan": registration,
        "blender": blender_report,
        "texture_state": texture_report,
        "runtime": "surface point-cloud similarity registration baked in model space + scaffold-root world placement + preserved hierarchy-child local matrices + defensive Blender import",
    }
    save_json(report, step / "stage_report.json")
    return report
