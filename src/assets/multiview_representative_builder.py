from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping

from PIL import Image, ImageFilter

from src.appearance.backend_factory import create_backend
from src.appearance.model_cache import ensure_backend_models
from src.assets.eligibility import objects_by_mode
from src.assets.representative_anchor_selection import DEFAULT_SELECTION_CONFIG, select_anchor_views
from src.assets.representative_prompting import (
    build_hero_summary,
    build_representative_negative_prompt,
    build_representative_prompt,
)
from src.io.json_io import load_json, save_json


def _merge(*items):
    result = {}
    for item in items:
        if isinstance(item, Mapping):
            for key, value in item.items():
                if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                    nested = dict(result[key])
                    nested.update(value)
                    result[key] = nested
                else:
                    result[key] = value
    return result


def build_object_prompt(record: Mapping, view: Mapping, generic_suffix: str = "", *, role: str = "hero", hero_summary: str = "") -> str:
    return build_representative_prompt(
        record,
        view,
        role=role,
        generic_suffix=generic_suffix,
        hero_summary=hero_summary,
    )

def _alpha_to_mask(source: Path, destination: Path, dilation_px: int = 6) -> None:
    alpha = Image.open(source).convert("RGBA").getchannel("A")
    if dilation_px > 0:
        alpha = alpha.filter(ImageFilter.MaxFilter(dilation_px * 2 + 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    alpha.save(destination)


def _rgba_to_reference(source: Path, destination: Path, background=(245, 245, 245)) -> None:
    rgba = Image.open(source).convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, background + (255,))
    Image.alpha_composite(canvas, rgba).convert("RGB").save(destination)


def _apply_original_alpha(generated: Path, mask_rgba: Path, output: Path) -> None:
    rgb = Image.open(generated).convert("RGB")
    alpha = Image.open(mask_rgba).convert("RGBA").getchannel("A").resize(rgb.size, Image.Resampling.NEAREST)
    alpha = alpha.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output)


def _run_capture(out: Path, object_id: str, output_dir: Path, blender: str, capture_script: str, asset_config_path: str | Path) -> None:
    command = [
        blender,
        "--background",
        "--python",
        str(capture_script),
        "--",
        "--out",
        str(out),
        "--object_id",
        object_id,
        "--output_dir",
        str(output_dir),
        "--asset_config",
        str(asset_config_path),
    ]
    subprocess.run(command, check=True)
    report = output_dir / "capture_report.json"
    if not report.exists() or report.stat().st_size == 0:
        raise RuntimeError(f"Multiview scaffold capture failed for {object_id}: {report}")


def _object_selection_config(record: Mapping, rep_cfg: Mapping) -> Dict:
    generation = dict(record.get("generation", {}))
    return _merge(DEFAULT_SELECTION_CONFIG, dict(rep_cfg.get("anchor_selection", {})), dict(generation.get("representative_anchor_selection", {})))


def generate_multiview_representative_images(out: str | Path, asset_config: Dict) -> Dict:
    out = Path(out)
    step = out / "03_object_representative_images"
    step.mkdir(parents=True, exist_ok=True)
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    rep_cfg = dict(asset_config.get("representative_image_generation", {}))
    capture_records = [
        record
        for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) == "asset_3d"
        and float(dict(record.get("camera", {})).get("importance", 1.0)) > 0.0
    ]
    asset_records = list(capture_records)

    backend_name = str(rep_cfg.get("backend", "flux1_depth_control_inpaint_nf4_16gb"))
    params = load_json("configs/parameters.json")
    backend_cfg = params["backend"]["profiles"][backend_name]
    auth_cfg = params["backend"].get("authentication", {})
    model_preparation = None
    backend = None
    if asset_records:
        model_preparation = ensure_backend_models(backend_name, backend_cfg, auth_cfg)
        backend = create_backend(backend_name, backend_cfg, auth_config=auth_cfg)

    blender = os.environ.get("BLENDER_BIN", "blender")
    capture_script = rep_cfg.get(
        "capture_script",
        "src/blender/prephysics_runtime/capture_object_multiview_inputs.py",
    )
    generic_prompt_suffix = str(rep_cfg.get("generic_prompt_suffix", "")).strip()

    records: List[Dict] = []
    asset_index = 0
    for record in capture_records:
        oid = str(record["object_id"])
        mode = str(record.get("generation_mode", ""))
        directory = step / oid
        directory.mkdir(parents=True, exist_ok=True)
        _run_capture(
            out,
            oid,
            directory,
            blender,
            capture_script,
            asset_config.get("__path__", "configs/asset_pipeline.json"),
        )
        capture_report = load_json(directory / "capture_report.json")

        def _base_report_entry(view: Mapping, *, selected_anchor: bool) -> Dict:
            view_dir = Path(view["output_dir"])
            return {
                "name": str(view["name"]),
                "ring": view.get("ring"),
                "azimuth_deg": float(view.get("azimuth_deg", 0.0)),
                "elevation_deg": float(view.get("elevation_deg", 0.0)),
                "output_dir": str(view_dir),
                "camera": view.get("camera", {}),
                "outputs": view.get("outputs", {}),
                "metrics": view.get("metrics", {}),
                "selection_base_score": float(view.get("selection_base_score", 0.0)),
                "selected_anchor": selected_anchor,
            }

        assert backend is not None
        eligible_views = [v for v in capture_report.get("views", []) if bool(dict(v.get("metrics", {})).get("fully_inside_frustum", False))]
        if not eligible_views:
            raise RuntimeError(f"Object {oid}: no candidate camera contains the full object inside the configured safe frustum")
        selection_input = dict(capture_report)
        selection_input["views"] = eligible_views
        selection = select_anchor_views(selection_input, _object_selection_config(record, rep_cfg))
        anchor_lookup = {
            str(item.get("name")): dict(item)
            for item in selection.get("selected_anchors", [])
        }
        selected_views = [
            dict(anchor_lookup[str(item.get("name"))])
            for item in selection.get("selected_anchors", [])
            if str(item.get("name")) in anchor_lookup
        ]
        selected_views.sort(
            key=lambda item: float(item.get("selection_gain", item.get("selection_base_score", 0.0))),
            reverse=True,
        )
        hero_view_name = str(selected_views[0]["name"]) if selected_views else ""
        hero_summary = ""
        hero_cfg = dict(rep_cfg.get("hero_generation", {}))
        consistency_cfg = dict(rep_cfg.get("consistency_generation", {}))
        selected_name_set = {str(item.get("name")) for item in selected_views}
        view_lookup = {str(item["name"]): item for item in capture_report.get("views", [])}
        view_reports = []
        anchor_reports = []

        for selected_index, selected_view in enumerate(selected_views):
            view_name = str(selected_view["name"])
            view = view_lookup[view_name]
            view_dir = Path(view["output_dir"])
            report_entry = _base_report_entry(view, selected_anchor=True)
            _rgba_to_reference(view_dir / "scaffold_rgba.png", view_dir / "scaffold_rgb.png")
            _alpha_to_mask(
                view_dir / "mask_rgba.png",
                view_dir / "generation_mask.png",
                int(rep_cfg.get("mask_dilation_px", 6)),
            )
            role = "hero" if selected_index == 0 else "consistency"
            prompt = build_object_prompt(
                record,
                selected_view,
                generic_suffix=generic_prompt_suffix,
                role=role,
                hero_summary=hero_summary,
            )
            negative = build_representative_negative_prompt(record, role=role)
            (view_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            raw = view_dir / "representative_raw.png"
            tuned = hero_cfg if role == "hero" else consistency_cfg
            request = {
                "prompt": prompt,
                "negative_prompt": negative,
                "init_image_path": str(view_dir / "scaffold_rgb.png"),
                "generation_mask_path": str(view_dir / "generation_mask.png"),
                "depth_image_path": str(view_dir / "depth_control.png"),
                "output_path": str(raw),
                "control_preview_path": str(view_dir / "depth_control_used.png"),
                "object_name": oid,
                "semantic_class": str(record.get("semantic_class", "")),
                "seed": int(rep_cfg.get("seed", 6100)) + 100 * asset_index + selected_index,
                "strength": float(tuned.get("strength", rep_cfg.get("strength", 0.82))),
                "guidance_scale": float(tuned.get("guidance_scale", rep_cfg.get("guidance_scale", backend_cfg.get("guidance_scale", 10.0)))),
                "num_inference_steps": int(tuned.get("num_inference_steps", rep_cfg.get("num_inference_steps", backend_cfg.get("num_inference_steps", 30)))),
                "width": int(rep_cfg.get("width", 1024)),
                "height": int(rep_cfg.get("height", 1024)),
                "require_depth_variation": True,
                "depth_min_unique_levels": int(rep_cfg.get("depth_min_unique_levels", 3)),
                "depth_valid_min_gray": int(rep_cfg.get("depth_valid_min_gray", 24)),
                "max_sequence_length": int(tuned.get("max_sequence_length", rep_cfg.get("max_sequence_length", backend_cfg.get("max_sequence_length", 256)))),
                "clip_max_tokens": int(tuned.get("clip_max_tokens", rep_cfg.get("clip_max_tokens", backend_cfg.get("clip_max_tokens", 75)))),
            }
            generation_result = backend.generate(request)
            representative = view_dir / "representative.png"
            _apply_original_alpha(Path(generation_result["output_path"]), view_dir / "mask_rgba.png", representative)
            hero_summary_used = hero_summary
            if role == "hero":
                hero_summary = build_hero_summary(prompt)
                (directory / "hero_anchor_summary.txt").write_text(hero_summary + "\n", encoding="utf-8")
            anchor_entry = dict(report_entry)
            anchor_entry.update({
                "prompt": prompt,
                "negative_prompt": negative,
                "representative": str(representative),
                "generation": generation_result,
                "canonical_view_slot": selected_view.get("canonical_view_slot"),
                "selection_gain": float(selected_view.get("selection_gain", 0.0)),
                "selection_components": selected_view.get("selection_components", {}),
                "generation_role": role,
                "hero_summary_used": hero_summary_used if role != "hero" else "",
            })
            anchor_reports.append(anchor_entry)
            view_reports.append(anchor_entry)

        for view in capture_report.get("views", []):
            if str(view["name"]) not in selected_name_set:
                view_reports.append(_base_report_entry(view, selected_anchor=False))

        anchor_reports.sort(
            key=lambda item: float(item.get("selection_gain", item.get("selection_base_score", 0.0))),
            reverse=True,
        )
        if not anchor_reports:
            raise RuntimeError(f"No selected representative anchor was generated for {oid}")
        primary = anchor_reports[0]
        primary_path = directory / "representative.png"
        Image.open(primary["representative"]).save(primary_path)
        representative_images = []
        for anchor in anchor_reports:
            slot = str(anchor.get("canonical_view_slot", anchor["name"]))
            slot_path = directory / f"anchor_{slot}.png"
            Image.open(anchor["representative"]).save(slot_path)
            representative_images.append({
                "slot": slot,
                "path": str(slot_path),
                "source_view": anchor["name"],
            })

        report = {
            "object_id": oid,
            "name": record.get("name", oid),
            "semantic_class": record.get("semantic_class", ""),
            "generation_mode": mode,
            "representative_image": str(primary_path),
            "primary_view": primary["name"],
            "hero_view": hero_view_name,
            "hero_anchor_summary": hero_summary,
            "selected_anchor_count": len(anchor_reports),
            "spatial_category": record.get("spatial_category", {}),
            "selected_anchors": anchor_reports,
            "representative_images": representative_images,
            "all_candidate_views": view_reports,
            "anchor_selection_summary": selection,
            "capture_report": str(directory / "capture_report.json"),
        }
        save_json(report, directory / "stage_report.json")
        records.append(report)
        asset_index += 1

    stage_report = {
        "status": "ok",
        "stage": "03_generate_object_representative_images",
        "objects": records,
        "model_preparation": model_preparation,
        "routing": "generation.mode == asset_3d only; isolated anchor generation is independent from Stage07",
        "selection_policy": "category-specific single-ring sampling -> strict full-frustum filtering -> exactly one best representative view for Pixal3D",
    }
    save_json(stage_report, step / "stage_report.json")
    return stage_report

