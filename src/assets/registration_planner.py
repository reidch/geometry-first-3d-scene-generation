from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from src.assets.eligibility import objects_by_mode
from src.assets.geometric_registration import (
    parse_obj_mesh,
    register_surface_similarity_alignment,
    sample_mesh_surface,
    sample_scaffold_surface,
)
from src.io.json_io import load_json, save_json


def _merged_registration_config(
    plan: Mapping[str, Any],
    record: Mapping[str, Any],
    asset_config: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = deepcopy(dict(asset_config.get("registration", {})))
    defaults = dict(plan.get("scene_defaults", {})).get("registration", {})
    merged.update(deepcopy(dict(defaults)))
    merged.update(deepcopy(dict(record.get("generation", {}).get("registration", {}))))
    return merged




def _stage04_generation_report(out: Path, object_id: str) -> Dict[str, Any]:
    report_path = out / "04_object_assets" / str(object_id) / "generation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing Stage04 generation report: {report_path}")
    return load_json(report_path)


def _scaffold_fallback_registration_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    oid = str(record["object_id"])
    return {
        "object_id": oid,
        "name": record.get("name", oid),
        "semantic_class": record.get("semantic_class", ""),
        "generation_mode": record.get("generation_mode"),
        "asset_path": None,
        "fallback_used": True,
        "fallback_source": "stage02_json_scaffold",
        "method": "identity_json_scaffold_fallback",
        "rotation_index": None,
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "uniform_scale": 1.0,
        "translation_local": [0.0, 0.0, 0.0],
        "normalized_loss": 0.0,
        "registration_config": deepcopy(dict(config)),
        "surface_sample_count": 0,
        "placement": deepcopy(dict(record.get("placement", {}))),
        "physics": deepcopy(dict(record.get("physics", {}))),
    }

def _resolve_asset_path(out: Path, record: Mapping[str, Any]) -> Path:
    oid = str(record["object_id"])
    mode = str(record.get("generation_mode", ""))
    generation = dict(record.get("generation", {}))
    if mode == "asset_3d":
        report_path = out / "04_object_assets" / oid / "generation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing Stage04 generation report: {report_path}")
        generation_report = load_json(report_path)
        raw = generation_report.get("blender_asset_path") or generation_report.get("obj_path", "")
        path = Path(raw).expanduser()
    elif mode == "external_asset":
        raw = str(generation.get("external_asset_path", "")).strip()
        if not raw:
            raise ValueError(f"Object {oid}: generation.external_asset_path is empty")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        raise ValueError(f"Object {oid}: unsupported registration mode {mode!r}")
    path = path.resolve()
    if path.suffix.lower() != ".obj":
        raise ValueError(
            f"Object {oid}: Stage05 standard-library importer currently accepts OBJ bundles; got {path}"
        )
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing OBJ asset for {oid}: {path}")
    return path


def build_registration_plan(out: str | Path, asset_config: Mapping[str, Any]) -> Dict[str, Any]:
    out = Path(out)
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    candidates = objects_by_mode(plan, "asset_3d") + objects_by_mode(plan, "external_asset")
    records = []
    for index, record in enumerate(candidates):
        oid = str(record["object_id"])
        config = _merged_registration_config(plan, record, asset_config)
        if str(record.get("generation_mode", "")) == "asset_3d":
            generation_report = _stage04_generation_report(out, oid)
            if bool(generation_report.get("fallback_used", False)):
                result = _scaffold_fallback_registration_record(record, config)
                result["stage04_fallback_reason"] = generation_report.get("fallback_reason")
                records.append(result)
                print(
                    f"[05][REGISTER][SCAFFOLD_FALLBACK] {oid}: Stage04 produced no asset; "
                    "registration skipped and Stage02 scaffold will be used.",
                    flush=True,
                )
                continue
        asset_path = _resolve_asset_path(out, record)
        sample_count = int(config.get("surface_sample_count", 3500))
        coarse_count = int(config.get("coarse_surface_sample_count", min(sample_count, 1600)))
        vertices, triangles = parse_obj_mesh(asset_path)
        asset_points = sample_mesh_surface(vertices, triangles, coarse_count, seed=1100 + index)
        scaffold_points = sample_scaffold_surface(
            record.get("scaffold", {}).get("parts", []),
            coarse_count,
            seed=2100 + index,
        )
        result = register_surface_similarity_alignment(asset_points, scaffold_points, config)
        result.update({
            "object_id": oid,
            "name": record.get("name", oid),
            "semantic_class": record.get("semantic_class", ""),
            "generation_mode": record.get("generation_mode"),
            "asset_path": str(asset_path),
            "registration_config": config,
            "surface_sample_count": coarse_count,
            "placement": deepcopy(dict(record.get("placement", {}))),
            "physics": deepcopy(dict(record.get("physics", {}))),
        })
        records.append(result)
        print(
            f"[05][REGISTER] {oid}: rotation={result['rotation_index']} "
            f"uniform_scale={result['uniform_scale']:.6f} loss={result['normalized_loss']:.8f}",
            flush=True,
        )
    report = {
        "status": "ok",
        "method": "generated/external assets use uniform mesh/scaffold surface sampling + model-space translation/rotation/uniform-scale similarity ICP + minimum symmetric point distance; Stage04 fallback objects bypass registration and reuse the authoritative Stage02 JSON scaffold; roots inherit scaffold world matrices and hierarchy children preserve local matrices",
        "routing": "generation.mode in {'asset_3d', 'external_asset'}; asset_3d with Stage04 fallback_used=true uses scaffold fallback",
        "objects": records,
    }
    destination = out / "05_scene_assets" / "registration_plan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, destination)
    return report
