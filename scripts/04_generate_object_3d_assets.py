#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assets.eligibility import objects_by_mode
from src.assets.pixal3d_backend import Pixal3DBackend, Pixal3DTimeoutError
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.assets.obj_candidate_quality import inspect_obj
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done


def _selected_image(representative_report: dict, representative_dir: Path) -> Path:
    image = Path(representative_report.get("representative_image", representative_dir / "representative.png")).expanduser().resolve()
    if not image.exists() or image.stat().st_size == 0:
        raise FileNotFoundError(f"Missing Pixal3D representative image: {image}")
    return image




def _remove_legacy_geometry_rejection_artifacts(object_dir: Path) -> None:
    for filename in ("geometry_constraint.json", "shape_geometry_report.json"):
        path = object_dir / filename
        if path.exists():
            path.unlink()
    for path in object_dir.glob("rejected_shape_attempt_*.obj"):
        path.unlink()
    fallback_dir = object_dir / "fallback_scaffold"
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset_config", default="configs/asset_pipeline.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    step = out / "04_object_assets"
    logger = setup_step_logger(Path("logs") / out.name / "04_generate_object_3d_assets.log")
    status("[04] Generating the first successful textured 3D asset for each explicit JSON asset_3d object, with scaffold fallback after exhausted timeouts...")
    try:
        config = load_json(args.asset_config)
        default_backend_cfg = dict(config["pixal3d"])
        plan = load_json(out / "01_world_ir" / "generation_plan.json")
        records = []
        for index, record in enumerate(objects_by_mode(plan, "asset_3d")):
            oid = str(record["object_id"])
            generation = dict(record.get("generation", {}))
            backend_name = str(generation.get("backend", "pixal3d"))
            if backend_name != "pixal3d":
                raise ValueError(
                    f"Object {oid}: backend {backend_name!r} is not registered in this package. "
                    "Use external_asset mode or add a generic backend adapter."
                )

            backend_cfg = dict(default_backend_cfg)
            backend_cfg.update(dict(generation.get("backend_options", {})))
            backend = Pixal3DBackend(backend_cfg)

            representative_dir = out / "03_object_representative_images" / oid
            representative_report = load_json(representative_dir / "stage_report.json")
            representative_image = _selected_image(representative_report, representative_dir)

            object_dir = step / oid
            object_dir.mkdir(parents=True, exist_ok=True)
            _remove_legacy_geometry_rejection_artifacts(object_dir)

            selection_cfg = dict(backend_cfg.get("candidate_selection", {}))
            attempt_count = max(1, int(selection_cfg.get("attempt_count", 3)))
            seed_stride = int(selection_cfg.get("seed_stride", 997))
            base_seed = int(backend_cfg.get("seed", 7100)) + index
            attempts_root = object_dir / "candidates"
            if args.force and attempts_root.exists():
                shutil.rmtree(attempts_root)
            attempts_root.mkdir(parents=True, exist_ok=True)

            selected_anchors = list(representative_report.get("selected_anchors", []))
            primary_anchor = selected_anchors[0] if selected_anchors else {}
            camera_context = {
                "object_id": oid,
                "primary_view": representative_report.get("primary_view"),
                "camera": dict(primary_anchor.get("camera", {})),
                "spatial_category": representative_report.get("spatial_category", record.get("spatial_category", {})),
                "bbox_center": primary_anchor.get("camera", {}).get("target"),
                "scaffold": record.get("scaffold", {}),
                "pose_correction": {
                    "enabled": True,
                    "method": "selected representative camera frame",
                },
            }
            attempts = []
            selected = None
            for attempt_index in range(attempt_count):
                seed = base_seed + attempt_index * seed_stride
                attempt_dir = attempts_root / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
                try:
                    attempt_result = backend.generate_asset(
                        representative_image,
                        attempt_dir,
                        oid,
                        seed=seed,
                        camera_context=camera_context,
                    )
                    diagnostics = inspect_obj(attempt_result["blender_asset_path"])
                    attempt_record = {
                        "attempt": attempt_index + 1,
                        "seed": seed,
                        "status": "candidate_generated",
                        "candidate_directory": str(attempt_dir),
                        "result": attempt_result,
                        "generic_geometry_diagnostics": diagnostics,
                        "selection_score": float(diagnostics["generic_quality_score"]),
                    }
                    attempts.append(attempt_record)
                    selected = attempt_record
                    status(
                        f"[04] {oid}: accepted first successful Pixal3D candidate "
                        f"from attempt {attempt_index + 1} (seed {seed}); later seeds are skipped."
                    )
                    break
                except Pixal3DTimeoutError as exc:
                    timeout_record = {
                        "attempt": attempt_index + 1,
                        "seed": seed,
                        "status": "timed_out_retrying_with_next_seed",
                        "candidate_directory": str(attempt_dir),
                        "error": str(exc),
                    }
                    attempts.append(timeout_record)
                    status(
                        f"[04][WARN] {oid}: seed {seed} timed out; "
                        "terminating it and retrying with the next configured seed."
                    )
                    continue

            if selected is None:
                result = {
                    "status": "fallback",
                    "object_id": oid,
                    "name": record.get("name", oid),
                    "semantic_class": record.get("semantic_class", ""),
                    "generation_mode": "asset_3d",
                    "backend": backend_name,
                    "visual_source": "json_scaffold_fallback",
                    "fallback_used": True,
                    "fallback_reason": f"all {attempt_count} configured Pixal3D attempts timed out",
                    "geometry_rejection_enabled": False,
                    "candidate_ranking_enabled": False,
                    "candidate_selection_policy": "first_success_then_scaffold_fallback",
                    "timeout_seed_retry_enabled": True,
                    "scaffold_fallback_after_exhaustion": True,
                    "selected_attempt": None,
                    "selected_seed": None,
                    "selection_reason": "no Pixal3D candidate completed within the configured attempts; use the authoritative JSON scaffold geometry",
                    "generation_attempts": attempts,
                    "input_representative_image": str(representative_image),
                    "selected_anchor_count": int(representative_report.get("selected_anchor_count", 1)),
                }
                save_json(result, object_dir / "generation_report.json")
                records.append(result)
                status(
                    f"[04][FALLBACK] {oid}: all {attempt_count} Pixal3D attempts timed out; "
                    "continuing the pipeline with the authoritative JSON scaffold."
                )
                continue
            selected_dir = Path(selected["candidate_directory"])
            # Publish the first successful candidate under the stable paths consumed by
            # Stage05. Timed-out attempts remain as diagnostics; later seeds are never generated.
            for source in selected_dir.iterdir():
                destination = object_dir / source.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)

            result = dict(selected["result"])
            prefix = str(selected_dir.resolve())
            for key, value in list(result.items()):
                if isinstance(value, str) and value.startswith(prefix):
                    result[key] = str(object_dir.resolve()) + value[len(prefix):]
            result.update({
                "name": record.get("name", oid),
                "semantic_class": record.get("semantic_class", ""),
                "generation_mode": "asset_3d",
                "backend": backend_name,
                "visual_source": "generated_asset",
                "fallback_used": False,
                "geometry_rejection_enabled": False,
                "candidate_ranking_enabled": False,
                "candidate_selection_policy": "first_success",
                "timeout_seed_retry_enabled": True,
                "selected_attempt": int(selected["attempt"]),
                "selected_seed": int(selected["seed"]),
                "selection_reason": "first successfully generated Pixal3D candidate in deterministic seed order; later seeds are skipped",
                "generation_attempts": attempts,
                "input_representative_image": str(representative_image),
                "selected_anchor_count": int(representative_report.get("selected_anchor_count", 1)),
            })
            save_json(result, object_dir / "generation_report.json")
            records.append(result)

        report = {
            "status": "ok",
            "stage": "04_generate_object_3d_assets",
            "objects": records,
            "routing": "generation.mode == asset_3d",
            "runtime": "Pixal3D single-image PBR generation -> first successful textured export; after all configured timeouts, continue with authoritative JSON scaffold fallback",
            "geometry_rejection_enabled": False,
            "candidate_ranking_enabled": False,
            "candidate_selection_policy": "first_success_then_scaffold_fallback",
            "timeout_seed_retry_enabled": True,
            "scaffold_fallback_after_exhaustion": True,
        }
        save_json(report, step / "stage_report.json")
        artifact_index = ArtifactIndex(scene_id=out.name, step="04_generate_object_3d_assets")
        artifact_index.add("stage_report", step / "stage_report.json")
        for record in records:
            if bool(record.get("fallback_used", False)):
                artifact_index.add(
                    f"{record['object_id']}_generation_report",
                    step / str(record["object_id"]) / "generation_report.json",
                )
                continue
            asset_path = Path(record["asset_path"])
            if asset_path.exists():
                artifact_index.add(f"{record['object_id']}_asset", asset_path)
            artifact_index.add(f"{record['object_id']}_obj", Path(record["blender_asset_path"]))
            artifact_index.add(f"{record['object_id']}_bundle", Path(record["asset_bundle_manifest"]))
        artifact_index.save(step / "artifact_index.json")
        mark_done(step)
        logger.info("Stage04 report: %s", report)
        status("[04] Done. First-success Pixal3D assets published; exhausted timeout cases continue with JSON scaffold fallback.")
    except Exception:
        step.mkdir(parents=True, exist_ok=True)
        (step / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
