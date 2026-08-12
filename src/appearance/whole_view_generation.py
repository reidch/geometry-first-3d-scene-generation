from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil

import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping

from PIL import Image

from src.appearance.model_cache import ensure_backend_models
from src.appearance.depth_control_image import load_depth_control_image
from src.appearance.monocular_depth_validation import (
    DepthAnythingV2Validator,
    image_validity,
    save_validation_images,
    validate_depth_structure,
)
from src.appearance.multiview_consistency import (
    build_stage07_overlap_runtime,
    fuse_reprojected_references,
    initialize_generated_neighbor_registry,
    multi_reference_overlap_error,
    reproject_reference_to_current,
    select_next_frontier_camera,
    select_restart_camera_from_completed_topology,
    update_generated_neighbor_registry,
)
from src.appearance.stage08_runtime import (
    FluxWorkerRuntimeError,
    FluxWorkerSettings,
    PersistentFluxWorker,
    Stage08EventLog,
    runtime_memory_snapshot,
)
from src.appearance.stage08_console import (
    Stage08ConsoleProgress,
    suppress_third_party_progress_bars,
    write_empty_marker_atomic,
)
from src.blender.scene_input import resolve_scene_for_textured_downstream
from src.io.json_io import load_json, save_json
from src.scene_ir.json_scene import flat_objects


def _stage08_generation_attempt_schedule(
    *,
    base_strength: float,
    retry_strength_scale: float,
    maximum_retries: int,
    seeds_per_strength: int,
    seed_base: int,
    event_index: int,
    seed_stride: int,
) -> list[Dict[str, Any]]:
    """Build the deterministic Stage08 strength/seed search schedule.

    ``maximum_retries`` keeps its historical meaning: the number of lower-strength
    levels after the initial strength.  Every strength level gets exactly
    ``seeds_per_strength`` independent seeds before the next, more conservative
    strength is tried.
    """
    strength_levels = 1 + max(int(maximum_retries), 0)
    seeds_per_strength = max(int(seeds_per_strength), 1)
    scale = float(retry_strength_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("whole_view_generation.validation.retry_strength_scale must be in (0, 1]")
    schedule: list[Dict[str, Any]] = []
    for strength_level_index in range(strength_levels):
        strength = float(base_strength) * (scale ** strength_level_index)
        for seed_attempt_index in range(seeds_per_strength):
            attempt_index = len(schedule)
            schedule.append({
                "attempt_index": attempt_index,
                "strength_level_index": strength_level_index,
                "strength_level_number": strength_level_index + 1,
                "total_strength_levels": strength_levels,
                "seed_attempt_index": seed_attempt_index,
                "seed_attempt_number": seed_attempt_index + 1,
                "seeds_per_strength": seeds_per_strength,
                "strength": strength,
                "seed": int(seed_base) + int(event_index) * int(seed_stride) + attempt_index,
            })
    return schedule


def _object_map(scene: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(record["object_id"]): dict(record) for record in flat_objects(scene)}


def _clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _natural_owner_label(record: Mapping[str, Any], owner_id: str) -> str:
    value = (
        record.get("name")
        or record.get("semantic_class")
        or record.get("semantic")
        or owner_id
    )
    return _clean_prompt_text(value).replace("_", " ")


def _owner_prompt(record: Mapping[str, Any], scene_prompt: str = "") -> str:
    """Compile Stage08-specific appearance text from the JSON owner record.

    The source JSON generation.prompt is shared with earlier stages and often
    contains geometry/rectified-texture instructions that are correct for asset or
    atlas generation but harmful for perspective whole-view synthesis. Stage08
    therefore prioritizes appearance/material text, uses the generation prompt only
    as a fallback, and omits a style_prompt when it exactly duplicates the scene
    style already present once in the global prompt.
    """
    generation = dict(record.get("generation", {}))
    appearance_record = dict(record.get("appearance", {}))
    mode = str(record.get("generation_mode") or generation.get("mode") or "")
    identity = _clean_prompt_text(generation.get("prompt"))
    appearance = _clean_prompt_text(
        generation.get("appearance_prompt")
        or generation.get("material_prompt")
        or appearance_record.get("prompt")
    )
    style = _clean_prompt_text(generation.get("style_prompt"))
    scene_clean = _clean_prompt_text(scene_prompt)

    values: list[str] = []
    # The natural owner label already carries identity. Prefer material/finish text
    # so Stage08 spends its limited token budget on appearance rather than asking
    # FLUX to re-invent geometry that the depth condition already fixes.
    if appearance:
        values.append(appearance)
    elif mode != "surface_texture" and identity:
        values.append(identity)

    # Scene-wide style prompts are commonly copied verbatim into every object.
    # Include only genuinely object-specific style text not already contained in
    # the global scene description.
    if style and style.casefold() not in scene_clean.casefold():
        values.append(style)
    return " ".join(values)




def _palette_color(record: Any) -> tuple[int, int, int] | None:
    if isinstance(record, Mapping):
        value = record.get("color_uint8_rgb", record.get("rgb"))
    else:
        value = record
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return tuple(int(round(float(component))) for component in value[:3])


def _semantic_layout(frame: Mapping[str, Any], owner_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    semantic_path = Path(str(frame.get("semantic", "")))
    palette_path = Path(str(frame.get("palette", "")))
    if not semantic_path.exists() or not palette_path.exists():
        return {}
    semantic = np.asarray(Image.open(semantic_path).convert("RGB"), dtype=np.uint8)
    palette = load_json(palette_path)
    height, width = semantic.shape[:2]
    result: Dict[str, Dict[str, Any]] = {}
    for owner in owner_ids:
        color = _palette_color(palette.get(owner))
        if color is None:
            continue
        mask = np.all(semantic == np.asarray(color, dtype=np.uint8), axis=-1)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        cx = float((xs.mean() + 0.5) / max(width, 1))
        cy = float((ys.mean() + 0.5) / max(height, 1))
        horizontal = "left" if cx < 1.0 / 3.0 else "right" if cx > 2.0 / 3.0 else "center"
        vertical = "upper" if cy < 1.0 / 3.0 else "lower" if cy > 2.0 / 3.0 else "middle"
        result[owner] = {
            "screen_region": f"{horizontal}-{vertical}",
            "centroid_normalized": [cx, cy],
            "bbox_normalized": [
                float(xs.min() / max(width, 1)),
                float(ys.min() / max(height, 1)),
                float((xs.max() + 1) / max(width, 1)),
                float((ys.max() + 1) / max(height, 1)),
            ],
            "pixel_fraction": float(xs.size / max(width * height, 1)),
        }
    return result


def _compile_prompt(
    scene: Mapping[str, Any],
    frame: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compile the Stage08 whole-view prompt from scene-level appearance only.

    Object-level generation/style/material prompts belong to upstream asset and
    texture synthesis stages. Reusing them here makes whole-view synthesis
    re-describe/re-design already-fixed geometry and wastes the short text-encoder
    budget. Stage08 therefore follows a WorldMesh-like prompt structure without
    inheriting WorldMesh-specific multi-image wording: a fixed pipeline contract
    first, followed by one compact scene-level appearance/theme description.
    """
    scene_payload = dict(scene.get("scene", scene))
    prompt_rule = _clean_prompt_text(config.get("prompt_rule"))
    scene_appearance = _clean_prompt_text(
        scene_payload.get("stage08_appearance_prompt")
        or scene_payload.get("appearance_prompt")
        or scene_payload.get("prompt")
        or "a coherent furnished interior"
    )
    scene_clause = f"Scene appearance: {scene_appearance}." if scene_appearance else ""
    prompt = " ".join(value for value in [prompt_rule, scene_clause] if value)
    return {
        "prompt": prompt,
        "prompt_structure": "fixed_stage08_contract_plus_scene_appearance",
        "scene_appearance_source": (
            "scene.stage08_appearance_prompt"
            if _clean_prompt_text(scene_payload.get("stage08_appearance_prompt"))
            else "scene.appearance_prompt"
            if _clean_prompt_text(scene_payload.get("appearance_prompt"))
            else "scene.prompt"
        ),
        "object_level_prompting_enabled": False,
        "visible_owner_ids_in_prompt": [],
        "visible_owner_count_in_prompt": 0,
        "target_owner_id": str(frame.get("target_owner_id") or "") or None,
        "target_owner_forced_even_if_non_significant": False,
        "semantic_layout": {},
        "semantic_layout_conditioning": "disabled_for_stage08_prompting",
    }

def _white_mask(reference_path: str | Path, output_path: Path) -> None:
    reference = Image.open(reference_path)
    mask = Image.new("L", reference.size, color=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


def _copy_required(source: str | Path, destination: Path) -> str:
    source = Path(source)
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _save_json_atomic(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    save_json(data, temporary)
    os.replace(temporary, path)




def _hardlink_or_copy(source: str | Path, destination: str | Path) -> str:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def _hardlink_tree(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _hardlink_or_copy(path, target)




def _effective_camera_z_depth_encoding(encoding: Mapping[str, Any]) -> Dict[str, Any]:
    """Publish the V109 effective contract without mutating Stage07 artifacts."""
    result = dict(encoding)
    result["depth_convention"] = "camera_z"
    for key in ("type", "encoding"):
        value = str(result.get(key, ""))
        if "euclidean_ray_distance" in value or "ray_distance" in value:
            result[key] = "uint16_normalized_camera_z_near_bright_background_zero"
    return result

def _reference_conditioning_key(attempt_config: Mapping[str, Any]) -> str:
    payload = {
        "depth_geometry_contract": str(attempt_config.get("depth_geometry_contract", "")),
        "reference_reprojection": dict(attempt_config.get("reference_reprojection", {})),
        "multi_reference_fusion": dict(attempt_config.get("multi_reference_fusion", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_signature(slot: Mapping[str, Any]) -> list[str]:
    return sorted(str(item["camera_id"]) for item in slot.get("generated_neighbors", []))


def _fallback_depth_scores(
    attempt: Mapping[str, Any],
    *,
    depth_geometry_contract: str | None = None,
) -> Dict[str, float] | None:
    """Return one-way mesh->predicted depth-recall fallback ranking scores.

    Deadlock recovery relaxes only the recall threshold. Reverse predicted-edge
    precision remains a saved diagnostic and never affects acceptance or ranking.
    """
    if depth_geometry_contract is not None and str(attempt.get("depth_geometry_contract", "")) != depth_geometry_contract:
        return None
    if bool(attempt.get("accepted", False)):
        return None
    if not bool(dict(attempt.get("image_validity", {})).get("accepted", False)):
        return None
    depth = dict(attempt.get("depth_validation", {}))
    try:
        recall = float(depth["depth_edge_recall"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(recall):
        return None
    precision_value = depth.get("predicted_depth_edge_precision")
    try:
        precision = float(precision_value) if precision_value is not None else float("nan")
    except (TypeError, ValueError):
        precision = float("nan")
    return {
        "mesh_to_predicted_recall": recall,
        "predicted_to_mesh_precision_diagnostic": precision,
    }


def _camera_round_history(directory: str | Path) -> list[Dict[str, Any]]:
    """Load every completed generation round for a camera from durable artifacts."""
    directory = Path(directory)
    history: list[Dict[str, Any]] = []
    for round_directory in sorted(directory.glob("round_*")):
        report_path = round_directory / "round_report.json"
        if not report_path.exists():
            continue
        report = load_json(report_path)
        history.append({
            "round_index": int(report.get("round_index", len(history))),
            "generated_neighbor_ids": list(report.get("generated_neighbor_ids", [])),
            "attempts": list(report.get("attempts", [])),
            "accepted_attempt_index": report.get("accepted_attempt_index"),
        })
    return history


def _select_deadlock_fallback_candidate(
    stage: str | Path,
    registry: Mapping[str, Mapping[str, Any]],
    frame_index_by_id: Mapping[str, int],
    *,
    require_image_validity: bool = True,
    depth_geometry_contract: str | None = None,
) -> Dict[str, Any] | None:
    """Choose the best previously-generated deferred result by one-way recall.

    Selection is lexicographic: maximize mesh->predicted depth-edge recall, then
    propagation support, then stable camera/round/attempt order. Reverse precision
    is diagnostic-only and cannot influence fallback promotion.
    """
    stage = Path(stage)
    candidates: list[Dict[str, Any]] = []
    for camera_id, slot in registry.items():
        if str(slot.get("status")) != "deferred":
            continue
        camera_directory = stage / "final_views" / str(camera_id)
        for validation_path in sorted(camera_directory.glob("round_*/attempt_*/validation.json")):
            attempt = load_json(validation_path)
            scores = _fallback_depth_scores(
                attempt,
                depth_geometry_contract=depth_geometry_contract,
            )
            if scores is None:
                if require_image_validity:
                    continue
                if depth_geometry_contract is not None and str(attempt.get("depth_geometry_contract", "")) != depth_geometry_contract:
                    continue
                depth = dict(attempt.get("depth_validation", {}))
                try:
                    recall = float(depth["depth_edge_recall"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(recall):
                    continue
                precision_value = depth.get("predicted_depth_edge_precision")
                try:
                    precision = float(precision_value) if precision_value is not None else float("nan")
                except (TypeError, ValueError):
                    precision = float("nan")
                scores = {
                    "mesh_to_predicted_recall": recall,
                    "predicted_to_mesh_precision_diagnostic": precision,
                }
            generated_path = validation_path.with_name("generated.png")
            if not generated_path.exists() or generated_path.stat().st_size == 0:
                continue
            round_name = validation_path.parent.parent.name
            try:
                round_index = int(round_name.rsplit("_", 1)[-1])
            except ValueError:
                round_index = int(attempt.get("round_index", 0))
            candidates.append({
                "camera_id": str(camera_id),
                "round_index": round_index,
                "attempt_index": int(attempt.get("attempt_index", 0)),
                "validation_path": str(validation_path),
                "generated_image": str(generated_path),
                "attempt": attempt,
                "propagation_support_score": float(slot.get("propagation_support_score", 0.0)),
                **scores,
            })
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            float(item["mesh_to_predicted_recall"]),
            float(item["propagation_support_score"]),
            -int(frame_index_by_id[str(item["camera_id"])]),
            -int(item["round_index"]),
            -int(item["attempt_index"]),
        ),
    )


def _restore_stage08_state(
    stage: Path,
    frames: list[Mapping[str, Any]],
    registry: Dict[str, Dict[str, Any]],
    adjacency: Mapping[str, Any],
    root_camera_id: str,
    registry_path: Path,
    depth_geometry_contract: str,
    execution_contract: str,
) -> Dict[str, Any]:
    """Restore atomically committed V124 single-round views only.

    Strict completions are replayed into the successful/trusted appearance
    frontier.  Fallback completions remain completed for traversal/Stage09 but
    are never replayed as appearance references.  Older execution contracts are
    intentionally ignored instead of mixing incompatible trust semantics.
    """
    previous = load_json(registry_path) if registry_path.exists() else {}
    previous_valid = bool(
        isinstance(previous, Mapping)
        and str(previous.get("depth_geometry_contract", "")) == depth_geometry_contract
        and str(previous.get("execution_contract", "")) == execution_contract
    )
    previous_events = list(previous.get("generation_events", [])) if previous_valid else []

    committed: list[tuple[int, str, Dict[str, Any]]] = []
    for frame in frames:
        camera_id = str(frame["camera_id"])
        directory = stage / "final_views" / camera_id
        marker = directory / "selected.txt"
        report_path = directory / "generation_report.json"
        final_view = directory / "final_view.png"
        if not marker.exists():
            continue
        if not report_path.exists() or not final_view.exists() or final_view.stat().st_size == 0:
            raise RuntimeError(
                f"Stage08 resume found selected.txt without complete committed artifacts for {camera_id}"
            )
        report = load_json(report_path)
        if (
            str(report.get("depth_geometry_contract", "")) != depth_geometry_contract
            or str(report.get("execution_contract", "")) != execution_contract
        ):
            # Do not silently adopt outputs from an incompatible lifecycle or depth contract.
            marker.unlink(missing_ok=True)
            continue
        committed.append((int(report["generation_order_index"]), camera_id, report))

    committed.sort(key=lambda item: item[0])
    if committed and committed[0][1] != root_camera_id:
        raise RuntimeError("Stage08 resume state does not begin with the configured bootstrap root")
    for expected_index, (order_index, camera_id, report) in enumerate(committed):
        if order_index != expected_index:
            raise RuntimeError(
                f"Stage08 resume completion-order gap: expected {expected_index}, found {order_index} for {camera_id}"
            )
        slot = registry[camera_id]
        successful = bool(report.get("successful", report.get("strict_validation_passed", False)))
        can_be_reference = bool(report.get("can_be_reference", successful)) and successful
        slot.update({
            "status": "completed",
            "completed": True,
            "successful": successful,
            "can_be_reference": can_be_reference,
            "reference_trust": "strict_success" if successful else "fallback_untrusted",
            "effective_reference_edge_weight": 1.0 if successful else 0.0,
            "generation_order_index": order_index,
            "attempt_round_count": 1,
        })
        if successful:
            update_generated_neighbor_registry(registry, camera_id, adjacency)

    normalized_events: list[Dict[str, Any]] = []
    committed_ids = {camera_id for _idx, camera_id, _report in committed}
    for raw_event in previous_events:
        event = dict(raw_event)
        camera_id = str(event.get("camera_id", ""))
        if camera_id in committed_ids and str(event.get("status", "")) in {"completed", "strict", "fallback_single_round"}:
            normalized_events.append(event)
    next_event_index = max(
        (int(item.get("event_index", -1)) for item in normalized_events),
        default=-1,
    ) + 1
    return {
        "committed": committed,
        "previous_events": normalized_events,
        "next_event_index": next_event_index,
        "resumed": bool(committed),
    }

def _spill_fusion_reference_records(fusion: Dict[str, Any], root: Path) -> list[Dict[str, Any]]:
    """Move diagnostic-only per-reference float arrays out of RAM before FLUX."""
    root.mkdir(parents=True, exist_ok=True)
    lightweight: list[Dict[str, Any]] = []
    for index, item in enumerate(list(fusion.get("reference_records", []))):
        path = root / f"reference_{index:03d}.npz"
        np.savez(
            path,
            warped_rgb=np.asarray(item["warped_rgb"], dtype=np.float32),
            final_weight=np.asarray(item["final_weight"], dtype=np.float32),
        )
        lightweight.append({
            "camera_id": str(item["camera_id"]),
            "overlap_cache_path": str(path),
        })
    fusion["reference_records"] = lightweight
    return lightweight


def _remove_overlap_spills(fusion: Mapping[str, Any]) -> None:
    for item in fusion.get("reference_records", []):
        value = str(item.get("overlap_cache_path", ""))
        if value:
            Path(value).unlink(missing_ok=True)



def _forward_snapshot_report_path(stage: Path, camera_id: str) -> Path:
    return stage / "final_views" / str(camera_id) / "forward_pass" / "generation_report.json"


def _write_forward_pass_snapshot(
    stage: Path,
    final_views_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    ordered = sorted(
        (dict(item) for item in final_views_by_id.values()),
        key=lambda item: int(item.get("generation_order_index", 10**9)),
    )
    entries: list[Dict[str, Any]] = []
    for report in ordered:
        camera_id = str(report["camera_id"])
        snapshot_path = _forward_snapshot_report_path(stage, camera_id)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_report = dict(report)
        snapshot_report.pop("repair_pass", None)
        save_json(snapshot_report, snapshot_path)
        accepted_attempt = int(snapshot_report["accepted_attempt_index"])
        source_image = stage / "final_views" / camera_id / "round_00" / f"attempt_{accepted_attempt:02d}" / "generated.png"
        if not source_image.exists() or source_image.stat().st_size == 0:
            raise FileNotFoundError(
                f"Stage08A forward snapshot source is missing for {camera_id}: {source_image}"
            )
        entries.append(
            {
                "camera_id": camera_id,
                "generation_order_index": int(snapshot_report.get("generation_order_index", len(entries))),
                "successful": bool(snapshot_report.get("successful", snapshot_report.get("strict_validation_passed", False))),
                "acceptance_mode": str(snapshot_report.get("acceptance_mode", "strict")),
                "accepted_attempt_index": accepted_attempt,
                "forward_generation_report": str(snapshot_path),
                "forward_selected_attempt_image": str(source_image),
            }
        )
    forward_state_root = stage / "forward_state"
    forward_state_root.mkdir(parents=True, exist_ok=True)
    state_snapshots: Dict[str, str] = {}
    for filename in ("generated_neighbor_registry.json", "weighted_frontier_generation_order.json"):
        source = stage / filename
        if source.exists():
            target = forward_state_root / filename
            shutil.copy2(source, target)
            state_snapshots[filename] = str(target)
    summary = {
        "schema_version": 1,
        "completed": True,
        "camera_count": len(entries),
        "generation_order": [item["camera_id"] for item in entries],
        "strict_success_camera_ids": [item["camera_id"] for item in entries if item["successful"]],
        "fallback_camera_ids": [item["camera_id"] for item in entries if not item["successful"]],
        "state_snapshots": state_snapshots,
        "entries": entries,
    }
    _save_json_atomic(summary, stage / "forward_pass_summary.json")
    return summary



def _adopt_existing_forward_pass(stage: Path, frames: list[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a V130 forward snapshot from immutable round_00 attempts.

    This migration path lets Stage08B operate directly on V128/V129 outputs
    without rerunning Stage08A.  It deliberately reconstructs the first-pass
    selection from round_00 rather than trusting a main final_view that may
    already have been replaced by an integrated V129 repair.
    """
    ordered_reports: list[Dict[str, Any]] = []
    entries: list[Dict[str, Any]] = []
    for frame in frames:
        camera_id = str(frame["camera_id"])
        directory = stage / "final_views" / camera_id
        report_path = directory / "generation_report.json"
        round_report_path = directory / "round_00" / "round_report.json"
        if not report_path.exists() or not round_report_path.exists():
            raise RuntimeError(
                "Stage08B cannot adopt existing outputs because the immutable forward artifacts are incomplete "
                f"for {camera_id}. Run bash run_stage08a.sh first."
            )
        current = load_json(report_path)
        round_report = load_json(round_report_path)
        accepted_attempt = int(round_report["accepted_attempt_index"])
        source_image = directory / "round_00" / f"attempt_{accepted_attempt:02d}" / "generated.png"
        if not source_image.exists() or source_image.stat().st_size == 0:
            raise RuntimeError(
                f"Stage08B cannot adopt existing forward image for {camera_id}: {source_image}"
            )
        selected_attempt = next(
            (
                dict(item)
                for item in round_report.get("attempts", [])
                if int(item.get("attempt_index", -1)) == accepted_attempt
            ),
            {},
        )
        successful = bool(round_report.get("successful", round_report.get("status") == "strict"))
        acceptance_mode = "strict" if successful else "fallback_single_round"
        forward_report = dict(current)
        forward_report.pop("repair_pass", None)
        forward_report.update(
            {
                "accepted_attempt_index": accepted_attempt,
                "acceptance_mode": acceptance_mode,
                "strict_validation_passed": successful,
                "completed": True,
                "successful": successful,
                "can_be_reference": successful,
                "reference_trust": "strict_success" if successful else "fallback_untrusted",
                "effective_reference_edge_weight": 1.0 if successful else 0.0,
                "accepted_for_stage09": True,
                "accepted_as_future_reference": successful,
                "generated_neighbor_ids_at_selection": list(round_report.get("generated_neighbor_ids", [])),
                "reference_camera_ids": list(selected_attempt.get("used_reference_camera_ids", [])),
                "reference_camera_id": (
                    str(selected_attempt.get("used_reference_camera_ids", [None])[0])
                    if selected_attempt.get("used_reference_camera_ids")
                    else None
                ),
                "fallback_mesh_to_predicted_depth_edge_recall": round_report.get(
                    "fallback_mesh_to_predicted_depth_edge_recall"
                ),
                "final_view": str(directory / "final_view.png"),
            }
        )
        ordered_reports.append(forward_report)

    ordered_reports.sort(key=lambda item: int(item.get("generation_order_index", 10**9)))
    expected_order = list(range(len(ordered_reports)))
    actual_order = [int(item.get("generation_order_index", -1)) for item in ordered_reports]
    if actual_order != expected_order:
        raise RuntimeError(
            f"Stage08B cannot adopt existing outputs with a broken forward generation order: {actual_order}"
        )
    for forward_report in ordered_reports:
        camera_id = str(forward_report["camera_id"])
        directory = stage / "final_views" / camera_id
        snapshot_path = _forward_snapshot_report_path(stage, camera_id)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(forward_report, snapshot_path)
        accepted_attempt = int(forward_report["accepted_attempt_index"])
        source_image = directory / "round_00" / f"attempt_{accepted_attempt:02d}" / "generated.png"
        entries.append(
            {
                "camera_id": camera_id,
                "generation_order_index": int(forward_report["generation_order_index"]),
                "successful": bool(forward_report["successful"]),
                "acceptance_mode": str(forward_report["acceptance_mode"]),
                "accepted_attempt_index": accepted_attempt,
                "forward_generation_report": str(snapshot_path),
                "forward_selected_attempt_image": str(source_image),
            }
        )
    summary = {
        "schema_version": 1,
        "completed": True,
        "adopted_from_pre_v130_stage08": True,
        "camera_count": len(entries),
        "generation_order": [item["camera_id"] for item in entries],
        "strict_success_camera_ids": [item["camera_id"] for item in entries if item["successful"]],
        "fallback_camera_ids": [item["camera_id"] for item in entries if not item["successful"]],
        "state_snapshots": {},
        "entries": entries,
    }
    _save_json_atomic(summary, stage / "forward_pass_summary.json")
    return summary

def _restore_forward_pass_snapshot(stage: Path, frames: list[Mapping[str, Any]]) -> Dict[str, Any]:
    summary_path = stage / "forward_pass_summary.json"
    if not summary_path.exists():
        _adopt_existing_forward_pass(stage, frames)
    summary = load_json(summary_path)
    if not bool(summary.get("completed", False)):
        raise RuntimeError("Stage08B refuses an incomplete Stage08A forward snapshot")
    expected_ids = {str(frame["camera_id"]) for frame in frames}
    entries = {str(item["camera_id"]): dict(item) for item in summary.get("entries", [])}
    if set(entries) != expected_ids:
        raise RuntimeError(
            "Stage08B forward snapshot camera set does not match the current Stage07 manifest"
        )
    state_snapshots = dict(summary.get("state_snapshots", {}))
    if state_snapshots:
        for filename, snapshot_value in state_snapshots.items():
            snapshot = Path(snapshot_value)
            if not snapshot.exists():
                raise FileNotFoundError(f"Stage08B forward state snapshot is missing: {snapshot}")
            shutil.copy2(snapshot, stage / filename)
    else:
        # Older forward-pass outputs may lack a pristine registry/order snapshot.
        # Remove potentially repair-modified state so _restore_stage08_state can
        # rebuild the trusted registry from reconstructed forward reports.
        (stage / "generated_neighbor_registry.json").unlink(missing_ok=True)
        (stage / "weighted_frontier_generation_order.json").unlink(missing_ok=True)
    for camera_id in summary.get("generation_order", []):
        camera_id = str(camera_id)
        entry = entries[camera_id]
        report_path = Path(entry["forward_generation_report"])
        source_image = Path(entry["forward_selected_attempt_image"])
        if not report_path.exists() or not source_image.exists():
            raise FileNotFoundError(
                f"Stage08B cannot restore forward baseline for {camera_id}: {report_path}, {source_image}"
            )
        directory = stage / "final_views" / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        report = load_json(report_path)
        report.pop("repair_pass", None)
        shutil.copy2(source_image, directory / "final_view.png")
        save_json(report, directory / "generation_report.json")
        repair_root = directory / "repair_pass"
        if repair_root.exists():
            shutil.rmtree(repair_root)
    (stage / "repair_pass_summary.json").unlink(missing_ok=True)
    (stage / "stage09_training_manifest.json").unlink(missing_ok=True)
    (stage / "stage_report.json").unlink(missing_ok=True)
    (stage / ".done").unlink(missing_ok=True)
    return summary

def generate_final_room_views(
    out: str | Path,
    refinement_config: Mapping[str, Any],
    scene_json: str | Path,
    prompts_json: str | Path,
    parameters_json: str | Path = "configs/parameters.json",
    *,
    execution_phase: str = "all",
    reset_repair_to_forward_snapshot: bool = False,
) -> Dict[str, Any]:
    out = Path(out)
    stage = out / "08_viewwise_refinement"
    stage.mkdir(parents=True, exist_ok=True)
    dataset_path = out / "07_refinement_cameras" / "reconstruction_dataset_manifest.json"
    dataset = load_json(dataset_path)
    scene = load_json(scene_json)
    prompts_cfg = load_json(prompts_json) if Path(prompts_json).exists() else {}
    config = dict(refinement_config.get("whole_view_generation", {}))
    depth_geometry_contract = str(config.get("depth_geometry_contract", "")).strip()
    if not depth_geometry_contract:
        raise KeyError("whole_view_generation.depth_geometry_contract")
    frames = list(dataset.get("frames", []))
    if not frames:
        raise RuntimeError("Stage08 requires at least one Stage07 reconstruction frame")
    execution_phase = str(execution_phase).strip().lower()
    if execution_phase not in {"all", "forward", "repair"}:
        raise ValueError(f"Unsupported Stage08 execution_phase: {execution_phase}")
    run_forward_phase = execution_phase in {"all", "forward"}
    run_repair_phase = execution_phase in {"all", "repair"}
    if reset_repair_to_forward_snapshot and execution_phase != "repair":
        raise ValueError("reset_repair_to_forward_snapshot is valid only for execution_phase='repair'")
    if reset_repair_to_forward_snapshot:
        _restore_forward_pass_snapshot(stage, frames)
    console_progress_cfg = dict(config.get("console_progress", {}))
    third_party_progress = (
        suppress_third_party_progress_bars()
        if bool(console_progress_cfg.get("suppress_third_party_progress_bars", True))
        else {"status": "not_requested"}
    )
    console_progress = Stage08ConsoleProgress(len(frames), console_progress_cfg)

    selection_report_path = out / "07_refinement_cameras" / "selection_report.json"
    selection_report = load_json(selection_report_path) if selection_report_path.exists() else {}
    scene_scale_m = float(selection_report.get("room_diagonal_m", 0.0) or 0.0) or None

    room_graph_value = str(dataset.get("room_coverage_graph", "")).strip()
    room_graph_path = (
        Path(room_graph_value)
        if room_graph_value
        else out / "07_refinement_cameras" / "room_coverage_graph.json"
    )
    if not room_graph_path.exists():
        room_graph_path = out / "07_refinement_cameras" / "room_coverage_graph.json"
    if not room_graph_path.exists():
        raise FileNotFoundError("Stage08 requires the Stage07 camera-correlation graph")
    room_graph = load_json(room_graph_path)
    graph_runtime = build_stage07_overlap_runtime(frames, room_graph)
    frame_by_id = {str(frame["camera_id"]): frame for frame in frames}
    frame_index_by_id = dict(graph_runtime["frame_index_by_id"])
    adjacency = dict(graph_runtime["adjacency"])
    root_camera_id = str(graph_runtime["initial_bootstrap_camera_id"])
    execution_contract = str(
        config.get(
            "execution_contract",
            "v125_flux2_three_reference_bootstrap_root_success_only_single_round_no_luminance_match",
        )
    )

    registry = initialize_generated_neighbor_registry(frames, graph_runtime)
    registry_path = stage / "generated_neighbor_registry.json"
    generation_order_path = stage / "weighted_frontier_generation_order.json"
    runtime_cfg = dict(config.get("engineering_runtime", {}))
    resume_cfg = dict(runtime_cfg.get("resume", {}))
    resume_state = (
        _restore_stage08_state(
            stage,
            frames,
            registry,
            adjacency,
            root_camera_id,
            registry_path,
            depth_geometry_contract,
            execution_contract,
        )
        if bool(resume_cfg.get("enabled", True))
        else {"committed": [], "previous_events": [], "resumed": False}
    )
    generation_events: list[Dict[str, Any]] = list(resume_state.get("previous_events", []))
    runtime_event_log = Stage08EventLog(
        stage / "stage08_events.jsonl",
        enabled=bool(runtime_cfg.get("event_log_enabled", True)),
    )
    runtime_event_log.write(
        "stage08_start",
        resumed=bool(resume_state.get("resumed", False)),
        committed_view_count=len(resume_state.get("committed", [])),
        depth_geometry_contract=depth_geometry_contract,
        execution_contract=execution_contract,
        memory=runtime_memory_snapshot(),
    )

    def persist_state(last_updates: list[Dict[str, Any]] | None = None) -> None:
        _save_json_atomic(
            {
                "schema_version": 4,
                "depth_geometry_contract": depth_geometry_contract,
                "execution_contract": execution_contract,
                "policy": (
                    "single-round completed-vs-successful lifecycle; successful/trusted views alone "
                    "populate appearance references; completed fallback views retain zero reference weight"
                ),
                "source_stage07_camera_correlation_graph": str(room_graph_path),
                "initial_bootstrap_camera_id": root_camera_id,
                "edge_weight": "correlation_score",
                "slots": registry,
                "generation_events": generation_events,
                "last_updates": list(last_updates or []),
            },
            registry_path,
        )
        _save_json_atomic(
            {
                "schema_version": 4,
                "depth_geometry_contract": depth_geometry_contract,
                "execution_contract": execution_contract,
                "scheduling": (
                    "root is restricted to the two Stage07 bootstrap cameras; prefer the successful/trusted "
                    "frontier, otherwise restart by original completed Stage07 topology without using fallback RGB"
                ),
                "entries": generation_events,
            },
            generation_order_path,
        )

    persist_state()

    backend_name = str(config.get("diffusion_backend", "flux2_klein_4b_multiref_16gb"))
    parameters = load_json(parameters_json)
    backend_cfg = dict(parameters["backend"]["profiles"][backend_name])
    auth_cfg = dict(parameters["backend"].get("authentication", {}))
    model_preparation = ensure_backend_models(backend_name, backend_cfg, auth_cfg)
    worker_settings = FluxWorkerSettings.from_mapping(runtime_cfg.get("flux_worker", {}))
    if not worker_settings.enabled:
        raise RuntimeError(
            "Stage08 requires engineering_runtime.flux_worker.enabled=true so CUDA hangs are isolated "
            "from the Stage08 controller process."
        )
    flux_worker = PersistentFluxWorker(
        backend_name,
        backend_cfg,
        auth_cfg,
        worker_settings,
        event_log=runtime_event_log,
    )

    stage07_rgb_cfg = dict(config.get("stage07_rgb_input", {}))
    if bool(stage07_rgb_cfg.get("rerender_in_stage08", False)):
        raise RuntimeError("Stage08 RGB rerender is forbidden; reuse Stage07 frames[].rgb")
    reference_reprojection_cfg = dict(config.get("reference_reprojection", {}))
    if "minimum_reference_valid_ratio" not in reference_reprojection_cfg:
        raise KeyError("whole_view_generation.reference_reprojection.minimum_reference_valid_ratio")
    multi_reference_cfg = dict(config.get("multi_reference_fusion", {}))
    if not bool(multi_reference_cfg.get("use_all_generated_neighbors", True)):
        raise RuntimeError("Stage08 requires all accepted generated neighbours for multi-reference fusion")

    validation_cfg = dict(config.get("validation", {}))
    monocular_cfg = dict(validation_cfg.get("monocular_depth", {}))
    depth_predictor = (
        DepthAnythingV2Validator(monocular_cfg)
        if bool(monocular_cfg.get("enabled", True))
        else None
    )
    maximum_retries = max(int(validation_cfg.get("maximum_retries", 3)), 0)
    seeds_per_strength = max(int(validation_cfg.get("seeds_per_strength", 3)), 1)
    base_strength = float(config.get("generation_strength", 0.64))
    retry_strength_scale = float(validation_cfg.get("retry_strength_scale", 0.75))
    lock_expansion_scale = float(validation_cfg.get("lock_expansion_erosion_scale", 0.5))

    repair_pass_cfg = dict(config.get("repair_pass", {}))
    repair_pass_enabled = bool(repair_pass_cfg.get("enabled", True)) and run_repair_phase
    repair_pass_summary_path = stage / "repair_pass_summary.json"
    repair_pass_state = (
        load_json(repair_pass_summary_path)
        if repair_pass_summary_path.exists()
        else {"completed": False, "processed_camera_ids": []}
    )
    repair_pass_completed = bool(repair_pass_state.get("completed", False))

    # Only strict/successful views are appearance references.  Fallback outputs
    # are still final Stage09 targets but are deliberately absent from this map.
    accepted_outputs: Dict[str, str] = {}
    final_views_by_id: Dict[str, Dict[str, Any]] = {}
    attempt_history_by_id: Dict[str, list[Dict[str, Any]]] = {str(frame["camera_id"]): [] for frame in frames}
    for _order_index, committed_camera_id, committed_report in resume_state.get("committed", []):
        final_path = Path(committed_report["final_view"])
        if bool(committed_report.get("successful", committed_report.get("strict_validation_passed", False))):
            accepted_outputs[str(committed_camera_id)] = str(final_path)
        final_views_by_id[str(committed_camera_id)] = dict(committed_report)
        attempt_history_by_id[str(committed_camera_id)] = list(committed_report.get("attempt_history", []))
    seed_base = int(config.get("seed", 9200))
    seed_stride = int(config.get("seed_stride", 1009))
    event_index = int(
        resume_state.get(
            "next_event_index",
            max((int(item.get("event_index", -1)) for item in generation_events), default=-1) + 1,
        )
    )
    accepted_order_index = len(final_views_by_id)
    if accepted_order_index:
        runtime_event_log.write(
            "stage08_resume_applied",
            accepted_view_count=accepted_order_index,
            next_event_index=event_index,
            memory=runtime_memory_snapshot(),
        )
    if execution_phase == "repair":
        forward_summary_path = stage / "forward_pass_summary.json"
        if not forward_summary_path.exists() or not bool(load_json(forward_summary_path).get("completed", False)):
            raise RuntimeError(
                "Stage08B requires a complete Stage08A forward pass; run bash run_stage08a.sh first."
            )
        if accepted_order_index != len(frames):
            raise RuntimeError(
                f"Stage08B refuses to generate missing forward views: restored {accepted_order_index}/{len(frames)}. "
                "Run bash run_stage08a.sh first."
            )

    repair_pass_active = False
    repair_pass_order: list[str] = []
    repair_pass_index = 0

    def serializable_reprojection(value: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"warped_rgb", "valid_mask", "confidence_map"}
        }

    while True:
        in_repair_pass = False
        if len(final_views_by_id) < len(frames):
            if not run_forward_phase:
                raise RuntimeError(
                    "Stage08B is repair-only and will not generate missing Stage08A forward views"
                )
            if accepted_order_index == 0:
                camera_id = root_camera_id
                if bool(registry[camera_id].get("completed", False)) or str(registry[camera_id].get("status")) != "pending":
                    raise RuntimeError("Stage08 bootstrap root state is invalid")
                selection_reason = "bootstrap_root_restricted_to_stage07_room_coverage_pair"
            else:
                camera_id = select_next_frontier_camera(registry, frame_index_by_id)
                if camera_id is not None:
                    selection_reason = "maximum_successful_trusted_multi_neighbour_noisy_or_frontier_support"
                else:
                    camera_id = select_restart_camera_from_completed_topology(
                        registry, frame_index_by_id, adjacency
                    )
                    if camera_id is None:
                        unfinished = {
                            key: {
                                "status": slot.get("status"),
                                "completed": slot.get("completed"),
                                "successful": slot.get("successful"),
                                "can_be_reference": slot.get("can_be_reference"),
                                "successful_reference_neighbors": slot.get("generated_neighbors", []),
                            }
                            for key, slot in registry.items()
                            if not bool(slot.get("completed", False))
                        }
                        persist_state()
                        raise RuntimeError(
                            "Stage08 successful frontier is empty and no unfinished camera is reachable "
                            f"from completed Stage07 topology. Unfinished state: {unfinished}"
                        )
                    selection_reason = "completed_stage07_topology_restart_without_appearance_reference"
        elif repair_pass_enabled and not repair_pass_completed:
            if not repair_pass_active:
                if not (stage / "forward_pass_summary.json").exists():
                    _write_forward_pass_snapshot(stage, final_views_by_id)
                ordered_reports = sorted(
                    final_views_by_id.values(),
                    key=lambda item: int(item.get("generation_order_index", 10**9)),
                )
                repair_pass_order = [str(item["camera_id"]) for item in ordered_reports]
                processed_ids = {str(item) for item in repair_pass_state.get("processed_camera_ids", [])}
                if processed_ids:
                    repair_pass_order = [camera for camera in repair_pass_order if camera not in processed_ids]
                repair_pass_active = True
                repair_pass_index = 0
                runtime_event_log.write(
                    "repair_pass_started",
                    camera_count=len(repair_pass_order),
                    generation_order=repair_pass_order,
                    memory=runtime_memory_snapshot(),
                )
            if repair_pass_index >= len(repair_pass_order):
                repair_pass_completed = True
                _save_json_atomic(
                    {
                        "completed": True,
                        "processed_camera_ids": list(repair_pass_state.get("processed_camera_ids", [])),
                        "camera_count": len(repair_pass_order),
                    },
                    repair_pass_summary_path,
                )
                runtime_event_log.write(
                    "repair_pass_completed",
                    processed_count=len(repair_pass_state.get("processed_camera_ids", [])),
                    memory=runtime_memory_snapshot(),
                )
                break
            camera_id = str(repair_pass_order[repair_pass_index])
            in_repair_pass = True
            selection_reason = "post_generation_repair_pass_original_generation_order"
        else:
            break

        frame = frame_by_id[camera_id]
        slot = registry[camera_id]
        if in_repair_pass:
            generated_neighbors = [
                {
                    "camera_id": str(edge["camera_id"]),
                    "correlation_score": float(edge["correlation_score"]),
                    "area_weighted_iou": float(edge.get("area_weighted_iou", 0.0)),
                    "area_weighted_dice": float(edge.get("area_weighted_dice", 0.0)),
                    "view_direction_cosine": float(edge.get("view_direction_cosine", 0.0)),
                    "view_angle_degrees": float(edge.get("view_angle_degrees", 0.0)),
                    "reference_trust": "strict_success",
                    "effective_reference_edge_weight": float(edge["correlation_score"]),
                }
                for edge in adjacency.get(camera_id, [])
                if str(edge.get("camera_id")) in accepted_outputs
            ]
            generated_neighbors.sort(
                key=lambda item: (-float(item["correlation_score"]), str(item["camera_id"]))
            )
        else:
            generated_neighbors = list(slot.get("generated_neighbors", []))
            if accepted_order_index > 0 and not generated_neighbors and selection_reason != "completed_stage07_topology_restart_without_appearance_reference":
                raise RuntimeError(f"Successful-frontier Stage08 camera has no trusted graph neighbour: {camera_id}")
            for item in generated_neighbors:
                neighbor_id = str(item["camera_id"])
                if neighbor_id not in accepted_outputs:
                    raise RuntimeError(
                        f"Stage08 registry references a non-accepted output: {camera_id} -> {neighbor_id}"
                    )

        slot["status"] = "generating"
        if not in_repair_pass:
            slot["generation_order_index"] = accepted_order_index
            if int(slot.get("attempt_round_count", 0)) != 0:
                raise RuntimeError(f"Stage08 V124 schedules each camera exactly once: {camera_id}")
            slot["attempt_round_count"] = 1
        round_index = 0
        if not in_repair_pass:
            slot["last_attempt_generated_neighbor_signature"] = sorted(
                str(item["camera_id"]) for item in generated_neighbors
            )
        propagation_support = (
            float(1.0 - np.prod([1.0 - min(max(float(item.get("correlation_score", 0.0)), 0.0), 1.0) for item in generated_neighbors], dtype=np.float64))
            if generated_neighbors
            else 0.0
        )
        display_view_number = repair_pass_index + 1 if in_repair_pass else accepted_order_index + 1
        generation_event: Dict[str, Any] = {
            "event_index": event_index,
            "pass_name": "repair" if in_repair_pass else "forward",
            "pass_index": repair_pass_index if in_repair_pass else 0,
            "accepted_order_index_if_successful": accepted_order_index,
            "camera_id": camera_id,
            "round_index": round_index,
            "selection_reason": selection_reason,
            "generated_neighbor_ids": [str(item["camera_id"]) for item in generated_neighbors],
            "generated_neighbor_edge_scores": {
                str(item["camera_id"]): float(item["correlation_score"])
                for item in generated_neighbors
            },
            "propagation_support_score": propagation_support,
            "status": "generating",
        }
        generation_events.append(generation_event)
        persist_state()
        console_progress.begin_view(
            camera_id=camera_id,
            view_number=display_view_number,
            round_index=round_index,
            reference_count=len(generated_neighbors),
            propagation_support=propagation_support,
            accepted_views=len(final_views_by_id),
            deferred_views=0,
        )

        directory = stage / "final_views" / camera_id
        if in_repair_pass:
            repair_root = directory / "repair_pass"
            round_directory = repair_root / f"round_{repair_pass_index:02d}"
        else:
            (directory / "selected.txt").unlink(missing_ok=True)
            round_directory = directory / f"round_{round_index:02d}"
        round_directory.mkdir(parents=True, exist_ok=True)
        source_rgb = Path(frame["rgb"])
        if not source_rgb.exists() or source_rgb.stat().st_size == 0:
            raise FileNotFoundError(
                f"Stage08 must reuse the Stage07 selected-view RGB for {camera_id}: {source_rgb}"
            )
        selected_marker = Path(
            frame.get("selected_marker") or Path(frame["triangle_id"]).with_name("selected.txt")
        )
        if not selected_marker.exists():
            raise RuntimeError(
                f"Stage08 refuses incomplete Stage07 view {camera_id}; selected.txt is missing: {selected_marker}"
            )
        prompt_meta = _compile_prompt(scene, frame, config)

        copied = {
            "source_rgb": _copy_required(source_rgb, directory / "source_rgb.png"),
            "depth": _copy_required(frame["depth"], directory / "depth.png"),
            "normal_world": _copy_required(frame["normal_world"], directory / "normal_world.png"),
            "semantic": _copy_required(frame["semantic"], directory / "semantic.png"),
            "palette": _copy_required(frame["palette"], directory / "semantic.palette.json"),
            "triangle_id": _copy_required(frame["triangle_id"], directory / "triangle_id.png"),
            "camera": _copy_required(frame["camera"], directory / "camera.json"),
            "stage07_selected_marker": str(selected_marker),
        }

        round_attempts: list[Dict[str, Any]] = []
        accepted_attempt: int | None = None
        selected_output: Path | None = None
        accepted_reference_ids: list[str] = []
        attempt_schedule = _stage08_generation_attempt_schedule(
            base_strength=base_strength,
            retry_strength_scale=retry_strength_scale,
            maximum_retries=maximum_retries,
            seeds_per_strength=seeds_per_strength,
            seed_base=seed_base,
            event_index=event_index,
            seed_stride=seed_stride,
        )
        total_attempts = len(attempt_schedule)
        conditioning_cache: Dict[str, Dict[str, Any]] = {}
        round_memory_cfg = dict(runtime_cfg.get("reference_memory", {}))
        spill_overlap_to_disk = bool(round_memory_cfg.get("spill_overlap_diagnostics_to_disk", True))
        for attempt_spec in attempt_schedule:
            attempt_index = int(attempt_spec["attempt_index"])
            strength_level_index = int(attempt_spec["strength_level_index"])
            seed_attempt_index = int(attempt_spec["seed_attempt_index"])
            strength = float(attempt_spec["strength"])
            generation_seed = int(attempt_spec["seed"])
            attempt_directory = round_directory / f"attempt_{attempt_index:02d}"
            attempt_directory.mkdir(parents=True, exist_ok=True)
            existing_validation_path = attempt_directory / "validation.json"
            existing_generated_path = attempt_directory / "generated.png"
            if existing_validation_path.exists() and existing_generated_path.exists() and existing_generated_path.stat().st_size > 0:
                existing_report = load_json(existing_validation_path)
                if (
                    str(existing_report.get("depth_geometry_contract", "")) == depth_geometry_contract
                    and int(existing_report.get("seed", generation_seed)) == generation_seed
                    and abs(float(existing_report.get("strength", strength)) - strength) <= 1e-9
                ):
                    round_attempts.append(existing_report)
                    runtime_event_log.write(
                        "generation_attempt_reused_from_resume",
                        camera_id=camera_id,
                        round_index=round_index,
                        attempt_index=attempt_index,
                        accepted=bool(existing_report.get("accepted", False)),
                    )
                    if bool(existing_report.get("accepted", False)):
                        accepted_attempt = attempt_index
                        selected_output = existing_generated_path
                        accepted_reference_ids = list(existing_report.get("used_reference_camera_ids", []))
                        break
                    continue

            attempt_config = copy.deepcopy(config)
            # Keep every seed within one strength level identical except for the seed.
            # The pre-existing lock expansion is tied to the conservative strength
            # level, not to the flat generation-attempt index.
            if strength_level_index >= 2:
                reprojection_cfg = dict(attempt_config.get("reference_reprojection", {}))
                reprojection_cfg["boundary_erosion_ratio"] = (
                    float(reprojection_cfg.get("boundary_erosion_ratio", 0.003))
                    * lock_expansion_scale
                )
                attempt_config["reference_reprojection"] = reprojection_cfg
            minimum_reference_valid_ratio = float(
                dict(attempt_config.get("reference_reprojection", {}))[
                    "minimum_reference_valid_ratio"
                ]
            )
            conditioning_key = _reference_conditioning_key(attempt_config)
            cached_conditioning = conditioning_cache.get(conditioning_key)
            is_root = (not in_repair_pass) and camera_id == root_camera_id and accepted_order_index == 0

            if cached_conditioning is None:
                reference_trials: list[Dict[str, Any]] = []
                valid_reference_records: list[Dict[str, Any]] = []
                references_root = attempt_directory / "references"
                references_root.mkdir(parents=True, exist_ok=True)
                runtime_event_log.write(
                    "reference_conditioning_build_started",
                    camera_id=camera_id,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    conditioning_key=conditioning_key,
                    generated_neighbor_count=len(generated_neighbors),
                    memory=runtime_memory_snapshot(),
                )
                for neighbor in generated_neighbors:
                    neighbor_id = str(neighbor["camera_id"])
                    candidate = reproject_reference_to_current(
                        accepted_outputs[neighbor_id],
                        frame_by_id[neighbor_id],
                        frame,
                        attempt_config,
                        scene_scale_m=scene_scale_m,
                    )
                    trial = {
                        "camera_id": neighbor_id,
                        "stage07_edge_correlation_score": float(neighbor["correlation_score"]),
                        "valid_pixel_count": int(candidate["valid_pixel_count"]),
                        "valid_ratio": float(candidate["valid_ratio"]),
                        "minimum_valid_ratio": minimum_reference_valid_ratio,
                        "accepted_for_fusion": bool(
                            candidate["valid_ratio"] >= minimum_reference_valid_ratio
                        ),
                    }
                    reference_trials.append(trial)
                    ref_dir = references_root / neighbor_id
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(candidate["warped_rgb"], mode="RGB").save(
                        ref_dir / "warped_rgb.png"
                    )
                    Image.fromarray(
                        np.asarray(candidate["valid_mask"], dtype=np.uint8) * 255,
                        mode="L",
                    ).save(ref_dir / "valid_mask.png")
                    Image.fromarray(
                        np.clip(np.asarray(candidate["confidence_map"]) * 255.0, 0, 255).astype(np.uint8),
                        mode="L",
                    ).save(ref_dir / "geometric_confidence.png")
                    save_json(
                        {
                            **trial,
                            "selected_for_fusion": bool(trial["accepted_for_fusion"]),
                            "participates_in_rgb_fusion": bool(trial["accepted_for_fusion"]),
                            "selection_reason": (
                                "sufficient_local_reprojection_support"
                                if trial["accepted_for_fusion"]
                                else "insufficient_local_reprojection_support"
                            ),
                        },
                        ref_dir / "reference_status.json",
                    )
                    if trial["accepted_for_fusion"]:
                        write_empty_marker_atomic(ref_dir / "selected.txt")
                        valid_reference_records.append({
                            "camera_id": neighbor_id,
                            "correlation_score": float(neighbor["correlation_score"]),
                            "reprojection": candidate,
                        })
                    else:
                        # The candidate's large arrays have no downstream use.
                        del candidate

                # Reference support is a write/fusion gate, never a node-generation gate.
                fusion = fuse_reprojected_references(
                    source_rgb,
                    valid_reference_records,
                    attempt_config,
                )
                condition_path = attempt_directory / "condition_rgb.png"
                mask_path = attempt_directory / "generation_mask.png"
                fusion["condition_image"].save(condition_path)
                fusion["generation_mask"].save(mask_path)
                fused_reference_path = attempt_directory / "fused_reference_rgb.png"
                Image.fromarray(fusion["fused_reference_rgb"], mode="RGB").save(fused_reference_path)
                for reference_record in fusion.get("reference_records", []):
                    reference_directory = references_root / str(reference_record["camera_id"])
                    raw_rgb = np.clip(
                        np.asarray(reference_record.get("warped_rgb_raw", reference_record["warped_rgb"])) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                    matched_rgb = np.clip(
                        np.asarray(reference_record["warped_rgb"]) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                    Image.fromarray(raw_rgb, mode="RGB").save(reference_directory / "warped_rgb_raw.png")
                    Image.fromarray(matched_rgb, mode="RGB").save(
                        reference_directory / "warped_rgb_luminance_matched.png"
                    )
                    save_json(
                        dict(reference_record.get("luminance_matching", {})),
                        reference_directory / "luminance_matching.json",
                    )
                reference_reliability_path = attempt_directory / "reference_reliability.png"
                reference_lock_path = attempt_directory / "reference_lock_mask.png"
                Image.fromarray(
                    np.clip(fusion["reference_reliability"] * 255.0, 0, 255).astype(np.uint8),
                    mode="L",
                ).save(reference_reliability_path)
                Image.fromarray(
                    np.asarray(fusion["lock_mask"], dtype=np.uint8) * 255,
                    mode="L",
                ).save(reference_lock_path)
                serializable_reference_records = list(fusion.get("serializable_reference_records", []))
                reference_fusion_path = attempt_directory / "reference_fusion.json"
                save_json(
                    {
                        "is_single_global_root_bootstrap": is_root,
                        "all_generated_neighbor_ids": [str(item["camera_id"]) for item in generated_neighbors],
                        "reference_trials": reference_trials,
                        "selected_reference_camera_ids": [
                            str(item["camera_id"]) for item in valid_reference_records
                        ],
                        "fused_reference_records": serializable_reference_records,
                        "reference_conditioning_mode": (
                            "multi_reference" if valid_reference_records else "stage07_rgb_only"
                        ),
                        "insufficient_reference_support_never_defers": True,
                        "fusion_metadata": fusion["metadata"],
                        "conditioning_key": conditioning_key,
                    },
                    reference_fusion_path,
                )
                overlap_cache_root = attempt_directory / ".runtime_overlap_cache"
                if spill_overlap_to_disk:
                    overlap_records = _spill_fusion_reference_records(fusion, overlap_cache_root)
                else:
                    overlap_records = list(fusion.get("reference_records", []))
                cached_conditioning = {
                    "owner_attempt_directory": str(attempt_directory),
                    "conditioning_key": conditioning_key,
                    "condition_path": str(condition_path),
                    "mask_path": str(mask_path),
                    "fused_reference_path": str(fused_reference_path),
                    "reference_reliability_path": str(reference_reliability_path),
                    "reference_lock_path": str(reference_lock_path),
                    "reference_fusion_path": str(reference_fusion_path),
                    "references_root": str(references_root),
                    "reference_trials": copy.deepcopy(reference_trials),
                    "selected_reference_camera_ids": [
                        str(item["camera_id"]) for item in valid_reference_records
                    ],
                    "serializable_reference_records": copy.deepcopy(serializable_reference_records),
                    "fusion_metadata": copy.deepcopy(fusion["metadata"]),
                    "reference_available": bool(fusion["reference_available"]),
                    "overlap_records": overlap_records,
                }
                conditioning_cache[conditioning_key] = cached_conditioning
                console_progress.set_reference_selection(
                    len(cached_conditioning["selected_reference_camera_ids"])
                )
                runtime_event_log.write(
                    "reference_conditioning_build_completed",
                    camera_id=camera_id,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    conditioning_key=conditioning_key,
                    selected_reference_count=len(cached_conditioning["selected_reference_camera_ids"]),
                    memory_before_release=runtime_memory_snapshot(),
                )
                # All large reprojected/fusion arrays are dead before FLUX starts.
                valid_reference_records.clear()
                del valid_reference_records
                # Break loop-variable references to the last large arrays before FLUX.
                candidate = None
                reference_record = None
                raw_rgb = None
                matched_rgb = None
                del fusion
                gc.collect()
                runtime_event_log.write(
                    "reference_working_memory_released",
                    camera_id=camera_id,
                    conditioning_key=conditioning_key,
                    memory=runtime_memory_snapshot(),
                )
            else:
                reference_trials = copy.deepcopy(cached_conditioning["reference_trials"])
                console_progress.set_reference_selection(
                    len(cached_conditioning["selected_reference_camera_ids"])
                )
                # Identical conditioning is reused for different seeds instead of
                # recomputing all reference reprojections and Huber fusion.
                condition_path = Path(_hardlink_or_copy(
                    cached_conditioning["condition_path"],
                    attempt_directory / "condition_rgb.png",
                ))
                mask_path = Path(_hardlink_or_copy(
                    cached_conditioning["mask_path"],
                    attempt_directory / "generation_mask.png",
                ))
                _hardlink_or_copy(
                    cached_conditioning["fused_reference_path"],
                    attempt_directory / "fused_reference_rgb.png",
                )
                _hardlink_or_copy(
                    cached_conditioning["reference_reliability_path"],
                    attempt_directory / "reference_reliability.png",
                )
                _hardlink_or_copy(
                    cached_conditioning["reference_lock_path"],
                    attempt_directory / "reference_lock_mask.png",
                )
                _hardlink_or_copy(
                    cached_conditioning["reference_fusion_path"],
                    attempt_directory / "reference_fusion.json",
                )
                _hardlink_tree(
                    cached_conditioning["references_root"],
                    attempt_directory / "references",
                )
                save_json(
                    {
                        "conditioning_key": conditioning_key,
                        "reused_from_attempt_directory": cached_conditioning["owner_attempt_directory"],
                        "reason": "identical_reference_and_fusion_configuration_for_this_seed",
                    },
                    attempt_directory / "conditioning_reuse.json",
                )
                runtime_event_log.write(
                    "reference_conditioning_reused",
                    camera_id=camera_id,
                    round_index=round_index,
                    attempt_index=attempt_index,
                    conditioning_key=conditioning_key,
                    source_attempt_directory=cached_conditioning["owner_attempt_directory"],
                    memory=runtime_memory_snapshot(),
                )

            condition_path = Path(attempt_directory / "condition_rgb.png")
            mask_path = Path(attempt_directory / "generation_mask.png")
            selected_reference_ids_for_attempt = list(cached_conditioning["selected_reference_camera_ids"])
            # Reconstruct only the small validation-facing pieces after FLUX.
            light_fusion = {
                "reference_available": bool(cached_conditioning["reference_available"]),
                "reference_records": list(cached_conditioning["overlap_records"]),
                "metadata": copy.deepcopy(cached_conditioning["fusion_metadata"]),
                "serializable_reference_records": copy.deepcopy(
                    cached_conditioning["serializable_reference_records"]
                ),
            }

            attempt_output = attempt_directory / "generated.png"
            attempt_output.unlink(missing_ok=True)
            inference_steps = int(config.get("num_inference_steps", 32))
            console_progress.begin_attempt(
                attempt_index,
                total_attempts,
                inference_steps,
                strength=strength,
                strength_level_index=strength_level_index,
                total_strength_levels=int(attempt_spec["total_strength_levels"]),
                seed_index=seed_attempt_index,
                seeds_per_strength=int(attempt_spec["seeds_per_strength"]),
            )
            common_request = {
                "prompt": prompt_meta["prompt"],
                "output_path": str(attempt_output),
                "object_name": "whole_room_view",
                "semantic_class": "whole_scene",
                "region_policy": {
                    "masked_object_only": False,
                    "preserve_geometry": True,
                    "continuous_surface": False,
                },
                "seed": generation_seed,
                # Kept in the request/report so the legacy FLUX.1 compatibility
                # schedule remains reproducible. FLUX.2 Klein does not expose a
                # denoising-strength parameter and records this as non-native.
                "strength": strength,
                "guidance_scale": float(config.get("guidance_scale", 1.0)),
                "num_inference_steps": inference_steps,
                "suppress_backend_progress_bars": bool(
                    console_progress_cfg.get("suppress_backend_progress_bars", True)
                ),
                "quiet_console": bool(console_progress_cfg.get("quiet_backend_console", True)),
                "width": int(load_json(frame["camera"])["width"]),
                "height": int(load_json(frame["camera"])["height"]),
            }

            # Native FLUX.2 references follow a strict two-reference contract:
            #   Image 1: target-view RGB condition (Stage07 RGB plus trusted manual warp).
            #   Image 2: pixel-aligned Stage07 camera-Z depth reference.
            # Cross-view appearance enters Image 1 only through geometry-aligned manual reprojection.

            if backend_name == "flux2_klein_4b_multiref_16gb":
                # Convert the Stage07 16-bit normalized camera-Z raster to an 8-bit RGB
                # visualization without clipping its gradients.  It remains pixel-aligned
                # with Image 1 and is supplied as native FLUX.2 Image 2.
                width = int(common_request["width"])
                height = int(common_request["height"])
                depth_reference_path = attempt_directory / "depth_reference.png"
                depth_reference, depth_reference_meta = load_depth_control_image(
                    frame["depth"], (width, height)
                )
                if depth_reference.size != (width, height):
                    raise RuntimeError(
                        f"Stage08 depth reference size {depth_reference.size} does not match target raster {(width, height)}"
                    )
                depth_reference.save(depth_reference_path)
                save_json(depth_reference_meta, attempt_directory / "depth_reference.json")

                native_reference_paths = [str(condition_path), str(depth_reference_path)]
                native_reference_roles = [
                    "target_geometry_layout_condition",
                    "aligned_camera_z_depth_geometry",
                ]
                request = {
                    **common_request,
                    "reference_image_paths": native_reference_paths,
                    "reference_roles": native_reference_roles,
                    "max_sequence_length": int(backend_cfg.get("max_sequence_length", 512)),
                }
            else:
                request = {
                    **common_request,
                    # FLUX.1 compatibility only.  The production FLUX.2 Klein
                    # request intentionally contains no negative-prompt field.
                    "negative_prompt": " ".join(filter(None, [
                        str(config.get("negative_prompt", "")),
                        str(prompts_cfg.get("global_negative", "")),
                    ])),
                    "init_image_path": str(condition_path),
                    "generation_mask_path": str(mask_path),
                    "depth_image_path": str(frame["depth"]),
                    "control_preview_path": str(attempt_directory / "depth_control_preview.png"),
                }

            def _runtime_recovery_notice(event: Mapping[str, Any]) -> None:
                console_progress.record_runtime_recovery(
                    camera_id=camera_id,
                    kind=str(event.get("kind", "runtime_failure")),
                    recovery_number=int(event.get("recovery_number", 1)),
                    maximum_recoveries=int(event.get("maximum_recoveries", worker_settings.runtime_retries)),
                )

            generation = flux_worker.generate_with_recovery(
                request,
                progress_callback=console_progress.diffusion_step,
                runtime_callback=console_progress.runtime_heartbeat,
                recovery_callback=_runtime_recovery_notice,
            )
            console_progress.finish_diffusion()
            if console_progress.is_tty:
                console_progress.set_phase("depth validation", force=True)
            if not attempt_output.exists() and generation.get("output_path"):
                attempt_output = Path(generation["output_path"])
            if not attempt_output.exists() or attempt_output.stat().st_size == 0:
                raise RuntimeError(
                    f"Stage08 generation produced no image for {camera_id} attempt {attempt_index}"
                )

            if depth_predictor is not None:
                depth_result = validate_depth_structure(
                    attempt_output,
                    frame,
                    depth_predictor,
                    dict(validation_cfg.get("depth_structure", {})),
                )
            else:
                depth_result = {
                    "accepted": True,
                    "depth_edge_recall": 1.0,
                    "minimum_depth_edge_recall": 0.0,
                    "depth_edge_recall_accepted": True,
                    "predicted_depth_edge_precision": 1.0,
                    "minimum_predicted_depth_edge_precision": None,
                    "predicted_depth_edge_precision_accepted": None,
                    "predicted_depth_edge_precision_diagnostic_only": True,
                    "extra_predicted_depth_edge_fraction": 0.0,
                    "mesh_edge": np.zeros((1, 1), dtype=bool),
                    "predicted_edge": np.zeros((1, 1), dtype=bool),
                    "predicted_extra_edge": np.zeros((1, 1), dtype=bool),
                    "predicted_depth": np.zeros((1, 1), dtype=np.float32),
                    "mesh_edge_metadata": {},
                    "predicted_edge_metadata": {},
                    "comparison": "disabled",
                }

            overlap_result = multi_reference_overlap_error(
                attempt_output,
                light_fusion,
                dict(validation_cfg.get("reference_overlap", {})),
            )
            if light_fusion["reference_available"]:
                generated_array = (
                    np.asarray(Image.open(attempt_output).convert("RGB"), dtype=np.float32) / 255.0
                )
                fused_array = (
                    np.asarray(
                        Image.open(cached_conditioning["fused_reference_path"]).convert("RGB"),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                difference = np.mean(np.abs(generated_array - fused_array), axis=2)
                overlap_mask = (
                    np.asarray(
                        Image.open(cached_conditioning["reference_lock_path"]).convert("L"),
                        dtype=np.uint8,
                    )
                    > 0
                )
                difference[~overlap_mask] = 0.0
            else:
                difference = None
                overlap_mask = None
            validity_result = image_validity(
                attempt_output,
                dict(validation_cfg.get("image_validity", {})),
            )
            # RGB disagreement is diagnostic-only. Geometry preservation and basic
            # image validity are the only Stage08 acceptance gates.
            accepted = bool(
                depth_result["accepted"]
                and validity_result["accepted"]
            )
            console_progress.record_validation(
                camera_id=camera_id,
                depth_result=depth_result,
                overlap_result=overlap_result,
                validity_result=validity_result,
                accepted=accepted,
            )
            diagnostic_paths = save_validation_images(
                attempt_directory,
                depth_result,
                overlap_mask,
                difference,
            )
            attempt_report = {
                "depth_geometry_contract": depth_geometry_contract,
                "attempt_index": attempt_index,
                "strength_level_index": strength_level_index,
                "strength_level_number": int(attempt_spec["strength_level_number"]),
                "total_strength_levels": int(attempt_spec["total_strength_levels"]),
                "seed_attempt_index": seed_attempt_index,
                "seed_attempt_number": int(attempt_spec["seed_attempt_number"]),
                "seeds_per_strength": int(attempt_spec["seeds_per_strength"]),
                "seed": generation_seed,
                "single_global_root_bootstrap": is_root,
                "all_generated_neighbor_ids": [str(item["camera_id"]) for item in generated_neighbors],
                "reference_trials": reference_trials,
                "used_reference_camera_ids": selected_reference_ids_for_attempt,
                "reference_conditioning_mode": (
                    "multi_reference" if selected_reference_ids_for_attempt else "stage07_rgb_only"
                ),
                "insufficient_reference_support_never_defers": True,
                "strength": strength,
                "fusion": light_fusion["metadata"],
                "fused_reference_records": light_fusion.get("serializable_reference_records", []),
                "conditioning_key": conditioning_key,
                "conditioning_reused": bool(
                    Path(cached_conditioning["owner_attempt_directory"]) != attempt_directory
                ),
                "generation": generation,
                "native_generator_reference_policy": (
                    "target_condition_plus_aligned_depth_only"
                    if backend_name == "flux2_klein_4b_multiref_16gb"
                    else "flux1_compatibility_condition_mask_plus_depth_control"
                ),
                "generated_image": str(attempt_output),
                "depth_validation": {
                    key: value
                    for key, value in depth_result.items()
                    if key not in {"mesh_edge", "predicted_edge", "predicted_extra_edge", "predicted_depth"}
                },
                "multi_reference_overlap_validation": overlap_result,
                "image_validity": validity_result,
                "diagnostics": diagnostic_paths,
                "accepted": accepted,
            }
            save_json(attempt_report, attempt_directory / "validation.json")
            round_attempts.append(attempt_report)
            runtime_event_log.write(
                "generation_attempt_validated",
                camera_id=camera_id,
                round_index=round_index,
                attempt_index=attempt_index,
                accepted=accepted,
                memory=runtime_memory_snapshot(),
            )
            # Validation images/reports are persisted; release per-attempt dense arrays
            # before the next seed/strength prepares its inputs.
            depth_result = None
            difference = None
            overlap_mask = None
            if "generated_array" in locals():
                generated_array = None
            if "fused_array" in locals():
                fused_array = None
            gc.collect()
            if accepted:
                accepted_attempt = attempt_index
                selected_output = attempt_output
                accepted_reference_ids = list(selected_reference_ids_for_attempt)
                break

        # Diagnostic spill files are only needed while attempts for this round are
        # being validated. Remove them before moving to the next graph node.
        for cached in conditioning_cache.values():
            for record in cached.get("overlap_records", []):
                overlap_cache_path = str(record.get("overlap_cache_path", ""))
                if overlap_cache_path:
                    Path(overlap_cache_path).unlink(missing_ok=True)
            overlap_root = Path(cached["owner_attempt_directory"]) / ".runtime_overlap_cache"
            try:
                overlap_root.rmdir()
            except OSError:
                pass
        gc.collect()

        attempt_history_by_id[camera_id].append({
            "round_index": round_index,
            "generated_neighbor_ids": [str(item["camera_id"]) for item in generated_neighbors],
            "attempts": round_attempts,
            "accepted_attempt_index": accepted_attempt,
        })

        strict_success = selected_output is not None
        fallback_source_validation = None
        fallback_recall = None
        if selected_output is None:
            # Each camera has one generation round. Publish the best usable result
            # as an untrusted fallback instead of deferring or requeueing the camera.
            usable_fallbacks: list[Dict[str, Any]] = []
            for attempt in round_attempts:
                if not bool(dict(attempt.get("image_validity", {})).get("accepted", False)):
                    continue
                try:
                    recall = float(dict(attempt.get("depth_validation", {}))["depth_edge_recall"])
                except (KeyError, TypeError, ValueError):
                    continue
                generated_path = Path(str(attempt.get("generated_image", "")))
                if not np.isfinite(recall) or not generated_path.exists() or generated_path.stat().st_size == 0:
                    continue
                usable_fallbacks.append({
                    "attempt": attempt,
                    "recall": recall,
                    "attempt_index": int(attempt.get("attempt_index", 0)),
                    "generated_path": generated_path,
                })
            if not usable_fallbacks:
                persist_state()
                raise RuntimeError(
                    f"Stage08 camera {camera_id} exhausted its single round without any usable generated image"
                )
            fallback = max(
                usable_fallbacks,
                key=lambda item: (float(item["recall"]), -int(item["attempt_index"])),
            )
            selected_output = Path(fallback["generated_path"])
            accepted_attempt = int(fallback["attempt_index"])
            selected_attempt_report = dict(fallback["attempt"])
            accepted_reference_ids = list(selected_attempt_report.get("used_reference_camera_ids", []))
            fallback_recall = float(fallback["recall"])
            fallback_source_validation = str(
                selected_output.with_name("validation.json")
            )

        main_final_output = directory / "final_view.png"
        final_output = main_final_output if not in_repair_pass else round_directory / "final_view_candidate.png"
        shutil.copy2(selected_output, final_output)
        slot["status"] = "completed"
        slot["completed"] = True
        slot["successful"] = bool(strict_success)
        slot["can_be_reference"] = bool(strict_success)
        slot["reference_trust"] = "strict_success" if strict_success else "fallback_untrusted"
        slot["effective_reference_edge_weight"] = 1.0 if strict_success else 0.0
        if strict_success and not in_repair_pass:
            accepted_outputs[camera_id] = str(final_output)
            neighbor_updates = update_generated_neighbor_registry(registry, camera_id, adjacency)
        else:
            neighbor_updates = []

        acceptance_mode = "strict" if strict_success else "fallback_single_round"
        generation_event["status"] = acceptance_mode
        generation_event["completed"] = True
        generation_event["successful"] = bool(strict_success)
        generation_event["can_be_reference"] = bool(strict_success)
        generation_event["effective_reference_edge_weight"] = 1.0 if strict_success else 0.0
        generation_event["accepted_attempt_index"] = accepted_attempt
        generation_event["used_reference_camera_ids"] = accepted_reference_ids
        generation_event["neighbor_list_updates"] = neighbor_updates
        if fallback_recall is not None:
            generation_event["fallback_mesh_to_predicted_depth_edge_recall"] = fallback_recall

        save_json(
            {
                "execution_contract": execution_contract,
                "camera_id": camera_id,
                "status": acceptance_mode,
                "completed": True,
                "successful": bool(strict_success),
                "can_be_reference": bool(strict_success),
                "effective_reference_edge_weight": 1.0 if strict_success else 0.0,
                "round_index": 0,
                "generated_neighbor_ids": [str(item["camera_id"]) for item in generated_neighbors],
                "attempts": round_attempts,
                "accepted_attempt_index": accepted_attempt,
                "accepted_for_stage09": True,
                "accepted_as_future_reference": bool(strict_success),
                "fallback_mesh_to_predicted_depth_edge_recall": fallback_recall,
            },
            round_directory / "round_report.json",
        )

        stage08_selected_marker = directory / "selected.txt"
        frame_report = {
            "execution_contract": execution_contract,
            "depth_geometry_contract": depth_geometry_contract,
            "camera_id": camera_id,
            "generation_order_index": accepted_order_index if not in_repair_pass else int(final_views_by_id[camera_id].get("generation_order_index", repair_pass_index)),
            "generation_event_index": event_index,
            "camera_role": frame.get("camera_role"),
            "target_owner_id": frame.get("target_owner_id"),
            "look_at_target_id": frame.get("look_at_target_id"),
            "look_at_target_type": frame.get("look_at_target_type"),
            "position_view_pair_id": frame.get("position_view_pair_id"),
            "single_global_root_bootstrap": (not in_repair_pass) and camera_id == root_camera_id,
            "root_restricted_to_stage07_bootstrap_pair": True,
            "generated_neighbor_ids_at_selection": [
                str(item["camera_id"]) for item in generated_neighbors
            ],
            "propagation_support_score": propagation_support,
            "reference_camera_ids": accepted_reference_ids,
            "reference_camera_id": accepted_reference_ids[0] if accepted_reference_ids else None,
            "reference_conditioning_mode": (
                "successful_trusted_manual_warp" if accepted_reference_ids else "stage07_rgb_only"
            ),
            "reference_sources_successful_only": True,
            "insufficient_reference_support_never_defers": True,
            "prompt": prompt_meta,
            "attempt_history": attempt_history_by_id[camera_id],
            "accepted_attempt_index": accepted_attempt,
            "acceptance_mode": acceptance_mode,
            "strict_validation_passed": bool(strict_success),
            "completed": True,
            "successful": bool(strict_success),
            "can_be_reference": bool(strict_success),
            "reference_trust": "strict_success" if strict_success else "fallback_untrusted",
            "effective_reference_edge_weight": 1.0 if strict_success else 0.0,
            "fallback_mesh_to_predicted_depth_edge_recall": fallback_recall,
            "fallback_source_validation": fallback_source_validation,
            "fallback_to_current_mesh_rgb": False,
            "accepted_for_stage09": True,
            "accepted_as_future_reference": bool(strict_success),
            "neighbor_list_updates": neighbor_updates,
            "final_view": str(main_final_output if in_repair_pass else final_output),
            "selected_marker": str(stage08_selected_marker),
            **copied,
            "depth_encoding": _effective_camera_z_depth_encoding(frame["depth_encoding"]),
            "normal_encoding": frame["normal_encoding"],
            "visible_semantics": frame.get("visible_semantics", {}),
        }

        if in_repair_pass:
            repair_report_path = round_directory / "generation_report.json"
            prior_report = dict(final_views_by_id[camera_id])
            repair_summary = {
                "enabled": True,
                "repair_order_index": repair_pass_index,
                "selection_reason": selection_reason,
                "accepted_attempt_index": accepted_attempt,
                "strict_validation_passed": bool(strict_success),
                "acceptance_mode": acceptance_mode,
                "reference_camera_ids": accepted_reference_ids,
                "generated_neighbor_ids_at_selection": [str(item["camera_id"]) for item in generated_neighbors],
                "propagation_support_score": propagation_support,
                "round_directory": str(round_directory),
                "candidate_final_view": str(final_output),
                "fallback_mesh_to_predicted_depth_edge_recall": fallback_recall,
                "replaced_main_result": bool(strict_success),
            }
            save_json({**frame_report, "repair_pass": repair_summary}, repair_report_path)
            prior_report["repair_pass"] = repair_summary
            if strict_success:
                shutil.copy2(final_output, main_final_output)
                accepted_outputs[camera_id] = str(main_final_output)
                prior_report.update(frame_report)
                prior_report["final_view"] = str(main_final_output)
                prior_report["repair_pass"] = repair_summary
                save_json(prior_report, directory / "generation_report.json")
                final_views_by_id[camera_id] = prior_report
            else:
                slot["successful"] = bool(prior_report.get("successful", prior_report.get("strict_validation_passed", False)))
                slot["can_be_reference"] = bool(prior_report.get("can_be_reference", slot["successful"]))
                slot["reference_trust"] = str(prior_report.get("reference_trust", "strict_success" if slot["successful"] else "fallback_untrusted"))
                slot["effective_reference_edge_weight"] = float(prior_report.get("effective_reference_edge_weight", 1.0 if slot["successful"] else 0.0))
                save_json(prior_report, directory / "generation_report.json")
                final_views_by_id[camera_id] = prior_report
            repair_pass_state["processed_camera_ids"] = list(dict.fromkeys([
                *repair_pass_state.get("processed_camera_ids", []),
                camera_id,
            ]))
            _save_json_atomic(
                {
                    "completed": False,
                    "processed_camera_ids": repair_pass_state.get("processed_camera_ids", []),
                    "camera_count": len(repair_pass_order),
                },
                repair_pass_summary_path,
            )
            persist_state()
            repair_pass_index += 1
            event_index += 1
            slot["status"] = "completed"
            continue

        save_json(frame_report, directory / "generation_report.json")
        write_empty_marker_atomic(stage08_selected_marker)
        final_views_by_id[camera_id] = frame_report
        persist_state(neighbor_updates)
        accepted_order_index += 1
        event_index += 1

    final_views = [final_views_by_id[str(frame["camera_id"])] for frame in frames]
    if len(final_views) != len(frames):
        raise RuntimeError("Stage08 did not accept every connected Stage07 graph node")

    forward_summary = (
        load_json(stage / "forward_pass_summary.json")
        if (stage / "forward_pass_summary.json").exists()
        else _write_forward_pass_snapshot(stage, final_views_by_id)
    )
    if execution_phase == "forward":
        runtime_event_log.write(
            "stage08a_forward_completed",
            accepted_view_count=len(final_views),
            generation_order=forward_summary.get("generation_order", []),
            memory=runtime_memory_snapshot(),
        )
        flux_worker.close()
        console_progress.complete(len(final_views))
        console_progress.close()
        return {
            "status": "ok",
            "execution_phase": "forward",
            "completed_views": len(final_views),
            "strict_accepted_views": sum(1 for item in final_views if item.get("acceptance_mode", "strict") == "strict"),
            "fallback_single_round_views": sum(1 for item in final_views if item.get("acceptance_mode") == "fallback_single_round"),
            "forward_pass_summary": str(stage / "forward_pass_summary.json"),
            "generated_neighbor_registry": str(registry_path),
            "weighted_frontier_generation_order": str(generation_order_path),
            "final_views_root": str(stage / "final_views"),
            "repair_pass_started": False,
            "stage09_training_manifest": None,
        }

    mesh_scene = resolve_scene_for_textured_downstream(out)
    training_manifest = {
        "schema_version": 6,
        "execution_contract": execution_contract,
        "depth_geometry_contract": depth_geometry_contract,
        "scene_id": out.name,
        "source_stage07_manifest": str(dataset_path),
        "active_owner_manifest": dataset.get("active_owner_manifest"),
        "active_owner_ids": dataset.get("active_owner_ids", []),
        "mesh_scene": str(mesh_scene),
        "whole_view_generation": True,
        "reference_guided_generation": True,
        "stage07_camera_correlation_graph": str(room_graph_path),
        "generated_neighbor_registry": str(registry_path),
        "weighted_frontier_generation_order": str(generation_order_path),
        "single_root_connected_frontier_expansion": True,
        "root_restricted_to_two_stage07_room_coverage_bootstraps": True,
        "all_generated_neighbors_used_for_robust_fusion": True,
        "appearance_references_successful_trusted_only": True,
        "fallback_reference_edge_weight": 0.0,
        "references_below_local_support_threshold_are_excluded_from_rgb_writeback": True,
        "insufficient_reference_support_never_defers_generation": True,
        "stage07_rgb_only_conditioning_when_no_reference_passes_threshold": True,
        "single_round_lifecycle": {
            "each_camera_scheduled_exactly_once": True,
            "strict_success_becomes_reference": True,
            "all_attempts_failed_publish_best_mesh_to_predicted_recall": True,
            "fallback_acceptance_mode": "fallback_single_round",
            "fallback_successful": False,
            "fallback_can_be_reference": False,
            "fallback_accepted_for_stage09": True,
            "defer_or_requeue": False,
        },
        "restart_policy": {
            "trigger": "no_successful_trusted_frontier_with_unfinished_cameras",
            "source": "original_completed_stage07_topology",
            "completed_fallback_may_drive_topology_only": True,
            "fallback_rgb_never_used_as_reference": True,
        },
        "engineering_runtime": {
            "algorithm_changed": True,
            "persistent_flux_worker": True,
            "worker_spawn_method": "spawn",
            "worker_runtime_retries_do_not_consume_generation_attempts": True,
            "heartbeat_watchdog": True,
            "worker_restart_count": flux_worker.restart_count,
            "identical_reference_conditioning_reused_across_seeds": True,
            "reference_working_arrays_released_before_flux": True,
            "resume_atomically_committed_views": bool(resume_cfg.get("enabled", True)),
            "event_log": str(stage / "stage08_events.jsonl"),
        },
        "generation_retry_schedule": {
            "base_strength": base_strength,
            "retry_strength_scale": retry_strength_scale,
            "strength_level_count": 1 + maximum_retries,
            "seeds_per_strength": seeds_per_strength,
            "maximum_actual_generation_attempts_per_round": (1 + maximum_retries) * seeds_per_strength,
            "policy": "try_all_seeds_at_current_strength_then_reduce_strength",
        },
        "stage07_selected_view_rgb_reused_without_rerender": True,
        "failed_views_deferred_until_new_neighbor_evidence_or_deadlock_recovery": False,
        "every_camera_completed_in_one_round": True,
        "stage08_selected_view_completion_marker": "final_views/<camera_id>/selected.txt",
        "completion_marker_field": "frames[].selected_marker",
        "console_progress": {
            "policy": "one controlling-terminal single-line status plus immutable result/runtime-recovery lines",
            "third_party_progress_bars": third_party_progress,
        },
        "monocular_depth_validation": (
            None if depth_predictor is None else depth_predictor.runtime_metadata()
        ),
        "semantic_subpass_fusion": False,
        "atlas_writeback": False,
        "source_frame_count": len(final_views),
        "frame_count": len(final_views),
        "strict_accepted_frame_count": sum(
            1 for item in final_views if item.get("acceptance_mode", "strict") == "strict"
        ),
        "fallback_single_round_frame_count": sum(
            1 for item in final_views if item.get("acceptance_mode") == "fallback_single_round"
        ),
        "frames": [
            {
                "camera_id": item["camera_id"],
                "camera_role": item["camera_role"],
                "target_owner_id": item["target_owner_id"],
                "look_at_target_id": item.get("look_at_target_id"),
                "look_at_target_type": item.get("look_at_target_type"),
                "position_view_pair_id": item.get("position_view_pair_id"),
                "reference_camera_ids": item.get("reference_camera_ids", []),
                "reference_camera_id": item.get("reference_camera_id"),
                "target_rgb": item["final_view"],
                "selected_marker": item["selected_marker"],
                "source_rgb": item["source_rgb"],
                "source_rgb_provenance": "stage07_selected_view_beauty_render",
                "depth": item["depth"],
                "depth_encoding": item["depth_encoding"],
                "normal_world": item["normal_world"],
                "normal_encoding": item["normal_encoding"],
                "semantic": item["semantic"],
                "palette": item["palette"],
                "triangle_id": item["triangle_id"],
                "camera": item["camera"],
                "visible_semantics": item["visible_semantics"],
                "acceptance_mode": item.get("acceptance_mode", "strict"),
                "completed": bool(item.get("completed", True)),
                "successful": bool(item.get("successful", item.get("acceptance_mode", "strict") == "strict")),
                "can_be_reference": bool(item.get("can_be_reference", item.get("acceptance_mode", "strict") == "strict")),
                "effective_reference_edge_weight": float(item.get("effective_reference_edge_weight", 0.0)),
                "fallback_to_current_mesh_rgb": False,
            }
            for item in final_views
        ],
        "repair_pass": {
            "enabled": repair_pass_enabled,
            "completed": repair_pass_completed,
            "policy": "post_generation_full_repair_pass_in_original_generation_order_using_all_current_strict_neighbors",
            "processed_camera_ids": list(repair_pass_state.get("processed_camera_ids", [])),
        },
    }
    save_json(training_manifest, stage / "stage09_training_manifest.json")
    runtime_event_log.write(
        "stage08_completed",
        accepted_view_count=len(final_views),
        depth_geometry_contract=depth_geometry_contract,
        execution_contract=execution_contract,
        worker_restart_count=flux_worker.restart_count,
        memory=runtime_memory_snapshot(),
    )
    flux_worker.close()
    console_progress.complete(len(final_views))
    console_progress.close()
    return {
        "status": "ok",
        "execution_phase": "repair" if execution_phase == "repair" else "all",
        "depth_geometry_contract": depth_geometry_contract,
        "completed_views": len(final_views),
        "accepted_generated_views": len(final_views),
        "strict_accepted_views": sum(
            1 for item in final_views if item.get("acceptance_mode", "strict") == "strict"
        ),
        "fallback_single_round_views": sum(
            1 for item in final_views if item.get("acceptance_mode") == "fallback_single_round"
        ),
        "deferred_retry_events": 0,
        "whole_view_generation": True,
        "reference_guided_generation": True,
        "stage07_camera_correlation_graph_reused": True,
        "single_root_connected_frontier_expansion": True,
        "root_restricted_to_two_stage07_room_coverage_bootstraps": True,
        "all_generated_neighbors_used_for_robust_fusion": True,
        "appearance_references_successful_trusted_only": True,
        "fallback_reference_edge_weight": 0.0,
        "stage07_selected_view_rgb_reused_without_rerender": True,
        "stage08_selected_view_completion_markers": True,
        "console_progress": {
            "single_line": bool(console_progress_cfg.get("single_line", True)),
            "third_party_progress_bars": third_party_progress,
        },
        "engineering_runtime": {
            "persistent_flux_worker": True,
            "worker_restart_count": flux_worker.restart_count,
            "runtime_retries_per_request": worker_settings.runtime_retries,
            "resume_enabled": bool(resume_cfg.get("enabled", True)),
            "resumed_from_incomplete_stage": bool(resume_state.get("resumed", False)),
            "event_log": str(stage / "stage08_events.jsonl"),
        },
        "generated_neighbor_registry": str(registry_path),
        "weighted_frontier_generation_order": str(generation_order_path),
        "semantic_subpass_fusion": False,
        "atlas_writeback": False,
        "model_preparation": model_preparation,
        "monocular_depth_validation": (
            None if depth_predictor is None else depth_predictor.runtime_metadata()
        ),
        "mesh_scene": str(mesh_scene),
        "stage09_training_manifest": str(stage / "stage09_training_manifest.json"),
        "final_views_root": str(stage / "final_views"),
        "forward_pass_summary": str(stage / "forward_pass_summary.json"),
        "repair_pass": {
            "enabled": repair_pass_enabled,
            "completed": repair_pass_completed,
            "processed_camera_ids": list(repair_pass_state.get("processed_camera_ids", [])),
            "summary_path": str(repair_pass_summary_path),
        },
    }

