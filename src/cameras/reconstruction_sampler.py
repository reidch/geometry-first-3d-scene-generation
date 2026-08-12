from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from src.appearance.triangle_id_map import load_triangle_id_map
from src.cameras.interior_probability_sampling import _aabb, _stage05_aabbs
from src.cameras.room_pair_sampling import (
    IncrementalRoomCoverageGraph,
    RoomCoverageState,
    build_room_surface_model,
    serialize_room_model,
)
from src.cameras.reconstruction_view_metrics import semantic_pixel_counts
from src.cameras.scene_geometry import scaffold_points, scaffold_collision_bodies
from src.cameras.worldmesh_coverage_sampling import (
    camera_room_sample_indices,
    choose_best_repair_camera,
    generate_repair_candidates,
    generate_worldmesh_base_cameras,
    uncovered_components,
)
from src.io.json_io import load_json, save_json
from src.scene_ir.json_scene import flat_objects


class Stage07Progress:
    """Compact Stage07 progress for deterministic base views and coverage repair."""

    def __init__(self, total_slots: int, total_targets: int = 0):
        self.total_slots = max(int(total_slots), 0)
        self.total_targets = max(int(total_targets), 0)
        self.accepted_count = 0
        self.attempt_count = 0
        self.coverage_ratio = 0.0
        self.covered_sample_count = 0
        self.total_sample_count = 0
        self.component_count = 0
        self.connected = False
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._started = False
        self._phase = "constructing WorldMesh-style base cameras"
        self._progress_text = self._format_progress(self._phase)
        self._current_text = "[07] Current: preparing deterministic camera layout..."

    def _format_progress(self, phase: str) -> str:
        base_text = f" base={self.total_slots}" if self.total_slots > 0 else ""
        return (
            f"[07] accepted={self.accepted_count}{base_text} "
            f"coverage={100.0 * self.coverage_ratio:7.3f}% "
            f"samples={self.covered_sample_count}/{self.total_sample_count} "
            f"components={self.component_count} connected={str(self.connected).lower()} "
            f"generated={self.attempt_count} | {phase}"
        )

    @staticmethod
    def _fit_terminal(text: str) -> str:
        width = max(shutil.get_terminal_size(fallback=(180, 24)).columns - 1, 40)
        if len(text) <= width:
            return text
        return text[: max(width - 1, 1)] + "…"

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._interactive:
            print(self._fit_terminal(self._progress_text), flush=True)
            print(self._fit_terminal(self._current_text), end="", flush=True)
        else:
            print(self._progress_text, flush=True)

    def _rewrite(self, *, progress: str | None = None, current: str | None = None) -> None:
        if progress is not None:
            self._progress_text = progress
        if current is not None:
            self._current_text = current
        if not self._started:
            self.start()
        if self._interactive:
            sys.stdout.write("\r\033[2K\033[1A\r\033[2K")
            sys.stdout.write(self._fit_terminal(self._progress_text) + "\n")
            sys.stdout.write(self._fit_terminal(self._current_text))
            sys.stdout.flush()

    def set_state(
        self,
        *,
        generated_count: int,
        accepted_count: int,
        hard_coverage_ratio: float,
        covered_sample_count: int,
        total_sample_count: int,
        component_count: int,
        connected: bool,
        phase: str,
        current: str,
    ) -> None:
        self.attempt_count = int(generated_count)
        self.accepted_count = int(accepted_count)
        self.coverage_ratio = float(hard_coverage_ratio)
        self.covered_sample_count = int(covered_sample_count)
        self.total_sample_count = int(total_sample_count)
        self.component_count = int(component_count)
        self.connected = bool(connected)
        self._phase = str(phase)
        progress = self._format_progress(self._phase)
        if self._interactive:
            self._rewrite(progress=progress, current=f"[07] Current: {current}")
        else:
            print(progress + f" | {current}", flush=True)

    def shared_buffer(self, camera_id: str, index: int, total: int) -> None:
        self._phase = "exporting selected RGB/depth/normal/semantic buffers"
        progress = self._format_progress(self._phase)
        current = f"[07] Current: {camera_id} | final buffers {index}/{total}"
        if self._interactive:
            self._rewrite(progress=progress, current=current)
        else:
            print(progress + " | " + current, flush=True)

    def finish(self) -> None:
        self._phase = "complete"
        progress = self._format_progress(self._phase)
        current = "[07] Current: complete"
        if self._interactive:
            self._rewrite(progress=progress, current=current)
            print(flush=True)
        else:
            print(progress, flush=True)


def _parse_shared_progress_line(line: str) -> tuple[str, int, int] | None:
    prefix = "[PGW_STAGE07_SHARED_PROGRESS] "
    if not line.startswith(prefix):
        return None
    fields: Dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    try:
        return fields["camera"], int(fields["index"]), int(fields["total"])
    except (KeyError, ValueError):
        return None


def _owner_records(scene: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(record["object_id"]): dict(record) for record in flat_objects(scene)}


def _build_context(out: Path, scene: Mapping[str, Any], camera_config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(camera_config.get("reconstruction_sampling", {}))
    records = _owner_records(scene)
    modes = {
        owner: str(dict(record.get("generation", {})).get("mode", ""))
        for owner, record in records.items()
    }
    points_by_id = scaffold_points(scene)
    collision_bodies = scaffold_collision_bodies(scene)
    if not collision_bodies:
        # Conservative fallback for unusual legacy scenes that lack scaffold parts.
        final_aabbs = _stage05_aabbs(out)
        for owner, mode in sorted(modes.items()):
            if mode in {"", "group", "surface_texture"}:
                continue
            box = final_aabbs.get(owner)
            if box is None:
                points = points_by_id.get(owner)
                if points is None or len(points) == 0:
                    continue
                box = _aabb(points.min(axis=0), points.max(axis=0))
            collision_bodies.append({
                "collider_type": "world_aabb_fallback",
                "owner_id": owner,
                "minimum": list(box["minimum"]),
                "maximum": list(box["maximum"]),
            })
    interior_ids = {str(body.get("owner_id", "object")) for body in collision_bodies}
    return {
        "config": cfg,
        "owner_records": records,
        "owner_generation_modes": modes,
        "object_boxes": collision_bodies,
        "interior_object_count": len(interior_ids),
    }


def _synthesize_semantic_image(
    triangle_map: np.ndarray,
    manifest: Mapping[str, Any],
    palette: Mapping[str, Any],
    output_path: Path,
) -> str:
    semantic = np.zeros((*triangle_map.shape, 3), dtype=np.uint8)
    for record in manifest.get("scene_triangle_ranges", []):
        owner = str(record["semantic_owner_id"])
        palette_record = palette.get(owner, {})
        color = palette_record.get("color_uint8_rgb") if isinstance(palette_record, Mapping) else None
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        start = int(record["scene_triangle_start"])
        end = int(record["scene_triangle_end_exclusive"])
        mask = (triangle_map >= start) & (triangle_map < end)
        semantic[mask] = np.asarray(color[:3], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(semantic, "RGB").save(output_path)
    return str(output_path)


def _render_final_buffers(
    out: Path,
    cameras: list[Dict[str, Any]],
    step: Path,
    refinement_config: Mapping[str, Any],
    active_owner_manifest: Path,
    progress: Stage07Progress | None = None,
) -> Dict[str, Any]:
    render_cfg = dict(refinement_config.get("stage07_render", refinement_config.get("candidate_render", {})))
    resolution = render_cfg.get("shared_buffer_resolution", [1376, 768])
    camera_file = step / "cameras.accepted.json"
    save_json({"cameras": cameras}, camera_file)
    lighting_config_path = step / "shared_buffers" / "stage07_lighting_config.json"
    save_json(dict(render_cfg.get("lighting", {})), lighting_config_path)
    if not bool(render_cfg.get("render_selected_view_rgb", True)):
        raise RuntimeError("Stage07 selected-view RGB rendering is mandatory for Stage08 reuse")
    command = [
        os.environ.get("BLENDER_BIN", "blender"),
        "--background",
        "--python",
        "src/blender/prephysics_runtime/render_refinement_shared_buffers_batch.py",
        "--",
        "--out", str(out),
        "--camera_file", str(camera_file),
        "--output_dir", str(step / "shared_buffers"),
        "--width", str(int(resolution[0])),
        "--height", str(int(resolution[1])),
        "--samples", str(int(render_cfg.get("shared_buffer_samples", 1))),
        "--lighting_config", str(lighting_config_path),
        "--active_owner_manifest", str(active_owner_manifest),
    ]
    log_path = step / "shared_buffers" / "shared_buffer_blender.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Stage07 shared-buffer Blender stdout pipe is unavailable")
        for line in process.stdout:
            log_stream.write(line)
            event = _parse_shared_progress_line(line)
            if event is not None and progress is not None:
                camera_id, index, total = event
                progress.shared_buffer(camera_id, index, total)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Stage07 shared-buffer Blender process exited with code {return_code}; see {log_path}"
        )
    report = load_json(step / "shared_buffers" / "batch_report.json")
    if int(report.get("success_count", 0)) != len(cameras):
        raise RuntimeError("Stage07 final reconstruction buffer render was incomplete")
    return report


def _stage07_step(out: str | Path) -> Path:
    return Path(out) / "07_refinement_cameras"


def _write_active_owner_manifest(step: Path, context: Mapping[str, Any]) -> tuple[Path, list[str]]:
    active_owner_manifest = step / "active_owner_ids.json"
    active_owner_ids = sorted(
        owner for owner, mode in context["owner_generation_modes"].items()
        if str(mode) not in {"", "group"}
    )
    active_owner_records = []
    for owner in active_owner_ids:
        record = context["owner_records"][owner]
        active_owner_records.append({
            "owner_id": owner,
            "parent_id": record.get("parent_id"),
            "generation_mode": str(dict(record.get("generation", {})).get("mode", "")),
            "transform": dict(record.get("transform", {})),
            "world_transform": dict(record.get("world_transform", {})),
        })
    save_json({
        "schema_version": 2,
        "source": "current scene JSON supplied to Stage07",
        "owner_ids": active_owner_ids,
        "owners": active_owner_records,
        "downstream_filtering_only": True,
        "downstream_hierarchy_overlay": True,
        "allows_reuse_of_older_stage06_scene": True,
    }, active_owner_manifest)
    return active_owner_manifest, active_owner_ids


def _camera_direction(camera: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(camera["target"], dtype=np.float64) - np.asarray(camera["position"], dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def select_reconstruction_cameras(
    out: str | Path,
    scene: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    refinement_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct WorldMesh-style base cameras, then greedily repair room-shell coverage."""
    out = Path(out)
    step = _stage07_step(out)
    step.mkdir(parents=True, exist_ok=True)

    # Explicitly remove artifacts from the retired stochastic candidate sampler.
    for stale in (
        step / "candidate_renders",
        step / "look_at_targets.json",
        step / "initial_pair_pool_manifest.json",
        step / "sampling_failure_state.json",
    ):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)

    context = _build_context(out, scene, camera_config)
    active_owner_manifest, active_owner_ids = _write_active_owner_manifest(step, context)
    cfg = context["config"]
    room_model = build_room_surface_model(scene, cfg)
    coverage_cfg = dict(cfg.get("coverage_repair", {}))
    termination_cfg = dict(cfg.get("termination", {}))

    np.savez_compressed(
        step / "room_surface_samples.npz",
        positions=np.asarray(room_model["positions"], dtype=np.float32),
        normals=np.asarray(room_model["normals"], dtype=np.float32),
        representative_areas=np.asarray(room_model["areas"], dtype=np.float32),
        surface_indices=np.asarray(room_model["surface_indices"], dtype=np.int32),
    )
    save_json(serialize_room_model(room_model), step / "room_surface_model.json")

    base_cameras = generate_worldmesh_base_cameras(
        room_model,
        cfg,
        context["object_boxes"],
    )
    maximum_repair = max(int(coverage_cfg.get("maximum_repair_cameras", 16)), 0)
    maximum_total = int(termination_cfg.get("maximum_camera_count", len(base_cameras) + maximum_repair))
    maximum_total = max(maximum_total, len(base_cameras))

    accepted: list[Dict[str, Any]] = []
    accepted_sample_sets: list[np.ndarray] = []
    coverage = RoomCoverageState.create(room_model, cfg)
    coverage_graph = IncrementalRoomCoverageGraph(
        np.asarray(room_model["areas"], dtype=np.float64),
        dict(cfg.get("camera_overlap_graph", {})),
    )
    repair_history: list[Dict[str, Any]] = []
    generated_count = 0
    progress = Stage07Progress(total_slots=len(base_cameras))
    progress.total_sample_count = int(len(room_model["positions"]))
    progress._progress_text = progress._format_progress(progress._phase)
    progress.start()
    areas = np.asarray(room_model["areas"], dtype=np.float64)

    def add_camera(camera: Dict[str, Any], indices: np.ndarray, *, phase: str, extra: Mapping[str, Any] | None = None) -> None:
        nonlocal generated_count
        generated_count += 1
        indices = np.unique(np.asarray(indices, dtype=np.int32))
        if indices.size == 0:
            raise RuntimeError(f"Stage07 camera sees no room-shell samples: {camera['camera_id']}")
        before = coverage.hard_covered.copy()
        newly_covered = indices[~before[indices]]
        coverage.update(indices)
        accepted_sample_sets.append(indices)
        coverage_graph.add(str(camera["camera_id"]), indices, _camera_direction(camera))
        camera["room_sample_set_index"] = len(accepted_sample_sets) - 1
        camera["selection_metrics"] = {
            "room_sample_count": int(indices.size),
            "covered_room_area": float(areas[indices].sum()),
            "newly_covered_room_sample_count": int(newly_covered.size),
            "newly_covered_room_area": float(areas[newly_covered].sum()) if newly_covered.size else 0.0,
            "room_hard_coverage_ratio_after_acceptance": coverage.hard_coverage_ratio(),
            **dict(extra or {}),
        }
        accepted.append(camera)
        progress.set_state(
            generated_count=generated_count,
            accepted_count=len(accepted),
            hard_coverage_ratio=coverage.hard_coverage_ratio(),
            covered_sample_count=int(np.count_nonzero(coverage.hard_covered)),
            total_sample_count=int(coverage.hard_covered.size),
            component_count=coverage_graph.component_count,
            connected=coverage_graph.connected,
            phase=phase,
            current=f"accepted {camera['camera_id']}",
        )

    for camera in base_cameras:
        add_camera(
            dict(camera),
            camera_room_sample_indices(camera, room_model, cfg),
            phase="adding WorldMesh-style base cameras",
        )

    repair_index = 0
    while not bool(coverage.hard_covered.all()):
        if repair_index >= maximum_repair or len(accepted) >= maximum_total:
            components = uncovered_components(coverage.hard_covered, room_model)
            save_json({
                "status": "failed",
                "reason": "coverage_repair_budget_exhausted",
                "accepted_camera_count": len(accepted),
                "repair_camera_count": repair_index,
                "room_hard_coverage_ratio": coverage.hard_coverage_ratio(),
                "largest_remaining_hole": components[0] if components else None,
            }, step / "coverage_repair_failure.json")
            raise RuntimeError("Stage07 coverage repair budget exhausted before full room-shell coverage")

        components = uncovered_components(coverage.hard_covered, room_model)
        if not components:
            break
        hole = components[0]
        candidates = generate_repair_candidates(
            hole,
            room_model,
            cfg,
            context["object_boxes"],
            accepted,
            repair_index,
        )
        chosen = choose_best_repair_camera(
            candidates,
            coverage.hard_covered,
            room_model,
            cfg,
            coverage_graph,
        )
        if chosen is None:
            save_json({
                "status": "failed",
                "reason": "no_safe_connected_repair_candidate_with_positive_coverage_gain",
                "hole": hole,
                "candidate_count": len(candidates),
                "room_hard_coverage_ratio": coverage.hard_coverage_ratio(),
            }, step / "coverage_repair_failure.json")
            raise RuntimeError(
                "Stage07 could not generate a safe graph-connected repair camera with positive new coverage"
            )
        camera, indices, repair_meta = chosen
        camera["camera_id"] = f"reconstruction_repair_{repair_index:03d}"
        camera["repair_target_surface_id"] = str(hole["surface_id"])
        camera["repair_target_centroid"] = list(hole["centroid"])
        camera["repair_target_area"] = float(hole["area"])
        camera["repair_target_sample_count"] = int(hole["sample_count"])
        add_camera(
            camera,
            indices,
            phase="repairing uncovered room-shell holes",
            extra=repair_meta,
        )
        repair_history.append({
            "repair_index": repair_index,
            "hole_surface_id": str(hole["surface_id"]),
            "hole_area_before": float(hole["area"]),
            "hole_sample_count_before": int(hole["sample_count"]),
            "hole_centroid": list(hole["centroid"]),
            "candidate_count": len(candidates),
            "selected_camera_id": str(camera["camera_id"]),
            **repair_meta,
            "room_hard_coverage_ratio_after": coverage.hard_coverage_ratio(),
        })
        repair_index += 1

    if bool(termination_cfg.get("require_room_sample_overlap_graph_connected", True)) and not coverage_graph.connected:
        save_json(coverage_graph.to_dict(), step / "room_coverage_graph.json")
        raise RuntimeError(
            "Stage07 WorldMesh-style camera set reached room-shell coverage but the Stage08 camera graph is disconnected"
        )

    graph = coverage_graph.to_dict()
    np.save(step / "room_hard_coverage.npy", coverage.hard_covered)
    np.save(step / "room_soft_coverage.npy", coverage.normalized().astype(np.float32))
    np.savez_compressed(
        step / "camera_room_sample_sets.npz",
        **{f"camera_{index:05d}": values for index, values in enumerate(accepted_sample_sets)},
    )
    save_json(graph, step / "room_coverage_graph.json")
    save_json({"cameras": accepted}, step / "cameras.accepted.json")
    save_json({"schema_version": 1, "repairs": repair_history}, step / "coverage_repair_history.json")

    camera_model = dict(cfg.get("camera_model", {}))
    if accepted:
        camera_model["resolved_focal_length_mm"] = float(accepted[0]["focal_length"])
    report = {
        "status": "ok",
        "policy": "worldmesh_deterministic_base_plus_largest_uncovered_component_repair",
        "camera_model": camera_model,
        "worldmesh_base_layout": dict(cfg.get("worldmesh_base_layout", {})),
        "base_camera_count": len(base_cameras),
        "repair_camera_count": repair_index,
        "accepted_camera_count": len(accepted),
        "generated_camera_count": generated_count,
        "room_volume_m3": float(room_model["room_volume_m3"]),
        "room_diagonal_m": float(room_model["room_diagonal_m"]),
        "room_surface_sample_count": int(len(room_model["positions"])),
        "room_hard_coverage_ratio": coverage.hard_coverage_ratio(),
        "covered_room_sample_count": int(np.count_nonzero(coverage.hard_covered)),
        "coverage_graph_component_count": graph["component_count"],
        "coverage_graph_connected": graph["connected"],
        "coverage_graph_edge_count": graph["edge_count"],
        "coverage_graph_overlap_metric": graph.get("overlap_metric"),
        "coverage_graph_minimum_edge_correlation_score": graph.get("minimum_edge_correlation_score"),
        "coverage_repair_history": str(step / "coverage_repair_history.json"),
        "active_owner_manifest": str(active_owner_manifest),
        "active_owner_count": len(active_owner_ids),
        "selection_uses_blender_candidate_renders": False,
        "selection_uses_probability_sampling": False,
        "selection_uses_semantic_or_depth_quality_gates": False,
    }
    save_json(report, step / "selection_report.json")
    return {
        "status": "ok",
        "context": context,
        "active_owner_manifest": str(active_owner_manifest),
        "active_owner_ids": active_owner_ids,
        "accepted_cameras": accepted,
        "base_camera_count": len(base_cameras),
        "repair_camera_count": repair_index,
        "generated_camera_count": generated_count,
        "room_model": room_model,
        "selection_report": report,
        "progress": progress,
    }


def render_reconstruction_buffers(
    out: str | Path,
    scene: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    refinement_config: Mapping[str, Any],
) -> Dict[str, Any]:
    out = Path(out)
    step = _stage07_step(out)
    step.mkdir(parents=True, exist_ok=True)
    context = _build_context(out, scene, camera_config)
    active_owner_manifest, active_owner_ids = _write_active_owner_manifest(step, context)
    accepted_payload = load_json(step / "cameras.accepted.json")
    accepted = list(accepted_payload.get("cameras", []))
    if not accepted:
        raise RuntimeError("Stage07 buffer export requires existing accepted cameras")
    selection_report_path = step / "selection_report.json"
    selection_report = load_json(selection_report_path) if selection_report_path.exists() else {}
    progress = Stage07Progress(total_slots=int(selection_report.get("base_camera_count", len(accepted))))
    progress.set_state(
        generated_count=int(selection_report.get("generated_camera_count", len(accepted))),
        accepted_count=len(accepted),
        hard_coverage_ratio=float(selection_report.get("room_hard_coverage_ratio", 0.0)),
        covered_sample_count=int(selection_report.get("covered_room_sample_count", 0)),
        total_sample_count=int(selection_report.get("room_surface_sample_count", 0)),
        component_count=int(selection_report.get("coverage_graph_component_count", 0)),
        connected=bool(selection_report.get("coverage_graph_connected", False)),
        phase="ready to export final buffers",
        current="starting shared-buffer renderer",
    )
    progress.start()
    shared_report = _render_final_buffers(out, accepted, step, refinement_config, active_owner_manifest, progress)
    progress.finish()
    by_id = {str(item["camera_id"]): item for item in shared_report["results"] if item.get("status") == "ok"}
    shared_manifest = load_json(shared_report["triangle_owner_manifest"])
    shared_palette_path = Path(shared_report["semantic_palette"])
    shared_palette = load_json(shared_palette_path)
    frames = []
    for camera in accepted:
        report = by_id[camera["camera_id"]]
        triangle_map = load_triangle_id_map(
            report["triangle_id"],
            valid_triangle_count=int(shared_manifest.get("triangle_count", 0)),
        )
        semantic_path = Path(report["triangle_id"]).with_name("semantic.png")
        report["semantic"] = _synthesize_semantic_image(triangle_map, shared_manifest, shared_palette, semantic_path)
        report["palette"] = str(shared_palette_path)
        selected_marker = Path(report.get("selected_marker") or Path(report["triangle_id"]).with_name("selected.txt"))
        if not selected_marker.exists():
            raise RuntimeError(f"Selected Stage07 view is missing selected.txt: {selected_marker}")
        report["selected_marker"] = str(selected_marker)
        save_json(report, Path(report["triangle_id"]).with_name("camera_report.json"))
        visible_counts = semantic_pixel_counts(triangle_map, shared_manifest)
        total_visible = max(sum(visible_counts.values()), 1)
        visible_semantics = {
            owner: {
                "pixel_count": int(count),
                "screen_fraction_of_visible_scene": float(count / total_visible),
                "generation_mode": context["owner_generation_modes"].get(owner, ""),
            }
            for owner, count in sorted(visible_counts.items(), key=lambda item: item[1], reverse=True)
        }
        frames.append({
            "camera_id": camera["camera_id"],
            "camera_role": camera["camera_role"],
            "camera_source": camera.get("camera_source"),
            "room_sample_set_index": camera.get("room_sample_set_index"),
            "selection_metrics": camera.get("selection_metrics", {}),
            "rgb": report.get("rgb") or report.get("albedo"),
            "rgb_source": "stage07_selected_view_beauty_render",
            "depth": report["depth"],
            "depth_encoding": report["depth_encoding"],
            "normal_world": report["normal_world"],
            "normal_encoding": report["normal_encoding"],
            "semantic": report["semantic"],
            "palette": report["palette"],
            "triangle_id": report["triangle_id"],
            "selected_marker": report["selected_marker"],
            "camera": report["camera"],
            "visible_semantics": visible_semantics,
        })
    shared_report["results"] = list(by_id.values())
    save_json(shared_report, step / "shared_buffers" / "batch_report.json")
    dataset_manifest = {
        "schema_version": 4,
        "scene_id": out.name,
        "policy": "WorldMesh-style deterministic base cameras plus greedy largest-hole room-shell coverage repair",
        "camera_model": selection_report.get("camera_model", {}),
        "camera_generation": {
            "base_camera_count": int(selection_report.get("base_camera_count", 0)),
            "repair_camera_count": int(selection_report.get("repair_camera_count", 0)),
            "room_hard_coverage_ratio": float(selection_report.get("room_hard_coverage_ratio", 0.0)),
            "coverage_graph_connected": bool(selection_report.get("coverage_graph_connected", False)),
        },
        "active_owner_manifest": str(active_owner_manifest),
        "active_owner_ids": active_owner_ids,
        "room_surface_samples": str(step / "room_surface_samples.npz"),
        "camera_room_sample_sets": str(step / "camera_room_sample_sets.npz"),
        "room_coverage_graph": str(step / "room_coverage_graph.json"),
        "coverage_repair_history": str(step / "coverage_repair_history.json"),
        "selected_view_rgb_contract": {
            "producer": "Stage07 final-buffer render",
            "render_only_for_accepted_cameras": True,
            "output_field": "frames[].rgb",
            "completion_marker_field": "frames[].selected_marker",
            "completion_marker_name": "selected.txt",
            "consumer": "Stage08 current mesh RGB; no Stage08 RGB rerender",
        },
        "frame_count": len(frames),
        "frames": frames,
    }
    save_json(dataset_manifest, step / "reconstruction_dataset_manifest.json")
    return {
        "status": "ok",
        "frame_count": len(frames),
        "dataset_manifest": str(step / "reconstruction_dataset_manifest.json"),
        "shared_buffer_report": str(step / "shared_buffers" / "batch_report.json"),
        "active_owner_manifest": str(active_owner_manifest),
        "active_owner_count": len(active_owner_ids),
    }


def prepare_reconstruction_cameras(
    out: str | Path,
    scene: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    refinement_config: Mapping[str, Any],
) -> Dict[str, Any]:
    selection = select_reconstruction_cameras(out, scene, camera_config, refinement_config)
    render = render_reconstruction_buffers(out, scene, camera_config, refinement_config)
    step = _stage07_step(out)
    return {
        "status": "ok",
        "base_camera_count": int(selection.get("base_camera_count", 0)),
        "repair_camera_count": int(selection.get("repair_camera_count", 0)),
        "generated_camera_count": int(selection.get("generated_camera_count", 0)),
        "accepted_camera_count": len(selection.get("accepted_cameras", [])),
        "room_diagonal_m": float(selection.get("room_model", {}).get("room_diagonal_m", 0.0)),
        "interior_object_count": int(selection.get("context", {}).get("interior_object_count", 0)),
        "dataset_manifest": render["dataset_manifest"],
        "selected_view_rgb_rendered": True,
        "selected_view_rgb_reused_by_stage08": True,
        "active_owner_manifest": render["active_owner_manifest"],
        "active_owner_count": int(render["active_owner_count"]),
        "shared_buffer_report": render["shared_buffer_report"],
        "selection_report": str(step / "selection_report.json"),
        "coverage_repair_history": str(step / "coverage_repair_history.json"),
        "room_coverage_graph": str(step / "room_coverage_graph.json"),
    }
