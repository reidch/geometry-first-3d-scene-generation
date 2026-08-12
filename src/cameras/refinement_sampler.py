from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from src.appearance.triangle_id_map import load_triangle_id_map
from src.cameras.interior_probability_sampling import (
    angular_separation_degrees,
    build_sampling_context,
    sample_candidate_batch,
)
from src.coverage.triangle_coverage import weighted_seen_ratio
from src.io.json_io import load_json, save_json
from src.cameras.target_depth_gate import (
    bounding_sphere_radius_from_aabb,
    decode_linear_target_depth,
    evaluate_target_depth_distribution,
    normalise_target_depth_config,
)


def _render_candidates_batch(out: Path, camera_file: Path, output_dir: Path, config: Dict) -> Dict:
    blender = os.environ.get("BLENDER_BIN", "blender")
    script = Path("src/blender/prephysics_runtime/render_refinement_candidates_batch.py")
    render_cfg = dict(config.get("candidate_render", {}))
    resolution = render_cfg.get("resolution", [384, 216])
    samples = int(render_cfg.get("samples", 1))
    cmd = [
        blender, "--background", "--python", str(script), "--",
        "--out", str(out),
        "--camera_file", str(camera_file),
        "--output_dir", str(output_dir),
        "--width", str(int(resolution[0])),
        "--height", str(int(resolution[1])),
        "--samples", str(samples),
        "--max_invalid_id_ratio", str(float(render_cfg.get("max_invalid_id_ratio", 0.0005))),
    ]
    subprocess.run(cmd, check=True)
    report_path = output_dir / "batch_report.json"
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError("Batch candidate renderer did not produce batch_report.json")
    report = load_json(report_path)
    if int(report.get("success_count", 0)) <= 0:
        raise RuntimeError("Batch candidate renderer produced no successful views")
    manifest = Path(report["triangle_owner_manifest"])
    if not manifest.exists() or manifest.stat().st_size == 0:
        raise RuntimeError(f"Triangle owner manifest missing: {manifest}")
    return report


class _CandidateRenderWorker:
    """Keep one Blender process alive and render one Stage07 candidate per request."""

    def __init__(self, out: Path, output_dir: Path, config: Dict):
        blender = os.environ.get("BLENDER_BIN", "blender")
        script = Path("src/blender/prephysics_runtime/render_refinement_candidates_batch.py")
        render_cfg = dict(config.get("candidate_render", {}))
        resolution = render_cfg.get("resolution", [384, 216])
        samples = int(render_cfg.get("samples", 1))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ready_file = self.output_dir / "worker_ready.json"
        if self.ready_file.exists():
            self.ready_file.unlink()
        cmd = [
            blender, "--background", "--python", str(script), "--",
            "--out", str(out),
            "--output_dir", str(self.output_dir),
            "--width", str(int(resolution[0])),
            "--height", str(int(resolution[1])),
            "--samples", str(samples),
            "--max_invalid_id_ratio", str(float(render_cfg.get("max_invalid_id_ratio", 0.0005))),
            "--worker_mode",
            "--ready_file", str(self.ready_file),
        ]
        self.start_timeout_seconds = float(render_cfg.get("worker_start_timeout_seconds", 300.0))
        self.render_timeout_seconds = float(render_cfg.get("worker_render_timeout_seconds", 300.0))
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, bufsize=1)
        try:
            self._wait_for_file(self.ready_file, self.start_timeout_seconds, "candidate renderer startup")
        except Exception:
            if self.process.poll() is None:
                self.process.terminate()
            raise
        ready = load_json(self.ready_file)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Stage07 candidate renderer did not become ready: {ready}")
        self.triangle_owner_manifest = Path(ready["triangle_owner_manifest"])
        if not self.triangle_owner_manifest.exists():
            raise RuntimeError(f"Triangle owner manifest missing: {self.triangle_owner_manifest}")

    def _wait_for_file(self, path: Path, timeout_seconds: float, operation: str) -> None:
        deadline = time.monotonic() + max(float(timeout_seconds), 1.0)
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size > 0:
                return
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"Stage07 Blender candidate renderer exited during {operation} with code {code}"
                )
            time.sleep(0.05)
        raise TimeoutError(f"Timed out during Stage07 {operation}: {path}")

    def _request(self, action: str, camera: Dict, output_dir: Path, response_name: str) -> Dict:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Stage07 Blender candidate renderer is not running (code={self.process.returncode})"
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        response_file = output_dir / response_name
        if response_file.exists():
            response_file.unlink()
        command = {
            "action": action,
            "camera": camera,
            "output_dir": str(output_dir),
            "response_file": str(response_file),
        }
        if self.process.stdin is None:
            raise RuntimeError("Stage07 Blender candidate renderer stdin is unavailable")
        self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        self._wait_for_file(
            response_file,
            self.render_timeout_seconds,
            f"candidate {action} {camera['camera_id']}",
        )
        return load_json(response_file)

    def render_isolated(self, camera: Dict, output_dir: Path) -> Dict:
        return self._request("render_isolated", camera, output_dir, "isolated_response.json")

    def render_full(self, camera: Dict, output_dir: Path) -> Dict:
        return self._request("render_full", camera, output_dir, "full_response.json")

    def render(self, camera: Dict, output_dir: Path) -> Dict:
        """Compatibility path that asks Blender for both candidate renders."""
        return self._request("render", camera, output_dir, "worker_response.json")

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.write('{"action":"shutdown"}\n')
                    self.process.stdin.flush()
                    self.process.stdin.close()
                self.process.wait(timeout=30)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except Exception:
                    self.process.kill()
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _adaptive_occlusion_config(validation_cfg: Dict, base_threshold: float) -> Dict:
    raw = dict(validation_cfg.get("adaptive_occlusion", {}))
    config = {
        "enabled": bool(raw.get("enabled", True)),
        "failure_streak_to_relax": int(raw.get("failure_streak_to_relax", 6)),
        "relax_step": float(raw.get("relax_step", 0.05)),
        "max_threshold": float(raw.get("max_threshold", max(0.75, base_threshold))),
        "success_streak_to_tighten": int(raw.get("success_streak_to_tighten", 2)),
        "tighten_step": float(raw.get("tighten_step", 0.03)),
    }
    if config["failure_streak_to_relax"] < 1:
        raise ValueError("adaptive_occlusion.failure_streak_to_relax must be >= 1")
    if config["success_streak_to_tighten"] < 1:
        raise ValueError("adaptive_occlusion.success_streak_to_tighten must be >= 1")
    if config["relax_step"] < 0.0 or config["tighten_step"] < 0.0:
        raise ValueError("adaptive_occlusion relax/tighten steps must be non-negative")
    if not base_threshold <= config["max_threshold"] <= 1.0:
        raise ValueError(
            "adaptive_occlusion.max_threshold must be within "
            f"[{base_threshold}, 1], got {config['max_threshold']}"
        )
    return config


def _new_adaptive_occlusion_state(base_threshold: float) -> Dict:
    return {
        "base_threshold": float(base_threshold),
        "current_threshold": float(base_threshold),
        "occlusion_failure_streak": 0,
        "accepted_success_streak": 0,
        "relax_event_count": 0,
        "tighten_event_count": 0,
        "history": [],
    }


def update_adaptive_occlusion_state(
    state: Dict,
    *,
    accepted: bool,
    rejection_reasons: List[str],
    config: Dict,
    candidate_id: str,
) -> Dict | None:
    """Update one target's threshold after an online candidate decision."""
    if not bool(config.get("enabled", True)):
        state["occlusion_failure_streak"] = 0
        state["accepted_success_streak"] = 0
        return None

    reasons = list(rejection_reasons)
    event = None
    if accepted:
        state["occlusion_failure_streak"] = 0
        state["accepted_success_streak"] = int(state.get("accepted_success_streak", 0)) + 1
        if state["accepted_success_streak"] >= int(config["success_streak_to_tighten"]):
            old = float(state["current_threshold"])
            new = max(float(state["base_threshold"]), old - float(config["tighten_step"]))
            state["accepted_success_streak"] = 0
            if new < old - 1e-12:
                state["current_threshold"] = new
                state["tighten_event_count"] = int(state.get("tighten_event_count", 0)) + 1
                event = {
                    "type": "tighten",
                    "candidate_id": candidate_id,
                    "old_threshold": old,
                    "new_threshold": new,
                }
    else:
        state["accepted_success_streak"] = 0
        occlusion_only_failure = reasons == ["target_occlusion_reduction_above_threshold"]
        if occlusion_only_failure:
            state["occlusion_failure_streak"] = int(state.get("occlusion_failure_streak", 0)) + 1
            if state["occlusion_failure_streak"] >= int(config["failure_streak_to_relax"]):
                old = float(state["current_threshold"])
                new = min(float(config["max_threshold"]), old + float(config["relax_step"]))
                state["occlusion_failure_streak"] = 0
                if new > old + 1e-12:
                    state["current_threshold"] = new
                    state["relax_event_count"] = int(state.get("relax_event_count", 0)) + 1
                    event = {
                        "type": "relax",
                        "candidate_id": candidate_id,
                        "old_threshold": old,
                        "new_threshold": new,
                    }
        else:
            state["occlusion_failure_streak"] = 0
    if event is not None:
        state.setdefault("history", []).append(event)
    return event


def _save_json_atomic(data: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_live_progress(
    step: Path,
    context: Mapping[str, Any],
    accepted: Mapping[str, list[Dict]],
    attempts_by_target: Mapping[str, int],
    adaptive_states: Mapping[str, Dict],
    *,
    processed_candidate_count: int,
    isolated_rendered_candidate_count: int,
    full_scene_rendered_candidate_count: int,
    depth_gate_rejected_count: int,
    target_depth_distribution: Mapping[str, Any],
    last_candidate: Dict | None,
    status: str,
) -> None:
    target_progress = {}
    for object_id, target in context["targets"].items():
        target_progress[object_id] = {
            "requested": int(target["quota"]),
            "accepted": len(accepted.get(object_id, [])),
            "attempts": int(attempts_by_target.get(object_id, 0)),
            "accepted_camera_ids": [str(item["camera_id"]) for item in accepted.get(object_id, [])],
            "adaptive_occlusion": adaptive_states[object_id],
            "target_depth_distribution": dict(target_depth_distribution),
        }
    _save_json_atomic({
        "status": status,
        "processed_candidate_count": int(processed_candidate_count),
        "isolated_rendered_candidate_count": int(isolated_rendered_candidate_count),
        "full_scene_rendered_candidate_count": int(full_scene_rendered_candidate_count),
        "depth_gate_rejected_count": int(depth_gate_rejected_count),
        "full_scene_render_avoidance_count": int(depth_gate_rejected_count),
        "target_depth_distribution": dict(target_depth_distribution),
        "accepted_camera_count": int(sum(len(items) for items in accepted.values())),
        "target_progress": target_progress,
        "last_candidate": last_candidate,
    }, step / "accepted_progress.json")
    live_cameras = []
    for items in accepted.values():
        live_cameras.extend(items)
    _save_json_atomic({"cameras": live_cameras}, step / "cameras.accepted.live.json")


def _render_shared_buffers_batch(
    out: Path,
    cameras: List[Dict],
    output_dir: Path,
    config: Dict,
) -> Dict:
    blender = os.environ.get("BLENDER_BIN", "blender")
    script = Path("src/blender/prephysics_runtime/render_refinement_shared_buffers_batch.py")
    render_cfg = dict(config.get("candidate_render", {}))
    resolution = render_cfg.get("shared_buffer_resolution", [768, 432])
    samples = int(render_cfg.get("shared_buffer_samples", 1))
    camera_file = output_dir / "cameras.shared.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"cameras": cameras}, camera_file)
    cmd = [
        blender, "--background", "--python", str(script), "--",
        "--out", str(out),
        "--camera_file", str(camera_file),
        "--output_dir", str(output_dir),
        "--width", str(int(resolution[0])),
        "--height", str(int(resolution[1])),
        "--samples", str(samples),
    ]
    if not bool(render_cfg.get("render_selected_view_rgb", True)):
        raise RuntimeError("Selected-view RGB rendering is mandatory")
    subprocess.run(cmd, check=True)
    report_path = output_dir / "batch_report.json"
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError("Shared refinement buffer renderer did not produce batch_report.json")
    report = load_json(report_path)
    if int(report.get("success_count", 0)) <= 0:
        raise RuntimeError("Shared refinement buffer renderer produced no successful views")
    return report


def compute_step08_semantic_eligibility(stats: Dict, config: Dict) -> Dict:
    """Apply the cached all-facing frustum-ratio gate for a non-main semantic."""
    minimum_frustum_ratio = float(config.get("minimum_frustum_triangle_ratio", 0.50))
    ratio = float(stats.get("frustum_triangle_ratio", 0.0))
    reasons = []
    if ratio < minimum_frustum_ratio:
        reasons.append("frustum_triangle_ratio_below_threshold")
    return {
        "eligible_for_step08": not reasons,
        "rejection_reasons": reasons,
        "thresholds": {"minimum_frustum_triangle_ratio": minimum_frustum_ratio},
        "gate_metric": "all-facing frustum-intersecting triangle ratio",
        "includes_back_facing_triangles": True,
        "ignores_occlusion": True,
    }


def _manifest_index(manifest: Dict):
    triangles = {int(item["scene_triangle_id"]): item for item in manifest.get("triangles", [])}
    owners = dict(manifest.get("owners", {}))
    return triangles, owners


def _candidate_range_records(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = [dict(item) for item in manifest.get("scene_triangle_ranges", [])]
    if records:
        return sorted(records, key=lambda item: int(item["scene_triangle_start"]))
    # Backward compatibility for candidate manifests that predate the current schema.
    triangles = sorted(manifest.get("triangles", []), key=lambda item: int(item["scene_triangle_id"]))
    return [
        {
            "scene_triangle_start": int(item["scene_triangle_id"]),
            "scene_triangle_end_exclusive": int(item["scene_triangle_id"]) + 1,
            "semantic_owner_id": str(item["semantic_owner_id"]),
            "owner_triangle_start": int(item["owner_triangle_id"]),
            "mesh_object_name": item.get("mesh_object_name"),
            "world_area": float(item.get("world_area", 0.0)),
            "uv_area_normalized": float(item.get("uv_area_normalized", 0.0)),
        }
        for item in triangles
    ]


def _range_lookup(manifest: Mapping[str, Any]):
    records = _candidate_range_records(manifest)
    starts = np.asarray([int(item["scene_triangle_start"]) for item in records], dtype=np.int64)
    ends = np.asarray([int(item["scene_triangle_end_exclusive"]) for item in records], dtype=np.int64)
    return records, starts, ends


def _metadata_for_scene_ids(manifest: Mapping[str, Any], scene_ids: np.ndarray) -> List[Dict[str, Any] | None]:
    ids = np.asarray(scene_ids, dtype=np.int64)
    records, starts, ends = _range_lookup(manifest)
    if not records or ids.size == 0:
        return [None] * int(ids.size)
    positions = np.searchsorted(starts, ids, side="right") - 1
    result: List[Dict[str, Any] | None] = []
    for scene_id, position in zip(ids.tolist(), positions.tolist()):
        if position < 0 or scene_id >= int(ends[position]):
            result.append(None)
            continue
        record = records[position]
        result.append({
            "scene_triangle_id": int(scene_id),
            "semantic_owner_id": str(record["semantic_owner_id"]),
            "owner_triangle_id": int(record["owner_triangle_start"]) + int(scene_id) - int(record["scene_triangle_start"]),
            "mesh_object_name": record.get("mesh_object_name"),
            "world_area": float(record.get("world_area", 0.0)),
            "uv_area_normalized": float(record.get("uv_area_normalized", 0.0)),
        })
    return result


def _semantic_mask_from_manifest(
    triangle_id_map: np.ndarray,
    manifest: Mapping[str, Any],
    owner_id: str,
) -> np.ndarray:
    mask = np.zeros(triangle_id_map.shape, dtype=bool)
    owner = str(owner_id)
    records = dict(manifest.get("owners", {})).get(owner, {}).get("scene_triangle_ranges")
    if not records:
        records = [
            item for item in _candidate_range_records(manifest)
            if str(item.get("semantic_owner_id", "")) == owner
        ]
    for record in records or []:
        start = int(record["scene_triangle_start"])
        end = int(record["scene_triangle_end_exclusive"])
        mask |= (triangle_id_map >= start) & (triangle_id_map < end)
    return mask


def evaluate_isolated_target_depth(
    camera: Mapping[str, Any],
    directory: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    render_report: Mapping[str, Any],
) -> Dict[str, Any]:
    target_owner = str(camera.get("target_object_id") or camera.get("target_source") or "")
    isolated_map, id_diagnostics = load_triangle_id_map(
        render_report["target_isolated_triangle_id"],
        valid_triangle_count=int(manifest.get("triangle_count", 0)),
        return_diagnostics=True,
    )
    target_mask = _semantic_mask_from_manifest(isolated_map, manifest, target_owner)
    depth = decode_linear_target_depth(
        render_report["target_isolated_depth"],
        dict(render_report["target_isolated_depth_encoding"]),
    )
    result = evaluate_target_depth_distribution(
        linear_depth=depth,
        target_mask=target_mask,
        target_bounding_sphere_radius=float(camera["target_bounding_sphere_radius"]),
        config=dict(config.get("target_depth_distribution", {})),
    )
    result.update({
        "target_object_id": target_owner,
        "target_isolated_triangle_id": str(render_report["target_isolated_triangle_id"]),
        "target_isolated_depth": str(render_report["target_isolated_depth"]),
        "target_isolated_depth_encoding": dict(render_report["target_isolated_depth_encoding"]),
        "target_isolated_triangle_id_diagnostics": id_diagnostics,
    })
    save_json(result, Path(directory) / "target_depth_distribution.json")
    return result


def evaluate_target_occlusion_reduction(
    isolated_screen_pixels: int,
    visible_screen_pixels: int,
    *,
    maximum_reduction_ratio: float = 0.40,
) -> Dict:
    """Compare target area before and after occlusion by other semantic bodies."""
    threshold = float(maximum_reduction_ratio)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "maximum_target_occlusion_reduction_ratio must be within [0, 1], "
            f"got {threshold}"
        )
    isolated_pixels = max(int(isolated_screen_pixels), 0)
    visible_pixels = max(int(visible_screen_pixels), 0)
    if isolated_pixels <= 0:
        retention_ratio = 0.0
        reduction_ratio = 1.0
    else:
        retention_ratio = min(max(float(visible_pixels / isolated_pixels), 0.0), 1.0)
        reduction_ratio = min(max(1.0 - retention_ratio, 0.0), 1.0)
    return {
        "isolated_screen_pixels": isolated_pixels,
        "visible_screen_pixels": visible_pixels,
        "occluded_pixel_count": max(isolated_pixels - visible_pixels, 0),
        "visible_retention_ratio": retention_ratio,
        "occlusion_reduction_ratio": reduction_ratio,
        "maximum_occlusion_reduction_ratio": threshold,
        "accepted": bool(isolated_pixels > 0 and reduction_ratio < threshold),
    }


def _semantic_pixel_count(triangle_id_map: np.ndarray, triangle_index: Dict[int, Dict], owner_id: str) -> int:
    scene_ids, counts = np.unique(triangle_id_map[triangle_id_map >= 0], return_counts=True)
    return int(sum(
        int(pixels)
        for scene_id, pixels in zip(scene_ids.tolist(), counts.tolist())
        if str(dict(triangle_index.get(int(scene_id), {})).get("semantic_owner_id", "")) == str(owner_id)
    ))


def evaluate_candidate(
    out: Path,
    camera: Dict,
    directory: Path,
    config: Dict,
    manifest: Dict,
    *,
    target_depth_distribution: Dict | None = None,
) -> Dict:
    """Evaluate one rendered candidate using the user's three hard gates.

    Gates: target pixels exist in isolated and full renders; isolated-to-full
    occlusion loss is below the adaptive threshold; and the fraction of isolated
    target pixels outside the configured normalized depth interval is acceptable.
    """
    owners = dict(manifest.get("owners", {}))
    valid_triangle_count = int(manifest.get("triangle_count", 0))
    triangle_id_map, id_diagnostics = load_triangle_id_map(
        directory / "triangle_id.png",
        valid_triangle_count=valid_triangle_count,
        return_diagnostics=True,
    )
    isolated_triangle_id_map, isolated_id_diagnostics = load_triangle_id_map(
        directory / "target_isolated_triangle_id.png",
        valid_triangle_count=valid_triangle_count,
        return_diagnostics=True,
    )
    if isolated_triangle_id_map.shape != triangle_id_map.shape:
        raise RuntimeError(
            "Target-isolated Triangle ID resolution does not match the full-scene render: "
            f"{isolated_triangle_id_map.shape} != {triangle_id_map.shape}"
        )
    height, width = triangle_id_map.shape
    frame_pixels = int(width * height)

    scene_ids, counts = np.unique(triangle_id_map[triangle_id_map >= 0], return_counts=True)
    metadata_records = _metadata_for_scene_ids(manifest, scene_ids)
    grouped: Dict[str, Dict[int, Dict]] = {}
    object_pixels: Dict[str, int] = {}
    for scene_id, pixels, metadata in zip(scene_ids.tolist(), counts.tolist(), metadata_records):
        if metadata is None:
            continue
        owner = str(metadata["semantic_owner_id"])
        owner_triangle_id = int(metadata["owner_triangle_id"])
        object_pixels[owner] = object_pixels.get(owner, 0) + int(pixels)
        records = grouped.setdefault(owner, {})
        record = records.setdefault(owner_triangle_id, {
            "global_triangle_id": owner_triangle_id,
            "mesh_object_name": metadata.get("mesh_object_name"),
            "visible_pixels": 0,
            "projected_pixels": 0,
            "frontality": 1.0,
            "world_area": float(metadata.get("world_area", 0.0)),
            "uv_area_normalized": float(metadata.get("uv_area_normalized", 0.0)),
        })
        record["visible_pixels"] += int(pixels)
        record["projected_pixels"] += int(pixels)

    visible_by_object: Dict[str, List[Dict]] = {
        owner: list(sorted(records.values(), key=lambda item: int(item["global_triangle_id"])))
        for owner, records in grouped.items()
    }
    object_stats: Dict[str, Dict[str, Any]] = {}
    for owner, records in visible_by_object.items():
        pixels = int(object_pixels.get(owner, 0))
        owner_meta = dict(owners.get(owner, {}))
        triangle_state_path = out / "05_texture_state" / owner / "triangle_seen.npy"
        triangle_count = int(len(np.load(triangle_state_path))) if triangle_state_path.exists() else int(
            owner_meta.get("triangle_count", 0)
        )
        visible_uv_area = float(sum(float(item.get("uv_area_normalized", 0.0)) for item in records))
        total_uv_area = float(owner_meta.get("uv_area_normalized", 0.0))
        object_stats[owner] = {
            "screen_pixels": pixels,
            "screen_ratio": float(pixels / max(frame_pixels, 1)),
            "visible_triangle_count": len(records),
            "triangle_count": triangle_count,
            "visible_triangle_ratio": float(len(records) / max(triangle_count, 1)),
            "visible_uv_area_normalized": visible_uv_area,
            "total_uv_area_normalized": total_uv_area,
            "surface_fraction": float(visible_uv_area / max(total_uv_area, 1e-12)) if total_uv_area > 0 else 0.0,
        }

    semantic_pixels = int((triangle_id_map >= 0).sum())
    target_owner = str(camera.get("target_object_id") or camera.get("target_source") or "")
    target_stats = dict(object_stats.get(target_owner, {}))
    target_screen_ratio = float(target_stats.get("screen_ratio", 0.0))
    target_visible_screen_pixels = int(target_stats.get("screen_pixels", 0))
    target_isolated_mask = _semantic_mask_from_manifest(isolated_triangle_id_map, manifest, target_owner)
    target_isolated_screen_pixels = int(np.count_nonzero(target_isolated_mask))
    target_projection_valid = bool(target_isolated_screen_pixels > 0 and target_visible_screen_pixels > 0)
    target_projection_failure_detail = []
    if target_isolated_screen_pixels <= 0:
        target_projection_failure_detail.append("not_projected_in_target_isolated_render")
    if target_visible_screen_pixels <= 0:
        target_projection_failure_detail.append("not_visible_in_full_scene")

    target_occlusion = evaluate_target_occlusion_reduction(
        target_isolated_screen_pixels,
        target_visible_screen_pixels,
        maximum_reduction_ratio=float(config.get("maximum_target_occlusion_reduction_ratio", 0.40)),
    )
    depth_gate = dict(target_depth_distribution or {})
    depth_available = bool(depth_gate)
    depth_valid = bool(depth_available and depth_gate.get("accepted", False))

    target_visible_triangle_ratio = float(target_stats.get("visible_triangle_ratio", 0.0))
    target_surface_fraction = float(target_stats.get("surface_fraction", 0.0))
    selection_visible = {target_owner: visible_by_object[target_owner]} if target_owner in visible_by_object else {}
    context = weighted_seen_ratio(selection_visible, out / "05_texture_state") if selection_visible else {
        "seen_ratio": 0.0,
        "new_surface_ratio": 0.0,
        "new_surface_weight": 0.0,
        "objects": {},
        "seen_pixels": 0,
        "visible_triangle_pixels": 0,
    }

    reasons: List[str] = []
    if not target_owner:
        reasons.append("missing_target_object_id")
    else:
        if not target_projection_valid:
            reasons.append("target_not_projected_or_visible")
        elif not bool(target_occlusion["accepted"]):
            reasons.append("target_occlusion_reduction_above_threshold")
        if not depth_available:
            reasons.append("target_depth_distribution_unavailable")
        elif not depth_valid:
            reasons.append("target_depth_out_of_good_range_fraction_above_threshold")
    if not bool(camera.get("target_valid", True)):
        reasons.append("camera_position_or_target_outside_room")
    if float(id_diagnostics.get("invalid_id_ratio", 0.0)) > float(config.get("max_invalid_id_ratio", 0.0005)):
        reasons.append("triangle_id_corruption")
    if float(isolated_id_diagnostics.get("invalid_id_ratio", 0.0)) > float(config.get("max_invalid_id_ratio", 0.0005)):
        reasons.append("target_isolated_triangle_id_corruption")

    validation_conditions = {
        "target_projected_and_visible": {
            "accepted": target_projection_valid,
            "target_isolated_screen_pixels": target_isolated_screen_pixels,
            "target_visible_screen_pixels": target_visible_screen_pixels,
            "failure_detail": target_projection_failure_detail,
        },
        "target_occlusion": {
            "accepted": bool(target_projection_valid and target_occlusion["accepted"]),
            "occlusion_reduction_ratio": target_occlusion["occlusion_reduction_ratio"],
            "maximum_occlusion_reduction_ratio": target_occlusion["maximum_occlusion_reduction_ratio"],
        },
        "target_depth_distribution": {
            **depth_gate,
            "accepted": depth_valid,
        },
    }

    return {
        "camera": camera,
        "valid_geometry": not reasons,
        "rejection_reasons": reasons,
        "validation_conditions": validation_conditions,
        "triangle_id_diagnostics": id_diagnostics,
        "target_isolated_triangle_id_diagnostics": isolated_id_diagnostics,
        "semantic_pixels": semantic_pixels,
        "frame_pixels": frame_pixels,
        "max_single_object_screen_ratio": max(
            (float(value.get("screen_ratio", 0.0)) for value in object_stats.values()), default=0.0
        ),
        "target_object_id": target_owner,
        "target_object_projection_and_visibility_valid": target_projection_valid,
        "target_object_projection_failure_detail": target_projection_failure_detail,
        "target_object_screen_ratio": target_screen_ratio,
        "target_object_visible_screen_pixels": target_visible_screen_pixels,
        "target_object_isolated_screen_pixels": target_isolated_screen_pixels,
        "target_object_isolated_screen_ratio": float(target_isolated_screen_pixels / max(frame_pixels, 1)),
        "target_object_occluded_pixel_count": target_occlusion["occluded_pixel_count"],
        "target_object_visible_retention_ratio": target_occlusion["visible_retention_ratio"],
        "target_object_occlusion_reduction_ratio": target_occlusion["occlusion_reduction_ratio"],
        "maximum_target_occlusion_reduction_ratio": target_occlusion["maximum_occlusion_reduction_ratio"],
        "target_object_occlusion_valid": bool(target_projection_valid and target_occlusion["accepted"]),
        "target_object_depth_distribution": depth_gate,
        "target_object_depth_distribution_valid": depth_valid,
        # Diagnostics only; neither field participates in acceptance.
        "target_object_visible_triangle_ratio": target_visible_triangle_ratio,
        "target_object_surface_fraction": target_surface_fraction,
        "visible_object_count": len(visible_by_object),
        "visible_objects": sorted(visible_by_object),
        "object_stats": object_stats,
        "initial_context": context,
        "new_triangle_gain": int(sum(v.get("new_triangle_count", 0) for v in context.get("objects", {}).values())),
        "new_surface_gain": float(context.get("new_surface_weight", 0.0)),
        "new_surface_gain_ratio": float(context.get("new_surface_ratio", 0.0)),
        "base_score": 0.0,
        "visible_triangles": visible_by_object,
        "render_directory": str(directory),
        "validation_policy": (
            "three semantic hard gates only: target projected-and-visible, adaptive isolated-versus-full "
            "occlusion, and JSON-configured isolated-target normalized depth distribution; screen area and "
            "visible triangle ratio remain diagnostics only; no aggregate camera score"
        ),
    }


def _attach_shared_buffers_and_eligibility(
    out: Path,
    evaluation: Dict,
    camera_report: Dict,
    shared_manifest: Dict,
    eligibility_config: Dict,
) -> Dict:
    triangle_index, owner_manifest = _manifest_index(shared_manifest)
    triangle_id_map, diagnostics = load_triangle_id_map(
        camera_report["triangle_id"],
        valid_triangle_count=int(shared_manifest.get("triangle_count", len(triangle_index))),
        return_diagnostics=True,
    )
    height, width = triangle_id_map.shape
    frame_pixels = int(width * height)
    scene_ids, counts = np.unique(triangle_id_map[triangle_id_map >= 0], return_counts=True)
    grouped: Dict[str, Dict[int, Dict]] = {}
    object_pixels: Dict[str, int] = {}
    for scene_id, pixels in zip(scene_ids.tolist(), counts.tolist()):
        metadata = triangle_index.get(int(scene_id))
        if metadata is None:
            continue
        owner = str(metadata["semantic_owner_id"])
        owner_triangle_id = int(metadata["owner_triangle_id"])
        object_pixels[owner] = object_pixels.get(owner, 0) + int(pixels)
        records = grouped.setdefault(owner, {})
        record = records.setdefault(owner_triangle_id, {
            "global_triangle_id": owner_triangle_id,
            "render_triangle_id": int(scene_id),
            "mesh_object_name": metadata.get("mesh_object_name"),
            "visible_pixels": 0,
            "projected_pixels": 0,
            "frontality": 1.0,
            "world_area": float(metadata.get("world_area", 0.0)),
            "uv_area_normalized": float(metadata.get("uv_area_normalized", 0.0)),
        })
        record["visible_pixels"] += int(pixels)
        record["projected_pixels"] += int(pixels)

    visible_by_object: Dict[str, List[Dict]] = {
        owner: list(sorted(records.values(), key=lambda item: int(item["global_triangle_id"])))
        for owner, records in grouped.items()
    }
    owner_projection = dict(camera_report.get("owners", {}))
    semantic_stats: Dict[str, Dict[str, Any]] = {}
    eligible: List[str] = []
    for owner in sorted(set(owner_projection) | set(visible_by_object)):
        projection = dict(owner_projection.get(owner, {}))
        records = visible_by_object.get(owner, [])
        pixels = int(object_pixels.get(owner, 0))
        total = int(projection.get("total_triangle_count", dict(owner_manifest.get(owner, {})).get("triangle_count", 0)))
        visible_triangle_count = len(records)
        visible_uv_area = float(sum(float(item.get("uv_area_normalized", 0.0)) for item in records))
        total_uv_area = float(dict(owner_manifest.get(owner, {})).get("uv_area_normalized", 0.0))
        stats = {
            **projection,
            "screen_pixels": pixels,
            "screen_ratio": float(pixels / max(frame_pixels, 1)),
            "visible_triangle_count": visible_triangle_count,
            "visible_triangle_ratio": float(visible_triangle_count / max(total, 1)),
            "visible_uv_area_normalized": visible_uv_area,
            "total_uv_area_normalized": total_uv_area,
            "visible_uv_ratio": float(visible_uv_area / max(total_uv_area, 1e-12)),
        }
        gate = compute_step08_semantic_eligibility(stats, eligibility_config)
        stats.update(gate)
        semantic_stats[owner] = stats
        if gate["eligible_for_step08"]:
            eligible.append(owner)

    target_owner = str(evaluation.get("camera", {}).get("target_object_id") or "")
    if target_owner in semantic_stats and int(semantic_stats[target_owner].get("screen_pixels", 0)) > 0:
        semantic_stats[target_owner]["main_target_forced_for_step08"] = True
        if target_owner not in eligible:
            eligible.append(target_owner)
    eligible.sort(key=lambda owner: (int(owner == target_owner), float(semantic_stats[owner].get("frustum_triangle_ratio", 0.0))), reverse=True)

    selection_visible = {target_owner: visible_by_object[target_owner]} if target_owner in visible_by_object else {}
    context = weighted_seen_ratio(selection_visible, out / "05_texture_state") if selection_visible else {
        "seen_ratio": 0.0,
        "new_surface_ratio": 0.0,
        "new_surface_weight": 0.0,
        "objects": {},
        "seen_pixels": 0,
        "visible_triangle_pixels": 0,
    }
    frustum_ratio_file = camera_report.get("semantic_frustum_ratios")
    if not frustum_ratio_file or not Path(frustum_ratio_file).exists():
        raise RuntimeError(f"Shared buffer report lacks semantic_frustum_ratios.json for {evaluation['camera']['camera_id']}")
    evaluation["semantic_frustum_ratios_file"] = str(frustum_ratio_file)
    evaluation["shared_buffers"] = {
        "transport": camera_report.get("buffer_transport", "standard_images_and_json"),
        "depth_encoding": dict(camera_report.get("depth_encoding", {})),
        "resolution": list(camera_report.get("resolution", [width, height])),
        "albedo": camera_report.get("albedo"),
        "depth": camera_report["depth"],
        "semantic": camera_report["semantic"],
        "palette": camera_report["palette"],
        "uv": camera_report["uv"],
        "triangle_id": camera_report["triangle_id"],
        "owner_manifests": {
            owner: stats.get("manifest_path")
            for owner, stats in semantic_stats.items()
            if stats.get("manifest_path")
        },
        "semantic_frustum_ratios": str(frustum_ratio_file),
    }
    evaluation["shared_triangle_id_diagnostics"] = diagnostics
    evaluation["semantic_visibility"] = semantic_stats
    evaluation["step08_eligible_semantics"] = eligible
    evaluation["main_target_object_id"] = target_owner
    evaluation["visible_triangles"] = visible_by_object
    evaluation["visible_objects"] = sorted(visible_by_object)
    evaluation["visible_object_count"] = len(visible_by_object)
    evaluation["frame_pixels"] = frame_pixels
    evaluation["semantic_pixels"] = int(sum(object_pixels.values()))
    evaluation["max_single_object_screen_ratio"] = max(
        (float(stats.get("screen_ratio", 0.0)) for stats in semantic_stats.values()), default=0.0
    )
    evaluation["initial_context"] = context
    evaluation["new_triangle_gain"] = int(sum(v.get("new_triangle_count", 0) for v in context.get("objects", {}).values()))
    evaluation["new_surface_gain"] = float(context.get("new_surface_weight", 0.0))
    evaluation["new_surface_gain_ratio"] = float(context.get("new_surface_ratio", 0.0))
    evaluation["object_stats"] = semantic_stats
    return evaluation


def _accepted_counts(accepted: Mapping[str, list[Dict]]) -> Dict[str, int]:
    return {object_id: len(items) for object_id, items in accepted.items()}


def _all_quotas_met(context: Mapping[str, Any], accepted: Mapping[str, list[Dict]]) -> bool:
    return all(len(accepted.get(object_id, [])) >= int(target["quota"]) for object_id, target in context["targets"].items())


def prepare_interior_probability_cameras(
    out: str | Path,
    scene: Mapping[str, Any],
    camera_config: Dict,
    refinement_config: Dict,
) -> Dict:
    """Online Stage07 sampling: render, evaluate, and accept one candidate at a time."""
    out = Path(out)
    step = out / "07_refinement_cameras"
    step.mkdir(parents=True, exist_ok=True)
    context = build_sampling_context(out, scene, camera_config)
    sampling_cfg = dict(context["config"])
    validation_cfg = dict(refinement_config.get("camera_candidate_validation", {}))
    # Candidate acceptance keeps exactly three hard gates. Legacy screen-area,
    # visible-triangle, and target-frustum keys are deliberately ignored.
    for removed_key in (
        "minimum_target_screen_ratio",
        "maximum_target_screen_ratio",
        "minimum_target_visible_triangle_ratio",
        "minimum_target_frustum_triangle_ratio",
        "frustum_triangle_chunk_size",
        "frustum_clip_epsilon",
    ):
        validation_cfg.pop(removed_key, None)
    target_depth_cfg = normalise_target_depth_config(
        validation_cfg.get("target_depth_distribution", {})
    )
    validation_cfg["target_depth_distribution"] = target_depth_cfg
    base_occlusion_threshold = float(
        validation_cfg.get("maximum_target_occlusion_reduction_ratio", 0.40)
    )
    if not 0.0 <= base_occlusion_threshold <= 1.0:
        raise ValueError(
            "camera_candidate_validation.maximum_target_occlusion_reduction_ratio "
            f"must be within [0, 1], got {base_occlusion_threshold}"
        )
    validation_cfg["maximum_target_occlusion_reduction_ratio"] = base_occlusion_threshold
    adaptive_cfg = _adaptive_occlusion_config(validation_cfg, base_occlusion_threshold)
    validation_cfg["adaptive_occlusion"] = adaptive_cfg

    rng = np.random.default_rng(int(sampling_cfg.get("seed", 7301)))
    require_full_quota = bool(sampling_cfg.get("require_full_quota", True))
    max_empty_sampling_cycles = max(
        1,
        int(sampling_cfg.get("max_empty_sampling_cycles", sampling_cfg.get("max_sampling_rounds", 32))),
    )

    accepted: Dict[str, list[Dict]] = {object_id: [] for object_id in context["targets"]}
    accepted_evaluations: Dict[str, Dict] = {}
    accepted_evaluation_paths: Dict[str, str] = {}
    accepted_directions: Dict[str, list[list[float]]] = {object_id: [] for object_id in context["targets"]}
    attempts_by_target = {object_id: 0 for object_id in context["targets"]}
    rejection_counts_by_target = {object_id: {} for object_id in context["targets"]}
    adaptive_states = {
        object_id: _new_adaptive_occlusion_state(base_occlusion_threshold)
        for object_id in context["targets"]
    }
    sampling_events = []
    isolated_rendered_candidate_count = 0
    full_scene_rendered_candidate_count = 0
    depth_gate_rejected_count = 0
    processed_candidate_count = 0
    sampling_sequence = 0
    empty_sampling_cycles = 0
    last_candidate = None
    candidate_root = step / "candidate_stream"
    candidate_root.mkdir(parents=True, exist_ok=True)
    worker_root = step / "candidate_renderer"

    single_context = dict(context)
    single_sampling_cfg = dict(sampling_cfg)
    single_sampling_cfg["candidate_batch_size"] = 1
    single_context["config"] = single_sampling_cfg

    _write_live_progress(
        step,
        context,
        accepted,
        attempts_by_target,
        adaptive_states,
        processed_candidate_count=0,
        isolated_rendered_candidate_count=0,
        full_scene_rendered_candidate_count=0,
        depth_gate_rejected_count=0,
        target_depth_distribution=target_depth_cfg,
        last_candidate=None,
        status="sampling",
    )

    with _CandidateRenderWorker(out, worker_root, refinement_config) as renderer:
        manifest = load_json(renderer.triangle_owner_manifest)
        while not _all_quotas_met(context, accepted):
            sampled = sample_candidate_batch(
                single_context,
                rng,
                accepted_directions,
                attempts_by_target,
                round_index=sampling_sequence,
            )
            sampling_sequence += 1
            cameras = list(sampled["cameras"])
            if not cameras:
                empty_sampling_cycles += 1
                sampling_events.append({
                    "sequence": sampling_sequence - 1,
                    "status": "no_candidate_sampled",
                    "sampling_draw_count": int(sampled["draw_count"]),
                    "sampling_rejection_counts": sampled["rejection_counts"],
                    "accepted_after": _accepted_counts(accepted),
                    "attempts_by_target": dict(attempts_by_target),
                })
                if empty_sampling_cycles >= max_empty_sampling_cycles:
                    break
                continue
            empty_sampling_cycles = 0
            camera = cameras[0]
            camera_id = str(camera["camera_id"])
            target_id = str(camera["target_object_id"])
            target_record = dict(context["targets"][target_id])
            camera["target_bounding_sphere_radius"] = bounding_sphere_radius_from_aabb(
                target_record["aabb"]
            )
            camera["target_depth_valid_min_gray"] = int(
                dict(refinement_config.get("candidate_render", {})).get("target_depth_valid_min_gray", 24)
            )
            directory = candidate_root / f"{processed_candidate_count:05d}_{camera_id}"
            save_json({"camera": camera}, directory / "camera.json")

            threshold_used = float(adaptive_states[target_id]["current_threshold"])
            candidate_validation_cfg = dict(validation_cfg)
            candidate_validation_cfg["maximum_target_occlusion_reduction_ratio"] = threshold_used
            processed_candidate_count += 1
            depth_result: Dict[str, Any] = {}

            isolated_report = renderer.render_isolated(camera, directory)
            if isolated_report.get("status") != "ok":
                evaluation = {
                    "camera": camera,
                    "valid_geometry": False,
                    "rejection_reasons": ["candidate_isolated_render_failed"],
                    "error": isolated_report.get("error"),
                    "full_scene_render_skipped": True,
                    "full_scene_render_skipped_reason": "candidate_isolated_render_failed",
                }
            else:
                isolated_rendered_candidate_count += 1
                try:
                    depth_result = evaluate_isolated_target_depth(
                        camera,
                        directory,
                        candidate_validation_cfg,
                        manifest,
                        isolated_report,
                    )
                except Exception:
                    detail = traceback.format_exc()
                    (directory / ".target_depth_evaluation_failed").write_text(detail, encoding="utf-8")
                    evaluation = {
                        "camera": camera,
                        "valid_geometry": False,
                        "rejection_reasons": ["target_depth_distribution_evaluation_failed"],
                        "error": detail[-4000:],
                        "full_scene_render_skipped": True,
                        "full_scene_render_skipped_reason": "target_depth_distribution_evaluation_failed",
                    }
                else:
                    isolated_invalid = float(
                        dict(depth_result.get("target_isolated_triangle_id_diagnostics", {})).get(
                            "invalid_id_ratio", 0.0
                        )
                    )
                    if isolated_invalid > float(candidate_validation_cfg.get("max_invalid_id_ratio", 0.0005)):
                        evaluation = {
                            "camera": camera,
                            "valid_geometry": False,
                            "rejection_reasons": ["target_isolated_triangle_id_corruption"],
                            "target_object_depth_distribution": depth_result,
                            "validation_conditions": {
                                "target_projected_and_visible": {
                                    "accepted": None,
                                    "not_evaluated_reason": "isolated_triangle_id_corruption",
                                },
                                "target_occlusion": {
                                    "accepted": None,
                                    "not_evaluated_reason": "isolated_triangle_id_corruption",
                                },
                                "target_depth_distribution": depth_result,
                            },
                            "full_scene_render_skipped": True,
                            "full_scene_render_skipped_reason": "isolated_triangle_id_corruption",
                        }
                    elif not bool(depth_result.get("accepted", False)):
                        depth_gate_rejected_count += 1
                        evaluation = {
                            "camera": camera,
                            "valid_geometry": False,
                            "rejection_reasons": [str(depth_result.get("reason") or "target_depth_distribution_rejected")],
                            "target_object_id": target_id,
                            "target_object_isolated_screen_pixels": int(depth_result.get("target_pixel_count", 0)),
                            "target_object_depth_distribution": depth_result,
                            "target_object_depth_distribution_valid": False,
                            "validation_conditions": {
                                "target_projected_and_visible": {
                                    "accepted": None,
                                    "target_isolated_screen_pixels": int(depth_result.get("target_pixel_count", 0)),
                                    "target_visible_screen_pixels": None,
                                    "not_evaluated_reason": "depth_gate_failed_before_full_scene_render",
                                },
                                "target_occlusion": {
                                    "accepted": None,
                                    "not_evaluated_reason": "depth_gate_failed_before_full_scene_render",
                                },
                                "target_depth_distribution": depth_result,
                            },
                            "full_scene_render_skipped": True,
                            "full_scene_render_skipped_reason": "target_depth_distribution_rejected",
                            "validation_policy": (
                                "isolated target depth distribution failed before the full-scene Triangle ID render; "
                                "projection/visibility and occlusion were therefore not evaluated"
                            ),
                        }
                    else:
                        full_report = renderer.render_full(camera, directory)
                        if full_report.get("status") == "ok":
                            full_scene_rendered_candidate_count += 1
                            try:
                                evaluation = evaluate_candidate(
                                    out,
                                    camera,
                                    directory,
                                    candidate_validation_cfg,
                                    manifest,
                                    target_depth_distribution=depth_result,
                                )
                                evaluation["full_scene_render_skipped"] = False
                            except Exception:
                                detail = traceback.format_exc()
                                (directory / ".evaluation_failed").write_text(detail, encoding="utf-8")
                                evaluation = {
                                    "camera": camera,
                                    "valid_geometry": False,
                                    "rejection_reasons": ["triangle_id_evaluation_failed"],
                                    "error": detail[-4000:],
                                    "target_object_depth_distribution": depth_result,
                                    "full_scene_render_skipped": False,
                                }
                        else:
                            evaluation = {
                                "camera": camera,
                                "valid_geometry": False,
                                "rejection_reasons": ["candidate_full_scene_render_failed"],
                                "error": full_report.get("error"),
                                "target_object_depth_distribution": depth_result,
                                "full_scene_render_skipped": False,
                            }

            reasons = list(evaluation.get("rejection_reasons", []))
            if not bool(camera.get("target_valid", True)) and "camera_position_or_target_outside_room" not in reasons:
                reasons.append("camera_position_or_target_outside_room")
            quota = int(context["targets"][target_id]["quota"])
            direction = list(camera.get("view_direction_from_target", []))
            minimum_separation = float(camera.get("minimum_view_separation_deg", 0.0))
            if evaluation.get("valid_geometry") and accepted_directions[target_id] and direction:
                if min(
                    angular_separation_degrees(direction, existing)
                    for existing in accepted_directions[target_id]
                ) < minimum_separation:
                    reasons.append("view_direction_too_similar_after_acceptance")
            evaluation["rejection_reasons"] = reasons
            evaluation["valid_geometry"] = bool(evaluation.get("valid_geometry")) and not reasons
            evaluation["accepted_for_target_quota"] = bool(evaluation["valid_geometry"])
            evaluation["adaptive_occlusion_threshold_before_decision"] = threshold_used

            if evaluation["valid_geometry"]:
                accepted[target_id].append(camera)
                accepted_directions[target_id].append(direction)
                accepted_evaluations[camera_id] = evaluation
                accepted_evaluation_paths[camera_id] = str(directory / "evaluation.json")
            else:
                counts = rejection_counts_by_target[target_id]
                for reason in reasons or ["unknown_rejection"]:
                    counts[reason] = int(counts.get(reason, 0)) + 1

            adaptive_event = update_adaptive_occlusion_state(
                adaptive_states[target_id],
                accepted=bool(evaluation["valid_geometry"]),
                rejection_reasons=reasons,
                config=adaptive_cfg,
                candidate_id=camera_id,
            )
            evaluation["adaptive_occlusion_event"] = adaptive_event
            evaluation["adaptive_occlusion_state_after_decision"] = dict(adaptive_states[target_id])
            evaluation_path = directory / "evaluation.json"
            save_json(evaluation, evaluation_path)
            if evaluation["valid_geometry"]:
                accepted_evaluation_paths[camera_id] = str(evaluation_path)

            occlusion_value = evaluation.get("target_object_occlusion_reduction_ratio")
            occlusion_text = "n/a" if occlusion_value is None else f"{float(occlusion_value):.3f}"
            depth_summary = dict(evaluation.get("target_object_depth_distribution", depth_result))
            bad_depth_value = depth_summary.get("bad_depth_fraction")
            bad_depth_text = "n/a" if bad_depth_value is None else f"{float(bad_depth_value):.3f}"
            maximum_bad_depth = float(target_depth_cfg["maximum_bad_depth_fraction"])
            visible_pixels = evaluation.get("target_object_visible_screen_pixels")
            isolated_pixels = evaluation.get("target_object_isolated_screen_pixels")
            projection_text = (
                "n/a"
                if visible_pixels is None or isolated_pixels is None
                else f"{int(visible_pixels)}/{int(isolated_pixels)}px"
            )
            if evaluation["valid_geometry"]:
                print(
                    f"[07][ACCEPT] {camera_id} target={target_id} "
                    f"quota={len(accepted[target_id])}/{quota} "
                    f"projected_visible={projection_text} "
                    f"bad_depth={bad_depth_text}<={maximum_bad_depth:.3f} "
                    f"occlusion={occlusion_text}<{threshold_used:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"[07][REJECT] {camera_id} target={target_id} "
                    f"reasons={','.join(reasons) or 'unknown'} "
                    f"bad_depth={bad_depth_text}/{maximum_bad_depth:.3f} "
                    f"occlusion={occlusion_text}/{threshold_used:.3f} "
                    f"full_render_skipped={bool(evaluation.get('full_scene_render_skipped', False))}",
                    flush=True,
                )
            if adaptive_event is not None:
                print(
                    f"[07][{adaptive_event['type'].upper()}] target={target_id} "
                    f"occlusion_threshold={adaptive_event['old_threshold']:.3f}"
                    f"->{adaptive_event['new_threshold']:.3f}",
                    flush=True,
                )
            progress_text = " ".join(
                f"{object_id}={len(accepted[object_id])}/{int(target['quota'])}"
                for object_id, target in context["targets"].items()
            )
            print(f"[07][PROGRESS] {progress_text}", flush=True)

            last_candidate = {
                "camera_id": camera_id,
                "target_object_id": target_id,
                "accepted": bool(evaluation["valid_geometry"]),
                "rejection_reasons": reasons,
                "target_object_occlusion_reduction_ratio": evaluation.get(
                    "target_object_occlusion_reduction_ratio"
                ),
                "occlusion_threshold_used": threshold_used,
                "target_object_depth_distribution": depth_summary,
                "full_scene_render_skipped": bool(evaluation.get("full_scene_render_skipped", False)),
                "adaptive_event": adaptive_event,
                "evaluation_file": str(evaluation_path),
            }
            _write_live_progress(
                step,
                context,
                accepted,
                attempts_by_target,
                adaptive_states,
                processed_candidate_count=processed_candidate_count,
                isolated_rendered_candidate_count=isolated_rendered_candidate_count,
                full_scene_rendered_candidate_count=full_scene_rendered_candidate_count,
                depth_gate_rejected_count=depth_gate_rejected_count,
                target_depth_distribution=target_depth_cfg,
                last_candidate=last_candidate,
                status="sampling" if not _all_quotas_met(context, accepted) else "camera_quota_complete",
            )
            sampling_events.append({
                "sequence": sampling_sequence - 1,
                "camera_id": camera_id,
                "target_object_id": target_id,
                "status": "accepted" if evaluation["valid_geometry"] else "rejected",
                "sampling_draw_count": int(sampled["draw_count"]),
                "sampling_rejection_counts": sampled["rejection_counts"],
                "rejection_reasons": reasons,
                "accepted_after": _accepted_counts(accepted),
                "attempts_by_target": dict(attempts_by_target),
                "occlusion_threshold_used": threshold_used,
                "target_object_depth_distribution": depth_summary,
                "full_scene_render_skipped": bool(evaluation.get("full_scene_render_skipped", False)),
                "adaptive_occlusion_event": adaptive_event,
            })


    quotas = {
        object_id: {
            "requested": int(target["quota"]),
            "accepted": len(accepted[object_id]),
            "attempts": int(attempts_by_target[object_id]),
            "complete": len(accepted[object_id]) >= int(target["quota"]),
            "aabb": target["aabb"],
            "aabb_source": target["aabb_source"],
            "rejection_counts": rejection_counts_by_target[object_id],
            "adaptive_occlusion": adaptive_states[object_id],
        }
        for object_id, target in context["targets"].items()
    }
    quota_complete = all(item["complete"] for item in quotas.values())
    _write_live_progress(
        step,
        context,
        accepted,
        attempts_by_target,
        adaptive_states,
        processed_candidate_count=processed_candidate_count,
        isolated_rendered_candidate_count=isolated_rendered_candidate_count,
        full_scene_rendered_candidate_count=full_scene_rendered_candidate_count,
        depth_gate_rejected_count=depth_gate_rejected_count,
        target_depth_distribution=target_depth_cfg,
        last_candidate=last_candidate,
        status="camera_quota_complete" if quota_complete else "incomplete_camera_quota",
    )
    if not quota_complete:
        failure_report = {
            "status": "incomplete_camera_quota",
            "camera_generation_policy": "online room-interior AABB-distance probability sampling with isolated-target depth-distribution validation",
            "processed_candidate_count": processed_candidate_count,
            "isolated_rendered_candidate_count": isolated_rendered_candidate_count,
            "full_scene_rendered_candidate_count": full_scene_rendered_candidate_count,
            "depth_gate_rejected_count": depth_gate_rejected_count,
            "full_scene_render_avoidance_count": depth_gate_rejected_count,
            "camera_candidate_validation": validation_cfg,
            "room_interior_aabb": context["room_interior_aabb"],
            "target_quotas": quotas,
            "sampling_events": sampling_events,
            "progress_file": str(step / "accepted_progress.json"),
        }
        save_json(failure_report, step / "sampling_failure_report.json")
        if require_full_quota:
            missing = ", ".join(
                f"{object_id}:{item['accepted']}/{item['requested']}"
                for object_id, item in quotas.items() if not item["complete"]
            )
            raise RuntimeError(f"Stage07 could not satisfy all per-object camera quotas: {missing}")

    accepted_cameras: List[Dict] = []
    for object_id in sorted(accepted):
        for camera in accepted[object_id]:
            camera = dict(camera)
            camera["deterministic_order"] = len(accepted_cameras)
            accepted_cameras.append(camera)
            evaluation = accepted_evaluations[camera["camera_id"]]
            evaluation["camera"] = camera
            save_json(evaluation, accepted_evaluation_paths[camera["camera_id"]])
    if not accepted_cameras:
        raise RuntimeError("No room-interior refinement camera passed the three Stage07 semantic gates")

    final_camera_file = step / "cameras.candidates.json"
    save_json({"cameras": accepted_cameras}, final_camera_file)
    shared_root = step / "shared_buffers"
    shared_report = _render_shared_buffers_batch(out, accepted_cameras, shared_root, refinement_config)
    shared_manifest = load_json(shared_report["triangle_owner_manifest"])
    shared_results = {
        str(item["camera_id"]): item
        for item in shared_report.get("results", [])
        if item.get("status") == "ok"
    }
    eligibility_config = dict(refinement_config.get("semantic_eligibility", {}))
    final_evaluation_files: List[str] = []
    final_summaries = []
    for camera in accepted_cameras:
        camera_id = str(camera["camera_id"])
        evaluation_path = Path(accepted_evaluation_paths[camera_id])
        evaluation = load_json(evaluation_path)
        camera_report = shared_results.get(camera_id)
        if camera_report is None:
            raise RuntimeError(f"Shared geometry buffer render failed for accepted camera {camera_id}")
        evaluation = _attach_shared_buffers_and_eligibility(
            out, evaluation, camera_report, shared_manifest, eligibility_config
        )
        save_json(evaluation, evaluation_path)
        final_evaluation_files.append(str(evaluation_path))
        final_summaries.append({
            "camera_id": camera_id,
            "target_object_id": camera["target_object_id"],
            "deterministic_order": camera["deterministic_order"],
            "target_object_visible_screen_pixels": evaluation.get("target_object_visible_screen_pixels", 0),
            "target_object_isolated_screen_pixels": evaluation.get("target_object_isolated_screen_pixels", 0),
            "target_object_projection_and_visibility_valid": evaluation.get(
                "target_object_projection_and_visibility_valid", False
            ),
            "target_object_depth_distribution": evaluation.get(
                "target_object_depth_distribution", {}
            ),
            "target_object_depth_distribution_valid": evaluation.get(
                "target_object_depth_distribution_valid", False
            ),
            "target_object_visible_retention_ratio": evaluation.get("target_object_visible_retention_ratio", 0.0),
            "target_object_occlusion_reduction_ratio": evaluation.get("target_object_occlusion_reduction_ratio", 1.0),
            "maximum_target_occlusion_reduction_ratio": evaluation.get("maximum_target_occlusion_reduction_ratio", 0.40),
            "adaptive_occlusion_threshold_before_decision": evaluation.get(
                "adaptive_occlusion_threshold_before_decision", base_occlusion_threshold
            ),
            "step08_eligible_semantics": evaluation.get("step08_eligible_semantics", []),
            "evaluation_file": str(evaluation_path),
        })

    report = {
        "status": "ok" if quota_complete else "partial",
        "candidate_count": processed_candidate_count,
        "processed_candidate_count": processed_candidate_count,
        "isolated_rendered_candidate_count": isolated_rendered_candidate_count,
        "full_scene_rendered_candidate_count": full_scene_rendered_candidate_count,
        "depth_gate_rejected_count": depth_gate_rejected_count,
        "full_scene_render_avoidance_count": depth_gate_rejected_count,
        "accepted_camera_count": len(accepted_cameras),
        "valid_geometry_count": len(accepted_cameras),
        "cameras_file": str(final_camera_file),
        "live_cameras_file": str(step / "cameras.accepted.live.json"),
        "progress_file": str(step / "accepted_progress.json"),
        "evaluation_files": final_evaluation_files,
        "candidate_summaries": final_summaries,
        "room_interior_aabb": context["room_interior_aabb"],
        "target_quotas": quotas,
        "quota_complete": quota_complete,
        "sampling_events": sampling_events,
        "candidate_triangle_owner_manifest": str(worker_root / "triangle_owner_manifest.json"),
        "shared_triangle_owner_manifest": shared_report["triangle_owner_manifest"],
        "shared_buffer_batch_report": str(shared_root / "batch_report.json"),
        "shared_buffer_resolution": shared_report.get("resolution"),
        "semantic_eligibility": eligibility_config,
        "camera_candidate_validation": validation_cfg,
        "camera_generation_policy": "online single-candidate room-interior AABB-distance probability sampling until every configured non-room target reaches k accepted views",
        "camera_filter_policy": (
            "exactly three semantic gates: target projected-and-visible, per-target adaptive "
            "isolated-versus-full occlusion, and JSON-configured isolated-target normalized depth distribution"
        ),
        "camera_ranking_policy": "none",
        "candidate_render_process_policy": (
            "one persistent Blender worker; emit isolated Triangle ID and 16-bit depth from one render, "
            "reject bad depth distributions before the full-scene Triangle ID render, and use compact "
            "scene-ID ranges instead of per-triangle world-geometry caches"
        ),
        "semantic_gate_policy": "main camera target always updates; other semantics require cached all-facing frustum triangle ratio >= configured threshold",
    }
    save_json(report, step / "stage_report.json")
    _write_live_progress(
        step,
        context,
        accepted,
        attempts_by_target,
        adaptive_states,
        processed_candidate_count=processed_candidate_count,
        isolated_rendered_candidate_count=isolated_rendered_candidate_count,
        full_scene_rendered_candidate_count=full_scene_rendered_candidate_count,
        depth_gate_rejected_count=depth_gate_rejected_count,
        target_depth_distribution=target_depth_cfg,
        last_candidate=last_candidate,
        status="complete",
    )
    return report


# Kept as an explicit failure to prevent old callers from silently using the removed policy.
def prepare_refinement_candidates(*args, **kwargs):
    raise RuntimeError(
        "prepare_refinement_candidates was removed in v25; use prepare_interior_probability_cameras"
    )
