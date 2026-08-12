from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import numpy as np
from PIL import Image


@dataclass
class TriangleCoverageConfig:
    threshold_init: float = 0.35
    threshold_min: float = 0.10
    threshold_max: float = 0.80
    threshold_growth_k: float = 1.12
    seen_upper_bound: float = 0.96
    max_single_object_screen_ratio: float = 0.62
    min_object_visible_fraction: float = 0.012
    min_visible_semantic_pixels: int = 3000


class DynamicThreshold:
    def __init__(self, cfg: TriangleCoverageConfig):
        self.cfg = cfg
        self.threshold = float(cfg.threshold_init)

    def reject_low_context(self) -> Dict:
        old = self.threshold
        self.threshold = max(self.cfg.threshold_min, self.threshold / self.cfg.threshold_growth_k)
        return {"old_threshold": old, "new_threshold": self.threshold}

    def accept_success(self) -> Dict:
        old = self.threshold
        self.threshold = min(self.cfg.threshold_max, self.threshold * self.cfg.threshold_growth_k)
        return {"old_threshold": old, "new_threshold": self.threshold}

    def decide(self, seen_ratio: float) -> Dict:
        if seen_ratio < self.threshold:
            update = self.reject_low_context()
            return {"accept": False, "reason": "not_enough_existing_texture_context", **update}
        if seen_ratio > self.cfg.seen_upper_bound:
            return {
                "accept": False,
                "reason": "view_too_redundant_already_seen",
                "old_threshold": self.threshold,
                "new_threshold": self.threshold,
            }
        return {
            "accept": True,
            "reason": "within_dynamic_context_band",
            "old_threshold": self.threshold,
            "new_threshold": self.threshold,
        }


def config_from_dict(data: Dict) -> TriangleCoverageConfig:
    allowed = TriangleCoverageConfig.__annotations__
    return TriangleCoverageConfig(**{k: v for k, v in data.items() if k in allowed})


def _barycentric(px, py, tri):
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(den)) < 1e-12:
        return None
    l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
    l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
    return l0, l1, 1.0 - l0 - l1


def projected_triangle_mask(clip, width: int, height: int, epsilon: float = 0.0025):
    clip = np.asarray(clip, np.float32)
    if clip.shape != (3, 4) or np.any(np.abs(clip[:, 3]) < 1e-8):
        return None
    ndc = clip[:, :2] / clip[:, 3:4]
    screen = np.stack([
        (ndc[:, 0] * 0.5 + 0.5) * (width - 1),
        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1),
    ], axis=1)
    x0 = max(0, int(np.floor(screen[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(screen[:, 0].max())))
    y0 = max(0, int(np.floor(screen[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(screen[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return None
    xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    bary = _barycentric(gx, gy, screen)
    if bary is None:
        return None
    l0, l1, l2 = bary
    inside = (l0 >= -epsilon) & (l1 >= -epsilon) & (l2 >= -epsilon)
    return x0, y0, inside


def visible_triangle_records(
    triangles: Iterable[Dict],
    object_mask: np.ndarray,
    minimum_pixels: int = 1,
) -> List[Dict]:
    object_mask = np.asarray(object_mask, dtype=bool)
    height, width = object_mask.shape
    records: List[Dict] = []
    for tri in triangles:
        if float(tri.get("frontality", 1.0)) <= 1e-6:
            continue
        raster = projected_triangle_mask(tri["clip"], width, height)
        if raster is None:
            continue
        x0, y0, inside = raster
        h, w = inside.shape
        visible = inside & object_mask[y0:y0 + h, x0:x0 + w]
        count = int(visible.sum())
        if count < int(minimum_pixels):
            continue
        records.append({
            "global_triangle_id": int(tri["global_triangle_id"]),
            "mesh_object_name": tri.get("mesh_object_name"),
            "visible_pixels": count,
            "projected_pixels": int(inside.sum()),
            "frontality": float(tri.get("frontality", 1.0)),
            "world_area": float(tri.get("world_area", 0.0)),
        })
    return records


def update_triangle_seen(state_path: str | Path, visible_records: Iterable[Dict]) -> Dict:
    state_path = Path(state_path)
    state = np.load(state_path).astype(np.bool_)
    before = int(state.sum())
    ids = sorted({int(r["global_triangle_id"]) for r in visible_records})
    valid = [idx for idx in ids if 0 <= idx < len(state)]
    if valid:
        state[np.asarray(valid, dtype=np.int64)] = True
    np.save(state_path, state)
    return {
        "triangle_count": int(len(state)),
        "seen_before": before,
        "seen_after": int(state.sum()),
        "newly_seen": int(state.sum()) - before,
        "visible_triangle_ids": valid,
    }


def weighted_seen_ratio(visible_records_by_object: Dict[str, List[Dict]], state_root: str | Path) -> Dict:
    """Measure refinement coverage using visible UV/world area, not mesh density.

    ``visible_pixels`` remains in the report for composition diagnostics, while
    selection uses ``uv_area_normalized`` when available.  This prevents a wall
    with twelve triangles and a generated object with hundreds of thousands of
    triangles from being scored on incompatible topology-dependent scales.
    """
    state_root = Path(state_root)
    total_pixels = 0
    seen_pixels = 0
    total_weight = 0.0
    seen_weight = 0.0
    new_weight = 0.0
    object_reports = {}
    for object_id, records in visible_records_by_object.items():
        path = state_root / object_id / "triangle_seen.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing triangle coverage state: {path}")
        state = np.load(path).astype(np.bool_)
        has_uv_weights = any(float(r.get("uv_area_normalized", 0.0)) > 0.0 for r in records)

        def record_weight(record):
            if has_uv_weights:
                return max(float(record.get("uv_area_normalized", 0.0)), 0.0)
            return max(float(record.get("world_area", 0.0)), float(record.get("visible_pixels", 0)))

        obj_total_pixels = sum(int(r.get("visible_pixels", 0)) for r in records)
        obj_seen_pixels = sum(
            int(r.get("visible_pixels", 0))
            for r in records
            if 0 <= int(r["global_triangle_id"]) < len(state) and bool(state[int(r["global_triangle_id"])])
        )
        obj_total_weight = sum(record_weight(r) for r in records)
        obj_seen_weight = sum(
            record_weight(r)
            for r in records
            if 0 <= int(r["global_triangle_id"]) < len(state) and bool(state[int(r["global_triangle_id"])])
        )
        obj_new_weight = max(0.0, obj_total_weight - obj_seen_weight)
        total_pixels += obj_total_pixels
        seen_pixels += obj_seen_pixels
        total_weight += obj_total_weight
        seen_weight += obj_seen_weight
        new_weight += obj_new_weight
        object_reports[object_id] = {
            "visible_triangle_count": len(records),
            "visible_triangle_pixels": obj_total_pixels,
            "seen_triangle_pixels": obj_seen_pixels,
            "visible_surface_weight": float(obj_total_weight),
            "seen_surface_weight": float(obj_seen_weight),
            "new_surface_weight": float(obj_new_weight),
            "seen_ratio": float(obj_seen_weight / max(obj_total_weight, 1e-12)),
            "new_triangle_count": sum(
                1 for r in records
                if 0 <= int(r["global_triangle_id"]) < len(state) and not bool(state[int(r["global_triangle_id"])])
            ),
        }
    return {
        "seen_pixels": int(seen_pixels),
        "visible_triangle_pixels": int(total_pixels),
        "visible_surface_weight": float(total_weight),
        "seen_surface_weight": float(seen_weight),
        "new_surface_weight": float(new_weight),
        "seen_ratio": float(seen_weight / max(total_weight, 1e-12)),
        "new_surface_ratio": float(new_weight / max(total_weight, 1e-12)),
        "objects": object_reports,
    }


def save_coverage_summary(texture_root: str | Path, output_path: str | Path) -> Dict:
    texture_root = Path(texture_root)
    objects = []
    for path in sorted(texture_root.glob("*/triangle_seen.npy")):
        state = np.load(path).astype(np.bool_)
        objects.append({
            "object_id": path.parent.name,
            "triangle_count": int(len(state)),
            "seen_count": int(state.sum()),
            "seen_fraction": float(state.mean()) if len(state) else 0.0,
        })
    report = {"objects": objects}
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
