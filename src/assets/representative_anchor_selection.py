from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

CANONICAL_MULTIVIEW_SLOTS = ("front", "right", "back", "left")

DEFAULT_CANDIDATE_ORBIT = {
    "floor_elevation_deg": 35.0,
    "ceiling_elevation_deg": -35.0,
    "floor_ceiling_view_count": 8,
    "wall_view_count": 4,
    "wall_ring_half_angle_deg": 35.0,
    "distance_scale": 2.2,
    "camera_projection": "PERSP",
    "fov_deg": 28.0,
    "frustum_margin_ndc": 0.86,
}

DEFAULT_SELECTION_CONFIG = {
    "max_anchors": 1,
    "minimum_anchor_count": 1,
    "score_weights": {
        "visible_area_fraction": 0.35,
        "normal_diversity": 0.15,
        "part_coverage_fraction": 0.20,
        "framing_score": 0.10,
        "levelness_score": 0.05,
    },
    "greedy_weights": {
        "coverage_gain": 0.65,
        "part_gain": 0.15,
        "view_diversity": 0.15,
        "canonical_slot_gain": 0.05,
        "base_score": 0.10,
    },
    "early_stop": {
        "after_2_coverage": 0.84,
        "after_2_remaining_gain": 0.04,
        "after_3_coverage": 0.92,
        "after_3_remaining_gain": 0.02,
    },
}


def _merged_dict(*items: Mapping[str, Any] | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if isinstance(item, Mapping):
            for key, value in item.items():
                if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                    result[key] = _merged_dict(result[key], value)
                else:
                    result[key] = value
    return result


def _normalize3(values):
    x, y, z = (float(values[0]), float(values[1]), float(values[2]))
    length = math.sqrt(x*x + y*y + z*z)
    if length <= 1e-9:
        raise ValueError("direction vector must be non-zero")
    return [x/length, y/length, z/length]


def build_candidate_views(config: Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    cfg = _merged_dict(DEFAULT_CANDIDATE_ORBIT, config or {})
    explicit = cfg.get("views")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) and explicit:
        return [dict(entry) for entry in explicit]
    spatial_role = str(cfg.get("spatial_category", "free")).lower()
    common = {
        "distance_scale": float(cfg.get("distance_scale", 2.2)),
        "camera_projection": str(cfg.get("camera_projection", "PERSP")),
        "fov_deg": float(cfg.get("fov_deg", 28.0)),
        "frustum_margin_ndc": float(cfg.get("frustum_margin_ndc", 0.86)),
    }
    candidates=[]
    if spatial_role in {"floor", "ceiling", "free"}:
        count=max(1,int(cfg.get("floor_ceiling_view_count",8)))
        elevation=float(cfg.get("floor_elevation_deg",35.0) if spatial_role != "ceiling" else cfg.get("ceiling_elevation_deg",-35.0))
        ring="upper" if elevation>=0 else "lower"
        for i in range(count):
            az=360.0*i/count
            candidates.append({**common,"name":f"{spatial_role}_{i:02d}","ring":ring,"azimuth_deg":az,"elevation_deg":elevation,
                "text_hint":_describe_view(az,elevation,ring_name=ring)})
        return candidates
    if spatial_role == "wall":
        n=_normalize3(cfg.get("wall_inward_normal_world", [0.0,-1.0,0.0]))
        nvec=n
        helper=[0.0,0.0,1.0] if abs(nvec[2])<0.9 else [1.0,0.0,0.0]
        def cross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        u=_normalize3(cross(nvec,helper))
        v=_normalize3(cross(nvec,u))
        count=max(1,int(cfg.get("wall_view_count",4)))
        theta=math.radians(float(cfg.get("wall_ring_half_angle_deg",35.0)))
        for i in range(count):
            phi=2*math.pi*i/count
            direction=[math.cos(theta)*nvec[j]+math.sin(theta)*(math.cos(phi)*u[j]+math.sin(phi)*v[j]) for j in range(3)]
            candidates.append({**common,"name":f"wall_{i:02d}","ring":"wall_middle","azimuth_deg":360.0*i/count,
                "elevation_deg":0.0,"world_direction":[float(x) for x in direction],
                "text_hint":"wall inward-hemisphere middle-ring view"})
        return candidates
    raise ValueError(f"Unsupported spatial_category: {spatial_role}")

def _describe_view(azimuth_deg: float, elevation_deg: float, ring_name: str | None = None) -> str:
    az = int(round(azimuth_deg)) % 360
    direction = f"azimuth {az} degrees"
    if elevation_deg > 1e-6:
        elevation = f"elevated by {abs(int(round(elevation_deg)))} degrees"
    elif elevation_deg < -1e-6:
        elevation = f"viewing from below by {abs(int(round(elevation_deg)))} degrees"
    else:
        elevation = "level with the object center"
    ring_text = f"{ring_name} orbit" if ring_name else "orbit"
    return f"isolated {ring_text} view, {direction}, {elevation}"


def _normalized_entropy(weights: Mapping[str, float]) -> float:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 1e-12:
        return 0.0
    entropy = 0.0
    count = 0
    for value in weights.values():
        probability = max(0.0, float(value)) / total
        if probability > 1e-12:
            entropy -= probability * math.log(probability)
            count += 1
    if count <= 1:
        return 0.0
    return float(entropy / math.log(count))


def compute_view_metrics(view: Mapping[str, Any]) -> Dict[str, float]:
    metrics = dict(view.get("metrics", {}))
    visible_area = float(metrics.get("visible_area_fraction", 0.0))
    part_fraction = float(metrics.get("part_coverage_fraction", 0.0))
    framing = float(metrics.get("framing_score", 0.0))
    normal_diversity = float(metrics.get("normal_diversity", _normalized_entropy(metrics.get("normal_bins", {}))))
    elevation = abs(float(view.get("elevation_deg", 0.0)))
    levelness = max(0.0, min(1.0, 1.0 - elevation / 60.0))
    return {
        "visible_area_fraction": max(0.0, min(1.0, visible_area)),
        "part_coverage_fraction": max(0.0, min(1.0, part_fraction)),
        "framing_score": max(0.0, min(1.0, framing)),
        "normal_diversity": max(0.0, min(1.0, normal_diversity)),
        "levelness_score": levelness,
    }


def score_candidate_view(view: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> float:
    cfg = _merged_dict(DEFAULT_SELECTION_CONFIG, config or {})
    weights = dict(cfg.get("score_weights", {}))
    metrics = compute_view_metrics(view)
    total = 0.0
    denom = 0.0
    for key, value in metrics.items():
        weight = float(weights.get(key, 0.0))
        total += weight * value
        denom += weight
    return total / max(denom, 1e-9)


def _circle_distance_deg(a: float, b: float) -> float:
    delta = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return min(delta, 180.0)


def _view_distance_normalized(candidate: Mapping[str, Any], selected: Iterable[Mapping[str, Any]]) -> float:
    selected = list(selected)
    if not selected:
        return 1.0
    values = []
    for item in selected:
        da = _circle_distance_deg(candidate.get("azimuth_deg", 0.0), item.get("azimuth_deg", 0.0)) / 180.0
        de = abs(float(candidate.get("elevation_deg", 0.0)) - float(item.get("elevation_deg", 0.0))) / 180.0
        values.append(math.sqrt(da * da + de * de) / math.sqrt(2.0))
    return max(0.0, min(1.0, min(values)))


def _coverage_gain(candidate: Mapping[str, Any], selected: Iterable[Mapping[str, Any]], triangle_areas: Mapping[str, float], total_area: float) -> float:
    selected_triangles: set[str] = set()
    for item in selected:
        selected_triangles.update(str(value) for value in item.get("visible_triangle_ids", []))
    candidate_triangles = {str(value) for value in candidate.get("visible_triangle_ids", [])}
    new_triangles = candidate_triangles - selected_triangles
    gain = sum(float(triangle_areas.get(key, 0.0)) for key in new_triangles)
    return gain / max(float(total_area), 1e-9)


def _union_coverage(selected: Iterable[Mapping[str, Any]], triangle_areas: Mapping[str, float], total_area: float) -> float:
    triangles: set[str] = set()
    for item in selected:
        triangles.update(str(value) for value in item.get("visible_triangle_ids", []))
    covered = sum(float(triangle_areas.get(key, 0.0)) for key in triangles)
    return covered / max(float(total_area), 1e-9)


def _part_gain(candidate: Mapping[str, Any], selected: Iterable[Mapping[str, Any]], total_parts: int) -> float:
    current: set[str] = set()
    for item in selected:
        current.update(str(value) for value in item.get("visible_part_ids", []))
    candidate_parts = {str(value) for value in candidate.get("visible_part_ids", [])}
    gain = len(candidate_parts - current)
    return gain / max(int(total_parts), 1)


def _best_remaining_gain(candidates: Iterable[Mapping[str, Any]], selected: Iterable[Mapping[str, Any]], triangle_areas: Mapping[str, float], total_area: float) -> float:
    selected_names = {str(item.get("name")) for item in selected}
    gains = []
    for item in candidates:
        if str(item.get("name")) in selected_names:
            continue
        gains.append(_coverage_gain(item, selected, triangle_areas, total_area))
    return max(gains) if gains else 0.0



def _nearest_canonical_slot(view: Mapping[str, Any]) -> str:
    preferred = {"front": -90.0, "right": 0.0, "back": 90.0, "left": 180.0}
    azimuth = float(view.get("azimuth_deg", 0.0))
    return min(preferred, key=lambda slot: _circle_distance_deg(azimuth, preferred[slot]))


def _canonical_slot_gain(candidate: Mapping[str, Any], selected: Iterable[Mapping[str, Any]]) -> float:
    used = {_nearest_canonical_slot(item) for item in selected}
    return 0.0 if _nearest_canonical_slot(candidate) in used else 1.0


def assign_canonical_slots(selected: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    preferred = {
        "front": -90.0,
        "right": 0.0,
        "back": 90.0,
        "left": 180.0,
    }
    remaining_slots = set(CANONICAL_MULTIVIEW_SLOTS)
    records = [dict(item) for item in selected]
    pairings: List[tuple[float, int, str]] = []
    for index, item in enumerate(records):
        azimuth = float(item.get("azimuth_deg", 0.0))
        for slot in remaining_slots:
            pairings.append((_circle_distance_deg(azimuth, preferred[slot]), index, slot))
    pairings.sort(key=lambda value: value[0])
    assigned_indices: set[int] = set()
    for _, index, slot in pairings:
        if index in assigned_indices or slot not in remaining_slots:
            continue
        records[index]["canonical_view_slot"] = slot
        assigned_indices.add(index)
        remaining_slots.remove(slot)
        if not remaining_slots:
            break
    fallback_slots = [slot for slot in CANONICAL_MULTIVIEW_SLOTS if slot in remaining_slots]
    for record in records:
        if "canonical_view_slot" not in record:
            record["canonical_view_slot"] = fallback_slots.pop(0) if fallback_slots else f"slot_{len(fallback_slots)}"
    return records


def select_anchor_views(capture_report: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = _merged_dict(DEFAULT_SELECTION_CONFIG, config or {})
    greedy_weights = dict(cfg.get("greedy_weights", {}))
    stop_cfg = dict(cfg.get("early_stop", {}))
    views = [dict(item) for item in capture_report.get("views", [])]
    if not views:
        raise ValueError("capture_report contains no candidate views")

    catalog = dict(capture_report.get("triangle_catalog", {}))
    triangle_areas = {str(key): float(value) for key, value in dict(catalog.get("areas", {})).items()}
    total_area = float(catalog.get("total_area", sum(triangle_areas.values())))
    total_parts = max(1, int(catalog.get("part_count", max(len(view.get("visible_part_ids", [])) for view in views))))

    for view in views:
        metrics = compute_view_metrics(view)
        view.setdefault("metrics", {}).update(metrics)
        view["selection_base_score"] = score_candidate_view(view, cfg)

    remaining = sorted(views, key=lambda item: float(item.get("selection_base_score", 0.0)), reverse=True)
    selected: List[Dict[str, Any]] = []
    max_anchors = max(1, min(int(cfg.get("max_anchors", 4)), len(remaining), 4))
    minimum_anchor_count = max(1, min(int(cfg.get("minimum_anchor_count", max_anchors)), max_anchors))

    while remaining and len(selected) < max_anchors:
        if not selected:
            choice = remaining.pop(0)
            choice["selection_gain"] = float(choice.get("selection_base_score", 0.0))
            selected.append(choice)
            continue
        scored: List[tuple[float, Dict[str, Any]]] = []
        for candidate in remaining:
            coverage_gain = _coverage_gain(candidate, selected, triangle_areas, total_area)
            part_gain = _part_gain(candidate, selected, total_parts)
            view_diversity = _view_distance_normalized(candidate, selected)
            canonical_slot_gain = _canonical_slot_gain(candidate, selected)
            base_score = float(candidate.get("selection_base_score", 0.0))
            gain = (
                float(greedy_weights.get("coverage_gain", 0.45)) * coverage_gain
                + float(greedy_weights.get("part_gain", 0.15)) * part_gain
                + float(greedy_weights.get("view_diversity", 0.15)) * view_diversity
                + float(greedy_weights.get("canonical_slot_gain", 0.20)) * canonical_slot_gain
                + float(greedy_weights.get("base_score", 0.05)) * base_score
            )
            enriched = dict(candidate)
            enriched["selection_components"] = {
                "coverage_gain": coverage_gain,
                "part_gain": part_gain,
                "view_diversity": view_diversity,
                "canonical_slot_gain": canonical_slot_gain,
                "base_score": base_score,
            }
            enriched["selection_gain"] = gain
            scored.append((gain, enriched))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_gain, choice = scored[0]
        selected.append(choice)
        remaining = [item for item in remaining if str(item.get("name")) != str(choice.get("name"))]

        union_cov = _union_coverage(selected, triangle_areas, total_area)
        best_remaining_gain = _best_remaining_gain(remaining, selected, triangle_areas, total_area)
        if len(selected) < minimum_anchor_count:
            continue
        if len(selected) >= 2:
            if (
                union_cov >= float(stop_cfg.get("after_2_coverage", 0.88))
                and best_remaining_gain <= float(stop_cfg.get("after_2_remaining_gain", 0.06))
            ):
                break
        if len(selected) >= 3:
            if (
                union_cov >= float(stop_cfg.get("after_3_coverage", 0.94))
                and best_remaining_gain <= float(stop_cfg.get("after_3_remaining_gain", 0.03))
            ):
                break

    selected = assign_canonical_slots(selected)
    selected_names = {str(item.get("name")) for item in selected}
    ranked_views = sorted(views, key=lambda item: float(item.get("selection_base_score", 0.0)), reverse=True)
    for view in ranked_views:
        view["selected_anchor"] = str(view.get("name")) in selected_names
        chosen = next((item for item in selected if str(item.get("name")) == str(view.get("name"))), None)
        if chosen is not None:
            view.update({
                "selected_anchor": True,
                "selection_gain": float(chosen.get("selection_gain", view.get("selection_base_score", 0.0))),
                "selection_components": dict(chosen.get("selection_components", {})),
                "canonical_view_slot": chosen.get("canonical_view_slot"),
            })

    summary = {
        "selected_anchors": selected,
        "ranked_candidates": ranked_views,
        "union_visible_area_fraction": _union_coverage(selected, triangle_areas, total_area),
        "selected_count": len(selected),
        "max_anchors": max_anchors,
        "minimum_anchor_count": minimum_anchor_count,
    }
    return summary
