from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.cameras.room_pair_sampling import room_samples_in_frustum

_EPS = 1e-12


def _vec3(values: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(values), dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected 3-vector, got {value.shape}")
    return value


def _point_in_polygon(point: Sequence[float], polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = int(len(polygon))
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if abs(float(p1y - p2y)) > _EPS:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return bool(inside)


def _floor_frame(room_model: Mapping[str, Any]) -> Mapping[str, Any]:
    horizontal = [
        frame for frame in room_model["frames"].values()
        if str(frame.get("orientation", "")) == "horizontal"
    ]
    if not horizontal:
        raise RuntimeError("WorldMesh-style Stage07 requires a horizontal floor surface")
    return min(horizontal, key=lambda frame: float(np.asarray(frame["center"], dtype=np.float64)[2]))


def room_floor_polygon(room_model: Mapping[str, Any]) -> np.ndarray:
    frame = _floor_frame(room_model)
    polygon = np.asarray(frame["corners"], dtype=np.float64)[:, :2]
    if polygon.shape[0] < 3:
        raise RuntimeError("Stage07 floor polygon has fewer than three corners")
    return polygon


def floor_height(room_model: Mapping[str, Any]) -> float:
    return float(np.asarray(_floor_frame(room_model)["center"], dtype=np.float64)[2])


def room_center_xy(room_model: Mapping[str, Any]) -> np.ndarray:
    polygon = room_floor_polygon(room_model)
    return np.asarray(polygon.mean(axis=0), dtype=np.float64)


def _vertical_wall_descriptors(room_model: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Return room-facing vertical wall planes and horizontal spans.

    ``build_room_surface_model`` already stores ``frame["center"]`` on the
    room-facing face of each thin surface and orients ``frame["normal"]``
    toward the room interior.  Stage07 therefore uses those physical inner
    faces directly instead of approximating the room boundary from the floor
    polygon or wall centre lines.
    """
    walls: list[Dict[str, Any]] = []
    for surface_id, frame in room_model["frames"].items():
        if str(frame.get("orientation", "")) != "vertical":
            continue
        center3 = np.asarray(frame["center"], dtype=np.float64)
        normal3 = np.asarray(frame["normal"], dtype=np.float64)
        normal_xy = normal3[:2]
        normal_norm = float(np.linalg.norm(normal_xy))
        if normal_norm <= _EPS:
            continue
        normal_xy = normal_xy / normal_norm

        u3 = np.asarray(frame["u_axis"], dtype=np.float64)
        v3 = np.asarray(frame["v_axis"], dtype=np.float64)
        if float(np.linalg.norm(u3[:2])) >= float(np.linalg.norm(v3[:2])):
            tangent3 = u3
            half_span = float(frame["half_u"])
        else:
            tangent3 = v3
            half_span = float(frame["half_v"])
        tangent_xy_raw = tangent3[:2]
        tangent_norm = float(np.linalg.norm(tangent_xy_raw))
        if tangent_norm <= _EPS:
            continue
        tangent_xy = tangent_xy_raw / tangent_norm
        half_length_xy = half_span * tangent_norm
        walls.append({
            "surface_id": str(surface_id),
            "center_xy": center3[:2],
            "normal_xy": normal_xy,
            "tangent_xy": tangent_xy,
            "half_length_xy": float(half_length_xy),
            "length_xy": float(2.0 * half_length_xy),
        })
    if not walls:
        raise RuntimeError("Stage07 requires room-facing vertical wall surfaces for camera placement")
    center = room_center_xy(room_model)
    walls.sort(key=lambda wall: math.atan2(
        float(wall["center_xy"][1] - center[1]),
        float(wall["center_xy"][0] - center[0]),
    ))
    return walls


def _room_wall_halfspaces(room_model: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "surface_id": str(wall["surface_id"]),
            "center_xy": np.asarray(wall["center_xy"], dtype=np.float64),
            "normal_xy": np.asarray(wall["normal_xy"], dtype=np.float64),
        }
        for wall in _vertical_wall_descriptors(room_model)
    ]


def _point_in_wall_safe_room(
    point_xy: Sequence[float],
    wall_halfspaces: Sequence[Mapping[str, Any]],
    wall_clearance: float,
) -> bool:
    """Check signed distance to *every* room-facing wall inner surface.

    For each vertical wall i, ``center_xy`` lies on the physical room-facing
    surface and ``normal_xy`` points inward.  A camera is valid iff

        n_i^T (p - c_i) >= clearance

    for every wall.
    """
    point = np.asarray(point_xy, dtype=np.float64)
    clearance = max(float(wall_clearance), 0.0)
    for wall in wall_halfspaces:
        center_xy = np.asarray(wall["center_xy"], dtype=np.float64)
        normal_xy = np.asarray(wall["normal_xy"], dtype=np.float64)
        signed_distance = float(np.dot(normal_xy, point - center_xy))
        if signed_distance + 1e-9 < clearance:
            return False
    return bool(wall_halfspaces)



def _safe_wall_tangent_interval(
    wall: Mapping[str, Any],
    wall_halfspaces: Sequence[Mapping[str, Any]],
    inward_offset: float,
    wall_clearance: float,
) -> tuple[float, float] | None:
    """Clip one physical wall span by every room-wall half-space.

    Candidate points on this wall have

        p(s) = c + inward_offset * n + s * t.

    Each wall half-space yields a linear constraint in ``s``.  Intersecting
    those intervals gives the exact tangential segment that is safe with
    respect to *all* room-facing wall surfaces.
    """
    center_xy = np.asarray(wall["center_xy"], dtype=np.float64)
    normal_xy = np.asarray(wall["normal_xy"], dtype=np.float64)
    tangent_xy = np.asarray(wall["tangent_xy"], dtype=np.float64)
    base = center_xy + float(inward_offset) * normal_xy
    lo = -float(wall["half_length_xy"])
    hi = float(wall["half_length_xy"])
    clearance = max(float(wall_clearance), 0.0)
    for constraint in wall_halfspaces:
        c = np.asarray(constraint["center_xy"], dtype=np.float64)
        n = np.asarray(constraint["normal_xy"], dtype=np.float64)
        a = float(np.dot(n, tangent_xy))
        b = float(np.dot(n, base - c))
        rhs = clearance - b
        if abs(a) <= _EPS:
            if rhs > 1e-9:
                return None
            continue
        bound = rhs / a
        if a > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
        if lo > hi + 1e-9:
            return None
    return float(lo), float(hi)

def _camera_model(config: Mapping[str, Any]) -> Dict[str, float]:
    model = dict(config.get("camera_model", {}))
    width, height = [int(v) for v in model.get("resolution", [1376, 768])]
    vertical_fov = float(model.get("vertical_fov_degrees", 60.0))
    sensor_width = float(model.get("sensor_width_mm", 36.0))
    aspect = float(width) / max(float(height), 1.0)
    focal = sensor_width / (2.0 * aspect * math.tan(math.radians(vertical_fov) / 2.0))
    return {
        "width": width,
        "height": height,
        "vertical_fov_degrees": vertical_fov,
        "sensor_width_mm": sensor_width,
        "focal_length_mm": float(focal),
    }


def _make_camera(
    camera_id: str,
    role: str,
    position: Sequence[float],
    target: Sequence[float],
    config: Mapping[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    model = _camera_model(config)
    return {
        "camera_id": str(camera_id),
        "camera_type": "perspective",
        "camera_role": str(role),
        "camera_source": str(source),
        "position": [float(v) for v in position],
        "target": [float(v) for v in target],
        "up": [0.0, 0.0, 1.0],
        "focal_length": float(model["focal_length_mm"]),
        "sensor_width_mm": float(model["sensor_width_mm"]),
        "sensor_fit": "HORIZONTAL",
        "vertical_fov_degrees": float(model["vertical_fov_degrees"]),
        "resolution": [int(model["width"]), int(model["height"])],
    }


def _sample_perimeter_positions(
    polygon: np.ndarray,
    count: int,
    wall_offset: float,
    height: float,
    center: np.ndarray,
    look_at_height: float,
) -> list[tuple[list[float], list[float]]]:
    edges: list[Dict[str, Any]] = []
    cumulative = 0.0
    n = int(len(polygon))
    for i in range(n):
        p1 = np.asarray(polygon[i], dtype=np.float64)
        p2 = np.asarray(polygon[(i + 1) % n], dtype=np.float64)
        edge_vec = p2 - p1
        length = float(np.linalg.norm(edge_vec))
        if length <= _EPS:
            continue
        direction = edge_vec / length
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        test = 0.5 * (p1 + p2) + 0.1 * normal
        if not _point_in_polygon(test, polygon):
            normal = -normal
        edges.append({
            "start": p1,
            "direction": direction,
            "normal": normal,
            "length": length,
            "cumulative": cumulative,
        })
        cumulative += length
    if cumulative <= _EPS or count <= 0:
        return []
    spacing = cumulative / float(count)
    result: list[tuple[list[float], list[float]]] = []
    edge_index = 0
    for index in range(count):
        distance = spacing * (index + 0.5)
        while edge_index < len(edges) - 1:
            edge_end = float(edges[edge_index]["cumulative"] + edges[edge_index]["length"])
            if distance <= edge_end + 1e-9:
                break
            edge_index += 1
        edge = edges[edge_index]
        local = float(np.clip(distance - edge["cumulative"], 0.0, edge["length"]))
        point = edge["start"] + local * edge["direction"] + float(wall_offset) * edge["normal"]
        position = [float(point[0]), float(point[1]), float(height)]
        target = [float(center[0]), float(center[1]), float(look_at_height)]
        result.append((position, target))
    return result


def _sample_room_wall_perimeter_positions(
    room_model: Mapping[str, Any],
    count: int,
    wall_offset: float,
    height: float,
    center: np.ndarray,
    look_at_height: float,
    wall_clearance: float,
) -> list[tuple[list[float], list[float]]]:
    """Sample the requested count on true inner-wall safe perimeter segments."""
    walls = _vertical_wall_descriptors(room_model)
    halfspaces = _room_wall_halfspaces(room_model)
    safe_segments: list[Dict[str, Any]] = []
    total_length = 0.0
    for wall in walls:
        interval = _safe_wall_tangent_interval(wall, halfspaces, wall_offset, wall_clearance)
        if interval is None:
            continue
        lo, hi = interval
        length = max(float(hi - lo), 0.0)
        if length <= _EPS:
            continue
        safe_segments.append({"wall": wall, "lo": lo, "hi": hi, "length": length, "cumulative": total_length})
        total_length += length
    if total_length <= _EPS or count <= 0:
        return []

    spacing = total_length / float(count)
    result: list[tuple[list[float], list[float]]] = []
    segment_index = 0
    for index in range(count):
        distance = spacing * (index + 0.5)
        while segment_index < len(safe_segments) - 1:
            segment = safe_segments[segment_index]
            if distance <= float(segment["cumulative"] + segment["length"]) + 1e-9:
                break
            segment_index += 1
        segment = safe_segments[segment_index]
        wall = segment["wall"]
        local = float(np.clip(distance - float(segment["cumulative"]), 0.0, float(segment["length"])))
        along = float(segment["lo"] + local)
        face_xy = np.asarray(wall["center_xy"], dtype=np.float64) + along * np.asarray(wall["tangent_xy"], dtype=np.float64)
        point = face_xy + float(wall_offset) * np.asarray(wall["normal_xy"], dtype=np.float64)
        if not _point_in_wall_safe_room(point, halfspaces, wall_clearance):
            raise RuntimeError("Stage07 internal error: analytically clipped perimeter sample is not wall-safe")
        result.append((
            [float(point[0]), float(point[1]), float(height)],
            [float(center[0]), float(center[1]), float(look_at_height)],
        ))
    return result

def _vertical_room_sample_mask(room_model: Mapping[str, Any]) -> np.ndarray:
    mask = np.zeros(len(room_model["positions"]), dtype=bool)
    for surface_id, record in room_model["surface_slices"].items():
        frame = room_model["frames"].get(surface_id, {})
        if str(frame.get("orientation", "")) != "vertical":
            continue
        mask[int(record["start"]):int(record["end"])] = True
    return mask


def _segment_hits_collider(
    origin: np.ndarray,
    endpoints: np.ndarray,
    collider: Mapping[str, Any],
    padding: float,
) -> np.ndarray:
    """Return which origin->endpoint segments enter an inflated scaffold body."""
    if endpoints.size == 0:
        return np.zeros(0, dtype=bool)
    directions = endpoints - origin[None, :]
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        body_center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
        local_origin = axes.T @ (origin - body_center)
        local_dir = directions @ axes
        lo = -half
        hi = half
    else:
        minimum = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
        maximum = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
        local_origin = origin
        local_dir = directions
        lo = minimum
        hi = maximum

    n = int(len(endpoints))
    t_enter = np.zeros(n, dtype=np.float64)
    t_exit = np.ones(n, dtype=np.float64)
    valid = np.ones(n, dtype=bool)
    for axis in range(3):
        d = local_dir[:, axis]
        o = float(local_origin[axis])
        parallel = np.abs(d) <= 1e-12
        valid &= ~(parallel & ((o < float(lo[axis])) | (o > float(hi[axis]))))
        nonparallel = ~parallel
        if np.any(nonparallel):
            t1 = (float(lo[axis]) - o) / d[nonparallel]
            t2 = (float(hi[axis]) - o) / d[nonparallel]
            near = np.minimum(t1, t2)
            far = np.maximum(t1, t2)
            t_enter[nonparallel] = np.maximum(t_enter[nonparallel], near)
            t_exit[nonparallel] = np.minimum(t_exit[nonparallel], far)
    # Ignore contact exactly at the room-surface endpoint; placed-object/scaffold
    # bodies are expected strictly before the structural surface target.
    return valid & (t_exit >= np.maximum(t_enter, 1e-6)) & (t_enter < 1.0 - 1e-5) & (t_exit > 1e-6)


def _bootstrap_visible_wall_indices(
    camera: Mapping[str, Any],
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    collision_bodies: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    indices = room_samples_in_frustum(
        camera["position"], camera["target"], room_model, config,
        focal_length_mm=float(camera["focal_length"]),
    )
    if indices.size == 0:
        return indices
    vertical_mask = _vertical_room_sample_mask(room_model)
    indices = indices[vertical_mask[indices]]
    if indices.size == 0 or not collision_bodies:
        return indices
    origin = _vec3(camera["position"])
    endpoints = np.asarray(room_model["positions"], dtype=np.float64)[indices]
    padding = 0.0  # visibility uses true scaffold geometry, never the camera-adjustment padding
    blocked = np.zeros(indices.size, dtype=bool)
    for collider in collision_bodies:
        blocked |= _segment_hits_collider(origin, endpoints, collider, padding)
        if bool(blocked.all()):
            break
    return indices[~blocked]


def _short_wall_pair(room_model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    walls = _vertical_wall_descriptors(room_model)
    if len(walls) < 2:
        raise RuntimeError("Stage07 requires at least two vertical walls for bootstrap cameras")
    min_length = min(float(wall["length_xy"]) for wall in walls)
    tolerance = max(1e-6, min_length * 1e-4)
    shortest = [wall for wall in walls if float(wall["length_xy"]) <= min_length + tolerance]
    pool = shortest if len(shortest) >= 2 else walls
    best_pair = None
    best_distance = -1.0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            distance = float(np.linalg.norm(
                np.asarray(pool[i]["center_xy"], dtype=np.float64)
                - np.asarray(pool[j]["center_xy"], dtype=np.float64)
            ))
            if distance > best_distance:
                best_distance = distance
                best_pair = [pool[i], pool[j]]
    if best_pair is None:
        raise RuntimeError("Stage07 could not identify opposite short walls")
    best_pair.sort(key=lambda wall: (float(wall["center_xy"][0]), float(wall["center_xy"][1]), str(wall["surface_id"])))
    return best_pair


def _coverage_wall_candidates(
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    side: int,
) -> list[Dict[str, Any]]:
    """WorldMesh short-wall bootstrap candidates on true room-facing wall planes."""
    layout = dict(config.get("worldmesh_base_layout", {}))
    pair = _short_wall_pair(room_model)
    source = pair[int(side)]
    opposite = pair[1 - int(side)]
    z = floor_height(room_model) + float(layout.get("camera_height_m", 1.6))
    offset = float(layout.get("coverage_wall_offset_m", 0.05))
    clearance = float(layout.get("camera_wall_clearance_m", 0.05))
    candidate_count = max(3, int(layout.get("bootstrap_wall_candidate_count", 13)))
    if candidate_count % 2 == 0:
        candidate_count += 1
    end_margin = max(float(layout.get("bootstrap_wall_end_margin_m", 0.4)), 0.0)
    halfspaces = _room_wall_halfspaces(room_model)
    safe_interval = _safe_wall_tangent_interval(source, halfspaces, offset, clearance)
    if safe_interval is None:
        return []
    safe_lo, safe_hi = safe_interval
    available_half = 0.5 * max(safe_hi - safe_lo, 0.0)
    effective_margin = min(end_margin, 0.90 * available_half)
    lo = safe_lo + effective_margin
    hi = safe_hi - effective_margin
    if lo > hi + 1e-9:
        return []
    tangential = np.linspace(lo, hi, candidate_count)
    source_center = np.asarray(source["center_xy"], dtype=np.float64)
    source_tangent = np.asarray(source["tangent_xy"], dtype=np.float64)
    source_normal = np.asarray(source["normal_xy"], dtype=np.float64)
    opposite_center = np.asarray(opposite["center_xy"], dtype=np.float64)
    opposite_tangent = np.asarray(opposite["tangent_xy"], dtype=np.float64)

    result: list[Dict[str, Any]] = []
    for index, along in enumerate(tangential):
        face_xy = source_center + float(along) * source_tangent
        position_xy = face_xy + float(offset) * source_normal
        if not _point_in_wall_safe_room(position_xy, halfspaces, clearance):
            continue
        # Aim at the corresponding location on the opposite physical inner wall.
        relative = face_xy - source_center
        opposite_along = float(np.dot(relative, opposite_tangent))
        opposite_along = float(np.clip(opposite_along, -float(opposite["half_length_xy"]), float(opposite["half_length_xy"])))
        target_xy = opposite_center + opposite_along * opposite_tangent
        result.append(_make_camera(
            f"bootstrap_candidate_{side}_{index:02d}",
            "room_coverage",
            [float(position_xy[0]), float(position_xy[1]), z],
            [float(target_xy[0]), float(target_xy[1]), z],
            config,
            source="worldmesh_bootstrap_candidate",
        ))
    return result

def _select_bootstrap_pair(
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    collision_bodies: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Pick two opposite short-wall bootstraps with minimal visibility-aware slide.

    WorldMesh fixes the two bootstraps at opposite short-wall midpoints. That is
    ideal for its first structure-only bootstrap, but Stage07 here runs after
    placed-object insertion and a wall-midpoint camera may look directly into a
    wardrobe or vanity. We retain the same two walls, eye height, wall offset,
    and straight-across orientation, while permitting only a tangential slide
    along each wall. For each side, we compute scaffold-occlusion-aware visible
    vertical room-shell area and choose the *closest-to-midpoint* candidate that
    reaches a configurable fraction of that side's best visibility. This keeps
    the WorldMesh midpoint prior whenever it remains useful rather than drifting
    unnecessarily toward a corner.
    """
    areas = np.asarray(room_model["areas"], dtype=np.float64)
    layout = dict(config.get("worldmesh_base_layout", {}))
    sufficiency = float(np.clip(layout.get("bootstrap_visibility_sufficiency_ratio", 0.75), 0.0, 1.0))

    selected: list[Dict[str, Any]] = []
    for side in (0, 1):
        raw_candidates = _coverage_wall_candidates(room_model, config, side=side)
        midpoint = np.asarray(raw_candidates[len(raw_candidates) // 2]["position"], dtype=np.float64)
        records: list[tuple[Dict[str, Any], np.ndarray, float, float]] = []
        for raw in raw_candidates:
            safe_candidates = apply_worldmesh_collision_avoidance([raw], room_model, config, collision_bodies)
            if not safe_candidates:
                continue
            safe = safe_candidates[0]
            visible = _bootstrap_visible_wall_indices(safe, room_model, config, collision_bodies)
            visible_area = float(areas[visible].sum()) if visible.size else 0.0
            displacement = float(np.linalg.norm(_vec3(safe["position"])[:2] - midpoint[:2]))
            records.append((safe, visible, visible_area, displacement))
        if not records:
            raise RuntimeError(f"Stage07 found no safe bootstrap candidate on short wall side {side}")

        best_area = max(item[2] for item in records)
        threshold = sufficiency * best_area
        sufficient = [item for item in records if item[2] + 1e-9 >= threshold]
        pool = sufficient or records
        chosen = min(pool, key=lambda item: (item[3], -item[2]))
        camera, visible, visible_area, displacement = chosen
        record = dict(camera)
        record["camera_id"] = f"reconstruction_coverage_{side:03d}"
        record["camera_source"] = "worldmesh_short_wall_bootstrap_scaffold_visibility_adjusted"
        record["bootstrap_visibility"] = {
            "visible_vertical_room_sample_count": int(visible.size),
            "visible_vertical_room_area": float(visible_area),
            "best_candidate_visible_vertical_room_area": float(best_area),
            "visibility_sufficiency_ratio": float(sufficiency),
            "wall_midpoint_tangential_displacement_m": float(displacement),
            "selection_policy": "nearest_short_wall_midpoint_candidate_reaching_visibility_sufficiency",
        }
        selected.append(record)
    return selected


def _coverage_cameras(
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    collision_bodies: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    return _select_bootstrap_pair(room_model, config, collision_bodies)


def _filter_perimeter_overlap(
    coverage: Sequence[Mapping[str, Any]],
    perimeter: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    minimum_angle_degrees: float,
) -> list[Dict[str, Any]]:
    azimuths = []
    for camera in coverage:
        p = np.asarray(camera["position"], dtype=np.float64)
        azimuths.append(float(np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) % 360.0))
    result = []
    for camera in perimeter:
        p = np.asarray(camera["position"], dtype=np.float64)
        azimuth = float(np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) % 360.0)
        duplicate = False
        for existing in azimuths:
            difference = abs(azimuth - existing)
            difference = min(difference, 360.0 - difference)
            if difference < float(minimum_angle_degrees):
                duplicate = True
                break
        if not duplicate:
            result.append(dict(camera))
    return result


def _collider_identifier(collider: Mapping[str, Any]) -> str:
    owner = str(collider.get("owner_id", "object"))
    part = str(collider.get("part_id", ""))
    return f"{owner}:{part}" if part else owner


def _point_inside_collider(position: Sequence[float], collider: Mapping[str, Any], padding: float) -> bool:
    p = _vec3(position)
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
        local = axes.T @ (p - center)
        return bool(np.all(np.abs(local) <= half + _EPS))
    lo = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
    hi = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _aabb_collisions(position: Sequence[float], boxes: Sequence[Mapping[str, Any]], padding: float) -> list[str]:
    p = _vec3(position)
    result = []
    for box in boxes:
        if _point_inside_collider(p, box, padding):
            result.append(_collider_identifier(box))
    return result



def _point_distance_to_collider(position: Sequence[float], collider: Mapping[str, Any]) -> float:
    """Exact Euclidean distance from a camera point to the *actual* collider geometry.

    Returns 0 for points inside/on the collider. Scaffold primitives are OBBs in
    world space; legacy fallback bodies are AABBs. No trigger/release padding is
    involved here.
    """
    p = _vec3(position)
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64)
        local = axes.T @ (p - center)
        outside = np.maximum(np.abs(local) - half, 0.0)
        return float(np.linalg.norm(outside))

    lo = np.asarray(collider["minimum"], dtype=np.float64)
    hi = np.asarray(collider["maximum"], dtype=np.float64)
    outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return float(np.linalg.norm(outside))


def _ray_interval_against_collider(
    position: np.ndarray,
    direction_xy: np.ndarray,
    collider: Mapping[str, Any],
    padding: float,
) -> tuple[float, float] | None:
    """Return the signed line/collider intersection interval ``(t_enter, t_exit)``.

    Stage07 uses a single 20 cm scaffold envelope.  The line is allowed to
    extend in both signs for the viewing direction so a nearby body whose
    relevant intersection lies behind the camera can be identified without
    forcing a relocation.  OBBs are solved in collider-local coordinates;
    legacy AABBs use the same slab calculation in world space.
    """
    norm = float(np.linalg.norm(direction_xy))
    if norm <= _EPS:
        return None
    direction_xy = np.asarray(direction_xy, dtype=np.float64) / norm
    d_world = np.asarray([float(direction_xy[0]), float(direction_xy[1]), 0.0], dtype=np.float64)
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
        o = axes.T @ (position - center)
        d = axes.T @ d_world
        lo, hi = -half, half
    else:
        lo = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
        hi = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
        o = position
        d = d_world

    t_enter = -float("inf")
    t_exit = float("inf")
    for axis in range(3):
        da = float(d[axis])
        oa = float(o[axis])
        loa = float(lo[axis])
        hia = float(hi[axis])
        if abs(da) <= _EPS:
            if oa < loa - _EPS or oa > hia + _EPS:
                return None
            continue
        t1 = (loa - oa) / da
        t2 = (hia - oa) / da
        near, far = (t1, t2) if t1 <= t2 else (t2, t1)
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)
        if t_exit < t_enter - _EPS:
            return None
    if not math.isfinite(t_enter) or not math.isfinite(t_exit):
        return None
    return float(t_enter), float(t_exit)


def _collider_center_xy(collider: Mapping[str, Any]) -> np.ndarray:
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        return np.asarray(collider["center"], dtype=np.float64)[:2]
    lo = np.asarray(collider["minimum"], dtype=np.float64)
    hi = np.asarray(collider["maximum"], dtype=np.float64)
    return (lo[:2] + hi[:2]) * 0.5


def _signed_view_exit_distance_from_collider(
    position: np.ndarray,
    view_direction_xy: np.ndarray,
    collider: Mapping[str, Any],
    padding: float,
    eps: float,
) -> float | None:
    """Signed exit distance for the single 20 cm envelope along the view line.

    The camera is already known to be within ``padding`` of the true scaffold
    geometry, hence it lies inside the axis-inflated 20 cm OBB/AABB envelope.
    We deliberately preserve sign for the viewing direction: if the nearby
    body's centre projects behind the camera, return the negative line exit;
    otherwise return the positive exit.  A negative result means the nearby
    body is most plausibly behind the current sight line and Stage07 leaves the
    intended camera pose unchanged.
    """
    norm = float(np.linalg.norm(view_direction_xy))
    if norm <= _EPS:
        return None
    direction = np.asarray(view_direction_xy, dtype=np.float64) / norm
    interval = _ray_interval_against_collider(position, direction, collider, padding)
    if interval is None:
        return None
    t_enter, t_exit = interval
    side = float(np.dot(_collider_center_xy(collider) - position[:2], direction))
    if side < 0.0:
        return float(t_enter - eps)
    return float(t_exit + eps)


def _positive_escape_along_direction(
    original: np.ndarray,
    direction_xy: np.ndarray,
    polygon: np.ndarray,
    colliders: Sequence[Mapping[str, Any]],
    padding: float,
    eps: float,
) -> np.ndarray | None:
    """Move only in the positive declared direction until outside all 20 cm envelopes."""
    norm = float(np.linalg.norm(direction_xy))
    if norm <= _EPS:
        return None
    direction_xy = np.asarray(direction_xy, dtype=np.float64) / norm
    d3 = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    position = original.copy()

    max_iterations = max(8, 4 * len(colliders) + 4)
    for _ in range(max_iterations):
        colliding = [body for body in colliders if _point_inside_collider(position, body, padding)]
        if not colliding:
            return position if _point_in_polygon(position[:2], polygon) else None

        positive_exits: list[float] = []
        for body in colliding:
            interval = _ray_interval_against_collider(position, direction_xy, body, padding)
            if interval is None:
                continue
            _t_enter, t_exit = interval
            if t_exit > eps:
                positive_exits.append(float(t_exit + eps))
        if not positive_exits:
            return None

        position = position + d3 * max(positive_exits)
        if not _point_in_polygon(position[:2], polygon):
            return None
    return None


def _candidate_exit_positions_xy(
    xy: np.ndarray,
    z: float,
    collider: Mapping[str, Any],
    padding: float,
    eps: float,
) -> list[np.ndarray]:
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
        point = np.asarray([float(xy[0]), float(xy[1]), float(z)], dtype=np.float64)
        local = axes.T @ (point - center)
        if abs(float(local[2])) > float(half[2]) + _EPS:
            return []
        exits = []
        for axis_idx in (0, 1):
            for sign in (-1.0, 1.0):
                target_local = local.copy()
                target_local[axis_idx] = sign * (float(half[axis_idx]) + eps)
                world = center + axes @ target_local
                exits.append(np.asarray(world[:2], dtype=np.float64))
        return exits
    lo = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
    hi = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
    if not (lo[2] <= z <= hi[2]):
        return []
    return [
        np.asarray([lo[0] - eps, xy[1]], dtype=np.float64),
        np.asarray([hi[0] + eps, xy[1]], dtype=np.float64),
        np.asarray([xy[0], lo[1] - eps], dtype=np.float64),
        np.asarray([xy[0], hi[1] + eps], dtype=np.float64),
    ]


def _ray_exit_distance_from_collider(
    position: np.ndarray,
    direction_xy: np.ndarray,
    collider: Mapping[str, Any],
    padding: float,
    eps: float,
) -> float | None:
    """Return the smallest positive XY-ray distance that exits an inflated collider.

    The caller only asks this for a point currently inside the collider.  The
    ray has no Z component because Stage07 collision correction must preserve
    camera height.  Scaffold primitive OBBs are solved in collider-local space;
    legacy AABB fallback bodies use the same slab calculation in world space.
    """
    d_world = np.asarray([float(direction_xy[0]), float(direction_xy[1]), 0.0], dtype=np.float64)
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        axes = np.asarray(collider["axes"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
        local_p = axes.T @ (position - center)
        local_d = axes.T @ d_world
        if abs(float(local_p[2])) > float(half[2]) + _EPS:
            return None
        exit_ts = []
        for axis in range(3):
            da = float(local_d[axis])
            if abs(da) <= _EPS:
                continue
            boundary = float(half[axis]) if da > 0.0 else -float(half[axis])
            t = (boundary - float(local_p[axis])) / da
            if t >= -_EPS:
                exit_ts.append(float(max(t, 0.0)))
        if not exit_ts:
            return None
        return min(exit_ts) + eps

    lo = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
    hi = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
    if not (lo[2] - _EPS <= position[2] <= hi[2] + _EPS):
        return None
    exit_ts = []
    for axis in (0, 1):
        da = float(direction_xy[axis])
        if abs(da) <= _EPS:
            continue
        boundary = float(hi[axis]) if da > 0.0 else float(lo[axis])
        t = (boundary - float(position[axis])) / da
        if t >= -_EPS:
            exit_ts.append(float(max(t, 0.0)))
    if not exit_ts:
        return None
    return min(exit_ts) + eps


def _escape_along_direction(
    original: np.ndarray,
    direction_xy: np.ndarray,
    polygon: np.ndarray,
    colliders: Sequence[Mapping[str, Any]],
    padding: float,
    eps: float,
) -> np.ndarray | None:
    """Analytically leave every intersecting scaffold body along one fixed XY direction."""
    norm = float(np.linalg.norm(direction_xy))
    if norm <= _EPS:
        return None
    direction_xy = np.asarray(direction_xy, dtype=np.float64) / norm
    position = original.copy()

    # Each iteration must cross at least one collider boundary.  A generous
    # body-count-derived bound protects against degenerate numerical loops.
    max_iterations = max(8, 4 * len(colliders) + 4)
    for _ in range(max_iterations):
        colliding = [body for body in colliders if _point_inside_collider(position, body, padding)]
        if not colliding:
            if _point_in_polygon(position[:2], polygon):
                return position
            return None

        exit_distances = [
            _ray_exit_distance_from_collider(position, direction_xy, body, padding, eps)
            for body in colliding
        ]
        exit_distances = [value for value in exit_distances if value is not None and value > 0.0]
        if not exit_distances:
            return None

        # To be outside *all currently overlapping* bodies, travel past the
        # furthest required exit on this same permitted direction.
        distance = max(exit_distances)
        position = position + np.asarray([direction_xy[0], direction_xy[1], 0.0]) * distance
        if not _point_in_polygon(position[:2], polygon):
            return None

    return None


def _distance_point_to_segment_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    denom = float(np.dot(segment, segment))
    if denom <= _EPS:
        return float(np.linalg.norm(point - a))
    u = float(np.clip(np.dot(point - a, segment) / denom, 0.0, 1.0))
    closest = a + u * segment
    return float(np.linalg.norm(point - closest))


def _point_in_safe_room(
    point_xy: Sequence[float],
    polygon: np.ndarray,
    wall_clearance: float,
    wall_halfspaces: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Production safety uses all physical room-facing wall half-spaces.

    ``polygon`` remains only as a backward-compatible fallback for historical
    direct/internal callers that do not have a room model. Production Stage07
    always supplies ``wall_halfspaces``.
    """
    if wall_halfspaces is not None:
        return _point_in_wall_safe_room(point_xy, wall_halfspaces, wall_clearance)
    point = np.asarray(point_xy, dtype=np.float64)
    if not _point_in_polygon(point, polygon):
        return False
    clearance = max(float(wall_clearance), 0.0)
    if clearance <= _EPS:
        return True
    distances = [
        _distance_point_to_segment_2d(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    return bool(distances and min(distances) + 1e-9 >= clearance)


def _inflated_collider_bounding_sphere(collider: Mapping[str, Any], padding: float) -> tuple[np.ndarray, float]:
    """Cheap broad-phase sphere enclosing one axis-inflated OBB/AABB collider."""
    ctype = str(collider.get("collider_type", ""))
    if ctype in {"scaffold_primitive_obb", "primitive_obb"}:
        center = np.asarray(collider["center"], dtype=np.float64)
        half = np.asarray(collider["half_extents"], dtype=np.float64) + float(padding)
    else:
        lo = np.asarray(collider["minimum"], dtype=np.float64) - float(padding)
        hi = np.asarray(collider["maximum"], dtype=np.float64) + float(padding)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
    return center, float(np.linalg.norm(half))


def _inside_inflated_collider_broad_phase(
    position: np.ndarray,
    collider: Mapping[str, Any],
    padding: float,
) -> bool:
    center, radius = _inflated_collider_bounding_sphere(collider, padding)
    return bool(float(np.dot(position - center, position - center)) <= radius * radius + 1e-12)


def _resolve_collision_detailed(
    position: Sequence[float],
    polygon: np.ndarray,
    boxes: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    *,
    padding: float = 0.30,
    view_target: Sequence[float] | None = None,
    wall_clearance: float = 0.05,
    max_left_moves: int = 5,
    max_right_moves: int = 5,
    wall_halfspaces: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Iteratively resolve Stage07 camera/scaffold collisions one primitive at a time.

    The true scaffold primitive OBB determines whether a padded collision should
    cause motion; the inflated OBB determines how far the camera must move.
    A primitive is ignored only when the current view line intersects its true
    (un-padded) collider and *all* such intersection parameters are negative.
    An empty true-collider intersection therefore still permits collision
    resolution, exactly as required by the Stage07 camera policy.

    A collision-triggered move is selected from positive-only view/left/right
    exits of the current primitive's inflated collider.  After any move, the
    entire primitive scan restarts because all previous intersection results are
    stale.  Left and right moves have independent finite budgets.  If the
    shortest valid move would use an already exhausted lateral direction, the
    camera is deleted rather than accepted in a potentially oscillatory pose.
    """
    padding = float(padding)
    wall_clearance = max(float(wall_clearance), 0.0)
    max_left_moves = max(int(max_left_moves), 0)
    max_right_moves = max(int(max_right_moves), 0)
    current = _vec3(position).copy()
    target3 = _vec3(view_target) if view_target is not None else np.asarray([center[0], center[1], current[2]], dtype=np.float64)
    eps = max(1e-6, padding * 1e-4)
    adjusted = False
    left_moves = 0
    right_moves = 0
    move_history: list[Dict[str, Any]] = []

    # Production cameras must already lie in the physical room-safe region.
    # Never allow a wall-embedded initial pose to survive merely because no
    # scene-object primitive triggers the collision resolver.
    if wall_halfspaces is not None and not _point_in_safe_room(current[:2], polygon, wall_clearance, wall_halfspaces):
        return {
            "position": current.astype(float).tolist(),
            "adjusted": False,
            "deleted": True,
            "status": "delete_initial_outside_physical_wall_safe_room",
            "left_move_count": 0,
            "right_move_count": 0,
            "move_history": move_history,
        }

    # A generous global bound protects against pathological forward-only loops.
    max_rounds = max(16, 8 * len(boxes) + max_left_moves + max_right_moves + 8)
    for round_index in range(max_rounds):
        moved_this_round = False

        view_direction = target3[:2] - current[:2]
        view_norm = float(np.linalg.norm(view_direction))
        if view_norm <= _EPS:
            view_direction = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            view_direction = view_direction / view_norm
        left = np.asarray([-view_direction[1], view_direction[0]], dtype=np.float64)
        right = -left

        for collider_index, body in enumerate(boxes):
            # Broad phase: if the camera is outside the inflated collider's
            # enclosing sphere, it cannot be inside the inflated OBB/AABB.
            if not _inside_inflated_collider_broad_phase(current, body, padding):
                continue
            if not _point_inside_collider(current, body, padding):
                continue

            # True geometry gate.  Empty intersection is intentionally *not* a
            # reason to ignore this primitive.  We ignore only a non-empty line
            # interval whose two endpoints are both behind the camera.
            true_interval = _ray_interval_against_collider(current, view_direction, body, 0.0)
            if true_interval is not None and max(true_interval) < 0.0:
                continue

            # Forward uses the already relevant view-line intersection with the
            # inflated collider.  Left/right only intersect the inflated body.
            candidates: list[tuple[float, int, str, np.ndarray]] = []
            for rank, (name, direction) in enumerate((
                ("view_direction", view_direction),
                ("left", left),
                ("right", right),
            )):
                interval = _ray_interval_against_collider(current, direction, body, padding)
                if interval is None:
                    continue
                _t_enter, t_exit = interval
                if t_exit <= eps:
                    continue
                distance = float(t_exit + eps)
                candidate = current + np.asarray([direction[0], direction[1], 0.0], dtype=np.float64) * distance
                if not _point_in_safe_room(candidate[:2], polygon, wall_clearance, wall_halfspaces):
                    continue
                candidates.append((distance, rank, name, candidate))

            # No legal positive move that remains safely inside the room: keep
            # the current pose and terminate adjustment for this camera.
            if not candidates:
                current_safe = _point_in_safe_room(current[:2], polygon, wall_clearance, wall_halfspaces)
                return {
                    "position": current.astype(float).tolist(),
                    "adjusted": bool(adjusted),
                    "deleted": bool(not current_safe),
                    "status": "keep_current_no_safe_candidate" if current_safe else "delete_no_safe_candidate_outside_physical_wall_safe_room",
                    "left_move_count": int(left_moves),
                    "right_move_count": int(right_moves),
                    "move_history": move_history,
                }

            distance, _rank, direction_name, candidate = min(candidates, key=lambda item: (item[0], item[1]))

            if direction_name == "left":
                if left_moves >= max_left_moves:
                    return {
                        "position": current.astype(float).tolist(),
                        "adjusted": bool(adjusted),
                        "deleted": True,
                        "status": "delete_left_move_limit",
                        "left_move_count": int(left_moves),
                        "right_move_count": int(right_moves),
                        "move_history": move_history,
                    }
                left_moves += 1
            elif direction_name == "right":
                if right_moves >= max_right_moves:
                    return {
                        "position": current.astype(float).tolist(),
                        "adjusted": bool(adjusted),
                        "deleted": True,
                        "status": "delete_right_move_limit",
                        "left_move_count": int(left_moves),
                        "right_move_count": int(right_moves),
                        "move_history": move_history,
                    }
                right_moves += 1

            old_position = current.copy()
            current = candidate
            adjusted = True
            moved_this_round = True
            move_history.append({
                "round": int(round_index),
                "collider_index": int(collider_index),
                "collider_id": _collider_identifier(body),
                "direction": direction_name,
                "distance_m": float(distance),
                "position_before": old_position.astype(float).tolist(),
                "position_after": current.astype(float).tolist(),
            })
            # Camera changed: restart from the first primitive immediately.
            break

        if not moved_this_round:
            current_safe = _point_in_safe_room(current[:2], polygon, wall_clearance, wall_halfspaces)
            return {
                "position": current.astype(float).tolist(),
                "adjusted": bool(adjusted),
                "deleted": bool(not current_safe),
                "status": "resolved_full_pass_without_move" if current_safe else "delete_final_outside_physical_wall_safe_room",
                "left_move_count": int(left_moves),
                "right_move_count": int(right_moves),
                "move_history": move_history,
            }

    # Defensive finite-loop fallback.  This is not a lateral-limit deletion;
    # preserve the last safe in-room pose rather than aborting Stage07.
    current_safe = _point_in_safe_room(current[:2], polygon, wall_clearance, wall_halfspaces)
    return {
        "position": current.astype(float).tolist(),
        "adjusted": bool(adjusted),
        "deleted": bool(not current_safe),
        "status": "keep_current_round_limit" if current_safe else "delete_round_limit_outside_physical_wall_safe_room",
        "left_move_count": int(left_moves),
        "right_move_count": int(right_moves),
        "move_history": move_history,
    }


def _resolve_collision(
    position: Sequence[float],
    polygon: np.ndarray,
    boxes: Sequence[Mapping[str, Any]],
    center: np.ndarray,
    *,
    padding: float = 0.30,
    view_target: Sequence[float] | None = None,
    wall_clearance: float = 0.05,
    max_left_moves: int = 5,
    max_right_moves: int = 5,
) -> tuple[list[float], bool]:
    """Backward-compatible pair-return wrapper around the detailed resolver.

    Production Stage07 uses ``_resolve_collision_detailed`` so a lateral-limit
    exhaustion can delete a camera.  Direct/internal historical callers still
    receive the familiar ``(position, adjusted)`` pair.
    """
    result = _resolve_collision_detailed(
        position,
        polygon,
        boxes,
        center,
        padding=padding,
        view_target=view_target,
        wall_clearance=wall_clearance,
        max_left_moves=max_left_moves,
        max_right_moves=max_right_moves,
    )
    return list(result["position"]), bool(result["adjusted"])

def apply_worldmesh_collision_avoidance(
    cameras: Sequence[Mapping[str, Any]],
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    object_boxes: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    layout = dict(config.get("worldmesh_base_layout", {}))
    polygon = room_floor_polygon(room_model)
    center = room_center_xy(room_model)
    padding = float(layout.get("collision_padding_m", 0.30))
    wall_clearance = float(layout.get("camera_wall_clearance_m", 0.05))
    max_left_moves = int(layout.get("maximum_left_adjustments", 5))
    max_right_moves = int(layout.get("maximum_right_adjustments", 5))
    wall_halfspaces = _room_wall_halfspaces(room_model)
    result = []
    for camera in cameras:
        record = dict(camera)
        resolution = _resolve_collision_detailed(
            record["position"],
            polygon,
            object_boxes,
            center,
            padding=padding,
            view_target=record.get("target"),
            wall_clearance=wall_clearance,
            max_left_moves=max_left_moves,
            max_right_moves=max_right_moves,
            wall_halfspaces=wall_halfspaces,
        )
        if bool(resolution["deleted"]):
            # Deliberately omit the camera from the returned set.  Coverage
            # repair will compensate later if its removal creates a room-shell
            # coverage hole.
            continue
        record["position"] = list(resolution["position"])
        record["collision_adjusted"] = bool(resolution["adjusted"])
        record["collision_adjustment_status"] = str(resolution["status"])
        record["collision_left_move_count"] = int(resolution["left_move_count"])
        record["collision_right_move_count"] = int(resolution["right_move_count"])
        record["collision_move_history"] = list(resolution["move_history"])
        result.append(record)
    return result

def generate_worldmesh_base_cameras(
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    object_boxes: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    layout = dict(config.get("worldmesh_base_layout", {}))
    polygon = room_floor_polygon(room_model)
    center = room_center_xy(room_model)
    floor_z = floor_height(room_model)
    eye_height = floor_z + float(layout.get("camera_height_m", 1.6))
    overhead_height = eye_height + float(layout.get("overhead_height_offset_m", 0.8))
    wall_offset = float(layout.get("perimeter_wall_offset_m", 0.4))

    coverage = _coverage_cameras(room_model, config, object_boxes)
    wall_clearance = float(layout.get("camera_wall_clearance_m", 0.05))
    perimeter_pairs = _sample_room_wall_perimeter_positions(
        room_model,
        int(layout.get("eye_level_perimeter_count", 16)),
        wall_offset,
        eye_height,
        center,
        eye_height,
        wall_clearance,
    )
    perimeter = [
        _make_camera(f"reconstruction_perimeter_{i:03d}", "perimeter_eye_level", pos, target, config, source="worldmesh_base")
        for i, (pos, target) in enumerate(perimeter_pairs)
    ]
    perimeter = _filter_perimeter_overlap(
        coverage,
        perimeter,
        center,
        float(layout.get("coverage_duplicate_azimuth_degrees", 10.0)),
    )
    for filtered_index, camera in enumerate(perimeter):
        camera["camera_id"] = f"reconstruction_perimeter_{filtered_index:03d}"

    overhead_pairs = _sample_room_wall_perimeter_positions(
        room_model,
        int(layout.get("overhead_count", 8)),
        wall_offset,
        overhead_height,
        center,
        eye_height,
        wall_clearance,
    )
    overhead = [
        _make_camera(f"reconstruction_overhead_{i:03d}", "perimeter_overhead", pos, target, config, source="worldmesh_base")
        for i, (pos, target) in enumerate(overhead_pairs)
    ]
    # Bootstrap coverage cameras are already collision-resolved while candidates
    # are evaluated in _select_bootstrap_pair().  Do not run them through the
    # iterative resolver a second time.  Only newly constructed perimeter and
    # overhead cameras still need collision adjustment here.
    remaining = apply_worldmesh_collision_avoidance(
        [*perimeter, *overhead], room_model, config, object_boxes
    )
    return [*coverage, *remaining]


def camera_room_sample_indices(
    camera: Mapping[str, Any],
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    return room_samples_in_frustum(
        camera["position"],
        camera["target"],
        room_model,
        config,
        focal_length_mm=float(camera["focal_length"]),
    )


def uncovered_components(
    hard_covered: np.ndarray,
    room_model: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    hard_covered = np.asarray(hard_covered, dtype=bool)
    positions = np.asarray(room_model["positions"], dtype=np.float64)
    areas = np.asarray(room_model["areas"], dtype=np.float64)
    result: list[Dict[str, Any]] = []
    for object_id, record in room_model["surface_slices"].items():
        start, end = int(record["start"]), int(record["end"])
        rows, cols = [int(v) for v in record["grid_shape"]]
        uncovered = (~hard_covered[start:end]).reshape(rows, cols)
        visited = np.zeros_like(uncovered, dtype=bool)
        for row in range(rows):
            for col in range(cols):
                if not uncovered[row, col] or visited[row, col]:
                    continue
                queue = deque([(row, col)])
                visited[row, col] = True
                local_indices = []
                while queue:
                    r, c = queue.popleft()
                    local_indices.append(r * cols + c)
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < rows and 0 <= nc < cols and uncovered[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                indices = start + np.asarray(local_indices, dtype=np.int32)
                component_areas = areas[indices]
                area = float(component_areas.sum())
                centroid = np.sum(positions[indices] * component_areas[:, None], axis=0) / max(area, _EPS)
                result.append({
                    "surface_id": str(object_id),
                    "sample_indices": indices,
                    "sample_count": int(indices.size),
                    "area": area,
                    "centroid": centroid.astype(float).tolist(),
                })
    result.sort(key=lambda item: (-float(item["area"]), -int(item["sample_count"]), str(item["surface_id"])))
    return result


def _view_direction(camera: Mapping[str, Any]) -> np.ndarray:
    value = _vec3(camera["target"]) - _vec3(camera["position"])
    return value / max(float(np.linalg.norm(value)), _EPS)


def _is_pose_duplicate(camera: Mapping[str, Any], existing: Sequence[Mapping[str, Any]], position_m: float, angle_deg: float) -> bool:
    p = _vec3(camera["position"])
    d = _view_direction(camera)
    for previous in existing:
        pp = _vec3(previous["position"])
        if float(np.linalg.norm(p - pp)) > float(position_m):
            continue
        pd = _view_direction(previous)
        cosine = float(np.clip(np.dot(d, pd), -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        if angle < float(angle_deg):
            return True
    return False


def generate_repair_candidates(
    component: Mapping[str, Any],
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    object_boxes: Sequence[Mapping[str, Any]],
    existing_cameras: Sequence[Mapping[str, Any]],
    repair_index: int,
) -> list[Dict[str, Any]]:
    repair = dict(config.get("coverage_repair", {}))
    layout = dict(config.get("worldmesh_base_layout", {}))
    center = room_center_xy(room_model)
    target = _vec3(component["centroid"])
    floor_z = floor_height(room_model)
    heights = [float(v) for v in repair.get("camera_heights_m", [1.6, 2.4])]
    fractions = [float(v) for v in repair.get("center_bias_fractions", [1.0, 0.75, 0.5])]
    polygon = room_floor_polygon(room_model)
    wall_halfspaces = _room_wall_halfspaces(room_model)
    wall_clearance = float(layout.get("camera_wall_clearance_m", 0.05))
    candidates: list[Dict[str, Any]] = []
    candidate_index = 0
    for height in heights:
        z = floor_z + height
        for fraction in fractions:
            fraction = float(np.clip(fraction, 0.0, 1.0))
            xy = target[:2] + fraction * (center - target[:2])
            if not _point_in_wall_safe_room(xy, wall_halfspaces, wall_clearance):
                continue
            camera = _make_camera(
                f"reconstruction_repair_{repair_index:03d}_{candidate_index:02d}",
                "coverage_repair",
                [xy[0], xy[1], z],
                target,
                config,
                source="coverage_repair",
            )
            candidate_index += 1
            safe_candidates = apply_worldmesh_collision_avoidance([camera], room_model, config, object_boxes)
            if not safe_candidates:
                continue
            camera = safe_candidates[0]
            if _is_pose_duplicate(
                camera,
                existing_cameras,
                float(repair.get("duplicate_position_threshold_m", 0.10)),
                float(repair.get("duplicate_view_angle_degrees", layout.get("coverage_duplicate_azimuth_degrees", 10.0))),
            ):
                continue
            candidates.append(camera)
    return candidates


def choose_best_repair_camera(
    candidates: Sequence[Mapping[str, Any]],
    hard_covered: np.ndarray,
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    coverage_graph,
) -> tuple[Dict[str, Any], np.ndarray, Dict[str, Any]] | None:
    if not candidates:
        return None
    areas = np.asarray(room_model["areas"], dtype=np.float64)
    hard = np.asarray(hard_covered, dtype=bool)
    center = room_center_xy(room_model)
    require_graph = bool(dict(config.get("coverage_repair", {})).get("require_graph_connection", True))
    scored = []
    for camera in candidates:
        indices = camera_room_sample_indices(camera, room_model, config)
        if indices.size == 0:
            continue
        new_indices = indices[~hard[indices]]
        gain = float(areas[new_indices].sum()) if new_indices.size else 0.0
        if gain <= _EPS:
            continue
        graph_connected = True
        prospective_edge_count = 0
        if coverage_graph.camera_ids:
            trial = copy.deepcopy(coverage_graph)
            previous_edges = trial.edge_count
            trial.add("__repair_candidate__", indices, _view_direction(camera))
            prospective_edge_count = trial.edge_count - previous_edges
            graph_connected = bool(prospective_edge_count > 0)
        if require_graph and not graph_connected:
            continue
        position = _vec3(camera["position"])
        center_distance = float(np.linalg.norm(position[:2] - center))
        scored.append((gain, -center_distance, prospective_edge_count, dict(camera), indices, new_indices))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    gain, neg_center_distance, edge_count, camera, indices, new_indices = scored[0]
    metadata = {
        "new_coverage_area": float(gain),
        "new_coverage_sample_count": int(new_indices.size),
        "center_distance_m": float(-neg_center_distance),
        "prospective_graph_edge_count": int(edge_count),
    }
    return camera, indices, metadata
