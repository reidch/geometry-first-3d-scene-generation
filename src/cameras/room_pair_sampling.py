from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.cameras.interior_probability_sampling import _aabb
from src.cameras.scene_geometry import scaffold_points
from src.scene_ir.json_scene import flat_objects


_EPS = 1e-12


def _vec(values: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(values), dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected 3-vector, got {value.shape}")
    return value


def _inflate_aabb(box: Mapping[str, Any], scale: float) -> Dict[str, list[float]]:
    if scale < 1.0:
        raise ValueError("AABB center scale must be >= 1")
    center = _vec(box["center"])
    half = 0.5 * _vec(box["extent"]) * float(scale)
    return _aabb(center - half, center + half)


def _orthonormal_surface_frame(points: np.ndarray, room_center: np.ndarray) -> Dict[str, Any]:
    """Recover the room-facing rectangle of a thin scaffold surface.

    Room surfaces are explicit JSON scaffold solids.  This routine uses their
    transformed scaffold points and a PCA frame, so it is independent of object
    names and remains valid when the surface is rotated.
    """
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.copy()
    projections = centered @ axes.T
    extents = projections.max(axis=0) - projections.min(axis=0)
    thin = int(np.argmin(extents))
    in_plane = [axis for axis in range(3) if axis != thin]
    # Stable ordering: longer in-plane axis first.
    in_plane.sort(key=lambda axis: float(extents[axis]), reverse=True)
    u_axis = axes[in_plane[0]]
    v_axis = axes[in_plane[1]]
    normal_axis = axes[thin]
    if float(np.dot(normal_axis, room_center - center)) < 0.0:
        normal_axis = -normal_axis
    # Choose the thin face that points into the room.
    normal_projection = centered @ normal_axis
    face_offset = float(normal_projection.max())
    face_center = center + normal_axis * face_offset
    u_projection = centered @ u_axis
    v_projection = centered @ v_axis
    half_u = 0.5 * float(u_projection.max() - u_projection.min())
    half_v = 0.5 * float(v_projection.max() - v_projection.min())
    corners = np.asarray(
        [
            face_center + su * half_u * u_axis + sv * half_v * v_axis
            for su, sv in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0))
        ],
        dtype=np.float64,
    )
    edges = np.asarray(
        [
            0.5 * (corners[0] + corners[1]),
            0.5 * (corners[1] + corners[2]),
            0.5 * (corners[2] + corners[3]),
            0.5 * (corners[3] + corners[0]),
        ],
        dtype=np.float64,
    )
    return {
        "center": face_center,
        "normal": normal_axis,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "half_u": half_u,
        "half_v": half_v,
        "corners": corners,
        "edge_midpoints": edges,
        "area": max(4.0 * half_u * half_v, _EPS),
    }


def build_room_surface_model(
    scene: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    records = {str(item["object_id"]): dict(item) for item in flat_objects(scene)}
    points_by_id = scaffold_points(scene)
    surface_ids = [
        object_id
        for object_id, record in records.items()
        if str(dict(record.get("generation", {})).get("mode", "")) == "surface_texture"
        and object_id in points_by_id
    ]
    if not surface_ids:
        raise RuntimeError("Stage07 room coverage requires explicit surface_texture scaffold objects")
    all_surface_points = np.concatenate([points_by_id[object_id] for object_id in surface_ids], axis=0)
    approximate_center = all_surface_points.mean(axis=0)
    frames: Dict[str, Dict[str, Any]] = {}
    for object_id in sorted(surface_ids):
        frames[object_id] = _orthonormal_surface_frame(points_by_id[object_id], approximate_center)

    face_points = np.concatenate([frame["corners"] for frame in frames.values()], axis=0)
    room_minimum = face_points.min(axis=0)
    room_maximum = face_points.max(axis=0)
    room_extent = np.maximum(room_maximum - room_minimum, 1e-6)
    room_diagonal = float(np.linalg.norm(room_extent))
    room_volume = float(np.prod(room_extent))
    up_axis = _vec(dict(scene.get("scene", scene)).get("coordinate_system", {}).get("up_vector", [0.0, 0.0, 1.0]))
    up_axis /= max(float(np.linalg.norm(up_axis)), _EPS)

    coverage_cfg = dict(config.get("room_surface_coverage", {}))
    spacing_ratio = float(coverage_cfg.get("sample_spacing_room_diagonal_ratio", 0.018))
    spacing = max(spacing_ratio * room_diagonal, 1e-5)
    all_positions: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    all_surface_indices: list[np.ndarray] = []
    surface_slices: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for surface_index, object_id in enumerate(sorted(frames)):
        frame = frames[object_id]
        count_u = max(2, int(math.ceil((2.0 * frame["half_u"]) / spacing)) + 1)
        count_v = max(2, int(math.ceil((2.0 * frame["half_v"]) / spacing)) + 1)
        us = np.linspace(-frame["half_u"], frame["half_u"], count_u)
        vs = np.linspace(-frame["half_v"], frame["half_v"], count_v)
        grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")
        positions = (
            frame["center"][None, None, :]
            + grid_u[..., None] * frame["u_axis"][None, None, :]
            + grid_v[..., None] * frame["v_axis"][None, None, :]
        ).reshape(-1, 3)
        count = len(positions)
        all_positions.append(positions)
        all_normals.append(np.repeat(frame["normal"][None, :], count, axis=0))
        all_areas.append(np.full(count, frame["area"] / count, dtype=np.float64))
        all_surface_indices.append(np.full(count, surface_index, dtype=np.int32))
        surface_slices[object_id] = {
            "surface_index": surface_index,
            "start": offset,
            "end": offset + count,
            "grid_shape": [count_v, count_u],
        }
        offset += count
        frame["orientation"] = "vertical" if abs(float(np.dot(frame["normal"], up_axis))) < float(
            coverage_cfg.get("horizontal_normal_alignment_threshold", 0.70)
        ) else "horizontal"

    return {
        "surface_ids": sorted(frames),
        "frames": frames,
        "positions": np.concatenate(all_positions, axis=0),
        "normals": np.concatenate(all_normals, axis=0),
        "areas": np.concatenate(all_areas, axis=0),
        "surface_indices": np.concatenate(all_surface_indices, axis=0),
        "surface_slices": surface_slices,
        "room_aabb": _aabb(room_minimum, room_maximum),
        "room_diagonal_m": room_diagonal,
        "room_volume_m3": room_volume,
        "sample_spacing_m": spacing,
    }


def camera_basis(position: Iterable[float], target: Iterable[float], up: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = _vec(position)
    target = _vec(target)
    forward = target - position
    length = float(np.linalg.norm(forward))
    if length <= 1e-8:
        raise ValueError("Camera position coincides with look-at target")
    forward /= length
    up_value = _vec(up)
    up_value /= max(float(np.linalg.norm(up_value)), _EPS)
    right = np.cross(forward, up_value)
    if float(np.linalg.norm(right)) <= 1e-8:
        fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, fallback))) > 0.95:
            fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback)
    right /= max(float(np.linalg.norm(right)), _EPS)
    true_up = np.cross(right, forward)
    true_up /= max(float(np.linalg.norm(true_up)), _EPS)
    return right, true_up, forward


def room_samples_in_frustum(
    position: Iterable[float],
    target: Iterable[float],
    room_model: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    focal_length_mm: float,
) -> np.ndarray:
    coverage_cfg = dict(config.get("room_surface_coverage", {}))
    camera_model = dict(config.get("camera_model", {}))
    width, height = [int(value) for value in camera_model.get("resolution", [1376, 768])]
    sensor_width = float(camera_model.get("sensor_width_mm", 36.0))
    fx = focal_length_mm / max(sensor_width, 1e-6) * width
    fy = fx
    margin_ratio = float(coverage_cfg.get("image_border_margin_ratio", 0.02))
    margin_x = margin_ratio * width
    margin_y = margin_ratio * height
    near_ratio = float(coverage_cfg.get("near_room_diagonal_ratio", 0.003))
    far_ratio = float(coverage_cfg.get("far_room_diagonal_ratio", 2.0))
    near = max(near_ratio * float(room_model["room_diagonal_m"]), 1e-5)
    far = max(far_ratio * float(room_model["room_diagonal_m"]), near + 1e-5)

    right, up, forward = camera_basis(position, target, [0.0, 0.0, 1.0])
    relative = np.asarray(room_model["positions"], dtype=np.float64) - _vec(position)[None, :]
    x = relative @ right
    y = relative @ up
    z = relative @ forward
    valid = (z > near) & (z < far)
    u = fx * x / np.maximum(z, _EPS) + 0.5 * width
    v = -fy * y / np.maximum(z, _EPS) + 0.5 * height
    valid &= (u >= margin_x) & (u < width - margin_x) & (v >= margin_y) & (v < height - margin_y)
    normals = np.asarray(room_model["normals"], dtype=np.float64)
    toward_camera = _vec(position)[None, :] - np.asarray(room_model["positions"], dtype=np.float64)
    valid &= np.sum(normals * toward_camera, axis=1) > 0.0
    return np.flatnonzero(valid).astype(np.int32)


def gaussian_kernel_1d(radius: int, sigma: float) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius == 0:
        return np.ones(1, dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(float(sigma), 1e-6)) ** 2)
    kernel /= max(float(kernel.sum()), _EPS)
    return kernel


def _convolve_same_1d(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full = np.convolve(values, kernel, mode="full")
    start = (len(kernel) - 1) // 2
    return full[start : start + len(values)]


@dataclass
class RoomCoverageState:
    hard_covered: np.ndarray
    accumulation: np.ndarray
    room_model: Mapping[str, Any]
    config: Mapping[str, Any]

    @classmethod
    def create(cls, room_model: Mapping[str, Any], config: Mapping[str, Any]) -> "RoomCoverageState":
        count = int(len(room_model["positions"]))
        return cls(
            hard_covered=np.zeros(count, dtype=bool),
            accumulation=np.zeros(count, dtype=np.float64),
            room_model=room_model,
            config=config,
        )

    def normalized(self) -> np.ndarray:
        cfg = dict(self.config.get("room_surface_coverage", {})).get("coverage_influence", {})
        saturation = max(float(cfg.get("saturation", 1.0)), 1e-8)
        return 1.0 - np.exp(-self.accumulation / saturation)

    def update(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        self.hard_covered[indices] = True
        cfg = dict(self.config.get("room_surface_coverage", {})).get("coverage_influence", {})
        radius = max(int(round(float(cfg.get("radius_in_sample_spacings", 3.0)))), 0)
        sigma = max(float(cfg.get("sigma_in_sample_spacings", max(radius / 2.0, 1.0))), 1e-6)
        kernel = gaussian_kernel_1d(radius, sigma)
        selected = np.zeros_like(self.accumulation)
        selected[indices] = 1.0
        for object_id, record in self.room_model["surface_slices"].items():
            start, end = int(record["start"]), int(record["end"])
            rows, cols = [int(value) for value in record["grid_shape"]]
            grid = selected[start:end].reshape(rows, cols)
            horizontal = np.vstack([_convolve_same_1d(row, kernel) for row in grid])
            blurred = np.vstack([_convolve_same_1d(col, kernel) for col in horizontal.T]).T
            # A selected region contributes one unit at its core.  Nearby points
            # receive a smoothly decaying value; repeated views add naturally.
            peak = max(float(blurred.max(initial=0.0)), _EPS)
            self.accumulation[start:end] += (blurred / peak).reshape(-1)

    def hard_coverage_ratio(self) -> float:
        areas = np.asarray(self.room_model["areas"], dtype=np.float64)
        return float(np.sum(areas[self.hard_covered]) / max(float(np.sum(areas)), _EPS))


class IncrementalRoomCoverageGraph:
    """Incrementally build the complete Stage07 camera-correlation graph.

    Every newly accepted camera is tested against every previously accepted
    camera.  The shared room-surface support is measured with an area-weighted
    symmetric overlap metric (Dice by default). Camera orientation is retained
    only as diagnostic metadata and does not attenuate the edge weight. Every pair
    whose overlap correlation reaches the configured threshold becomes an edge.
    Union-find is updated only for those retained edges, so the live component
    count exactly matches the graph later consumed by Stage08.
    """

    def __init__(
        self,
        sample_areas: np.ndarray | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        if sample_areas is None:
            self.sample_areas: np.ndarray | None = None
        else:
            areas = np.asarray(sample_areas, dtype=np.float64).reshape(-1)
            if np.any(~np.isfinite(areas)) or np.any(areas < 0.0):
                raise ValueError("sample_areas must be finite and non-negative")
            self.sample_areas = areas
        cfg = dict(config or {})
        self.overlap_metric = str(cfg.get("overlap_metric", "area_weighted_dice"))
        if self.overlap_metric not in {"area_weighted_dice", "area_weighted_iou"}:
            raise ValueError(f"Unsupported Stage07 overlap metric: {self.overlap_metric}")
        self.minimum_edge_correlation_score = float(
            cfg.get("minimum_edge_correlation_score", 0.05)
        )
        if not 0.0 <= self.minimum_edge_correlation_score <= 1.0:
            raise ValueError("minimum_edge_correlation_score must be in [0, 1]")
        self.camera_ids: list[str] = []
        self.sample_sets: list[np.ndarray] = []
        self.covered_areas: list[float] = []
        self.view_directions: list[np.ndarray] = []
        self.edges: list[Dict[str, Any]] = []
        self._parent: list[int] = []
        self._rank: list[int] = []
        self._component_count = 0

    def _area_for(self, indices: np.ndarray) -> float:
        if indices.size == 0:
            return 0.0
        if self.sample_areas is None:
            return float(indices.size)
        if int(indices.min()) < 0 or int(indices.max()) >= len(self.sample_areas):
            raise IndexError("room sample index is outside sample_areas")
        return float(self.sample_areas[indices].sum())

    @staticmethod
    def _normalise_direction(value: np.ndarray | Sequence[float] | None) -> np.ndarray:
        if value is None:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        direction = np.asarray(value, dtype=np.float64).reshape(-1)
        if direction.shape != (3,) or np.any(~np.isfinite(direction)):
            raise ValueError("camera view direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:
            raise ValueError("camera view direction must be non-zero")
        return direction / norm

    def _find(self, node: int) -> int:
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def _union(self, first: int, second: int) -> bool:
        first_root = self._find(first)
        second_root = self._find(second)
        if first_root == second_root:
            return False
        if self._rank[first_root] < self._rank[second_root]:
            first_root, second_root = second_root, first_root
        self._parent[second_root] = first_root
        if self._rank[first_root] == self._rank[second_root]:
            self._rank[first_root] += 1
        self._component_count -= 1
        return True

    @property
    def component_count(self) -> int:
        return int(self._component_count)

    @property
    def connected(self) -> bool:
        return bool(self.camera_ids and self._component_count == 1)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def add(
        self,
        camera_id: str,
        sample_indices: np.ndarray,
        view_direction: np.ndarray | Sequence[float] | None = None,
    ) -> int:
        camera_id = str(camera_id)
        if camera_id in self.camera_ids:
            raise ValueError(f"Duplicate Stage07 graph camera ID: {camera_id}")
        values = np.unique(np.asarray(sample_indices, dtype=np.int32))
        direction = self._normalise_direction(view_direction)
        index = len(self.camera_ids)
        covered_area = self._area_for(values)
        self.camera_ids.append(camera_id)
        self.sample_sets.append(values)
        self.covered_areas.append(covered_area)
        self.view_directions.append(direction)
        self._parent.append(index)
        self._rank.append(0)
        self._component_count += 1

        # Test the new camera against every previously accepted camera.  Keep
        # every pair that passes the *final* correlation test; no tree reduction
        # is performed here or later in Stage08.
        for previous in range(index):
            shared = np.intersect1d(self.sample_sets[previous], values, assume_unique=True)
            if shared.size == 0:
                continue
            shared_area = self._area_for(shared)
            previous_area = self.covered_areas[previous]
            union_area = previous_area + covered_area - shared_area
            weighted_iou = float(shared_area / union_area) if union_area > _EPS else 0.0
            weighted_dice = float(
                (2.0 * shared_area) / (previous_area + covered_area)
            ) if previous_area + covered_area > _EPS else 0.0
            overlap_score = weighted_dice if self.overlap_metric == "area_weighted_dice" else weighted_iou
            raw_cosine = float(np.dot(self.view_directions[previous], direction))
            correlation_score = float(overlap_score)
            if correlation_score <= _EPS or (
                correlation_score + _EPS < self.minimum_edge_correlation_score
            ):
                continue
            angle_degrees = float(math.degrees(math.acos(max(-1.0, min(1.0, raw_cosine)))))
            self.edges.append({
                "source_index": previous,
                "target_index": index,
                "source_camera_id": self.camera_ids[previous],
                "target_camera_id": camera_id,
                "shared_room_sample_count": int(shared.size),
                "shared_room_area": shared_area,
                "source_covered_room_area": float(previous_area),
                "target_covered_room_area": float(covered_area),
                "area_weighted_iou": weighted_iou,
                "area_weighted_dice": weighted_dice,
                "overlap_metric": self.overlap_metric,
                "overlap_score": float(overlap_score),
                "view_direction_cosine": raw_cosine,
                "view_angle_degrees": angle_degrees,
                "correlation_score": correlation_score,
            })
            self._union(previous, index)
        return index

    def to_dict(self) -> Dict[str, Any]:
        groups: Dict[int, list[int]] = {}
        for index in range(len(self.camera_ids)):
            groups.setdefault(self._find(index), []).append(index)
        components = sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])
        nodes = [
            {
                "index": index,
                "camera_id": camera_id,
                "covered_room_sample_count": int(self.sample_sets[index].size),
                "covered_room_area": float(self.covered_areas[index]),
                "view_direction": self.view_directions[index].astype(float).tolist(),
            }
            for index, camera_id in enumerate(self.camera_ids)
        ]
        return {
            "schema_version": 5,
            "graph_type": "stage07_room_surface_overlap_correlation",
            "construction": "incremental_new_camera_against_all_previous_accepted_cameras",
            "overlap_test": "area_weighted_symmetric_overlap_only",
            "edge_policy": "retain_every_accepted_camera_pair_whose_final_correlation_reaches_threshold",
            "overlap_metric": self.overlap_metric,
            "minimum_edge_correlation_score": self.minimum_edge_correlation_score,
            "edge_weight": "correlation_score",
            "node_count": len(self.camera_ids),
            "edge_count": len(self.edges),
            "nodes": nodes,
            "edges": list(self.edges),
            "components": components,
            "components_camera_ids": [
                [self.camera_ids[index] for index in component]
                for component in components
            ],
            "component_count": self.component_count,
            "connected": self.connected,
            "connectivity_runtime": "incremental_union_find_on_final_correlation_edges",
        }


def graph_from_sample_sets(
    sample_sets: Sequence[np.ndarray],
    sample_areas: np.ndarray | None = None,
    camera_ids: Sequence[str] | None = None,
    camera_directions: Sequence[Sequence[float]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the Stage07 graph using the same incremental implementation."""
    count = len(sample_sets)
    if camera_ids is not None and len(camera_ids) != count:
        raise ValueError("camera_ids length must match sample_sets length")
    if camera_directions is not None and len(camera_directions) != count:
        raise ValueError("camera_directions length must match sample_sets length")
    ids = list(camera_ids) if camera_ids is not None else [str(index) for index in range(count)]
    directions = list(camera_directions) if camera_directions is not None else [None] * count
    state = IncrementalRoomCoverageGraph(sample_areas, config)
    for camera_id, values, direction in zip(ids, sample_sets, directions):
        state.add(str(camera_id), values, direction)
    graph = state.to_dict()
    if camera_ids is None:
        for node in graph["nodes"]:
            node.pop("camera_id", None)
        for edge in graph["edges"]:
            edge.pop("source_camera_id", None)
            edge.pop("target_camera_id", None)
        graph["components_camera_ids"] = []
    return graph


def serialize_room_model(room_model: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "surface_ids": list(room_model["surface_ids"]),
        "room_aabb": dict(room_model["room_aabb"]),
        "room_diagonal_m": float(room_model["room_diagonal_m"]),
        "room_volume_m3": float(room_model["room_volume_m3"]),
        "sample_spacing_m": float(room_model["sample_spacing_m"]),
        "sample_count": int(len(room_model["positions"])),
        "surface_slices": dict(room_model["surface_slices"]),
        "frames": {
            object_id: {
                "center": frame["center"].astype(float).tolist(),
                "normal": frame["normal"].astype(float).tolist(),
                "u_axis": frame["u_axis"].astype(float).tolist(),
                "v_axis": frame["v_axis"].astype(float).tolist(),
                "half_u": float(frame["half_u"]),
                "half_v": float(frame["half_v"]),
                "area": float(frame["area"]),
                "orientation": str(frame["orientation"]),
                "corners": frame["corners"].astype(float).tolist(),
                "edge_midpoints": frame["edge_midpoints"].astype(float).tolist(),
            }
            for object_id, frame in room_model["frames"].items()
        },
    }
