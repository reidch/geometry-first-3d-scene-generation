from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from src.appearance.triangle_id_map import load_triangle_id_map
from src.cameras.reconstruction_view_metrics import decode_metric_depth
from src.cameras.depth_geometry import backproject_camera_z, depth_convention
from src.io.json_io import load_json


_EPS = 1e-8


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _match_warp_luminance_to_source(
    warped_srgb: np.ndarray,
    source_srgb: np.ndarray,
    valid_mask: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Match each valid warped pixel to the source pixel luminance.

    V82 deliberately changes only luminance before reference fusion.  Hue and
    chromatic ratios come from the warped reference, while target brightness
    comes from the current Stage07 RGB at the same pixel.  The operation is
    performed in linear RGB so PNG/sRGB gamma does not create dark-region
    colour shifts.
    """
    cfg = dict(config.get("luminance_matching", {}))
    if not bool(cfg.get("enabled", False)):
        return np.asarray(warped_srgb, dtype=np.float32), {
            "enabled": False,
            "method": "disabled",
        }

    coefficients = np.asarray(
        cfg.get("linear_rgb_coefficients", [0.2126, 0.7152, 0.0722]),
        dtype=np.float32,
    )
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all() or float(coefficients.sum()) <= 0.0:
        raise ValueError("luminance_matching.linear_rgb_coefficients must contain three finite positive-sum values")
    coefficients = coefficients / float(coefficients.sum())

    epsilon = max(float(cfg.get("epsilon", 1.0e-4)), _EPS)
    minimum_gain = float(cfg.get("minimum_gain", 0.25))
    maximum_gain = float(cfg.get("maximum_gain", 4.0))
    if not (0.0 < minimum_gain <= maximum_gain):
        raise ValueError("luminance matching requires 0 < minimum_gain <= maximum_gain")

    warped = np.clip(np.asarray(warped_srgb, dtype=np.float32), 0.0, 1.0)
    source = np.clip(np.asarray(source_srgb, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(valid_mask, dtype=bool)
    if warped.shape != source.shape or warped.shape[:2] != mask.shape:
        raise ValueError("warped/source/mask shapes must agree for luminance matching")

    warped_linear = _srgb_to_linear(warped)
    source_linear = _srgb_to_linear(source)
    warped_luminance = np.sum(warped_linear * coefficients[None, None, :], axis=2)
    target_luminance = np.sum(source_linear * coefficients[None, None, :], axis=2)

    raw_gain = (target_luminance + epsilon) / (warped_luminance + epsilon)
    gain = np.clip(raw_gain, minimum_gain, maximum_gain).astype(np.float32)
    matched_linear = np.clip(warped_linear * gain[..., None], 0.0, 1.0)
    matched = _linear_to_srgb(matched_linear)
    matched[~mask] = warped[~mask]

    matched_luminance = np.sum(matched_linear * coefficients[None, None, :], axis=2)
    valid_count = int(mask.sum())
    if valid_count:
        before_error = float(np.mean(np.abs(warped_luminance[mask] - target_luminance[mask])))
        after_error = float(np.mean(np.abs(matched_luminance[mask] - target_luminance[mask])))
        clipped_fraction = float(np.mean((raw_gain[mask] < minimum_gain) | (raw_gain[mask] > maximum_gain)))
        mean_gain = float(np.mean(gain[mask]))
        median_gain = float(np.median(gain[mask]))
        mean_before = float(np.mean(warped_luminance[mask]))
        mean_target = float(np.mean(target_luminance[mask]))
        mean_after = float(np.mean(matched_luminance[mask]))
    else:
        before_error = after_error = clipped_fraction = mean_gain = median_gain = 0.0
        mean_before = mean_target = mean_after = 0.0

    return matched, {
        "enabled": True,
        "method": "per_pixel_linear_rgb_luminance_gain",
        "valid_pixel_count": valid_count,
        "minimum_gain": minimum_gain,
        "maximum_gain": maximum_gain,
        "mean_gain": mean_gain,
        "median_gain": median_gain,
        "clipped_gain_fraction": clipped_fraction,
        "mean_linear_luminance_before": mean_before,
        "mean_linear_luminance_target": mean_target,
        "mean_linear_luminance_after": mean_after,
        "mean_absolute_luminance_error_before": before_error,
        "mean_absolute_luminance_error_after": after_error,
    }


def _load_camera(frame: Mapping[str, Any]) -> Dict[str, Any]:
    source = frame.get("camera")
    if isinstance(source, Mapping):
        return dict(source)
    return load_json(source)


def build_stage07_overlap_runtime(
    frames: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and index the complete Stage07 correlation graph.

    Stage08 consumes every Stage07 edge that passed the final symmetric-overlap
    times view-direction test.  It never rebuilds a pairwise image graph and
    never reduces the graph to a tree.
    """
    if not frames:
        raise ValueError("At least one Stage07 frame is required")
    if not bool(graph.get("connected", False)) or int(graph.get("component_count", 0)) != 1:
        raise RuntimeError("Stage08 requires the connected Stage07 camera-correlation graph")
    if str(graph.get("edge_weight", "")) != "correlation_score":
        raise RuntimeError("Stage08 requires Stage07 graph edges weighted by correlation_score")

    frame_index_by_id = {str(frame["camera_id"]): index for index, frame in enumerate(frames)}
    if len(frame_index_by_id) != len(frames):
        raise ValueError("Stage07 frame camera IDs must be unique")

    coverage_by_id: Dict[str, Dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        camera_id = str(node.get("camera_id", ""))
        if not camera_id and "index" in node:
            index = int(node["index"])
            if 0 <= index < len(frames):
                camera_id = str(frames[index]["camera_id"])
        if camera_id:
            coverage_by_id[camera_id] = {
                "covered_room_area": float(node.get("covered_room_area", 0.0)),
                "covered_room_sample_count": int(node.get("covered_room_sample_count", 0)),
                "view_direction": list(node.get("view_direction", [])),
            }

    missing = sorted(set(frame_index_by_id) - set(coverage_by_id))
    if missing:
        raise RuntimeError(f"Stage07 graph is missing frame nodes: {missing}")

    adjacency: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    weighted_degree = {camera_id: 0.0 for camera_id in frame_index_by_id}
    for edge in graph.get("edges", []):
        source_id = str(edge.get("source_camera_id", ""))
        target_id = str(edge.get("target_camera_id", ""))
        if not source_id and "source_index" in edge:
            source_id = str(frames[int(edge["source_index"])]["camera_id"])
        if not target_id and "target_index" in edge:
            target_id = str(frames[int(edge["target_index"])]["camera_id"])
        if source_id not in frame_index_by_id or target_id not in frame_index_by_id:
            raise RuntimeError("Stage07 graph edge references an unknown camera")
        score = float(edge.get("correlation_score", 0.0))
        if not 0.0 < score <= 1.0:
            raise RuntimeError("Stage07 retained edge has an invalid correlation_score")
        common = {
            "correlation_score": score,
            "area_weighted_iou": float(edge.get("area_weighted_iou", 0.0)),
            "area_weighted_dice": float(edge.get("area_weighted_dice", 0.0)),
            "overlap_score": float(edge.get("overlap_score", 0.0)),
            "view_direction_cosine": float(edge.get("view_direction_cosine", 0.0)),
            "nonnegative_view_direction_cosine": float(
                edge.get("nonnegative_view_direction_cosine", 0.0)
            ),
            "view_direction_score": float(edge.get("view_direction_score", 0.0)),
            "view_angle_degrees": float(edge.get("view_angle_degrees", 0.0)),
            "shared_room_area": float(edge.get("shared_room_area", 0.0)),
            "shared_room_sample_count": int(edge.get("shared_room_sample_count", 0)),
        }
        adjacency[source_id].append({"camera_id": target_id, **common})
        adjacency[target_id].append({"camera_id": source_id, **common})
        weighted_degree[source_id] += score
        weighted_degree[target_id] += score

    for camera_id in frame_index_by_id:
        adjacency[camera_id].sort(
            key=lambda item: (
                -float(item["correlation_score"]),
                -float(item["shared_room_area"]),
                str(item["camera_id"]),
            )
        )

    # Stage08 must bootstrap appearance from one of the two dedicated Stage07
    # short-wall bootstrap/coverage cameras.  Graph-central perimeter or
    # overhead views can be excellent propagation nodes but are intentionally
    # forbidden from becoming the first no-reference generation view.
    bootstrap_camera_ids = [
        str(frame["camera_id"])
        for frame in frames
        if str(frame.get("camera_role", "")) == "room_coverage"
    ]
    if len(bootstrap_camera_ids) != 2:
        raise RuntimeError(
            "Stage08 requires exactly two Stage07 room_coverage bootstrap cameras; "
            f"found {len(bootstrap_camera_ids)}: {bootstrap_camera_ids}"
        )

    # Between the two bootstraps, retain the existing graph criterion: prefer
    # the one with the larger total incident correlation, then room coverage,
    # then stable Stage07 frame order as the deterministic final tie breaker.
    root_camera_id = max(
        bootstrap_camera_ids,
        key=lambda camera_id: (
            weighted_degree[camera_id],
            coverage_by_id[camera_id]["covered_room_area"],
            coverage_by_id[camera_id]["covered_room_sample_count"],
            -frame_index_by_id[camera_id],
        ),
    )
    return {
        "schema_version": 2,
        "source_graph_type": str(graph.get("graph_type", "")),
        "graph_policy": "complete_stage07_correlation_graph_without_tree_reduction",
        "edge_weight": "correlation_score",
        "frame_index_by_id": frame_index_by_id,
        "coverage_by_id": coverage_by_id,
        "weighted_degree_by_id": weighted_degree,
        "adjacency": dict(adjacency),
        "bootstrap_candidate_camera_ids": list(bootstrap_camera_ids),
        "initial_bootstrap_camera_id": root_camera_id,
    }


def initialize_generated_neighbor_registry(
    frames: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Create the V112/V124 completed-vs-successful Stage08 registry.

    ``generated_neighbors`` is intentionally a *successful/trusted appearance*
    neighbour list.  Fallback-completed cameras never enter it.  Completion is
    tracked separately so the original Stage07 topology may still be used to
    choose a restart node when the successful frontier is empty.
    """
    coverage_by_id = dict(runtime["coverage_by_id"])
    degree_by_id = dict(runtime["weighted_degree_by_id"])
    return {
        str(frame["camera_id"]): {
            "status": "pending",
            "completed": False,
            "successful": False,
            "can_be_reference": False,
            "reference_trust": "none",
            "effective_reference_edge_weight": 0.0,
            "covered_room_area": float(coverage_by_id[str(frame["camera_id"])]["covered_room_area"]),
            "covered_room_sample_count": int(
                coverage_by_id[str(frame["camera_id"])]["covered_room_sample_count"]
            ),
            "weighted_graph_degree": float(degree_by_id[str(frame["camera_id"])]),
            "generated_neighbors": [],
            "propagation_support_score": 0.0,
            "last_attempt_generated_neighbor_signature": [],
            "generation_order_index": None,
            "attempt_round_count": 0,
        }
        for frame in frames
    }


def generated_neighbor_signature(slot: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item["camera_id"]) for item in slot.get("generated_neighbors", [])))


def propagation_support_score(generated_neighbors: Sequence[Mapping[str, Any]]) -> float:
    complement = 1.0
    for item in generated_neighbors:
        score = min(max(float(item.get("correlation_score", 0.0)), 0.0), 1.0)
        complement *= 1.0 - score
    return float(1.0 - complement)


def select_next_frontier_camera(
    registry: Mapping[str, Mapping[str, Any]],
    frame_index_by_id: Mapping[str, int],
) -> str | None:
    """Choose the best unfinished node reachable from successful references only."""
    eligible: list[str] = []
    for camera_id, slot in registry.items():
        if bool(slot.get("completed", False)) or str(slot.get("status", "pending")) != "pending":
            continue
        neighbors = list(slot.get("generated_neighbors", []))
        if neighbors:
            eligible.append(camera_id)
    if not eligible:
        return None

    def key(camera_id: str) -> tuple[float, float, int, float, int]:
        slot = registry[camera_id]
        neighbors = list(slot.get("generated_neighbors", []))
        support = propagation_support_score(neighbors)
        strongest = max(float(item["correlation_score"]) for item in neighbors)
        return (
            -support,
            -strongest,
            -len(neighbors),
            -float(slot.get("covered_room_area", 0.0)),
            int(frame_index_by_id[camera_id]),
        )

    return min(eligible, key=key)


def select_restart_camera_from_completed_topology(
    registry: Mapping[str, Mapping[str, Any]],
    frame_index_by_id: Mapping[str, int],
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str | None:
    """Choose one unfinished restart using topology only, never fallback RGB.

    This restores the V112 separation between traversal topology and appearance
    evidence.  Any completed camera (strict or fallback) can make an adjacent
    unfinished node topologically reachable, but only strict/successful cameras
    populate ``generated_neighbors`` and therefore become image references.
    """
    completed = {
        camera_id for camera_id, slot in registry.items()
        if bool(slot.get("completed", False))
    }
    unfinished = [
        camera_id for camera_id, slot in registry.items()
        if not bool(slot.get("completed", False)) and str(slot.get("status", "pending")) == "pending"
    ]
    if not unfinished:
        return None

    candidates: list[tuple[str, list[float]]] = []
    for camera_id in unfinished:
        scores = [
            float(edge.get("correlation_score", 0.0))
            for edge in adjacency.get(camera_id, [])
            if str(edge.get("camera_id")) in completed
        ]
        if scores:
            candidates.append((camera_id, scores))
    if not candidates:
        return None

    def key(item: tuple[str, list[float]]) -> tuple[float, float, float, int]:
        camera_id, scores = item
        support = propagation_support_score([{"correlation_score": value} for value in scores])
        strongest = max(scores)
        return (
            -support,
            -strongest,
            -float(registry[camera_id].get("covered_room_area", 0.0)),
            int(frame_index_by_id[camera_id]),
        )

    return min(candidates, key=key)[0]


def update_generated_neighbor_registry(
    registry: Dict[str, Dict[str, Any]],
    generated_camera_id: str,
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    """Propagate one strict/successful Stage08 view as trusted appearance evidence."""
    source_slot = registry[str(generated_camera_id)]
    if not bool(source_slot.get("successful", False)) or not bool(source_slot.get("can_be_reference", False)):
        # Completed fallback views deliberately have zero appearance propagation.
        return []
    updates: list[Dict[str, Any]] = []
    for edge in adjacency.get(str(generated_camera_id), []):
        neighbor_id = str(edge["camera_id"])
        slot = registry[neighbor_id]
        if bool(slot.get("completed", False)) or str(slot.get("status")) == "generating":
            continue
        existing = {str(item["camera_id"]) for item in slot.get("generated_neighbors", [])}
        if str(generated_camera_id) in existing:
            continue
        record = {
            "camera_id": str(generated_camera_id),
            "correlation_score": float(edge["correlation_score"]),
            "area_weighted_iou": float(edge.get("area_weighted_iou", 0.0)),
            "area_weighted_dice": float(edge.get("area_weighted_dice", 0.0)),
            "view_direction_cosine": float(edge.get("view_direction_cosine", 0.0)),
            "view_angle_degrees": float(edge.get("view_angle_degrees", 0.0)),
            "reference_trust": "strict_success",
            "effective_reference_edge_weight": float(edge["correlation_score"]),
        }
        slot.setdefault("generated_neighbors", []).append(record)
        slot["generated_neighbors"].sort(
            key=lambda item: (-float(item["correlation_score"]), str(item["camera_id"]))
        )
        slot["propagation_support_score"] = propagation_support_score(
            slot["generated_neighbors"]
        )
        updates.append({
            "camera_id": neighbor_id,
            "added_generated_neighbor": record,
            "propagation_support_score": slot["propagation_support_score"],
        })
    return updates

def _resize_array(array: np.ndarray, width: int, height: int, *, nearest: bool) -> np.ndarray:
    if array.shape[:2] == (height, width):
        return array
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if array.ndim == 2:
        image = Image.fromarray(array)
        return np.asarray(image.resize((width, height), mode))
    image = Image.fromarray(array)
    return np.asarray(image.resize((width, height), mode))


def _backproject_current_pixels(
    current_depth: np.ndarray,
    current_depth_encoding: Mapping[str, Any],
    current_camera: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject project metric depth under the camera-Z contract.

    Pixel centers are (x+0.5, y+0.5).  For Z-depth, K^{-1}[u,v,1]^T
    already has third component one, therefore P_cam = Z * ray.
    """

    height, width = current_depth.shape
    if int(current_camera["width"]) != width or int(current_camera["height"]) != height:
        raise ValueError(
            "Current depth resolution must match its camera calibration: "
            f"depth={width}x{height}, camera={current_camera['width']}x{current_camera['height']}"
        )
    convention = depth_convention(current_depth_encoding)
    if convention != "camera_z":
        raise ValueError(f"Stage08 requires camera-Z depth, got {convention}")
    valid = np.isfinite(current_depth) & (current_depth > 0.0)
    camera_points = backproject_camera_z(
        current_depth, np.asarray(current_camera["K"], dtype=np.float64), pixel_center_offset=0.5
    )
    camera_points_xyz = camera_points[valid]
    homogeneous = np.concatenate(
        [camera_points_xyz.T, np.ones((1, camera_points_xyz.shape[0]), dtype=np.float64)],
        axis=0,
    )
    c2w = np.asarray(current_camera["camera_to_world_opencv"], dtype=np.float64)
    world = c2w @ homogeneous
    ys, xs = np.nonzero(valid)
    return world, valid, np.stack([ys, xs], axis=1).astype(np.int32)

def reproject_reference_to_current(
    reference_rgb_path: str | Path,
    reference_frame: Mapping[str, Any],
    current_frame: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    scene_scale_m: float | None = None,
) -> Dict[str, Any]:
    current_camera = _load_camera(current_frame)
    reference_camera = _load_camera(reference_frame)
    current_width = int(current_camera["width"])
    current_height = int(current_camera["height"])
    reference_width = int(reference_camera["width"])
    reference_height = int(reference_camera["height"])

    reference_rgb = np.asarray(Image.open(reference_rgb_path).convert("RGB"))
    if reference_rgb.shape[:2] != (reference_height, reference_width):
        raise ValueError(
            "Reference RGB resolution must match reference camera calibration; "
            "resizing an image without changing K corrupts reprojection geometry"
        )
    current_depth = decode_metric_depth(current_frame["depth"], current_frame["depth_encoding"]).astype(np.float32)
    reference_depth = decode_metric_depth(reference_frame["depth"], reference_frame["depth_encoding"]).astype(np.float32)
    if current_depth.shape != (current_height, current_width):
        raise ValueError("Current depth resolution does not match current camera calibration")
    if reference_depth.shape != (reference_height, reference_width):
        raise ValueError("Reference depth resolution does not match reference camera calibration")
    current_triangle = load_triangle_id_map(current_frame["triangle_id"]).astype(np.int32)
    reference_triangle = load_triangle_id_map(reference_frame["triangle_id"]).astype(np.int32)
    if current_triangle.shape != current_depth.shape or reference_triangle.shape != reference_depth.shape:
        raise ValueError("Triangle-ID and metric-depth buffers must share each camera's native resolution")

    # Semantic-owner buffers are the stable cross-view surface identity contract.
    # Exact rasterized triangle IDs are too brittle for reprojection because a
    # sub-pixel projection shift can land on an adjacent triangle of the same
    # wall/object and incorrectly discard otherwise valid reference support.
    current_semantic = np.asarray(Image.open(current_frame["semantic"]).convert("RGB"), dtype=np.uint8)
    reference_semantic = np.asarray(Image.open(reference_frame["semantic"]).convert("RGB"), dtype=np.uint8)
    if current_semantic.shape[:2] != current_depth.shape or reference_semantic.shape[:2] != reference_depth.shape:
        raise ValueError("Semantic-owner and metric-depth buffers must share each camera's native resolution")

    world, valid_current, current_pixels = _backproject_current_pixels(
        current_depth, current_frame["depth_encoding"], current_camera
    )
    w2c_ref = np.asarray(reference_camera["world_to_camera_opencv"], dtype=np.float64)
    reference_points = w2c_ref @ world
    reference_xyz = reference_points[:3]
    projected_z = reference_xyz[2]
    K_ref = np.asarray(reference_camera["K"], dtype=np.float64)
    projected = K_ref @ reference_xyz
    u = projected[0] / np.maximum(projected[2], _EPS) - 0.5
    v = projected[1] / np.maximum(projected[2], _EPS) - 0.5
    xi = np.rint(u).astype(np.int64)
    yi = np.rint(v).astype(np.int64)
    inside = (projected_z > 0.0) & (xi >= 0) & (xi < reference_width) & (yi >= 0) & (yi < reference_height)

    warped = np.zeros((current_height, current_width, 3), dtype=np.uint8)
    mask = np.zeros((current_height, current_width), dtype=bool)
    confidence = np.zeros((current_height, current_width), dtype=np.float32)
    if inside.any():
        current_y = current_pixels[inside, 0]
        current_x = current_pixels[inside, 1]
        ref_y = yi[inside]
        ref_x = xi[inside]
        sampled_ref_depth = reference_depth[ref_y, ref_x]
        sampled_ref_triangle = reference_triangle[ref_y, ref_x]
        sampled_current_triangle = current_triangle[current_y, current_x]
        sampled_ref_semantic = reference_semantic[ref_y, ref_x]
        sampled_current_semantic = current_semantic[current_y, current_x]

        reference_convention = depth_convention(reference_frame["depth_encoding"])
        if reference_convention != "camera_z":
            raise ValueError(f"Stage08 requires camera-Z reference depth, got {reference_convention}")
        projected_depth_for_compare = projected_z[inside]

        reproj_cfg = dict(config.get("reference_reprojection", {}))
        relative_tolerance = float(reproj_cfg.get("depth_consistency_relative_tolerance", 0.02))
        absolute_ratio = float(reproj_cfg.get("depth_consistency_room_diagonal_ratio", 0.001))
        valid_depth_values = current_depth[current_depth > 0.0]
        fallback_scale = float(np.quantile(valid_depth_values, 0.98)) if valid_depth_values.size else 1.0
        room_scale = max(float(scene_scale_m or fallback_scale), 1e-6)
        tolerance = np.maximum(
            relative_tolerance * np.maximum(projected_depth_for_compare, sampled_ref_depth),
            absolute_ratio * room_scale,
        )
        depth_residual = np.abs(sampled_ref_depth - projected_depth_for_compare)
        reliable = (
            (sampled_ref_depth > 0.0)
            & (sampled_ref_triangle >= 0)
            & (sampled_current_triangle >= 0)
        )
        if bool(reproj_cfg.get("require_depth_consistency", True)):
            reliable &= depth_residual <= tolerance
        if bool(reproj_cfg.get("require_owner_match", True)):
            # Semantic palette colors are stable per semantic owner across all
            # Stage07 cameras. This permits continuous propagation across the
            # triangles composing one wall/object while still blocking cross-
            # owner leakage at object and room-surface boundaries.
            reliable &= np.all(sampled_ref_semantic == sampled_current_semantic, axis=1)
        if bool(reproj_cfg.get("require_triangle_match", False)):
            # Optional debug/legacy gate only. Owner + metric depth is the
            # production cross-view contract; exact triangle equality is much
            # more brittle under rasterization.
            reliable &= sampled_ref_triangle == sampled_current_triangle
        local_confidence = np.clip(
            1.0 - depth_residual / np.maximum(tolerance, _EPS), 0.0, 1.0
        ).astype(np.float32)
        current_y = current_y[reliable]
        current_x = current_x[reliable]
        ref_y = ref_y[reliable]
        ref_x = ref_x[reliable]
        local_confidence = local_confidence[reliable]
        warped[current_y, current_x] = reference_rgb[ref_y, ref_x]
        mask[current_y, current_x] = True
        confidence[current_y, current_x] = local_confidence

    reproj_cfg = dict(config.get("reference_reprojection", {}))
    erosion_ratio = float(reproj_cfg.get("boundary_erosion_ratio", 0.003))
    radius = max(int(round(min(current_width, current_height) * erosion_ratio)), 0)
    if radius > 0 and mask.any():
        kernel = 2 * radius + 1
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask = np.asarray(mask_image.filter(ImageFilter.MinFilter(kernel)), dtype=np.uint8) > 0
        warped[~mask] = 0
        confidence[~mask] = 0.0

    return {
        "warped_rgb": warped,
        "valid_mask": mask,
        "confidence_map": confidence,
        "valid_pixel_count": int(mask.sum()),
        "valid_ratio": float(mask.mean()),
        "width": current_width,
        "height": current_height,
        "reference_camera_id": str(reference_frame["camera_id"]),
        "current_camera_id": str(current_frame["camera_id"]),
        "current_depth_convention": depth_convention(current_frame["depth_encoding"]),
        "reference_depth_convention": reference_convention if inside.any() else depth_convention(reference_frame["depth_encoding"]),
        "depth_comparison_contract": "camera_z_ref_buffer_vs_projected_reference_camera_z",
    }

def compose_condition_image(
    source_rgb_path: str | Path,
    reprojection: Mapping[str, Any] | None,
) -> tuple[Image.Image, Image.Image, Dict[str, Any]]:
    source = Image.open(source_rgb_path).convert("RGB")
    width, height = source.size
    if reprojection is None:
        return source, Image.new("L", source.size, 255), {
            "reference_available": False,
            "locked_reference_ratio": 0.0,
            "generation_mask_white_ratio": 1.0,
        }
    warped = np.asarray(reprojection["warped_rgb"], dtype=np.uint8)
    mask = np.asarray(reprojection["valid_mask"], dtype=bool)
    if warped.shape[:2] != (height, width):
        warped = np.asarray(Image.fromarray(warped).resize((width, height), Image.Resampling.LANCZOS))
        mask = np.asarray(Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((width, height), Image.Resampling.NEAREST)) > 0
    source_array = np.asarray(source, dtype=np.uint8)
    condition = source_array.copy()
    condition[mask] = warped[mask]
    generation_mask = (~mask).astype(np.uint8) * 255
    return Image.fromarray(condition, mode="RGB"), Image.fromarray(generation_mask, mode="L"), {
        "reference_available": True,
        "locked_reference_ratio": float(mask.mean()),
        "generation_mask_white_ratio": float((generation_mask > 0).mean()),
    }


def fuse_reprojected_references(
    source_rgb_path: str | Path,
    reference_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Robustly fuse all locally valid accepted neighbours into one condition.

    Each reference contributes with a per-pixel weight equal to the Stage07
    edge correlation multiplied by its local geometric confidence and valid
    reprojection mask.  A one-step Huber reweighting suppresses RGB outliers
    before a continuous support map blends the fused reference with Stage07 RGB.
    """
    source_image = Image.open(source_rgb_path).convert("RGB")
    source = np.asarray(source_image, dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    fusion_cfg = dict(config.get("multi_reference_fusion", {}))
    robust_cfg = dict(fusion_cfg.get("robust_rgb_fusion", {}))
    delta = max(float(robust_cfg.get("rgb_residual_delta", 0.08)), _EPS)
    support_scale = max(float(fusion_cfg.get("reference_support_scale", 0.35)), _EPS)
    lock_threshold = float(fusion_cfg.get("reference_lock_threshold", 0.60))

    prepared: list[Dict[str, Any]] = []
    for item in reference_records:
        reprojection = dict(item["reprojection"])
        warped = np.asarray(reprojection["warped_rgb"], dtype=np.uint8)
        mask = np.asarray(reprojection["valid_mask"], dtype=bool)
        confidence = np.asarray(
            reprojection.get("confidence_map", mask.astype(np.float32)),
            dtype=np.float32,
        )
        if warped.shape[:2] != (height, width):
            warped = np.asarray(
                Image.fromarray(warped).resize((width, height), Image.Resampling.LANCZOS)
            )
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                    (width, height), Image.Resampling.NEAREST
                )
            ) > 0
            confidence = np.asarray(
                Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        edge_score = min(max(float(item["correlation_score"]), 0.0), 1.0)
        warped_float = warped.astype(np.float32) / 255.0
        matched_warped, luminance_metadata = _match_warp_luminance_to_source(
            warped_float,
            source,
            mask,
            fusion_cfg,
        )
        base_weight = edge_score * np.clip(confidence, 0.0, 1.0) * mask.astype(np.float32)
        prepared.append({
            "camera_id": str(item["camera_id"]),
            "correlation_score": edge_score,
            "warped_rgb_raw": warped_float,
            "warped_rgb": matched_warped,
            "luminance_matching": luminance_metadata,
            "valid_mask": mask,
            "base_weight": base_weight,
            "reprojection": reprojection,
        })

    if not prepared:
        return {
            "reference_available": False,
            "condition_image": source_image,
            "generation_mask": Image.new("L", source_image.size, 255),
            "fused_reference_rgb": np.zeros((height, width, 3), dtype=np.uint8),
            "reference_reliability": np.zeros((height, width), dtype=np.float32),
            "lock_mask": np.zeros((height, width), dtype=bool),
            "reference_records": [],
            "metadata": {
                "reference_count": 0,
                "locked_reference_ratio": 0.0,
                "generation_mask_white_ratio": 1.0,
                "mean_reference_reliability": 0.0,
            },
        }

    rgb_stack = np.stack([item["warped_rgb"] for item in prepared], axis=0)
    base_weights = np.stack([item["base_weight"] for item in prepared], axis=0)
    base_sum = base_weights.sum(axis=0)
    initial = (
        (rgb_stack * base_weights[..., None]).sum(axis=0)
        / np.maximum(base_sum[..., None], _EPS)
    )
    residual = np.mean(np.abs(rgb_stack - initial[None, ...]), axis=3)
    robust_multiplier = np.minimum(1.0, delta / np.maximum(residual, _EPS)).astype(np.float32)
    final_weights = base_weights * robust_multiplier
    final_sum = final_weights.sum(axis=0)
    fused = (
        (rgb_stack * final_weights[..., None]).sum(axis=0)
        / np.maximum(final_sum[..., None], _EPS)
    )
    reliability = (1.0 - np.exp(-final_sum / support_scale)).astype(np.float32)
    no_support = final_sum <= _EPS
    fused[no_support] = source[no_support]
    # Use the reliability produced by the geometry-aware multi-reference fusion
    # directly.  Do not apply an additional manual-warp cap: the computed weight
    # is the condition blend weight.
    condition_reliability = reliability
    condition = condition_reliability[..., None] * fused + (1.0 - condition_reliability[..., None]) * source
    lock_mask = reliability >= lock_threshold
    generation_mask = (~lock_mask).astype(np.uint8) * 255

    serializable_records = []
    for index, item in enumerate(prepared):
        item["final_weight"] = final_weights[index]
        serializable_records.append({
            "camera_id": item["camera_id"],
            "correlation_score": item["correlation_score"],
            "valid_pixel_count": int(item["valid_mask"].sum()),
            "valid_ratio": float(item["valid_mask"].mean()),
            "base_weight_sum": float(item["base_weight"].sum()),
            "final_weight_sum": float(item["final_weight"].sum()),
            "luminance_matching": dict(item.get("luminance_matching", {})),
        })

    return {
        "reference_available": True,
        "condition_image": Image.fromarray(
            np.clip(condition * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        ),
        "generation_mask": Image.fromarray(generation_mask, mode="L"),
        "fused_reference_rgb": np.clip(fused * 255.0, 0, 255).astype(np.uint8),
        "reference_reliability": reliability,
        "lock_mask": lock_mask,
        "reference_records": prepared,
        "serializable_reference_records": serializable_records,
        "metadata": {
            "reference_count": len(prepared),
            "robust_fusion_method": "one_step_huber",
            "rgb_residual_delta": delta,
            "reference_support_scale": support_scale,
            "reference_lock_threshold": lock_threshold,
            "locked_reference_ratio": float(lock_mask.mean()),
            "generation_mask_white_ratio": float((generation_mask > 0).mean()),
            "mean_reference_reliability": float(reliability.mean()),
            "maximum_reference_reliability": float(reliability.max()),
            "condition_reference_blend_policy": "computed_reference_reliability_without_additional_cap",
            "mean_condition_reference_blend": float(condition_reliability.mean()),
            "maximum_applied_condition_reference_blend": float(condition_reliability.max()),
            "luminance_matching_enabled": bool(
                dict(fusion_cfg.get("luminance_matching", {})).get("enabled", False)
            ),
            "luminance_matching_method": (
                "per_pixel_linear_rgb_luminance_gain"
                if bool(dict(fusion_cfg.get("luminance_matching", {})).get("enabled", False))
                else "disabled_preserve_reference_rgb_exactly"
            ),
        },
    }


def multi_reference_overlap_error(
    generated_rgb_path: str | Path,
    fusion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure multi-reference RGB disagreement for diagnostics only.

    Stage08 acceptance is intentionally geometry-led: this metric is recorded for
    inspection and ablation, but it never rejects an otherwise valid generated
    view. The ``mode`` field is retained in the report so downstream consumers do
    not accidentally reinterpret this diagnostic as an active gate.
    """
    records = list(fusion.get("reference_records", []))
    mode = str(config.get("mode", "diagnostic_only"))
    if mode != "diagnostic_only":
        raise ValueError(
            "Stage08 reference RGB consistency only supports mode='diagnostic_only'"
        )
    if not records:
        return {
            "accepted": True,
            "used_for_acceptance": False,
            "mode": mode,
            "multi_reference_rgb_l1": 0.0,
            "reference_pixel_weight_sum": 0.0,
            "reference_count": 0,
            "reference_overlap_available": False,
        }
    generated = np.asarray(Image.open(generated_rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    numerator = 0.0
    denominator = 0.0
    per_reference = []
    for item in records:
        overlap_cache_path = str(item.get("overlap_cache_path", ""))
        if overlap_cache_path:
            with np.load(overlap_cache_path) as cached:
                reference = np.asarray(cached["warped_rgb"], dtype=np.float32)
                weight = np.asarray(cached["final_weight"], dtype=np.float32)
        else:
            reference = np.asarray(item["warped_rgb"], dtype=np.float32)
            weight = np.asarray(item["final_weight"], dtype=np.float32)
        error = np.mean(np.abs(generated - reference), axis=2)
        weighted_sum = float((error * weight).sum())
        weight_sum = float(weight.sum())
        numerator += weighted_sum
        denominator += weight_sum
        per_reference.append({
            "camera_id": str(item["camera_id"]),
            "weighted_rgb_l1": float(weighted_sum / max(weight_sum, _EPS)),
            "weight_sum": weight_sum,
        })
    value = float(numerator / max(denominator, _EPS))
    return {
        "accepted": True,
        "used_for_acceptance": False,
        "mode": mode,
        "multi_reference_rgb_l1": value,
        "reference_pixel_weight_sum": denominator,
        "reference_count": len(records),
        "reference_overlap_available": True,
        "per_reference": per_reference,
    }
