from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from src.appearance.triangle_id_map import load_triangle_id_map
from src.io.json_io import load_json, save_json
from src.room_surfaces.surface_commit import sha256_file


def _triangle_records_by_id(payload: Dict) -> Dict[int, Dict]:
    return {
        int(record["global_triangle_id"]): dict(record)
        for record in payload.get("triangles", [])
        if "global_triangle_id" in record
    }


def _select_room_facing_triangles(
    triangle_payload: Dict,
    triangle_id_path: str | Path,
    exact_mask_path: str | Path,
) -> List[Dict]:
    """Return only the triangles belonging to the directly viewed interior face.

    The canonical renderer isolates one semantic surface object and points an
    orthographic camera directly at its room-facing side. Triangle-ID pixels under
    the exact alpha mask identify the visible face. A frontality filter removes
    accidental thickness/side triangles without introducing semantic category
    branches or a shape heuristic.
    """
    records_by_id = _triangle_records_by_id(triangle_payload)
    decoded = load_triangle_id_map(triangle_id_path)
    mask = np.asarray(Image.open(exact_mask_path).convert("L"), dtype=np.uint8) > 0
    if decoded.shape != mask.shape:
        raise RuntimeError(
            f"Triangle-ID/mask resolution mismatch: triangle_id={decoded.shape}, mask={mask.shape}"
        )
    visible_ids = {
        int(value)
        for value in np.unique(decoded[mask & (decoded >= 0)]).tolist()
        if int(value) in records_by_id
    }
    if not visible_ids:
        raise RuntimeError("Canonical surface capture contains no visible target triangle IDs")

    visible = [records_by_id[value] for value in sorted(visible_ids)]
    best_frontality = max(float(record.get("frontality", 0.0)) for record in visible)
    threshold = max(0.90, best_frontality - 0.03)
    selected = [
        record for record in visible
        if float(record.get("frontality", 0.0)) >= threshold
    ]
    if not selected:
        raise RuntimeError(
            f"No room-facing triangles survived frontality filtering: best={best_frontality:.6f}"
        )
    return selected


def _ndc_xy(clip: Iterable[float]) -> Tuple[float, float]:
    values = [float(value) for value in clip]
    if len(values) < 4 or abs(values[3]) <= 1e-12:
        raise RuntimeError(f"Invalid homogeneous clip coordinate: {values}")
    return values[0] / values[3], values[1] / values[3]


def _surface_frame(selected: List[Dict]) -> Dict[str, float]:
    points = [
        _ndc_xy(clip)
        for record in selected
        for clip in record.get("clip", [])
    ]
    if not points:
        raise RuntimeError("Selected room-facing triangles contain no clip-space vertices")
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if max_x - min_x <= 1e-9 or max_y - min_y <= 1e-9:
        raise RuntimeError(
            f"Degenerate room-facing surface frame: x=({min_x}, {max_x}), y=({min_y}, {max_y})"
        )
    return {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
    }


def _source_surface_coordinates(record: Dict, frame: Dict[str, float]) -> np.ndarray:
    result = []
    span_x = frame["max_x"] - frame["min_x"]
    span_y = frame["max_y"] - frame["min_y"]
    for clip in record.get("clip", []):
        x, y = _ndc_xy(clip)
        result.append(
            [
                (x - frame["min_x"]) / span_x,
                (y - frame["min_y"]) / span_y,
            ]
        )
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (3, 2):
        raise RuntimeError(
            f"A triangulated surface record must contain exactly three clip vertices: {array.shape}"
        )
    return array


def _destination_uv(record: Dict) -> np.ndarray:
    array = np.asarray(record.get("uv", []), dtype=np.float64)
    if array.shape != (3, 2):
        raise RuntimeError(
            f"A triangulated surface record must contain exactly three loop UVs: {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"Non-finite UV coordinates: {array.tolist()}")
    if np.any(array < -1e-6) or np.any(array > 1.0 + 1e-6):
        raise RuntimeError(
            "Exact room-surface commit requires the object-owned atlas UVs to lie in [0,1]; "
            f"received {array.tolist()}"
        )
    return np.clip(array, 0.0, 1.0)


def _triangle_twice_area(points: np.ndarray) -> float:
    a, b, c = points
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _barycentric(points: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = points
    denominator = _triangle_twice_area(points)
    if abs(denominator) <= 1e-14:
        raise RuntimeError(f"Degenerate UV triangle: {points.tolist()}")
    w0 = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / denominator
    w1 = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / denominator
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(x, 0.0, max(width - 1, 0))
    y = np.clip(y, 0.0, max(height - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (x - x0)[..., None]
    ty = (y - y0)[..., None]
    c00 = image[y0, x0].astype(np.float64)
    c10 = image[y0, x1].astype(np.float64)
    c01 = image[y1, x0].astype(np.float64)
    c11 = image[y1, x1].astype(np.float64)
    top = c00 * (1.0 - tx) + c10 * tx
    bottom = c01 * (1.0 - tx) + c11 * tx
    return top * (1.0 - ty) + bottom * ty


def _rasterize_triangle(
    atlas: np.ndarray,
    observed: np.ndarray,
    generated: np.ndarray,
    destination_uv: np.ndarray,
    source_surface: np.ndarray,
    epsilon: float = 1e-7,
) -> int:
    atlas_height, atlas_width = atlas.shape[:2]
    generated_height, generated_width = generated.shape[:2]

    min_u = float(np.min(destination_uv[:, 0]))
    max_u = float(np.max(destination_uv[:, 0]))
    min_v = float(np.min(destination_uv[:, 1]))
    max_v = float(np.max(destination_uv[:, 1]))
    x0 = max(0, int(np.floor(min_u * atlas_width - 0.5)) - 1)
    x1 = min(atlas_width - 1, int(np.ceil(max_u * atlas_width - 0.5)) + 1)
    y0 = max(0, int(np.floor((1.0 - max_v) * atlas_height - 0.5)) - 1)
    y1 = min(atlas_height - 1, int(np.ceil((1.0 - min_v) * atlas_height - 0.5)) + 1)
    if x1 < x0 or y1 < y0:
        return 0

    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sample_u = (grid_x + 0.5) / float(atlas_width)
    sample_v = 1.0 - (grid_y + 0.5) / float(atlas_height)
    w0, w1, w2 = _barycentric(destination_uv, sample_u, sample_v)
    inside = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
    if not np.any(inside):
        return 0

    source_u = w0 * source_surface[0, 0] + w1 * source_surface[1, 0] + w2 * source_surface[2, 0]
    source_v = w0 * source_surface[0, 1] + w1 * source_surface[1, 1] + w2 * source_surface[2, 1]
    source_x = source_u * max(generated_width - 1, 0)
    source_y = (1.0 - source_v) * max(generated_height - 1, 0)
    sampled = _bilinear_sample(generated, source_x[inside], source_y[inside])

    destination_y = grid_y[inside].astype(np.int64)
    destination_x = grid_x[inside].astype(np.int64)
    atlas[destination_y, destination_x] = np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
    observed[destination_y, destination_x] = True
    return int(destination_x.size)


def commit_rectified_surface_by_exact_triangles(
    atlas,
    generated_image_path: str | Path,
    object_id: str,
    directory: str | Path,
) -> Dict:
    """Write a complete generated surface into its exact UV-triangle footprint.

    Generated-image corners correspond directly to the room-facing surface frame.
    Every selected surface triangle is cut from that generated image according to
    its front-view geometry vertices and rasterized only into the same triangle's
    three recorded loop UVs. No atlas-wide replacement, UV bounding-box fill, UV
    rewrite, polygon approximation, or semantic category branch is used.
    """
    directory = Path(directory)
    generated_image_path = Path(generated_image_path)
    capture = directory / "capture"
    triangles_path = capture / "triangles.json"
    triangle_id_path = capture / "triangle_id.png"
    exact_mask_path = directory / "exact_mask.png"
    for path in (generated_image_path, triangles_path, triangle_id_path, exact_mask_path, atlas.color_path):
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            raise FileNotFoundError(f"Exact surface commit input is missing: {path}")

    triangle_payload = load_json(triangles_path)
    selected = _select_room_facing_triangles(triangle_payload, triangle_id_path, exact_mask_path)
    frame = _surface_frame(selected)
    capture_report_path = capture / "capture_report.json"
    capture_report = load_json(capture_report_path) if capture_report_path.exists() else {}
    exact_mask_array = np.asarray(Image.open(exact_mask_path).convert("L"), dtype=np.uint8) > 0
    if np.any(exact_mask_array):
        ys, xs = np.where(exact_mask_array)
        mask_bbox_width_fraction = float((int(xs.max()) - int(xs.min()) + 1) / max(exact_mask_array.shape[1], 1))
        mask_bbox_height_fraction = float((int(ys.max()) - int(ys.min()) + 1) / max(exact_mask_array.shape[0], 1))
    else:
        mask_bbox_width_fraction = 0.0
        mask_bbox_height_fraction = 0.0

    atlas_before_u8 = np.asarray(Image.open(atlas.color_path).convert("RGB"), dtype=np.uint8)
    atlas_after_u8 = atlas_before_u8.copy()
    generated = np.asarray(Image.open(generated_image_path).convert("RGB"), dtype=np.uint8)
    observed = np.zeros(atlas_before_u8.shape[:2], dtype=bool)
    mapping_records = []
    total_rasterized_samples = 0

    for record in selected:
        source_surface = _source_surface_coordinates(record, frame)
        destination_uv = _destination_uv(record)
        if abs(_triangle_twice_area(source_surface)) <= 1e-12:
            raise RuntimeError(
                f"Degenerate front-view source triangle for {object_id}: "
                f"triangle={record.get('global_triangle_id')}"
            )
        rasterized = _rasterize_triangle(
            atlas_after_u8,
            observed,
            generated,
            destination_uv,
            source_surface,
        )
        if rasterized <= 0:
            raise RuntimeError(
                f"UV triangle received no atlas texels for {object_id}: "
                f"triangle={record.get('global_triangle_id')}, uv={destination_uv.tolist()}"
            )
        total_rasterized_samples += rasterized
        mapping_records.append(
            {
                "global_triangle_id": int(record["global_triangle_id"]),
                "mesh_object_name": str(record.get("mesh_object_name", "")),
                "local_triangle_id": int(record.get("local_triangle_id", -1)),
                "frontality": float(record.get("frontality", 0.0)),
                "source_surface_coordinates": source_surface.tolist(),
                "destination_loop_uv": destination_uv.tolist(),
                "rasterized_texel_samples": int(rasterized),
            }
        )

    unique_observed_texels = int(observed.sum())
    if unique_observed_texels <= 0:
        raise RuntimeError(f"Exact triangle writeback produced no texels for {object_id}")

    before_sha256 = sha256_file(atlas.color_path)
    Image.fromarray(atlas_after_u8, "RGB").save(atlas.color_path)
    after_sha256 = sha256_file(atlas.color_path)
    changed = np.any(atlas_after_u8 != atlas_before_u8, axis=2)
    changed_texels = int(changed.sum())
    if changed_texels <= 0:
        raise RuntimeError(f"Exact triangle writeback did not change the atlas for {object_id}")

    observed_mask_path = directory / "writeback_observed_uv_mask.png"
    Image.fromarray((observed.astype(np.uint8) * 255), "L").save(observed_mask_path)
    committed_preview_path = directory / "committed_base_color.png"
    Image.fromarray(atlas_after_u8, "RGB").save(committed_preview_path)
    mapping_report_path = directory / "exact_surface_triangle_mapping.json"
    mapping_report = {
        "schema_version": 1,
        "status": "ok",
        "object_id": str(object_id),
        "policy": "front_view_surface_vertices_to_recorded_loop_uv_per_triangle",
        "generated_image": str(generated_image_path),
        "atlas": str(atlas.color_path),
        "surface_frame_ndc": frame,
        "selected_triangle_count": len(selected),
        "selected_triangle_ids": [int(record["global_triangle_id"]) for record in selected],
        "triangles": mapping_records,
        "unique_observed_texels": unique_observed_texels,
        "total_rasterized_samples": int(total_rasterized_samples),
        "writes_outside_selected_uv_triangles": False,
        "uses_uv_bbox_fill": False,
        "rewrites_mesh_uvs": False,
    }
    save_json(mapping_report, mapping_report_path)

    delta = np.abs(atlas_after_u8.astype(np.int16) - atlas_before_u8.astype(np.int16))
    atlas_commit = {
        "committed": True,
        "object_id": str(object_id),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "unique_observed_texels": unique_observed_texels,
        "changed_texels": changed_texels,
        "changed_texel_fraction": float(changed_texels / max(changed.size, 1)),
        "maximum_channel_delta_u8": int(delta.max()) if delta.size else 0,
        "writeback_core": "src.room_surfaces.exact_surface_uv_commit.commit_rectified_surface_by_exact_triangles",
        "writeback_policy": "exact_front_view_vertex_to_loop_uv_triangle_rasterization",
        "canonical_visible_triangle_count": len(selected),
        "canonical_observed_triangle_count": len(selected),
        "canonical_visible_world_area": float(sum(float(item.get("world_area", 0.0)) for item in selected)),
        "canonical_observed_world_area": float(sum(float(item.get("world_area", 0.0)) for item in selected)),
        "canonical_observed_world_area_ratio_diagnostic": 1.0,
        "target_bbox_width_fraction": mask_bbox_width_fraction,
        "target_bbox_height_fraction": mask_bbox_height_fraction,
        "target_mask_pixel_fraction": float(exact_mask_array.mean()),
        "non_target_meshes_hidden": bool(
            capture_report.get("visibility", {}).get("non_target_meshes_hidden", False)
        ),
        "observed_uv_mask": str(observed_mask_path),
        "coverage_is_not_an_acceptance_gate": True,
        "committed_base_color_preview": str(committed_preview_path),
        "exact_triangle_mapping_report": str(mapping_report_path),
        "surface_texture_fills_exact_uv_triangles": True,
        "surface_uv_policy": "preserve_original_surface_loop_uvs",
        "uses_uv_bbox_fill": False,
        "rewrites_mesh_uvs": False,
    }
    return {
        "fusion": {
            str(object_id): {
                "direct_commit": True,
                "writeback_core": atlas_commit["writeback_core"],
                "unique_observed_texels": unique_observed_texels,
                "changed_texels": changed_texels,
                "observed_triangle_ids": atlas_commit["canonical_observed_triangle_count"] and [
                    int(record["global_triangle_id"]) for record in selected
                ],
                "observed_uv_mask": str(observed_mask_path),
                "exact_triangle_mapping_report": str(mapping_report_path),
            }
        },
        "atlas_commit": atlas_commit,
        "triangle_update": {
            "updated": False,
            "reason": "exact_surface_triangle_commit",
        },
    }
