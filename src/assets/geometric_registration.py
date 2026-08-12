from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def proper_axis_rotations() -> list[np.ndarray]:
    """Return the 24 right-handed signed axis-permutation rotations."""
    rotations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(permutation):
                matrix[output_axis, input_axis] = signs[output_axis]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    unique: list[np.ndarray] = []
    for matrix in rotations:
        if not any(np.allclose(matrix, existing) for existing in unique):
            unique.append(matrix)
    if len(unique) != 24:
        raise RuntimeError(f"Expected 24 proper axis rotations, got {len(unique)}")
    return unique

def signed_axis_transforms() -> list[np.ndarray]:
    """Return all 48 signed axis permutations, including reflections."""
    transforms: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(permutation):
                matrix[output_axis, input_axis] = signs[output_axis]
            transforms.append(matrix)
    unique: list[np.ndarray] = []
    for matrix in transforms:
        if not any(np.allclose(matrix, existing) for existing in unique):
            unique.append(matrix)
    if len(unique) != 48:
        raise RuntimeError(f"Expected 48 signed axis transforms, got {len(unique)}")
    return unique


def parse_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            tokens = line.split()[1:]
            indices: list[int] = []
            for token in tokens:
                value = int(token.split("/")[0])
                indices.append(value - 1 if value > 0 else len(vertices) + value)
            for index in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[index], indices[index + 1]])
    if not vertices or not triangles:
        raise ValueError(f"OBJ has no usable triangle mesh: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    valid_index = np.all((triangle_array >= 0) & (triangle_array < len(vertex_array)), axis=1)
    triangle_array = triangle_array[valid_index]
    if len(triangle_array) == 0:
        raise ValueError(f"OBJ has no in-range triangles: {path}")
    tri_vertices = vertex_array[triangle_array]
    finite = np.all(np.isfinite(tri_vertices), axis=(1, 2))
    areas = np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )
    triangle_array = triangle_array[finite & (areas > 1e-14)]
    if len(triangle_array) == 0:
        raise ValueError(f"OBJ has no finite positive-area triangles: {path}")
    return vertex_array, triangle_array.astype(np.int32, copy=False)


def sample_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-14
    if not np.any(valid):
        raise ValueError("Mesh contains no positive-area triangles")
    tri = tri[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(tri), size=max(16, int(count)), p=probabilities)
    selected = tri[chosen]
    r1 = np.sqrt(rng.random(len(selected)))
    r2 = rng.random(len(selected))
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    return (
        a[:, None] * selected[:, 0]
        + b[:, None] * selected[:, 1]
        + c[:, None] * selected[:, 2]
    )


def _transform_matrix(transform: Mapping[str, Any] | None) -> np.ndarray:
    transform = dict(transform or {})
    position = np.asarray(transform.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    rx, ry, rz = [math.radians(float(v)) for v in transform.get("rotation_deg", [0.0, 0.0, 0.0])]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = mz @ my @ mx @ np.diag(scale)
    matrix[:3, 3] = position
    return matrix


def _apply_matrix(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3][None, :]


def _box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        list(itertools.product((-0.5, 0.5), repeat=3)),
        dtype=np.float64,
    )
    # Vertex order from itertools: xyz binary. Faces are split into triangles.
    faces = [
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ]
    return vertices, np.asarray(faces, dtype=np.int32)


def _lathe_mesh(kind: str, segments: int = 32, rings: int = 16) -> tuple[np.ndarray, np.ndarray]:
    if kind == "sphere":
        vertices = []
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            z = 0.5 * math.cos(phi)
            radius = 0.5 * math.sin(phi)
            for segment in range(segments):
                theta = 2 * math.pi * segment / segments
                vertices.append([radius * math.cos(theta), radius * math.sin(theta), z])
        faces = []
        for ring in range(rings):
            for segment in range(segments):
                nxt = (segment + 1) % segments
                a = ring * segments + segment
                b = ring * segments + nxt
                c = (ring + 1) * segments + nxt
                d = (ring + 1) * segments + segment
                if ring > 0:
                    faces.append([a, b, d])
                if ring < rings - 1:
                    faces.append([b, c, d])
        return np.asarray(vertices), np.asarray(faces, dtype=np.int32)

    vertices = []
    for z, radius in [(-0.5, 0.5), (0.5, 0.5 if kind in {"cylinder", "capsule"} else 0.0)]:
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            vertices.append([radius * math.cos(theta), radius * math.sin(theta), z])
    vertices.extend([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    bottom_center, top_center = len(vertices) - 2, len(vertices) - 1
    faces = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        a, b = segment, nxt
        c, d = segments + nxt, segments + segment
        faces.extend([[a, b, d], [b, c, d], [bottom_center, b, a]])
        if kind in {"cylinder", "capsule"}:
            faces.append([top_center, d, c])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def primitive_mesh(primitive: str) -> tuple[np.ndarray, np.ndarray]:
    if primitive == "box":
        return _box_mesh()
    if primitive == "sphere":
        return _lathe_mesh("sphere")
    if primitive in {"cylinder", "cone", "capsule"}:
        return _lathe_mesh(primitive)
    raise ValueError(f"Unsupported primitive: {primitive}")


def sample_scaffold_surface(parts: Sequence[Mapping[str, Any]], count: int, *, seed: int = 0) -> np.ndarray:
    if not parts:
        raise ValueError("Scaffold contains no parts")
    meshes = []
    areas = []
    for part in parts:
        vertices, triangles = primitive_mesh(str(part["primitive"]))
        matrix = _transform_matrix(part.get("transform", {}))
        transformed = _apply_matrix(vertices, matrix)
        tri = transformed[triangles]
        area = float((np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5).sum())
        meshes.append((transformed, triangles))
        areas.append(max(area, 1e-12))
    rng = np.random.default_rng(int(seed))
    counts = np.maximum(16, np.round(np.asarray(areas) / sum(areas) * int(count)).astype(int))
    points = []
    for index, ((vertices, triangles), part_count) in enumerate(zip(meshes, counts)):
        points.append(sample_mesh_surface(vertices, triangles, int(part_count), seed=int(rng.integers(0, 2**31 - 1))))
    merged = np.concatenate(points, axis=0)
    if len(merged) > count:
        chosen = rng.choice(len(merged), size=int(count), replace=False)
        merged = merged[chosen]
    return merged


def robust_bounds(points: np.ndarray, trim: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    lower = np.quantile(points, float(trim), axis=0)
    upper = np.quantile(points, 1.0 - float(trim), axis=0)
    center = 0.5 * (lower + upper)
    size = np.maximum(upper - lower, 1e-8)
    return lower, upper, center, size


def robust_bidirectional_distance(
    transformed_asset: np.ndarray,
    scaffold_points: np.ndarray,
    scaffold_tree,
    truncation: float,
) -> tuple[float, float, float]:
    from scipy.spatial import cKDTree

    d_asset, _ = scaffold_tree.query(transformed_asset, workers=1)
    asset_tree = cKDTree(transformed_asset)
    d_scaffold, _ = asset_tree.query(scaffold_points, workers=1)
    tau = max(float(truncation), 1e-8)
    loss_a = float(np.mean(np.minimum(d_asset, tau) ** 2))
    loss_s = float(np.mean(np.minimum(d_scaffold, tau) ** 2))
    return loss_a + loss_s, loss_a, loss_s


def _initial_uniform_scale(rotated: np.ndarray, target: np.ndarray, trim: float) -> float:
    _, _, _, source_size = robust_bounds(rotated, trim)
    _, _, _, target_size = robust_bounds(target, trim)
    ratios = target_size / np.maximum(source_size, 1e-8)
    return float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-8)))))


def _translation_from_centers(rotated: np.ndarray, target: np.ndarray, scale: float, trim: float) -> np.ndarray:
    _, _, source_center, _ = robust_bounds(rotated, trim)
    _, _, target_center, _ = robust_bounds(target, trim)
    return target_center - float(scale) * source_center


def _optimize_candidate(
    asset_points: np.ndarray,
    scaffold_points: np.ndarray,
    rotation: np.ndarray,
    config: Mapping[str, Any],
    *,
    maxiter: int,
    maxfev: int,
):
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    trim = float(config.get("robust_trim_fraction", 0.01))
    rotated = asset_points @ rotation.T
    s0 = _initial_uniform_scale(rotated, scaffold_points, trim)
    t0 = _translation_from_centers(rotated, scaffold_points, s0, trim)
    _, _, _, target_size = robust_bounds(scaffold_points, trim)
    target_diag = max(float(np.linalg.norm(target_size)), 1e-8)
    tau = float(config.get("truncation_ratio", 0.08)) * target_diag
    lower_factor = float(config.get("scale_lower_factor", 0.70))
    upper_factor = float(config.get("scale_upper_factor", 1.35))
    translation_radius = np.minimum(
        np.full(3, float(config.get("translation_radius_max_m", 0.30))),
        np.maximum(target_size * float(config.get("translation_radius_ratio", 0.18)), 0.02),
    )
    scaffold_tree = cKDTree(scaffold_points)

    def objective(parameters):
        scale = math.exp(float(parameters[0]))
        translation = np.asarray(parameters[1:4], dtype=np.float64)
        transformed = scale * rotated + translation[None, :]
        loss, _, _ = robust_bidirectional_distance(transformed, scaffold_points, scaffold_tree, tau)
        return loss / (target_diag * target_diag)

    x0 = np.concatenate([[math.log(max(s0, 1e-8))], t0])
    bounds = [
        (math.log(max(lower_factor * s0, 1e-8)), math.log(max(upper_factor * s0, 1e-8))),
        (t0[0] - translation_radius[0], t0[0] + translation_radius[0]),
        (t0[1] - translation_radius[1], t0[1] + translation_radius[1]),
        (t0[2] - translation_radius[2], t0[2] + translation_radius[2]),
    ]
    result = minimize(
        objective,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter), "maxfev": int(maxfev), "xtol": 1e-4, "ftol": 1e-6},
    )
    scale = math.exp(float(result.x[0]))
    translation = np.asarray(result.x[1:4], dtype=np.float64)
    transformed = scale * rotated + translation[None, :]
    raw, a2s, s2a = robust_bidirectional_distance(transformed, scaffold_points, scaffold_tree, tau)
    return {
        "rotation": rotation,
        "uniform_scale": float(scale),
        "translation": translation,
        "normalized_loss": float(raw / (target_diag * target_diag)),
        "asset_to_scaffold_loss": float(a2s / (target_diag * target_diag)),
        "scaffold_to_asset_loss": float(s2a / (target_diag * target_diag)),
        "target_diagonal_m": target_diag,
        "truncation_m": tau,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": int(getattr(result, "nfev", 0)),
    }


def register_uniform_bidirectional(
    asset_points: np.ndarray,
    scaffold_points: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Find a proper axis rotation, uniform scale, and translation by robust symmetric distance."""
    config = dict(config or {})
    rotations = proper_axis_rotations()
    trim = float(config.get("robust_trim_fraction", 0.01))
    from scipy.spatial import cKDTree

    target_tree = cKDTree(scaffold_points)
    _, _, _, target_size = robust_bounds(scaffold_points, trim)
    target_diag = max(float(np.linalg.norm(target_size)), 1e-8)
    tau = float(config.get("truncation_ratio", 0.08)) * target_diag

    coarse = []
    for index, rotation in enumerate(rotations):
        rotated = asset_points @ rotation.T
        scale = _initial_uniform_scale(rotated, scaffold_points, trim)
        translation = _translation_from_centers(rotated, scaffold_points, scale, trim)
        transformed = scale * rotated + translation[None, :]
        raw, a2s, s2a = robust_bidirectional_distance(transformed, scaffold_points, target_tree, tau)
        coarse.append({
            "rotation_index": index,
            "rotation": rotation,
            "uniform_scale": scale,
            "translation": translation,
            "normalized_loss": raw / (target_diag * target_diag),
            "asset_to_scaffold_loss": a2s / (target_diag * target_diag),
            "scaffold_to_asset_loss": s2a / (target_diag * target_diag),
        })
    coarse.sort(key=lambda item: item["normalized_loss"])
    refine_count = max(1, min(int(config.get("rotation_refine_count", 8)), len(coarse)))
    refined = []
    for candidate in coarse[:refine_count]:
        result = _optimize_candidate(
            asset_points,
            scaffold_points,
            candidate["rotation"],
            config,
            maxiter=int(config.get("optimizer_maxiter", 28)),
            maxfev=int(config.get("optimizer_maxfev", 180)),
        )
        result["rotation_index"] = int(candidate["rotation_index"])
        result["coarse_loss"] = float(candidate["normalized_loss"])
        refined.append(result)
    refined.sort(key=lambda item: item["normalized_loss"])
    best = refined[0]
    second_loss = refined[1]["normalized_loss"] if len(refined) > 1 else None
    return {
        "method": "truncated_bidirectional_surface_distance",
        "rotation_candidates": 24,
        "scale_mode": "uniform",
        "rotation_index": int(best["rotation_index"]),
        "rotation_matrix": [[float(value) for value in row] for row in best["rotation"]],
        "uniform_scale": float(best["uniform_scale"]),
        "translation_local": [float(value) for value in best["translation"]],
        "normalized_loss": float(best["normalized_loss"]),
        "asset_to_scaffold_loss": float(best["asset_to_scaffold_loss"]),
        "scaffold_to_asset_loss": float(best["scaffold_to_asset_loss"]),
        "second_best_loss": float(second_loss) if second_loss is not None else None,
        "confidence_margin": float(second_loss - best["normalized_loss"]) if second_loss is not None else None,
        "target_diagonal_m": float(best["target_diagonal_m"]),
        "truncation_m": float(best["truncation_m"]),
        "optimizer_success": bool(best["optimizer_success"]),
        "optimizer_message": best["optimizer_message"],
        "evaluations": int(best["evaluations"]),
        "coarse_candidates": [
            {
                "rotation_index": int(item["rotation_index"]),
                "normalized_loss": float(item["normalized_loss"]),
            }
            for item in coarse
        ],
        "refined_candidates": [
            {
                "rotation_index": int(item["rotation_index"]),
                "normalized_loss": float(item["normalized_loss"]),
                "uniform_scale": float(item["uniform_scale"]),
                "translation_local": [float(value) for value in item["translation"]],
            }
            for item in refined
        ],
    }



def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return robust center and a right-handed PCA basis (columns are axes)."""
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center[None, :]
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return center, basis


def _solve_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares proper rotation, uniform scale, and translation (Umeyama)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity alignment requires matching Nx3 point arrays")
    if len(source) < 3:
        raise ValueError("Similarity alignment requires at least three correspondences")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (target_zero.T @ source_zero) / float(len(source))
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_zero * source_zero, axis=1)))
    if variance <= 1e-14:
        raise ValueError("Source point cloud has zero variance")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Similarity solver produced an invalid uniform scale")
    translation = target_center - scale * (rotation @ source_center)
    return rotation, scale, translation


def _surface_chamfer_loss(
    asset_points: np.ndarray,
    scaffold_points: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> tuple[float, float, float]:
    """Symmetric nearest-surface squared distance, normalized by scaffold size."""
    from scipy.spatial import cKDTree

    transformed = scale * (asset_points @ rotation.T) + translation[None, :]
    scaffold_tree = cKDTree(scaffold_points)
    asset_tree = cKDTree(transformed)
    asset_distance, _ = scaffold_tree.query(transformed, workers=1)
    scaffold_distance, _ = asset_tree.query(scaffold_points, workers=1)
    _, _, _, target_size = robust_bounds(scaffold_points, 0.0)
    denominator = max(float(np.dot(target_size, target_size)), 1e-12)
    asset_to_scaffold = float(np.mean(asset_distance ** 2) / denominator)
    scaffold_to_asset = float(np.mean(scaffold_distance ** 2) / denominator)
    return asset_to_scaffold + scaffold_to_asset, asset_to_scaffold, scaffold_to_asset


def register_surface_similarity_alignment(
    asset_points: np.ndarray,
    scaffold_points: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Align sampled mesh and scaffold surfaces with similarity ICP in model space.

    Both inputs are uniformly sampled surface point clouds.  The solver estimates only
    translation, a proper 3-D rotation, and one uniform scale.  It evaluates 24 PCA-axis
    initial poses, refines every pose with nearest-neighbour similarity ICP, and returns
    the transform with the smallest symmetric point-cloud distance.  World placement is intentionally left to Stage05, where generated roots inherit scaffold world matrices and generated children preserve hierarchy-local matrices.
    """
    config = dict(config or {})
    asset_points = np.asarray(asset_points, dtype=np.float64)
    scaffold_points = np.asarray(scaffold_points, dtype=np.float64)
    if asset_points.ndim != 2 or asset_points.shape[1] != 3 or len(asset_points) < 16:
        raise ValueError("Asset surface point cloud must contain at least 16 Nx3 points")
    if scaffold_points.ndim != 2 or scaffold_points.shape[1] != 3 or len(scaffold_points) < 16:
        raise ValueError("Scaffold surface point cloud must contain at least 16 Nx3 points")

    from scipy.spatial import cKDTree

    # Keep the solver bounded even when called directly with a very dense cloud.
    # Stage05 already supplies uniformly sampled surfaces, so deterministic reduction
    # preserves the intended distribution rather than changing the registration model.
    max_points = max(64, int(config.get("similarity_icp_max_points", 2000)))
    rng = np.random.default_rng(int(config.get("similarity_icp_seed", 7319)))
    if len(asset_points) > max_points:
        asset_points = asset_points[rng.choice(len(asset_points), size=max_points, replace=False)]
    if len(scaffold_points) > max_points:
        scaffold_points = scaffold_points[rng.choice(len(scaffold_points), size=max_points, replace=False)]

    iterations = max(1, int(config.get("similarity_icp_iterations", 18)))
    tolerance = max(0.0, float(config.get("similarity_icp_tolerance", 1e-7)))
    scaffold_tree = cKDTree(scaffold_points)
    asset_center, pca_basis = _pca_frame(asset_points)
    centered_asset = asset_points - asset_center[None, :]
    _, _, scaffold_center, scaffold_size = robust_bounds(scaffold_points, 0.01)

    candidates = []
    for index, axis_rotation in enumerate(proper_axis_rotations()):
        rotation = axis_rotation @ pca_basis.T
        rotated = centered_asset @ rotation.T
        _, _, _, source_size = robust_bounds(rotated, 0.01)
        ratios = scaffold_size / np.maximum(source_size, 1e-8)
        scale = float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-8)))))
        translation = scaffold_center - scale * ((asset_center @ rotation.T))
        previous_loss = float("inf")
        performed = 0

        for iteration in range(iterations):
            transformed = scale * (asset_points @ rotation.T) + translation[None, :]
            _, nearest_scaffold = scaffold_tree.query(transformed, workers=1)
            asset_tree = cKDTree(transformed)
            _, nearest_asset = asset_tree.query(scaffold_points, workers=1)

            # Symmetric correspondences: every asset sample finds scaffold surface and
            # every scaffold sample finds generated mesh surface.
            source_pairs = np.concatenate([asset_points, asset_points[nearest_asset]], axis=0)
            target_pairs = np.concatenate([scaffold_points[nearest_scaffold], scaffold_points], axis=0)
            rotation, scale, translation = _solve_similarity_transform(source_pairs, target_pairs)
            loss, a2s, s2a = _surface_chamfer_loss(
                asset_points, scaffold_points, rotation, scale, translation
            )
            performed = iteration + 1
            if abs(previous_loss - loss) <= tolerance:
                break
            previous_loss = loss

        loss, a2s, s2a = _surface_chamfer_loss(
            asset_points, scaffold_points, rotation, scale, translation
        )
        candidates.append({
            "candidate_index": int(index),
            "rotation": rotation,
            "uniform_scale": float(scale),
            "translation": translation,
            "normalized_loss": float(loss),
            "asset_to_scaffold_loss": float(a2s),
            "scaffold_to_asset_loss": float(s2a),
            "iterations": int(performed),
        })

    if not candidates:
        raise RuntimeError("No valid point-cloud similarity registration candidate")
    candidates.sort(key=lambda item: item["normalized_loss"])
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    records = [
        {
            "candidate_index": int(candidate["candidate_index"]),
            "rotation_matrix": [[float(value) for value in row] for row in candidate["rotation"]],
            "uniform_scale": float(candidate["uniform_scale"]),
            "translation_local": [float(value) for value in candidate["translation"]],
            "normalized_loss": float(candidate["normalized_loss"]),
            "asset_to_scaffold_loss": float(candidate["asset_to_scaffold_loss"]),
            "scaffold_to_asset_loss": float(candidate["scaffold_to_asset_loss"]),
            "iterations": int(candidate["iterations"]),
        }
        for candidate in candidates
    ]
    return {
        "method": "uniform_surface_pointcloud_similarity_icp",
        "selection_policy": "uniform_surface_samples_then_minimum_symmetric_nearest_point_distance",
        "rotation_candidates": 24,
        "scale_mode": "uniform",
        "rotation_index": int(best["candidate_index"]),
        "selected_candidate_index": int(best["candidate_index"]),
        "rotation_matrix": [[float(value) for value in row] for row in best["rotation"]],
        "determinant": float(np.linalg.det(best["rotation"])),
        "is_reflection": False,
        "uniform_scale": float(best["uniform_scale"]),
        "translation_local": [float(value) for value in best["translation"]],
        "normalized_loss": float(best["normalized_loss"]),
        "asset_to_scaffold_loss": float(best["asset_to_scaffold_loss"]),
        "scaffold_to_asset_loss": float(best["scaffold_to_asset_loss"]),
        "second_best_loss": float(second["normalized_loss"]) if second else None,
        "confidence_margin": (
            float(second["normalized_loss"] - best["normalized_loss"]) if second else None
        ),
        "optimizer_success": True,
        "optimizer_message": "24 PCA initial poses refined by proper-rotation uniform-scale symmetric surface ICP",
        "evaluations": len(candidates),
        "surface_alignment_space": "model_space",
        "world_placement_policy": "stage05_scaffold_root_world_and_preserved_child_local_matrix",
        "candidates": records,
        "coarse_candidates": [
            {
                "rotation_index": int(candidate["candidate_index"]),
                "normalized_loss": float(candidate["normalized_loss"]),
            }
            for candidate in candidates
        ],
        "refined_candidates": records,
    }


def register_pca_scaffold_alignment(
    asset_points: np.ndarray,
    scaffold_points: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Backward-compatible entry point for the surface point-cloud similarity solver."""
    return register_surface_similarity_alignment(asset_points, scaffold_points, config)

