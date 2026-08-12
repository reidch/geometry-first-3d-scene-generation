from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _vec3(values: Sequence[float] | None, default: Sequence[float]) -> list[float]:
    source = default if values is None else values
    return [float(source[0]), float(source[1]), float(source[2])]


def euler_xyz_matrix_deg(rotation_deg: Sequence[float] | None = None) -> np.ndarray:
    """Return a right-handed XYZ Euler rotation matrix from degrees."""
    rx, ry, rz = [math.radians(v) for v in _vec3(rotation_deg, [0.0, 0.0, 0.0])]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def transform_matrix(
    position: Sequence[float] | None = None,
    rotation_deg: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
) -> np.ndarray:
    position = _vec3(position, [0.0, 0.0, 0.0])
    scale = _vec3(scale, [1.0, 1.0, 1.0])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler_xyz_matrix_deg(rotation_deg) @ np.diag(scale)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix.astype(np.float32)


def matrix_from_transform(transform: Mapping | None) -> np.ndarray:
    transform = dict(transform or {})
    rotation = transform.get("rotation_deg", transform.get("rotation", [0.0, 0.0, 0.0]))
    return transform_matrix(
        transform.get("position", [0.0, 0.0, 0.0]),
        rotation,
        transform.get("scale", [1.0, 1.0, 1.0]),
    )


def compose_matrices(parent: np.ndarray, local: np.ndarray) -> np.ndarray:
    return np.asarray(parent, dtype=np.float64) @ np.asarray(local, dtype=np.float64)


def decompose_matrix(matrix: np.ndarray) -> dict:
    """Decompose an affine matrix into position, XYZ rotation degrees, and scale.

    This assumes no shear. Parent/child transforms in the JSON are TRS only, so this
    is sufficient and keeps the normalized scene human-readable.
    """
    m = np.asarray(matrix, dtype=np.float64)
    position = m[:3, 3].copy()
    linear = m[:3, :3].copy()
    scale = np.linalg.norm(linear, axis=0)
    safe = np.where(scale > 1e-12, scale, 1.0)
    r = linear / safe
    # XYZ extraction for R = Rz @ Ry @ Rx.
    sy = -float(r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rx = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        rz = 0.0
    return {
        "position": [float(v) for v in position],
        "rotation_deg": [math.degrees(rx), math.degrees(ry), math.degrees(rz)],
        "scale": [float(v) for v in scale],
        "matrix": [[float(v) for v in row] for row in m],
    }


def compose_part_transform(object_world_matrix: np.ndarray, part_transform: Mapping | None) -> dict:
    world = compose_matrices(object_world_matrix, matrix_from_transform(part_transform))
    return decompose_matrix(world)
