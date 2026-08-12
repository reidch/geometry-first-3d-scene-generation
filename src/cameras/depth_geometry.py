from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Project-wide metric-depth contract for Blender-produced buffers.
# Blender Render Layers ``Depth`` buffers are interpreted as positive OpenCV
# camera-space forward Z after the Blender -> OpenCV axis flip.
CAMERA_Z_CONVENTION = "camera_z"
RAY_DISTANCE_CONVENTION = "euclidean_ray_distance_from_camera_origin"


def depth_convention(encoding: Mapping[str, Any]) -> str:
    """Return the project geometric convention of a decoded metric-depth image.

    V109 establishes a single contract: every Blender-produced metric depth map
    in this project is camera-space Z depth.  Historical project versions wrote
    ``euclidean_ray_distance_from_camera_origin`` into JSON even though their
    normalization bounds were computed from camera-space ``-Z``.  Those legacy
    labels therefore *must* be reinterpreted as camera-Z so existing Stage07
    caches can be reused without rerendering.

    This function intentionally does not expose an active ray-distance path for
    project buffers.  If a future external source truly stores ray distance it
    must be converted explicitly before entering the project depth contract.
    """

    explicit = str(encoding.get("depth_convention", "")).strip().lower()
    if explicit in {CAMERA_Z_CONVENTION, RAY_DISTANCE_CONVENTION}:
        return CAMERA_Z_CONVENTION

    kind = str(encoding.get("type", encoding.get("encoding", ""))).strip().lower()
    if not kind:
        return CAMERA_Z_CONVENTION
    if "camera_z" in kind:
        return CAMERA_Z_CONVENTION
    if "euclidean_ray_distance" in kind or "ray_distance" in kind:
        # Legacy project metadata alias: historical Blender Z-pass buffers were
        # mislabeled as ray distance.  Reinterpret, do not numerically convert.
        return CAMERA_Z_CONVENTION
    raise ValueError(f"Unknown metric-depth convention: {kind!r}")


def pixel_camera_rays_opencv(
    width: int,
    height: int,
    K: np.ndarray,
    *,
    pixel_center_offset: float = 0.5,
) -> np.ndarray:
    """Return HxWx3 OpenCV pinhole rays with z=1 (not unit-normalized).

    For camera-Z depth Z, the 3D camera point is exactly ``ray * Z`` because
    each returned ray has third component one.
    """

    intrinsic = np.asarray(K, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Camera intrinsic matrix K must be finite with shape (3, 3)")
    ys, xs = np.indices((int(height), int(width)), dtype=np.float64)
    pixels = np.stack(
        [
            xs.reshape(-1) + float(pixel_center_offset),
            ys.reshape(-1) + float(pixel_center_offset),
            np.ones(int(width) * int(height), dtype=np.float64),
        ],
        axis=0,
    )
    rays = np.linalg.inv(intrinsic) @ pixels
    if np.any(rays[2] <= 1e-12):
        raise ValueError("Camera intrinsics produced a non-forward pinhole ray")
    rays /= rays[2:3]
    return rays.T.reshape(int(height), int(width), 3)


def pixel_unit_rays_opencv(
    width: int,
    height: int,
    K: np.ndarray,
    *,
    pixel_center_offset: float = 0.5,
) -> np.ndarray:
    """Return HxWx3 unit rays for utilities that explicitly need directions."""

    rays = pixel_camera_rays_opencv(
        width, height, K, pixel_center_offset=pixel_center_offset
    )
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Camera intrinsics produced a degenerate pixel ray")
    return rays / norms


def backproject_camera_z(
    camera_z_depth: np.ndarray,
    K: np.ndarray,
    *,
    pixel_center_offset: float = 0.5,
) -> np.ndarray:
    """Backproject HxW positive camera-Z depth to HxWx3 OpenCV camera points."""

    depth = np.asarray(camera_z_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"camera-Z depth must be HxW, got {depth.shape}")
    rays = pixel_camera_rays_opencv(
        depth.shape[1], depth.shape[0], K, pixel_center_offset=pixel_center_offset
    )
    points = rays * depth[..., None]
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    points[invalid] = 0.0
    return points


def ray_distance_to_camera_z(
    ray_depth: np.ndarray,
    K: np.ndarray,
    *,
    pixel_center_offset: float = 0.5,
) -> np.ndarray:
    """Explicit utility for genuinely external ray-distance data only.

    Project Blender buffers never call this conversion after V109.
    """

    values = np.asarray(ray_depth, dtype=np.float32)
    rays = pixel_unit_rays_opencv(
        values.shape[1], values.shape[0], K, pixel_center_offset=pixel_center_offset
    )
    camera_z = values.astype(np.float64) * rays[..., 2]
    camera_z[~np.isfinite(values) | (values <= 0.0)] = 0.0
    return camera_z.astype(np.float32)


def metric_depth_to_camera_z(
    depth: np.ndarray,
    encoding: Mapping[str, Any],
    K: np.ndarray | None = None,
    *,
    pixel_center_offset: float = 0.5,
) -> np.ndarray:
    """Return decoded project metric depth as camera-Z without re-scaling.

    ``K`` and ``pixel_center_offset`` remain in the signature for backwards API
    compatibility.  Under the V109 project contract no Blender buffer requires
    ray-distance conversion.
    """

    del K, pixel_center_offset
    convention = depth_convention(encoding)
    if convention != CAMERA_Z_CONVENTION:  # pragma: no cover - contract guard
        raise ValueError(f"Unsupported project depth convention: {convention}")
    return np.asarray(depth, dtype=np.float32).copy()
