from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
from PIL import Image

from src.cameras.depth_geometry import depth_convention
from src.cameras.reconstruction_view_metrics import decode_metric_depth
from src.io.json_io import load_json, save_json


def _rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Numerically stable 3x3 rotation matrix -> COLMAP (qw,qx,qy,qz)."""
    R = np.asarray(R, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / max(s, 1e-15)
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / max(s, 1e-15)
            q[3] = (R[0, 2] + R[2, 0]) / max(s, 1e-15)
        elif i == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / max(s, 1e-15)
            q[1] = (R[0, 1] + R[1, 0]) / max(s, 1e-15)
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / max(s, 1e-15)
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / max(s, 1e-15)
            q[1] = (R[0, 2] + R[2, 0]) / max(s, 1e-15)
            q[2] = (R[1, 2] + R[2, 1]) / max(s, 1e-15)
            q[3] = 0.25 * s
    q /= max(float(np.linalg.norm(q)), 1e-15)
    if q[0] < 0.0:
        q *= -1.0
    return tuple(float(v) for v in q)


def _load_camera(camera_value: Any) -> Dict[str, Any]:
    if isinstance(camera_value, Mapping):
        return dict(camera_value)
    return dict(load_json(camera_value))


def _scaled_intrinsics(camera: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    K = np.asarray(camera["K"], dtype=np.float64).copy()
    src_w = float(camera.get("width", width))
    src_h = float(camera.get("height", height))
    sx = float(width) / max(src_w, 1.0)
    sy = float(height) / max(src_h, 1.0)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def _resize_depth_nearest(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape == (height, width):
        return np.asarray(depth, dtype=np.float32)
    # PIL mode F keeps floating-point depth and nearest-neighbour preserves discontinuities.
    pil = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.asarray(pil.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def _write_depth_mm(path: Path, depth_m: np.ndarray, scale: float) -> int:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    scaled = np.zeros(depth_m.shape, dtype=np.uint16)
    values = np.rint(np.clip(depth_m[valid] * float(scale), 1.0, 65535.0)).astype(np.uint16)
    scaled[valid] = values
    Image.fromarray(scaled, mode="I;16").save(path)
    return int(valid.sum())


def _prepare_image(src: Path, dst: Path, copy_images: bool) -> Tuple[int, int]:
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if copy_images:
            rgb.save(dst)
        else:
            # A symlink avoids duplicate large images, but only if no conversion was required.
            if image.mode == "RGB" and src.suffix.lower() == ".png":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                rgb.save(dst)
    return width, height


def _camera_self_check(K: np.ndarray, c2w: np.ndarray, width: int, height: int) -> Dict[str, float]:
    # Validate the exact project convention: OpenCV camera coordinates, +Z forward,
    # and array index (x,y) corresponding to pixel center (x+0.5,y+0.5).
    w2c = np.linalg.inv(c2w)
    probes = [
        (0, 0, 2.0),
        (max(width // 2, 0), max(height // 2, 0), 4.0),
        (max(width - 1, 0), max(height - 1, 0), 7.0),
    ]
    pixel_errors = []
    depth_errors = []
    for x, y, z in probes:
        pc = np.asarray([
            (float(x) + 0.5 - K[0, 2]) * z / K[0, 0],
            (float(y) + 0.5 - K[1, 2]) * z / K[1, 1],
            z,
            1.0,
        ])
        pw = c2w @ pc
        pr = w2c @ pw
        u = K[0, 0] * pr[0] / pr[2] + K[0, 2] - 0.5
        v = K[1, 1] * pr[1] / pr[2] + K[1, 2] - 0.5
        pixel_errors.append(max(abs(u - x), abs(v - y)))
        depth_errors.append(abs(pr[2] - z))
    return {
        "maximum_pixel_roundtrip_error": float(max(pixel_errors)),
        "maximum_camera_z_roundtrip_error_m": float(max(depth_errors)),
    }


def export_nerfstudio_colmap_dataset(
    training_manifest_path: str | Path,
    stage09_root: str | Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = Path(training_manifest_path)
    manifest = load_json(manifest_path)
    frames = list(manifest.get("frames", []))
    if not frames:
        raise RuntimeError("Stage09 requires at least one Stage08 training frame")

    dataset_cfg = dict(config.get("dataset", {}))
    init_cfg = dict(config.get("initialization", {}))
    export_cfg = dict(config.get("export", {}))
    include_fallback = bool(dataset_cfg.get("include_fallback_views", True))
    if not include_fallback:
        frames = [f for f in frames if bool(f.get("successful", f.get("acceptance_mode", "strict") == "strict"))]
    if not frames:
        raise RuntimeError("No Stage09 frames remain after dataset filtering")

    stage09_root = Path(stage09_root)
    dataset_root = stage09_root / str(export_cfg.get("dataset_root", "dataset"))
    images_dir = dataset_root / "images"
    depths_dir = dataset_root / "depths"
    sparse_dir = dataset_root / "colmap" / "sparse" / "0"
    for path in (images_dir, depths_dir, sparse_dir):
        path.mkdir(parents=True, exist_ok=True)

    depth_scale = float(dataset_cfg.get("depth_scale", 1000.0))
    copy_images = bool(dataset_cfg.get("copy_images", True))
    max_points = max(int(init_cfg.get("max_points", 5_000_000)), 1)
    pixel_center_offset = float(init_cfg.get("pixel_center_offset", 0.5))
    if abs(pixel_center_offset - 0.5) > 1e-12:
        raise ValueError("This project requires a 0.5 pixel-center offset")

    prepared: list[Dict[str, Any]] = []
    total_valid_depth = 0
    self_checks = []
    for image_id, frame in enumerate(frames, start=1):
        selected_marker = frame.get("selected_marker")
        if selected_marker:
            marker = Path(str(selected_marker))
            if not marker.is_file():
                raise RuntimeError(f"Stage09 refuses incomplete Stage08 view {frame.get('camera_id')}: missing selected_marker {marker}")
        src_rgb = Path(str(frame["target_rgb"]))
        if not src_rgb.is_file():
            raise FileNotFoundError(src_rgb)
        filename = f"{image_id:05d}.png"
        width, height = _prepare_image(src_rgb, images_dir / filename, copy_images)
        camera = _load_camera(frame["camera"])
        K = _scaled_intrinsics(camera, width, height)
        c2w = np.asarray(camera["camera_to_world_opencv"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Invalid camera_to_world_opencv for {frame['camera_id']}")

        encoding = frame["depth_encoding"]
        if depth_convention(encoding) != "camera_z":
            raise ValueError(f"Stage09 requires camera-Z depth, got {depth_convention(encoding)!r}")
        depth = decode_metric_depth(frame["depth"], encoding)
        depth = _resize_depth_nearest(np.asarray(depth, dtype=np.float32), width, height)
        valid_count = _write_depth_mm(depths_dir / filename, depth, depth_scale)
        total_valid_depth += valid_count
        check = _camera_self_check(K, c2w, width, height)
        if check["maximum_pixel_roundtrip_error"] > 1e-6 or check["maximum_camera_z_roundtrip_error_m"] > 1e-8:
            raise RuntimeError(f"Camera/depth projection contract failed for {frame['camera_id']}: {check}")
        self_checks.append({"camera_id": str(frame["camera_id"]), **check})
        prepared.append({
            "image_id": image_id,
            "camera_id": str(frame["camera_id"]),
            "filename": filename,
            "acceptance_mode": str(frame.get("acceptance_mode", "strict")),
            "successful": bool(frame.get("successful", frame.get("acceptance_mode", "strict") == "strict")),
            "source_rgb": str(src_rgb),
            "source_depth": str(frame["depth"]),
            "width": int(width),
            "height": int(height),
            "K": K,
            "c2w": c2w,
            "depth": depth,
        })

    subsample_mode = init_cfg.get("subsample", "auto")
    if str(subsample_mode).lower() == "auto":
        subsample = max(int(math.ceil(total_valid_depth / max_points)), 1)
    else:
        subsample = max(int(subsample_mode), 1)

    # One PINHOLE camera entry per frame intentionally supports differing image sizes/intrinsics.
    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(prepared)}\n")
        for item in prepared:
            K = item["K"]
            f.write(
                f"{item['image_id']} PINHOLE {item['width']} {item['height']} "
                f"{K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}\n"
            )

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(prepared)}\n")
        for item in prepared:
            w2c = np.linalg.inv(item["c2w"])
            qw, qx, qy, qz = _rotation_matrix_to_quaternion_wxyz(w2c[:3, :3])
            tx, ty, tz = (float(v) for v in w2c[:3, 3])
            f.write(
                f"{item['image_id']} {qw:.15g} {qx:.15g} {qy:.15g} {qz:.15g} "
                f"{tx:.15g} {ty:.15g} {tz:.15g} {item['image_id']} {item['filename']}\n\n"
            )

    point_count = 0
    points_path = sparse_dir / "points3D.txt"
    with points_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        # We deliberately write no feature tracks, matching WorldMesh's generated initialization cloud.
        for item in prepared:
            depth = item["depth"]
            valid = np.isfinite(depth) & (depth > 0.0)
            y, x = np.nonzero(valid)
            if len(x) == 0:
                continue
            selection = np.arange(0, len(x), subsample, dtype=np.int64)
            x = x[selection]
            y = y[selection]
            z = depth[y, x].astype(np.float64)
            K = item["K"]
            pc = np.stack([
                (x.astype(np.float64) + 0.5 - K[0, 2]) * z / K[0, 0],
                (y.astype(np.float64) + 0.5 - K[1, 2]) * z / K[1, 1],
                z,
            ], axis=-1)
            world = pc @ item["c2w"][:3, :3].T + item["c2w"][:3, 3]
            rgb = np.asarray(Image.open(images_dir / item["filename"]).convert("RGB"), dtype=np.uint8)[y, x]
            for p, c in zip(world, rgb):
                point_count += 1
                f.write(
                    f"{point_count} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0\n"
                )

    if point_count > max_points:
        # ceil(total/max) should keep us below max; catch any unexpected counting mismatch.
        raise RuntimeError(f"Initialization point budget exceeded: {point_count} > {max_points}")

    report = {
        "schema_version": 1,
        "backend_contract": "worldmesh_style_colmap_depth_splatfacto",
        "training_manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "frame_count": len(prepared),
        "strict_frame_count": sum(int(i["successful"]) for i in prepared),
        "fallback_frame_count": sum(int(not i["successful"]) for i in prepared),
        "camera_convention": "opencv_colmap_x_right_y_down_z_forward",
        "depth_convention": "camera_z",
        "pixel_center_offset": 0.5,
        "depth_storage": "uint16_png",
        "depth_scale_m_to_storage": depth_scale,
        "depth_unit_scale_factor_storage_to_m": 1.0 / depth_scale,
        "valid_depth_pixels_total": int(total_valid_depth),
        "initialization_subsample": int(subsample),
        "initialization_point_count": int(point_count),
        "initialization_max_points": int(max_points),
        "pointcloud_source": "all Stage07 camera-Z depth maps + Stage08 final RGB",
        "mesh_locking": False,
        "camera_self_checks": self_checks,
        "frames": [
            {
                "camera_id": i["camera_id"],
                "image_id": i["image_id"],
                "filename": i["filename"],
                "acceptance_mode": i["acceptance_mode"],
                "successful": i["successful"],
            }
            for i in prepared
        ],
    }
    init_report_path = stage09_root / str(export_cfg.get("initialization_report", "initialization/initialization_report.json"))
    init_report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, init_report_path)
    save_json(report, dataset_root / "dataset_manifest.json")
    return report
