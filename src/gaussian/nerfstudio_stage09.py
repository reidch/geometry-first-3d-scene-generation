from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.cameras.depth_geometry import depth_convention
from src.cameras.reconstruction_view_metrics import decode_metric_depth
from src.io.json_io import load_json, save_json


def _rotation_matrix_to_colmap_qvec(R: np.ndarray) -> np.ndarray:
    """Return COLMAP scalar-first quaternion [qw,qx,qy,qz] from a 3x3 rotation."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0.0:
        q = -q
    return q


def _copy_rgb(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst)


def _encode_camera_z_millimeters(depth_m: np.ndarray, dst: Path) -> Dict[str, Any]:
    depth_m = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    mm = np.zeros(depth_m.shape, dtype=np.uint16)
    if np.any(valid):
        values = np.rint(depth_m[valid] * 1000.0)
        if float(values.max(initial=0.0)) > 65535.0:
            raise ValueError(
                "Stage09 Nerfstudio depth export uses uint16 millimetres and encountered depth > 65.535 m"
            )
        mm[valid] = np.clip(values, 1.0, 65535.0).astype(np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mm).save(dst)
    return {
        "valid_pixels": int(np.count_nonzero(valid)),
        "minimum_depth_m": float(depth_m[valid].min()) if np.any(valid) else None,
        "maximum_depth_m": float(depth_m[valid].max()) if np.any(valid) else None,
    }


def _frame_valid_count(frame: Mapping[str, Any]) -> int:
    depth = np.asarray(decode_metric_depth(frame["depth"], frame["depth_encoding"]), dtype=np.float64)
    if depth_convention(frame["depth_encoding"]) != "camera_z":
        raise ValueError("Stage09 WorldMesh/Nerfstudio exporter requires Stage07 camera-Z depth")
    return int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))


def _subsample_indices(total: int, maximum: int) -> np.ndarray:
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    # Deterministic evenly-spaced selection across the global concatenated valid-pixel stream.
    return np.linspace(0, total - 1, num=maximum, dtype=np.int64)


def export_nerfstudio_colmap_dataset(
    manifest_path: str | Path,
    dataset_dir: str | Path,
    maximum_initial_points: int = 5_000_000,
) -> Dict[str, Any]:
    """Export Stage08 RGB + Stage07 camera-Z depth/cameras to COLMAP text format.

    The project camera contract is OpenCV: +x right, +y down, +z forward. COLMAP's
    pinhole camera uses the same camera-coordinate handedness/convention, so the stored
    world_to_camera_opencv matrix can be written directly as COLMAP R,t.
    """
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    frames = list(manifest.get("frames", []))
    if not frames:
        raise RuntimeError("Stage09 requires at least one Stage08 frame")

    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    depths_dir = dataset_dir / "depths"
    sparse_dir = dataset_dir / "colmap" / "sparse" / "0"
    for path in (images_dir, depths_dir, sparse_dir):
        path.mkdir(parents=True, exist_ok=True)

    frame_meta: list[Dict[str, Any]] = []
    valid_counts: list[int] = []
    camera_lines: list[str] = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines: list[str] = ["# Image list with two lines of data per image:", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]

    for index, frame in enumerate(frames, start=1):
        camera = load_json(frame["camera"])
        K = np.asarray(camera["K"], dtype=np.float64)
        w2c = np.asarray(camera["world_to_camera_opencv"], dtype=np.float64)
        width = int(camera.get("width", camera.get("image_width", 0)))
        height = int(camera.get("height", camera.get("image_height", 0)))
        if width <= 0 or height <= 0:
            with Image.open(frame["target_rgb"]) as image:
                width, height = image.size
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        name = f"{index:04d}.png"
        _copy_rgb(frame["target_rgb"], images_dir / name)

        depth = np.asarray(decode_metric_depth(frame["depth"], frame["depth_encoding"]), dtype=np.float64)
        if depth_convention(frame["depth_encoding"]) != "camera_z":
            raise ValueError(f"Camera {frame['camera_id']} does not provide camera-Z depth")
        depth_stats = _encode_camera_z_millimeters(depth, depths_dir / name)
        valid_counts.append(int(depth_stats["valid_pixels"]))

        camera_lines.append(f"{index} PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}")
        q = _rotation_matrix_to_colmap_qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        image_lines.append(
            f"{index} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {index} {name}"
        )
        image_lines.append("")
        frame_meta.append({
            "index": index,
            "camera_id": str(frame["camera_id"]),
            "rgb": str(images_dir / name),
            "depth": str(depths_dir / name),
            "camera": str(frame["camera"]),
            "acceptance_mode": str(frame.get("acceptance_mode", "strict")),
            "successful": bool(frame.get("successful", True)),
            "depth_stats": depth_stats,
        })

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    total_valid = int(sum(valid_counts))
    maximum_initial_points = max(int(maximum_initial_points), 1)
    selected_global = _subsample_indices(total_valid, maximum_initial_points)
    selected_cursor = 0
    stream_start = 0
    point_id = 1
    points_path = sparse_dir / "points3D.txt"
    with points_path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for frame_index, frame in enumerate(frames, start=1):
            count = valid_counts[frame_index - 1]
            stream_end = stream_start + count
            left = int(np.searchsorted(selected_global, stream_start, side="left"))
            right = int(np.searchsorted(selected_global, stream_end, side="left"))
            selected_local_stream = selected_global[left:right] - stream_start
            if len(selected_local_stream):
                camera = load_json(frame["camera"])
                K = np.asarray(camera["K"], dtype=np.float64)
                c2w = np.asarray(camera["camera_to_world_opencv"], dtype=np.float64)
                depth = np.asarray(decode_metric_depth(frame["depth"], frame["depth_encoding"]), dtype=np.float64)
                valid = np.isfinite(depth) & (depth > 0.0)
                yy, xx = np.nonzero(valid)
                yy = yy[selected_local_stream]
                xx = xx[selected_local_stream]
                z = depth[yy, xx]
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])
                cam = np.stack([
                    (xx.astype(np.float64) + 0.5 - cx) * z / fx,
                    (yy.astype(np.float64) + 0.5 - cy) * z / fy,
                    z,
                ], axis=-1)
                world = cam @ c2w[:3, :3].T + c2w[:3, 3]
                rgb = np.asarray(Image.open(frame["target_rgb"]).convert("RGB"), dtype=np.uint8)[yy, xx]
                for xyz, colour in zip(world, rgb):
                    handle.write(
                        f"{point_id} {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                        f"{int(colour[0])} {int(colour[1])} {int(colour[2])} 0\n"
                    )
                    point_id += 1
            stream_start = stream_end

    report = {
        "schema_version": 1,
        "backend": "worldmesh_nerfstudio_depth_splatfacto",
        "source_manifest": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "frame_count": len(frames),
        "images_dir": str(images_dir),
        "depths_dir": str(depths_dir),
        "colmap_sparse_dir": str(sparse_dir),
        "depth_encoding": "uint16_millimeters_camera_z",
        "depth_unit_scale_factor": 1e-3,
        "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
        "pixel_center_offset": 0.5,
        "valid_mesh_depth_point_count": total_valid,
        "exported_initial_point_count": point_id - 1,
        "maximum_initial_points": maximum_initial_points,
        "initial_point_subsampling": "deterministic_global_even_spacing" if total_valid > maximum_initial_points else "none",
        "frames": frame_meta,
    }
    save_json(report, dataset_dir / "dataset_export_report.json")
    return report


def build_nerfstudio_train_command(config: Mapping[str, Any], dataset_dir: str | Path, output_dir: str | Path) -> list[str]:
    training = dict(config.get("training", {}))
    env_name = str(training.get("conda_env", "worldmesh-nerfstudio"))
    method = str(training.get("method", "depth-splatfacto"))
    depth_mult = float(training.get("depth_loss_mult", 0.7))
    ssim_lambda = float(training.get("ssim_lambda", 0.2))
    use_scale_regularization = bool(training.get("use_scale_regularization", False))
    max_gauss_ratio = float(training.get("max_gauss_ratio", 10.0))
    if max_gauss_ratio <= 1.0:
        raise ValueError("Stage09 training.max_gauss_ratio must be > 1.0")
    depth_scale = float(dict(config.get("dataset", {})).get("depth_unit_scale_factor", 1e-3))
    command = [
        "conda", "run", "-n", env_name, "--no-capture-output",
        "ns-train", method,
        "--output-dir", str(output_dir),
        "--pipeline.model.depth-loss-mult", str(depth_mult),
        "--pipeline.model.ssim-lambda", str(ssim_lambda),
        "--pipeline.model.use-scale-regularization", str(use_scale_regularization),
        "--pipeline.model.max-gauss-ratio", str(max_gauss_ratio),
        "--pipeline.model.camera-optimizer.mode", str(training.get("camera_optimizer_mode", "off")),
        "--viewer.quit-on-train-completion", "True",
    ]
    if bool(training.get("disable_periodic_full_image_eval", True)):
        command += [
            "--steps-per-eval-image", str(int(training.get("steps_per_eval_image", 1_000_000_000))),
            "--steps-per-eval-all-images", str(int(training.get("steps_per_eval_all_images", 1_000_000_000))),
        ]
    command += [
        "colmap",
        "--data", str(dataset_dir),
        "--depth-unit-scale-factor", str(depth_scale),
        "--eval-mode", str(training.get("eval_mode", "all")),
    ]
    return command


def build_nerfstudio_eval_command(config: Mapping[str, Any], config_yml: str | Path, stage_dir: str | Path) -> list[str]:
    training = dict(config.get("training", {}))
    env_name = str(training.get("conda_env", "worldmesh-nerfstudio"))
    final_dir = Path(stage_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    return [
        "conda", "run", "-n", env_name, "--no-capture-output",
        "ns-eval",
        "--load-config", str(config_yml),
        "--output-path", str(final_dir / "evaluation.json"),
        "--render-output-path", str(final_dir / "evaluation_renders"),
    ]


def _newest_config_yml(root: Path) -> Path:
    candidates = [*root.rglob("config.yml"), *root.rglob("config.yaml")]
    if not candidates:
        raise FileNotFoundError(f"Nerfstudio training completed without a config.yml under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def probe_nerfstudio_runtime(config: Mapping[str, Any]) -> None:
    training = dict(config.get("training", {}))
    env_name = str(training.get("conda_env", "worldmesh-nerfstudio"))
    method = str(training.get("method", "depth-splatfacto"))
    subprocess.run(
        ["conda", "run", "-n", env_name, "--no-capture-output", "ns-train", method, "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_worldmesh_nerfstudio_stage09(
    manifest_path: str | Path,
    config: Mapping[str, Any],
    stage_dir: str | Path,
) -> Dict[str, Any]:
    stage_dir = Path(stage_dir)
    dataset_cfg = dict(config.get("dataset", {}))
    dataset_dir = stage_dir / str(dataset_cfg.get("output_dir", "dataset"))
    dataset_report = export_nerfstudio_colmap_dataset(
        manifest_path,
        dataset_dir,
        maximum_initial_points=int(dataset_cfg.get("maximum_initial_points", 5_000_000)),
    )
    probe_nerfstudio_runtime(config)

    ns_output = stage_dir / "nerfstudio_output"
    train_command = build_nerfstudio_train_command(config, dataset_dir, ns_output)
    subprocess.run(train_command, check=True)
    config_yml = _newest_config_yml(ns_output)
    (stage_dir / "nerfstudio_config_path.txt").write_text(str(config_yml) + "\n", encoding="utf-8")

    training = dict(config.get("training", {}))
    env_name = str(training.get("conda_env", "worldmesh-nerfstudio"))
    viewer_command = f"conda run -n {env_name} --no-capture-output ns-viewer --load-config {config_yml}"
    (stage_dir / "view_scene.txt").write_text(viewer_command + "\n", encoding="utf-8")

    eval_command = build_nerfstudio_eval_command(config, config_yml, stage_dir)
    subprocess.run(eval_command, check=True)

    report = {
        "schema_version": 1,
        "backend": "worldmesh_nerfstudio_depth_splatfacto",
        "training_manifest": str(manifest_path),
        "dataset_export": dataset_report,
        "train_command": train_command,
        "nerfstudio_config": str(config_yml),
        "viewer_command": viewer_command,
        "evaluation_command": eval_command,
        "evaluation_json": str(stage_dir / "final" / "evaluation.json"),
        "evaluation_renders": str(stage_dir / "final" / "evaluation_renders"),
        "depth_loss_mult": float(training.get("depth_loss_mult", 0.7)),
        "ssim_lambda": float(training.get("ssim_lambda", 0.2)),
        "camera_optimizer_mode": str(training.get("camera_optimizer_mode", "off")),
        "eval_mode": str(training.get("eval_mode", "all")),
        "framework_default_optimization_schedule": bool(training.get("framework_default_optimization_schedule", True)),
        "expected_framework_default_iterations": int(training.get("expected_framework_default_iterations", 30000)),
        "periodic_full_image_eval_disabled": bool(training.get("disable_periodic_full_image_eval", True)),
        "final_lpips_policy": "ns-eval_once_after_training_over_eval_mode_all",
    }
    save_json(report, stage_dir / "stage_report.json")
    return report
