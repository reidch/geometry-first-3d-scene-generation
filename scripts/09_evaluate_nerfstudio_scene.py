#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().float().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3:
        arr = arr[..., 0]
    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="L").save(path)


def _depth_preview(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup

    output = Path(args.output)
    rgb_dir = output / "renders"
    depth_dir = output / "depth_renders"
    acc_dir = output / "accumulation"
    for d in (rgb_dir, depth_dir, acc_dir):
        d.mkdir(parents=True, exist_ok=True)

    _, pipeline, checkpoint_path, _ = eval_setup(Path(args.load_config))
    pipeline.eval()
    datamanager = pipeline.datamanager
    loader = datamanager.fixed_indices_eval_dataloader
    dataset = datamanager.eval_dataset
    rows = []

    print(f"[09][EVAL] checkpoint={checkpoint_path}", flush=True)
    total = len(loader)
    with torch.no_grad():
        for ordinal, (camera, batch) in enumerate(loader, start=1):
            outputs = pipeline.model.get_outputs_for_camera(camera)
            # Nerfstudio computes PSNR/SSIM/LPIPS here. This call occurs exactly once
            # per final view and never participates in training.
            metrics, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
            image_idx_value = batch.get("image_idx", ordinal - 1)
            if torch.is_tensor(image_idx_value):
                image_idx = int(image_idx_value.reshape(-1)[0].item())
            else:
                image_idx = int(image_idx_value)
            try:
                source_name = Path(dataset.image_filenames[image_idx]).stem
            except Exception:
                source_name = f"{image_idx + 1:05d}"
            name = source_name

            rgb = outputs["rgb"]
            _save_rgb(rgb_dir / f"{name}.png", rgb)
            depth = outputs.get("depth")
            accumulation = outputs.get("accumulation")
            depth_mae = float(metrics.get("depth_mae", float("nan")))
            depth_rmse = float("nan")
            if depth is not None:
                depth_np = depth.detach().float().cpu().numpy().squeeze(-1)
                np.save(depth_dir / f"{name}.npy", depth_np.astype(np.float32))
                valid = np.isfinite(depth_np) & (depth_np > 0)
                Image.fromarray(_depth_preview(depth_np, valid), mode="L").save(depth_dir / f"{name}.png")
                gt = batch.get("depth_image")
                if gt is not None:
                    gt_np = gt.detach().float().cpu().numpy().squeeze(-1)
                    if gt_np.shape == depth_np.shape:
                        m = np.isfinite(gt_np) & (gt_np > 0) & valid
                        if np.any(m):
                            diff = depth_np[m] - gt_np[m]
                            depth_mae = float(np.mean(np.abs(diff)))
                            depth_rmse = float(np.sqrt(np.mean(diff * diff)))
            if accumulation is not None:
                _save_gray(acc_dir / f"{name}.png", accumulation)

            row = {
                "camera_index": image_idx,
                "camera_name": name,
                "psnr": float(metrics["psnr"]),
                "ssim": float(metrics["ssim"]),
                "lpips": float(metrics["lpips"]),
                "depth_mae_m": depth_mae,
                "depth_rmse_m": depth_rmse,
            }
            rows.append(row)
            print(
                f"\r[09][EVAL] {ordinal}/{total} {ordinal/total*100:5.1f}% "
                f"PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}",
                end="", flush=True,
            )
    print(flush=True)

    fields = ["camera_index", "camera_name", "psnr", "ssim", "lpips", "depth_mae_m", "depth_rmse_m"]
    with (output / "metrics_per_view.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    gaussian_count = int(getattr(pipeline.model, "num_points", 0))
    summary = {
        "frame_count": len(rows),
        "lpips_evaluations": len(rows),
        "lpips_during_training": 0,
        "final_gaussian_count": gaussian_count,
    }
    for key in ("psnr", "ssim", "lpips", "depth_mae_m", "depth_rmse_m"):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
