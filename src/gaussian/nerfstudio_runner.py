from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.io.json_io import save_json


def _conda_prefix(env_name: str) -> list[str]:
    if not env_name:
        return []
    if shutil.which("conda") is None:
        raise RuntimeError(
            f"Nerfstudio conda environment {env_name!r} was requested but 'conda' is not on PATH. "
            "Set NERFSTUDIO_ENV='' to use ns-train directly, or install the WorldMesh Nerfstudio environment."
        )
    return ["conda", "run", "-n", env_name, "--no-capture-output"]


def _run_visible(command: Sequence[str], *, cwd: Path) -> None:
    print("[09] $ " + shlex.join(str(x) for x in command), flush=True)
    result = subprocess.run(list(command), cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {shlex.join(command)}")


def verify_depth_splatfacto_available(config: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    ns_cfg = dict(config.get("nerfstudio", {}))
    env_name = os.environ.get("NERFSTUDIO_ENV", str(ns_cfg.get("conda_env", "worldmesh-nerfstudio")))
    prefix = _conda_prefix(env_name)
    probe = [*prefix, "python", "-c", (
        "from nerfstudio.models.depth_splatfacto import DepthSplatfactoModelConfig; "
        "from nerfstudio.configs.method_configs import all_methods; "
        "assert 'depth-splatfacto' in all_methods; "
        "from nerfstudio.models.splatfacto import SplatfactoModel; "
        "assert hasattr(SplatfactoModel, 'num_points'); print('depth-splatfacto OK')"
    )]
    proc = subprocess.run(probe, cwd=str(project_root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Stage09 requires the WorldMesh-compatible Nerfstudio fork with depth-splatfacto.\n"
            "Run: bash tools/install_worldmesh_nerfstudio.sh\n"
            f"Probe stderr:\n{proc.stderr.strip()}"
        )
    return {"conda_env": env_name, "probe_stdout": proc.stdout.strip()}


def train_depth_splatfacto(
    dataset_root: str | Path,
    stage09_root: str | Path,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> Dict[str, Any]:
    project_root = Path(project_root)
    stage09_root = Path(stage09_root)
    dataset_root = Path(dataset_root)
    ns_cfg = dict(config.get("nerfstudio", {}))
    export_cfg = dict(config.get("export", {}))
    availability = verify_depth_splatfacto_available(config, project_root)
    env_name = availability["conda_env"]
    prefix = _conda_prefix(env_name)
    output_dir = stage09_root / str(export_cfg.get("nerfstudio_output", "nerfstudio_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *prefix,
        "ns-train", str(ns_cfg.get("method", "depth-splatfacto")),
        "--output-dir", str(output_dir),
        "--pipeline.model.depth-loss-mult", str(float(ns_cfg.get("depth_loss_mult", 0.7))),
        "--pipeline.model.depth-loss-type", str(ns_cfg.get("depth_loss_type", "l1")),
        "--pipeline.model.camera-optimizer.mode", str(ns_cfg.get("camera_optimizer_mode", "off")),
        "--viewer.quit-on-train-completion", str(bool(ns_cfg.get("viewer_quit_on_train_completion", True))),
    ]
    if not bool(ns_cfg.get("periodic_image_evaluation_during_training", False)):
        disabled_eval_step = int(ns_cfg.get("periodic_eval_interval_disabled_value", 1_000_000_000))
        command.extend([
            "--steps-per-eval-image", str(disabled_eval_step),
            "--steps-per-eval-all-images", str(disabled_eval_step),
        ])
    command.extend([
        "colmap",
        "--data", str(dataset_root),
        "--depth-unit-scale-factor", str(float(ns_cfg.get("depth_unit_scale_factor", 0.001))),
        "--eval-mode", str(ns_cfg.get("eval_mode", "all")),
    ])
    (stage09_root / "training_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    print("[09] Nerfstudio owns the live training console below (progress, loss/PSNR and gaussian_count).", flush=True)
    _run_visible(command, cwd=project_root)

    configs = sorted(output_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    if not configs:
        raise RuntimeError(f"Nerfstudio training completed but no config.yml was found below {output_dir}")
    trained_config = configs[-1]
    checkpoints = sorted(trained_config.parent.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    result = {
        "nerfstudio_config": str(trained_config),
        "checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "output_dir": str(output_dir),
        "command": list(command),
        "framework_console": "nerfstudio_rich_live_writer",
        "gaussian_count_metric": "model.metrics.gaussian_count",
    }
    save_json(result, stage09_root / "nerfstudio_training_report.json")
    viewer_cmd = [*prefix, "ns-viewer", "--load-config", str(trained_config)]
    (stage09_root / str(export_cfg.get("viewer_command", "view_scene.txt"))).write_text(
        shlex.join(viewer_cmd) + "\n", encoding="utf-8"
    )
    return result


def evaluate_depth_splatfacto(
    trained_config: str | Path,
    stage09_root: str | Path,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> Dict[str, Any]:
    ns_cfg = dict(config.get("nerfstudio", {}))
    export_cfg = dict(config.get("export", {}))
    env_name = os.environ.get("NERFSTUDIO_ENV", str(ns_cfg.get("conda_env", "worldmesh-nerfstudio")))
    prefix = _conda_prefix(env_name)
    eval_root = Path(stage09_root) / str(export_cfg.get("evaluation_root", "evaluation"))
    eval_root.mkdir(parents=True, exist_ok=True)
    command = [
        *prefix,
        "python", "scripts/09_evaluate_nerfstudio_scene.py",
        "--load-config", str(trained_config),
        "--output", str(eval_root),
    ]
    _run_visible(command, cwd=Path(project_root))
    metrics = eval_root / "metrics_summary.json"
    if not metrics.is_file():
        raise RuntimeError(f"Final evaluation did not publish {metrics}")
    from src.io.json_io import load_json
    return dict(load_json(metrics))
