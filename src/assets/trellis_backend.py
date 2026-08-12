from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import shutil
import subprocess
import sys

from src.io.json_io import load_json


def _has_weights(path: Path) -> bool:
    return path.exists() and (
        (path / "pipeline.json").exists()
        or any(path.rglob("*.safetensors"))
        or any(path.rglob("*.bin"))
    )


def ensure_trellis_repo(config: Dict[str, Any]) -> Dict[str, Any]:
    configured = os.environ.get("TRELLIS_REPO") or config.get("trellis_repo", "external/TRELLIS")
    repo = Path(configured).expanduser().resolve()
    report = {"path": str(repo), "actions": []}
    if (repo / "trellis").exists() and (repo / "example.py").exists():
        report["status"] = "ready_already_local"
        return report
    if not bool(config.get("auto_clone_repo", True)):
        raise FileNotFoundError(
            f"TRELLIS repo missing: {repo}. Set TRELLIS_REPO or enable auto_clone_repo."
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone Microsoft TRELLIS automatically.")
    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists():
        shutil.rmtree(repo)
    cmd = [
        "git", "clone", "--recurse-submodules",
        config.get("repo_url", "https://github.com/microsoft/TRELLIS.git"),
        str(repo),
    ]
    subprocess.run(cmd, check=True)
    report["actions"].append({"command": cmd})
    if not (repo / "trellis").exists():
        raise RuntimeError(f"TRELLIS clone completed but package directory is absent: {repo}")
    report["status"] = "ready_after_clone"
    return report


def ensure_trellis_models(config: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = config.get("model_id", "microsoft/TRELLIS-image-large")
    local_dir = Path(config.get("local_path", "models/trellis-image-large"))
    auto_download = bool(config.get("auto_download", True))
    ready = _has_weights(local_dir)
    report: Dict[str, Any] = {
        "model_id": repo_id,
        "local_path": str(local_dir),
        "already_present": bool(ready),
        "auto_download": auto_download,
        "actions": [],
    }
    if ready:
        report["status"] = "ready_already_local"
        return report
    if not auto_download:
        raise RuntimeError(f"TRELLIS model missing and auto_download=false: {repo_id} -> {local_dir}")
    from huggingface_hub import snapshot_download
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(config.get("token_env_var", "HF_TOKEN"))
    resolved = snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=token)
    report["actions"].append({"downloaded": True, "resolved_path": resolved})
    if not _has_weights(local_dir):
        raise RuntimeError(f"TRELLIS model download appears incomplete: {local_dir}")
    report["status"] = "ready_after_download"
    return report



def _resolve_runtime_python(config: Dict[str, Any]) -> str:
    explicit = os.environ.get("TRELLIS_PYTHON")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TRELLIS_PYTHON does not exist: {path}")
        return str(path.resolve())
    configured = str(config.get("runtime_python", "")).strip()
    if configured and configured not in {"python", "python3"}:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path.resolve())
    candidates = [
        Path.home() / ".conda/envs/trellis/bin/python",
        Path.home() / "miniconda3/envs/trellis/bin/python",
        Path.home() / "anaconda3/envs/trellis/bin/python",
    ]
    conda = shutil.which("conda")
    if conda:
        proc = subprocess.run(
            [conda, "info", "--base"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            candidates.insert(0, Path(proc.stdout.strip()) / "envs/trellis/bin/python")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return sys.executable


class TrellisImageLargeBackend:
    """Subprocess-isolated, real TRELLIS image-to-GLB backend.

    It never emits placeholder geometry and never reports success unless a nonempty
    GLB exists.  The inference implementation follows Microsoft's official example.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_report = ensure_trellis_repo(config)
        self.model_report = ensure_trellis_models(config)
        self.trellis_repo = Path(self.repo_report["path"])
        self.model_source = str(Path(config.get("local_path", "models/trellis-image-large")).resolve())
        self.python = _resolve_runtime_python(config)

    def _check_runtime_import(self) -> None:
        code = (
            "import sys, torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable in TRELLIS runtime'; "
            "cap=torch.cuda.get_device_capability(0); "
            "cuda=tuple(int(x) for x in str(torch.version.cuda).split('.')[:2]); "
            "assert cap < (12,0) or cuda >= (12,8), "
            "f'Blackwell GPU {cap} requires a PyTorch build with CUDA 12.8+, found {torch.version.cuda}'; "
            "x=torch.ones(1, device='cuda'); torch.cuda.synchronize(); "
            "print('CUDA', torch.version.cuda, torch.cuda.get_device_name(0), cap, float(x.item())); "
            "sys.path.insert(0, r'%s'); "
            "from trellis.pipelines import TrellisImageTo3DPipeline; "
            "from trellis.utils import postprocessing_utils; print('TRELLIS_RUNTIME_OK')"
        ) % str(self.trellis_repo)
        proc = subprocess.run(
            [self.python, "-c", code], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            setup_hint = (
                f"TRELLIS Python runtime is not ready under {self.python}.\n"
                f"Repo: {self.trellis_repo}\n"
                "Install the official TRELLIS dependencies in that Python environment, or set TRELLIS_PYTHON.\n"
                "Official setup example:\n"
                f"  cd {self.trellis_repo}\n"
                "  . ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast\n"
            )
            raise RuntimeError(setup_hint + "\nImport stderr:\n" + proc.stderr[-6000:])

    def generate_mesh(self, image_path: str | Path, output_dir: str | Path, object_id: str, seed: int = 1) -> Dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not image_path.exists() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"TRELLIS input image missing: {image_path}")
        self._check_runtime_import()

        output_glb = output_dir / "asset.glb"
        report_path = output_dir / "trellis_runtime_report.json"
        runner = Path(__file__).with_name("trellis_runner.py").resolve()
        cmd = [
            self.python, str(runner),
            "--trellis_repo", str(self.trellis_repo),
            "--model", self.model_source,
            "--image", str(image_path.resolve()),
            "--output", str(output_glb.resolve()),
            "--report", str(report_path.resolve()),
            "--seed", str(seed),
            "--sparse_steps", str(int(self.config.get("sparse_steps", 12))),
            "--sparse_cfg", str(float(self.config.get("sparse_cfg", 7.5))),
            "--slat_steps", str(int(self.config.get("slat_steps", 12))),
            "--slat_cfg", str(float(self.config.get("slat_cfg", 3.0))),
            "--simplify", str(float(self.config.get("simplify", 0.95))),
            "--texture_size", str(int(self.config.get("texture_size", 1024))),
        ]
        proc = subprocess.run(
            cmd, cwd=str(self.trellis_repo), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        runtime_report = load_json(report_path) if report_path.exists() else {}
        if proc.returncode != 0 or not output_glb.exists() or output_glb.stat().st_size == 0:
            raise RuntimeError(
                "TRELLIS failed to produce a GLB.\n"
                f"object_id={object_id}\ncommand={cmd}\n"
                f"runtime_report={runtime_report}\n"
                f"stdout_tail={proc.stdout[-6000:]}\n"
                f"stderr_tail={proc.stderr[-6000:]}"
            )
        return {
            "status": "ok",
            "object_id": object_id,
            "asset_path": str(output_glb),
            "source_image": str(image_path),
            "runtime_report": runtime_report,
            "repo_preparation": self.repo_report,
            "model_preparation": self.model_report,
            "command": cmd,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
