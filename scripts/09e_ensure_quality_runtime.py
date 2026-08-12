#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def _venv_python(venv: Path) -> Path:
    linux = venv / "bin" / "python"
    if linux.exists():
        return linux
    windows = venv / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    return linux


def _requirements_digest(path: Path, base_env: str) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(base_env.encode("utf-8"))
    h.update(b"stage09e-quality-isolated-v1")
    return h.hexdigest()


def _smoke(python: Path) -> None:
    # pkg_resources is explicitly checked because openai-clip imports
    # `from pkg_resources import packaging`.
    code = (
        "import importlib.metadata as m; "
        "import pkg_resources, torch, pyiqa, open_clip; "
        "assert m.version('setuptools') == '80.9.0'; "
        "assert m.version('pyiqa') == '0.1.16'; "
        "assert m.version('open_clip_torch') == '3.3.0'; "
        "print('[09E-B][QUALITY] isolated runtime smoke test OK')"
    )
    subprocess.run([str(python), "-c", code], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-conda-env", required=True)
    ap.add_argument("--venv-dir", required=True)
    ap.add_argument("--requirements", required=True)
    ap.add_argument("--write-python-path", required=True)
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    venv = Path(args.venv_dir).resolve()
    req = Path(args.requirements).resolve()
    path_file = Path(args.write_python_path)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    if not req.exists():
        raise FileNotFoundError(req)

    python = _venv_python(venv)
    if not python.exists():
        if not args.auto_install:
            raise RuntimeError(f"Stage09E isolated quality runtime does not exist: {venv}")
        venv.parent.mkdir(parents=True, exist_ok=True)
        print(f"[09E-B][QUALITY] Creating isolated quality runtime at {venv}", flush=True)
        # Reuse the proven Stage09 torch/CUDA build without modifying its site-packages.
        subprocess.run([
            "conda", "run", "-n", args.base_conda_env, "--no-capture-output",
            "python", "-m", "venv", "--system-site-packages", str(venv),
        ], check=True)
        python = _venv_python(venv)

    digest = _requirements_digest(req, args.base_conda_env)
    stamp = venv / ".stage09e_quality_requirements.sha256"
    needs_install = not stamp.exists() or stamp.read_text(encoding="utf-8").strip() != digest

    if needs_install:
        if not args.auto_install:
            raise RuntimeError("Stage09E quality runtime requirements are not installed for the current lock file")
        print("[09E-B][QUALITY] Installing pinned image-quality packages into the isolated runtime only...", flush=True)
        subprocess.run([
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "-r", str(req),
        ], check=True)
        stamp.write_text(digest + "\n", encoding="utf-8")

    try:
        _smoke(python)
    except subprocess.CalledProcessError:
        if not args.auto_install:
            raise
        print("[09E-B][QUALITY] Runtime smoke test failed; reinstalling pinned isolated packages...", flush=True)
        subprocess.run([
            str(python), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "--force-reinstall",
            "-r", str(req),
        ], check=True)
        stamp.write_text(digest + "\n", encoding="utf-8")
        _smoke(python)

    path_file.write_text(str(python) + "\n", encoding="utf-8")
    metadata = {
        "base_conda_env": args.base_conda_env,
        "isolated_venv": str(venv),
        "python": str(python),
        "requirements": str(req),
        "requirements_sha256": digest,
        "system_site_packages": True,
    }
    (path_file.parent / "B_quality_runtime.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
