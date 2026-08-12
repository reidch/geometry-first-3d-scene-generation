#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--conda-env", default="worldmesh-nerfstudio")
    args = parser.parse_args()
    config_path_file = Path(args.out) / "09_gaussian_splat" / "nerfstudio_config_path.txt"
    if not config_path_file.is_file():
        raise FileNotFoundError(f"Missing Stage09 Nerfstudio config pointer: {config_path_file}")
    config = config_path_file.read_text(encoding="utf-8").strip()
    if not config:
        raise RuntimeError(f"Empty Nerfstudio config pointer: {config_path_file}")
    subprocess.run([
        "conda", "run", "-n", args.conda_env, "--no-capture-output",
        "ns-viewer", "--load-config", config,
    ], check=True)


if __name__ == "__main__":
    main()
