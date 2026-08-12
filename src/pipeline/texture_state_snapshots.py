from __future__ import annotations

import shutil
from pathlib import Path


def replace_tree(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"Texture-state snapshot is missing: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def create_snapshot(source: str | Path, destination: str | Path) -> Path:
    return replace_tree(source, destination)
