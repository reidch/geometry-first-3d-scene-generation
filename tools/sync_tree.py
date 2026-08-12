#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace a directory with a deterministic copy of another directory")
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    if not source.is_dir():
        raise FileNotFoundError(f"Snapshot source is missing: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
