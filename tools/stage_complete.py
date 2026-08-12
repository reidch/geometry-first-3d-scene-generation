#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from stage_manifest import validate_manifest


def _nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(child.is_file() and child.stat().st_size > 0 for child in path.rglob("*"))
    return False


def _artifact_paths(index_path: Path) -> Iterable[Path]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for values in dict(payload.get("artifacts", {})).values():
        for value in values:
            yield Path(value)


def check(stage_dir: Path, required: list[Path]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not (stage_dir / ".done").is_file():
        problems.append("missing .done")
    if (stage_dir / ".failed").exists():
        problems.append(".failed exists")
    if (stage_dir / ".output_manifest.json").exists():
        manifest_ok, manifest_problems = validate_manifest(stage_dir)
        if not manifest_ok:
            problems.extend(manifest_problems)
    index_path = stage_dir / "artifact_index.json"
    if not _nonempty(index_path):
        problems.append("missing or empty artifact_index.json")
    else:
        try:
            for path in _artifact_paths(index_path):
                if not _nonempty(path):
                    problems.append(f"missing or empty artifact: {path}")
        except Exception as exc:  # malformed index must never count as complete
            problems.append(f"invalid artifact_index.json: {exc}")
    for path in required:
        if not _nonempty(path):
            problems.append(f"missing or empty required output: {path}")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that a pipeline stage is fully materialized")
    parser.add_argument("stage_dir")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    ok, problems = check(Path(args.stage_dir), [Path(value) for value in args.require])
    if args.explain and problems:
        for problem in problems:
            print(problem)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
