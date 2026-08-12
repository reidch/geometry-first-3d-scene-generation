#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IGNORED_NAMES = {".input_hash", ".output_manifest.json", ".done", ".failed"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(path: Path):
    if path.is_file():
        if path.name not in IGNORED_NAMES:
            yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.name not in IGNORED_NAMES:
                yield child


def _artifact_roots(stage_dir: Path) -> list[Path]:
    index = stage_dir / "artifact_index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    roots = [index]
    for values in dict(payload.get("artifacts", {})).values():
        roots.extend(Path(value) for value in values)
    return roots


def write_manifest(stage_dir: Path) -> Path:
    entries = {}
    for root in _artifact_roots(stage_dir):
        if not root.exists():
            raise FileNotFoundError(f"Cannot mark stage complete; artifact is missing: {root}")
        for path in _files(root):
            key = str(path)
            entries[key] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
    if not entries:
        raise RuntimeError(f"Cannot mark stage complete without materialized artifact files: {stage_dir}")
    output = stage_dir / ".output_manifest.json"
    output.write_text(
        json.dumps({"schema_version": 1, "files": entries}, indent=2),
        encoding="utf-8",
    )
    return output


def validate_manifest(stage_dir: Path) -> tuple[bool, list[str]]:
    path = stage_dir / ".output_manifest.json"
    if not path.is_file() or path.stat().st_size == 0:
        return False, ["missing .output_manifest.json"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid .output_manifest.json: {exc}"]
    problems = []
    for raw, expected in dict(payload.get("files", {})).items():
        file_path = Path(raw)
        if not file_path.is_file():
            problems.append(f"manifest file missing: {file_path}")
            continue
        size = int(file_path.stat().st_size)
        if size != int(expected.get("size_bytes", -1)):
            problems.append(f"manifest size changed: {file_path}")
            continue
        if _sha256(file_path) != str(expected.get("sha256", "")):
            problems.append(f"manifest hash changed: {file_path}")
    if not payload.get("files"):
        problems.append("output manifest has no files")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["write", "validate"])
    parser.add_argument("stage_dir")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    stage_dir = Path(args.stage_dir)
    if args.command == "write":
        write_manifest(stage_dir)
        return 0
    ok, problems = validate_manifest(stage_dir)
    if args.explain:
        for problem in problems:
            print(problem)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
