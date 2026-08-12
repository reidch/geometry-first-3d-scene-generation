#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def files_under(path: Path):
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            if any(part in IGNORED_PARTS for part in child.parts):
                continue
            if child.suffix in IGNORED_SUFFIXES:
                continue
            yield child


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic stage input fingerprint")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--value", action="append", default=[])
    args = parser.parse_args()
    digest = hashlib.sha256()
    for value in args.value:
        digest.update(b"VALUE\0")
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    for raw in args.path:
        path = Path(raw)
        digest.update(b"PATH\0")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        if not path.exists():
            digest.update(b"MISSING\0")
            continue
        root = path if path.is_dir() else path.parent
        for child in files_under(path):
            try:
                name = child.relative_to(root)
            except ValueError:
                name = child
            digest.update(str(name).encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
    print(digest.hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
