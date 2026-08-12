#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys

# These are the exact pins shipped by the vendored WorldMesh/Nerfstudio 1.1.5
# runtime in this project. Stage09E quality packages must never replace them.
LOCKED = {
    "timm": "0.6.7",
    "transformers": "4.29.2",
}


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args()

    mismatched = {name: (_version(name), expected) for name, expected in LOCKED.items() if _version(name) != expected}
    if mismatched and not args.repair:
        details = ", ".join(f"{k}={got!r} (expected {want})" for k, (got, want) in mismatched.items())
        raise RuntimeError("Nerfstudio runtime dependency drift detected: " + details)

    if mismatched:
        details = ", ".join(f"{k}={got!r}->{want}" for k, (got, want) in mismatched.items())
        print("[09E-B][RUNTIME] Repairing Nerfstudio dependency drift caused by external evaluation packages: " + details, flush=True)
        specs = [f"{name}=={expected}" for name, expected in LOCKED.items()]
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *specs], check=True)

    remaining = {name: (_version(name), expected) for name, expected in LOCKED.items() if _version(name) != expected}
    if remaining:
        details = ", ".join(f"{k}={got!r} (expected {want})" for k, (got, want) in remaining.items())
        raise RuntimeError("Could not restore Nerfstudio locked dependencies: " + details)

    print("[09E-B][RUNTIME] Nerfstudio runtime guard OK: timm==0.6.7, transformers==4.29.2", flush=True)


if __name__ == "__main__":
    main()
