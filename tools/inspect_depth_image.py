#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=2)
    foreground = array[array > 0]
    report = {
        "path": str(path),
        "mode": image.mode,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
        "foreground_pixels": int(foreground.size),
        "foreground_unique_levels": int(np.unique(foreground).size) if foreground.size else 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
