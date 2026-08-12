#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print("Depth Anything V2 Transformers API available")
    print("checkpoint:", args.checkpoint)
    if args.download:
        AutoImageProcessor.from_pretrained(args.checkpoint)
        AutoModelForDepthEstimation.from_pretrained(args.checkpoint)
        print("checkpoint downloaded and loadable")
    else:
        print("model download skipped; pass --download to prefetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
