#!/usr/bin/env python
"""Compatibility entry point for the removed legacy depth inspector."""

raise SystemExit(
    "Floating-image depth files are no longer part of the pipeline. "
    "Inspect depth_control.png directly, or use tools/inspect_depth_image.py."
)
