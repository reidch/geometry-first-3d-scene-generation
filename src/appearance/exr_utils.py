"""Deprecated compatibility boundary for the removed floating-image transport.

Active pipeline stages use only PNG images and JSON manifests. These functions
remain solely so stale external imports fail with a clear migration message.
"""


def _removed(*_args, **_kwargs):
    raise RuntimeError(
        "Legacy floating-image buffers are disabled. Re-run the producing stage "
        "with v44 to generate standard PNG images and JSON manifests."
    )


read_exr_first_channel = _removed
read_exr_rgb = _removed
depth_exr_to_pil = _removed
read_exr_rgb_values = _removed
read_position_exr_values = _removed
