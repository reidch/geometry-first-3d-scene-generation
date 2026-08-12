from __future__ import annotations

import numpy as np
from PIL import Image


def load_triangle_id_map(path, valid_triangle_count=None, return_diagnostics=False):
    """Decode 24-bit ``id+1`` RGB.

    Background and any decoded value outside the supplied manifest range become
    ``-1``.  The optional diagnostics make ID-pass corruption visible instead of
    silently mapping stray RGB values to unrelated semantic owners.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    code = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
    decoded = np.where(code > 0, code - 1, -1).astype(np.int32)
    foreground = decoded >= 0
    invalid = np.zeros(decoded.shape, dtype=bool)
    if valid_triangle_count is not None:
        invalid = foreground & (decoded >= int(valid_triangle_count))
        decoded[invalid] = -1
    diagnostics = {
        "pixel_count": int(decoded.size),
        "encoded_foreground_pixels": int(foreground.sum()),
        "valid_foreground_pixels": int((decoded >= 0).sum()),
        "invalid_id_pixels": int(invalid.sum()),
        "invalid_id_ratio": float(invalid.sum() / max(int(foreground.sum()), 1)),
        "decoded_max_before_validation": int(np.max(np.where(foreground, code - 1, -1))) if foreground.any() else -1,
        "valid_triangle_count": None if valid_triangle_count is None else int(valid_triangle_count),
    }
    if return_diagnostics:
        return decoded, diagnostics
    return decoded


def visible_records_from_id_map(id_map, object_mask, triangle_metadata=None):
    id_map = np.asarray(id_map, dtype=np.int32)
    object_mask = np.asarray(object_mask, dtype=bool)
    values, counts = np.unique(id_map[object_mask & (id_map >= 0)], return_counts=True)
    metadata = triangle_metadata or {}
    records = []
    for triangle_id, pixels in zip(values.tolist(), counts.tolist()):
        meta = metadata.get(int(triangle_id), {})
        records.append({
            "global_triangle_id": int(triangle_id),
            "mesh_object_name": meta.get("mesh_object_name"),
            "visible_pixels": int(pixels),
            "projected_pixels": int(pixels),
            "frontality": float(meta.get("frontality", 1.0)),
            "world_area": float(meta.get("world_area", 0.0)),
            "uv_area_normalized": float(meta.get("uv_area_normalized", 0.0)),
        })
    return records
