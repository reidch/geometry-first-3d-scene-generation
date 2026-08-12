from __future__ import annotations

from typing import Dict, Mapping


def summarize_atlas_visibility(semantic_pixel_fraction: Mapping[str, float], atlases: Mapping) -> Dict:
    values = {str(name): float(fraction) for name, fraction in semantic_pixel_fraction.items() if name in atlases}
    return {
        "visible_object_count": len(values),
        "total_semantic_fraction": float(sum(values.values())),
        "per_object_fraction": values,
        "largest_object_fraction": max(values.values(), default=0.0),
    }
