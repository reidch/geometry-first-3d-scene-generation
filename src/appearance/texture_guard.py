from __future__ import annotations

from typing import Dict, Mapping


def texture_policy(record: Mapping) -> Dict:
    appearance = dict(record.get("appearance", {}))
    generation = dict(record.get("generation", {}))
    return {
        "preserve_geometry": bool(generation.get("preserve_geometry", True)),
        "preserve_boundary": bool(generation.get("preserve_boundary", True)),
        "base_color": appearance.get("base_color"),
        "uv_mode": dict(record.get("scaffold", {})).get("uv", {}).get("mode", "auto"),
    }
