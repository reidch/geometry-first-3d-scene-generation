from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


def stable_debug_color(identifier: str) -> tuple[int, int, int]:
    """Return a deterministic neutral debug colour without interpreting semantics."""
    digest = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    return tuple(72 + int(value) % 144 for value in digest[:3])


class ObjectAtlas:
    """Persistent object-owned texture state keyed only by the JSON object ID.

    The persistent appearance state is intentionally only the texture itself.
    Observation masks, visit counts, and coverage files are not pipeline state.
    """

    _LEGACY_TRANSIENT_FILES = (
        "coverage.png",
        "initial_coverage.png",
        "visit_count.npy",
    )

    def __init__(self, root, object_name, semantic_class="", resolution=1024, base_color=None):
        self.dir = Path(root) / object_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.object_name = str(object_name)
        self.semantic_class = str(semantic_class)
        self.resolution = int(resolution)
        self.base_color = tuple(base_color) if base_color is not None else stable_debug_color(self.object_name)
        self.color_path = self.dir / "base_color.png"
        self.reachable_path = self.dir / "reachable.png"
        self.island_path = self.dir / "uv_islands.npy"
        self.meta_path = self.dir / "metadata.json"

    def initialize(self):
        resolution = self.resolution
        if not self.color_path.exists():
            Image.new("RGB", (resolution, resolution), self.base_color).save(self.color_path)
        if not self.reachable_path.exists():
            Image.new("L", (resolution, resolution), 0).save(self.reachable_path)
        for name in self._LEGACY_TRANSIENT_FILES:
            legacy = self.dir / name
            if legacy.exists():
                legacy.unlink()
        self.meta_path.write_text(
            json.dumps(
                {
                    "object_name": self.object_name,
                    "semantic_class": self.semantic_class,
                    "resolution": resolution,
                    "base_color": list(self.base_color),
                    "fusion_state": "base_color_only_uv_triangle_baking",
                    "persistent_texture_files": ["base_color.png"],
                    "observation_state_persisted": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def metadata(self):
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def update_metadata(self, updates):
        metadata = self.metadata()
        metadata.update(dict(updates))
        self.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def load(self):
        return __import__("numpy").asarray(Image.open(self.color_path).convert("RGB")).copy()

    def save(self, color):
        import numpy as np
        Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB").save(self.color_path)



def remove_legacy_texture_observation_state(texture_root) -> list[str]:
    """Delete obsolete texel-observation files from an existing texture tree."""
    root = Path(texture_root)
    removed = []
    for name in ObjectAtlas._LEGACY_TRANSIENT_FILES:
        for path in root.glob(f"*/{name}"):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed

def load_registry(texture_root):
    root = Path(texture_root)
    result = {}
    for meta_path in root.glob("*/metadata.json"):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        result[metadata["object_name"]] = ObjectAtlas(
            root,
            metadata["object_name"],
            metadata.get("semantic_class", ""),
            metadata["resolution"],
            metadata.get("base_color"),
        )
    return result
