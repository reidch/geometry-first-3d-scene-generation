from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Material:
    material_id: int
    name: str
    albedo_texture: Optional[str] = None
    base_color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    prompt: str = ""

    def to_dict(self):
        return {
            "material_id": self.material_id,
            "name": self.name,
            "albedo_texture": self.albedo_texture,
            "base_color": list(self.base_color),
            "prompt": self.prompt,
        }

@dataclass
class MaterialLibrary:
    materials: List[Material] = field(default_factory=list)

    def add(self, name: str, albedo_texture=None, base_color=(0.7, 0.7, 0.7, 1.0), prompt: str = "") -> int:
        material_id = len(self.materials)
        self.materials.append(Material(material_id, name, albedo_texture, tuple(base_color), prompt))
        return material_id

    def to_list(self):
        return [m.to_dict() for m in self.materials]
