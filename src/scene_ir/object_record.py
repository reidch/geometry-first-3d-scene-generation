from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectRecord:
    object_id: int
    name: str
    display_name: str
    semantic_class: str
    parent_id: Optional[int] = None
    child_ids: list[int] = field(default_factory=list)
    generation_mode: str = "scaffold_only"
    render_id: Optional[int] = None
    physics_id: Optional[int] = None
    binding_id: Optional[int] = None
    active: bool = True

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "name": self.name,
            "display_name": self.display_name,
            "semantic_class": self.semantic_class,
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
            "generation_mode": self.generation_mode,
            "render_id": self.render_id,
            "physics_id": self.physics_id,
            "binding_id": self.binding_id,
            "active": self.active,
        }
