from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RenderMeshView:
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int

    def to_dict(self):
        return {
            "vertex_offset": self.vertex_offset,
            "vertex_count": self.vertex_count,
            "index_offset": self.index_offset,
            "index_count": self.index_count,
        }

@dataclass
class RenderPart:
    part_id: str
    primitive: str
    mesh_view: RenderMeshView

    def to_dict(self):
        return {"part_id": self.part_id, "primitive": self.primitive, "mesh_view": self.mesh_view.to_dict()}

@dataclass
class RenderObject:
    render_id: int
    object_id: int
    material_id: int
    local_to_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    visible: bool = True
    parts: list[RenderPart] = field(default_factory=list)

    def to_dict(self):
        return {
            "render_id": self.render_id,
            "object_id": self.object_id,
            "material_id": self.material_id,
            "visible": self.visible,
            "local_to_world": self.local_to_world.tolist(),
            "parts": [p.to_dict() for p in self.parts],
        }
