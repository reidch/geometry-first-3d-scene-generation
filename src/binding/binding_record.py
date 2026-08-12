from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BindingRecord:
    binding_id: int
    object_id: int
    render_id: int | None
    physics_id: int | None
    binding_type: str
    rigid_body_index: int = -1
    render_vertex_offset: int = 0
    render_vertex_count: int = 0
    physics_particle_offset: int = 0
    physics_particle_count: int = 0

    def to_dict(self):
        return {
            "binding_id": self.binding_id,
            "object_id": self.object_id,
            "render_id": self.render_id,
            "physics_id": self.physics_id,
            "binding_type": self.binding_type,
            "rigid_body_index": self.rigid_body_index,
            "render_vertex_offset": self.render_vertex_offset,
            "render_vertex_count": self.render_vertex_count,
            "physics_particle_offset": self.physics_particle_offset,
            "physics_particle_count": self.physics_particle_count,
        }
