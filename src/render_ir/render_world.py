from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from src.render_ir.render_buffers import RenderBuffers
from src.render_ir.render_object import RenderObject, RenderPart, RenderMeshView

@dataclass
class RenderWorld:
    buffers: RenderBuffers = field(default_factory=RenderBuffers)
    objects: List[RenderObject] = field(default_factory=list)

    def add_object(self, object_id: int, material_id: int, local_to_world=None) -> int:
        render_id = len(self.objects)
        local_to_world = np.eye(4, dtype=np.float32) if local_to_world is None else np.asarray(local_to_world, dtype=np.float32)
        self.objects.append(RenderObject(render_id=render_id, object_id=object_id, material_id=material_id, local_to_world=local_to_world))
        return render_id

    def add_part(self, render_id: int, part_id: str, primitive: str, mesh_view: RenderMeshView):
        self.objects[render_id].parts.append(RenderPart(part_id=part_id, primitive=primitive, mesh_view=mesh_view))

    def to_list(self):
        return [o.to_dict() for o in self.objects]
