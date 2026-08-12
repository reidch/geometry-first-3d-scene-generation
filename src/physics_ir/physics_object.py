from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from src.physics_ir.enums import PhysicsKind, DynamicMode

@dataclass
class PhysicsObject:
    physics_id: int
    object_id: int
    kind: PhysicsKind
    active: bool = True
    dynamic_mode: DynamicMode = DynamicMode.STATIC
    backend_handle: Optional[int] = None
    collider_ids: list[int] = None
    rigid_body_index: Optional[int] = None
    deformable_id: Optional[int] = None
    fluid_id: Optional[int] = None

    def __post_init__(self):
        if self.collider_ids is None:
            self.collider_ids = []

    def to_dict(self):
        return {
            "physics_id": self.physics_id,
            "object_id": self.object_id,
            "kind": self.kind.value,
            "active": self.active,
            "dynamic_mode": self.dynamic_mode.value,
            "backend_handle": self.backend_handle,
            "collider_ids": self.collider_ids,
            "rigid_body_index": self.rigid_body_index,
            "deformable_id": self.deformable_id,
            "fluid_id": self.fluid_id,
        }
