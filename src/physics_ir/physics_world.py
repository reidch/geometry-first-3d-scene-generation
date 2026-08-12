from __future__ import annotations
from dataclasses import dataclass, field
from src.physics_ir.physics_object import PhysicsObject
from src.physics_ir.rigid_buffers import RigidBodyBuffers
from src.physics_ir.collider_buffers import ColliderBuffers
from src.physics_ir.deformable_buffers import DeformableBuffers
from src.physics_ir.fluid_buffers import FluidBuffers

@dataclass
class PhysicsWorld:
    objects: list[PhysicsObject] = field(default_factory=list)
    rigid: RigidBodyBuffers = field(default_factory=RigidBodyBuffers)
    colliders: ColliderBuffers = field(default_factory=ColliderBuffers)
    deformable: DeformableBuffers = field(default_factory=DeformableBuffers)
    fluid: FluidBuffers = field(default_factory=FluidBuffers)

    def add_object(self, obj: PhysicsObject):
        self.objects.append(obj)
        return obj.physics_id

    def to_list(self):
        return [o.to_dict() for o in self.objects]
