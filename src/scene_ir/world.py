from __future__ import annotations
from dataclasses import dataclass, field
from src.scene_ir.object_registry import ObjectRegistry
from src.render_ir.render_world import RenderWorld
from src.render_ir.materials import MaterialLibrary
from src.physics_ir.physics_world import PhysicsWorld
from src.binding.binding_world import BindingWorld

@dataclass
class World:
    scene_id: str
    prompt: str = ""
    objects: ObjectRegistry = field(default_factory=ObjectRegistry)
    render_world: RenderWorld = field(default_factory=RenderWorld)
    materials: MaterialLibrary = field(default_factory=MaterialLibrary)
    physics_world: PhysicsWorld = field(default_factory=PhysicsWorld)
    binding_world: BindingWorld = field(default_factory=BindingWorld)
