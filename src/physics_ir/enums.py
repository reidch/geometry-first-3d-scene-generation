from enum import Enum

class PhysicsKind(str, Enum):
    STATIC = "static"
    RIGID = "rigid"
    DEFORMABLE = "deformable"
    FLUID = "fluid"
    VISUAL_ONLY = "visual_only"

class DynamicMode(str, Enum):
    STATIC = "static"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"

class ColliderShape(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    CONE = "cone"
    CONVEX_MESH = "convex_mesh"
    TRIANGLE_MESH = "triangle_mesh"
