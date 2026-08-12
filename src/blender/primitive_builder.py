from __future__ import annotations

import math

from src.blender.blender_runtime import require_bpy


def _deg_to_rad_xyz(rotation):
    return tuple(math.radians(float(value)) for value in rotation)


def create_primitive_object(name, primitive, position, rotation, scale):
    """Create one generic scaffold primitive from explicit JSON geometry."""
    bpy = require_bpy()
    rot = _deg_to_rad_xyz(rotation)
    pos = tuple(float(value) for value in position)
    scl = tuple(float(value) for value in scale)

    if primitive == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=pos, rotation=rot)
    elif primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.5, location=pos, rotation=rot)
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.5, depth=1.0, location=pos, rotation=rot)
    elif primitive == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.5, radius2=0.0, depth=1.0, location=pos, rotation=rot)
    elif primitive == "capsule":
        # Blender has no single capsule operator in the supported runtime. A
        # cylinder proxy preserves the declared extent for scaffold and physics.
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.5, depth=1.0, location=pos, rotation=rot)
    else:
        raise ValueError("Unsupported scaffold primitive: " + str(primitive))

    obj = bpy.context.object
    obj.scale = scl
    obj.name = name
    return obj
