from __future__ import annotations
from src.blender.blender_runtime import require_bpy

def clear_scene():
    bpy = require_bpy()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def setup_units_and_world():
    bpy = require_bpy()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    try:
        # Dim neutral background; does not create white overexposure.
        bpy.context.scene.world.color = (0.025, 0.025, 0.025)
    except Exception:
        pass

def _look_at(obj, target):
    from mathutils import Vector
    loc = Vector(obj.location)
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def setup_preview_camera_and_lights():
    bpy = require_bpy()

    # Preview camera is inside the sealed room, below the upper scene bound, looking down.
    # Use orthographic projection for readable layout inspection.
    bpy.ops.object.camera_add(location=(0.0, -0.05, 2.70))
    cam = bpy.context.object
    cam.name = "preview_camera_layout_under_ceiling"
    _look_at(cam, (0.0, -0.05, 0.35))
    bpy.context.scene.camera = cam
    try:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = 5.2
        cam.data.clip_start = 0.03
        cam.data.clip_end = 20.0
    except Exception:
        pass

    # Use soft, moderate indoor light. Avoid huge energy area light right in camera view.
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 2.58))
    light = bpy.context.object
    light.name = "preview_soft_ceiling_area"
    light.data.energy = 180
    light.data.size = 4.5

    bpy.ops.object.light_add(type="POINT", location=(1.6, -1.4, 1.5))
    fill = bpy.context.object
    fill.name = "preview_weak_fill_point"
    fill.data.energy = 25

def configure_preview_render(resolution=(1280, 720)):
    bpy = require_bpy()
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass

    try:
        # Natural non-overexposed scaffold preview.
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = -1.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    try:
        scene.eevee.taa_render_samples = 64
    except Exception:
        pass
