from __future__ import annotations
import math
from src.blender.blender_runtime import require_bpy

def _look_at(obj, target):
    from mathutils import Vector
    loc = Vector(obj.location)
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def create_or_update_camera(cam_data):
    bpy = require_bpy()
    camera_id = cam_data["camera_id"]
    obj = bpy.data.objects.get(camera_id)
    if obj is None:
        bpy.ops.object.camera_add()
        obj = bpy.context.object
        obj.name = camera_id

    obj.location = tuple(float(v) for v in cam_data["position"])
    _look_at(obj, cam_data["target"])

    camera_type = cam_data.get("camera_type", "perspective")
    if camera_type == "orthographic":
        obj.data.type = "ORTHO"
        obj.data.ortho_scale = float(cam_data.get("ortho_scale") or 5.0)
    else:
        obj.data.type = "PERSP"
        obj.data.lens = float(cam_data.get("focal_length", 28.0))
        obj.data.sensor_width = float(cam_data.get("sensor_width_mm", 36.0))
        obj.data.sensor_fit = str(cam_data.get("sensor_fit", "HORIZONTAL"))

    obj.data.clip_start = 0.03
    obj.data.clip_end = 50.0
    return obj

def set_active_camera(cam_data):
    bpy = require_bpy()
    cam = create_or_update_camera(cam_data)
    bpy.context.scene.camera = cam
    return cam
