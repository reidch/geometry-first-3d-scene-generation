from __future__ import annotations

from src.blender.blender_runtime import require_bpy


def _matrix_to_lists(matrix):
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def _invert_4x4(matrix):
    return _matrix_to_lists(matrix.inverted())


def calculate_intrinsics(camera_obj, scene):
    """Return the exact pinhole K used by Blender for the current render."""
    cam = camera_obj.data
    scale = float(scene.render.resolution_percentage) / 100.0
    width = float(scene.render.resolution_x) * scale
    height = float(scene.render.resolution_y) * scale
    pixel_aspect = float(scene.render.pixel_aspect_x) / float(scene.render.pixel_aspect_y)

    if cam.type != 'PERSP':
        return None

    sensor_fit = cam.sensor_fit
    if sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if width * pixel_aspect >= height else 'VERTICAL'

    if sensor_fit == 'VERTICAL':
        sensor_size = float(cam.sensor_height)
        fy = float(cam.lens) / sensor_size * height
        fx = fy / pixel_aspect
    else:
        sensor_size = float(cam.sensor_width)
        fx = float(cam.lens) / sensor_size * width
        fy = fx * pixel_aspect

    cx = width * (0.5 - float(cam.shift_x))
    cy = height * (0.5 + float(cam.shift_y))

    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def export_camera_calibration(camera_obj, camera_data, scene):
    c2w = _matrix_to_lists(camera_obj.matrix_world)
    w2c = _invert_4x4(camera_obj.matrix_world)
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

    return {
        'camera_id': camera_data['camera_id'],
        'camera_type': camera_obj.data.type,
        'render_width': width,
        'render_height': height,
        'pixel_aspect_x': float(scene.render.pixel_aspect_x),
        'pixel_aspect_y': float(scene.render.pixel_aspect_y),
        'lens_mm': float(camera_obj.data.lens),
        'sensor_width_mm': float(camera_obj.data.sensor_width),
        'sensor_height_mm': float(camera_obj.data.sensor_height),
        'sensor_fit': str(camera_obj.data.sensor_fit),
        'shift_x': float(camera_obj.data.shift_x),
        'shift_y': float(camera_obj.data.shift_y),
        'K': calculate_intrinsics(camera_obj, scene),
        'camera_to_world_blender': c2w,
        'world_to_camera_blender': w2c,
        'camera_coordinates': {
            'right_axis': '+X',
            'up_axis': '+Y',
            'view_axis': '-Z',
            'pixel_origin': 'top_left',
        },
        'depth_convention': 'camera_z',
        'source_camera_data': camera_data,
    }
