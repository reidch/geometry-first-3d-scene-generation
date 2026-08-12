from __future__ import annotations
from pathlib import Path
import json

from src.cameras.depth_geometry import depth_convention


class CalibrationError(RuntimeError):
    pass


def load_calibration(scene_out_dir, camera_id, strict=True):
    path = Path(scene_out_dir) / '04_sparse_conditions' / 'camera_calibration' / f'{camera_id}.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        validate_calibration(data, path)
        return data
    if strict:
        raise CalibrationError(
            f'Missing exact Blender camera calibration: {path}\n'
            'Rerun Stage 04 with the final project before running Stage 05. '
            'The strict multiview pipeline does not guess camera intrinsics or depth convention.'
        )
    raise CalibrationError(f'Calibration unavailable: {path}')


def validate_calibration(data, source='<memory>'):
    required = ['render_width','render_height','K','camera_to_world_blender','world_to_camera_blender','depth_convention']
    missing = [k for k in required if data.get(k) is None]
    if missing:
        raise CalibrationError(f'Incomplete calibration {source}; missing: {missing}')
    if depth_convention(data) != 'camera_z':
        raise CalibrationError(f'Unsupported depth convention in {source}: {data["depth_convention"]}')
    if data.get('camera_type') != 'PERSP':
        raise CalibrationError(f'Strict sparse reprojection currently requires perspective cameras: {source}')
