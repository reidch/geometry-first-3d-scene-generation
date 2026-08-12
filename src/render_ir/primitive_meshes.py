from __future__ import annotations
import numpy as np

def box_mesh(scale=(1.0, 1.0, 1.0)):
    sx, sy, sz = [float(v) for v in scale]
    x, y, z = sx / 2.0, sy / 2.0, sz / 2.0

    positions = np.array([
        [-x, -y, -z], [ x, -y, -z], [ x,  y, -z], [-x,  y, -z],
        [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3,
        3, 7, 4, 3, 4, 0,
    ], dtype=np.uint32)

    normals = np.zeros_like(positions, dtype=np.float32)
    # Simple MVP normals: normalized position vectors for now.
    length = np.linalg.norm(positions, axis=1, keepdims=True)
    normals = np.divide(positions, np.maximum(length, 1e-8)).astype(np.float32)

    tangents = np.zeros((positions.shape[0], 4), dtype=np.float32)
    tangents[:, 0] = 1.0
    tangents[:, 3] = 1.0

    uvs = np.array([
        [0,0], [1,0], [1,1], [0,1],
        [0,0], [1,0], [1,1], [0,1],
    ], dtype=np.float32)

    return positions, normals, tangents, uvs, indices

def primitive_mesh(primitive: str, scale=(1.0, 1.0, 1.0)):
    # Stage 01 MVP: sphere/cylinder/capsule use box proxy visual mesh.
    # Later stages can replace this without changing World IR.
    return box_mesh(scale)
