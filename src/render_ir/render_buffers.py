from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RenderBuffers:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    tangents: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint32))

    def append_mesh(self, positions, normals, tangents, uvs, indices):
        positions = np.asarray(positions, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        tangents = np.asarray(tangents, dtype=np.float32)
        uvs = np.asarray(uvs, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)

        vertex_offset = int(self.positions.shape[0])
        index_offset = int(self.indices.shape[0])

        self.positions = np.concatenate([self.positions, positions], axis=0)
        self.normals = np.concatenate([self.normals, normals], axis=0)
        self.tangents = np.concatenate([self.tangents, tangents], axis=0)
        self.uvs = np.concatenate([self.uvs, uvs], axis=0)
        self.indices = np.concatenate([self.indices, indices + vertex_offset], axis=0)

        return vertex_offset, int(positions.shape[0]), index_offset, int(indices.shape[0])

    def save_npz(self, path):
        np.savez_compressed(
            path,
            positions=self.positions,
            normals=self.normals,
            tangents=self.tangents,
            uvs=self.uvs,
            indices=self.indices,
        )
