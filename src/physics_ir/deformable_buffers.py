from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DeformableView:
    deformable_id: int
    object_id: int
    particle_offset: int
    particle_count: int
    spring_offset: int
    spring_count: int
    render_vertex_offset: int = 0
    render_vertex_count: int = 0

    def to_dict(self):
        return {
            "deformable_id": self.deformable_id,
            "object_id": self.object_id,
            "particle_offset": self.particle_offset,
            "particle_count": self.particle_count,
            "spring_offset": self.spring_offset,
            "spring_count": self.spring_count,
            "render_vertex_offset": self.render_vertex_offset,
            "render_vertex_count": self.render_vertex_count,
        }

@dataclass
class DeformableBuffers:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    masses: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    fixed: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    spring_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    rest_lengths: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    stiffness: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    damping: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    views: list[DeformableView] = field(default_factory=list)

    def add_grid_cloth(self, object_id: int, nx: int, ny: int, size=(2.0, 1.2), mass=0.03, k=0.7, d=0.15, render_vertex_offset=0, render_vertex_count=0):
        p_offset = int(self.positions.shape[0])
        s_offset = int(self.spring_pairs.shape[0])

        sx, sy = float(size[0]), float(size[1])
        pts = []
        for yy in range(ny):
            for xx in range(nx):
                x = (xx / max(nx - 1, 1) - 0.5) * sx
                y = (yy / max(ny - 1, 1) - 0.5) * sy
                pts.append([x, y, 0.0])
        pts = np.asarray(pts, dtype=np.float32)

        fixed = np.zeros((nx * ny,), dtype=bool)
        fixed[:nx] = True

        pairs = []
        rests = []
        def add_pair(a, b):
            pairs.append([p_offset + a, p_offset + b])
            rests.append(float(np.linalg.norm(pts[b] - pts[a])))

        for yy in range(ny):
            for xx in range(nx):
                i = yy * nx + xx
                if xx + 1 < nx:
                    add_pair(i, yy * nx + xx + 1)
                if yy + 1 < ny:
                    add_pair(i, (yy + 1) * nx + xx)
                if xx + 1 < nx and yy + 1 < ny:
                    add_pair(i, (yy + 1) * nx + xx + 1)
                if xx - 1 >= 0 and yy + 1 < ny:
                    add_pair(i, (yy + 1) * nx + xx - 1)

        self.positions = np.concatenate([self.positions, pts], axis=0)
        self.velocities = np.concatenate([self.velocities, np.zeros_like(pts)], axis=0)
        self.masses = np.concatenate([self.masses, np.full((nx * ny,), mass, dtype=np.float32)], axis=0)
        self.fixed = np.concatenate([self.fixed, fixed], axis=0)

        pair_arr = np.asarray(pairs, dtype=np.int32)
        self.spring_pairs = np.concatenate([self.spring_pairs, pair_arr], axis=0)
        self.rest_lengths = np.concatenate([self.rest_lengths, np.asarray(rests, dtype=np.float32)], axis=0)
        self.stiffness = np.concatenate([self.stiffness, np.full((len(pairs),), k, dtype=np.float32)], axis=0)
        self.damping = np.concatenate([self.damping, np.full((len(pairs),), d, dtype=np.float32)], axis=0)

        deformable_id = len(self.views)
        view = DeformableView(
            deformable_id=deformable_id,
            object_id=object_id,
            particle_offset=p_offset,
            particle_count=nx * ny,
            spring_offset=s_offset,
            spring_count=len(pairs),
            render_vertex_offset=render_vertex_offset,
            render_vertex_count=render_vertex_count,
        )
        self.views.append(view)
        return deformable_id

    def to_list(self):
        return [v.to_dict() for v in self.views]

    def save_npz(self, path):
        np.savez_compressed(
            path,
            positions=self.positions,
            velocities=self.velocities,
            masses=self.masses,
            fixed=self.fixed,
            spring_pairs=self.spring_pairs,
            rest_lengths=self.rest_lengths,
            stiffness=self.stiffness,
            damping=self.damping,
        )
