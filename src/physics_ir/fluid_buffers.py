from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FluidView:
    fluid_id: int
    object_id: int
    particle_offset: int
    particle_count: int
    viscosity: float = 0.1
    rest_density: float = 1000.0
    smoothing_radius: float = 0.12

    def to_dict(self):
        return {
            "fluid_id": self.fluid_id,
            "object_id": self.object_id,
            "particle_offset": self.particle_offset,
            "particle_count": self.particle_count,
            "viscosity": self.viscosity,
            "rest_density": self.rest_density,
            "smoothing_radius": self.smoothing_radius,
        }

@dataclass
class FluidBuffers:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    densities: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    pressures: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    radii: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    views: list[FluidView] = field(default_factory=list)

    def to_list(self):
        return [v.to_dict() for v in self.views]

    def save_npz(self, path):
        np.savez_compressed(
            path,
            positions=self.positions,
            velocities=self.velocities,
            densities=self.densities,
            pressures=self.pressures,
            radii=self.radii,
        )
