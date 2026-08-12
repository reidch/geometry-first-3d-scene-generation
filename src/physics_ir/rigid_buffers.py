from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RigidBodyBuffers:
    transforms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), dtype=np.float32))
    linear_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    angular_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    masses: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    frictions: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    restitutions: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    def add(self, transform, mass=0.0, friction=0.5, restitution=0.1):
        idx = int(self.transforms.shape[0])
        transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        self.transforms = np.concatenate([self.transforms, transform[None, :, :]], axis=0)
        self.linear_velocities = np.concatenate([self.linear_velocities, np.zeros((1, 3), dtype=np.float32)], axis=0)
        self.angular_velocities = np.concatenate([self.angular_velocities, np.zeros((1, 3), dtype=np.float32)], axis=0)
        self.masses = np.concatenate([self.masses, np.array([mass], dtype=np.float32)])
        self.frictions = np.concatenate([self.frictions, np.array([friction], dtype=np.float32)])
        self.restitutions = np.concatenate([self.restitutions, np.array([restitution], dtype=np.float32)])
        return idx

    def save_npz(self, path):
        np.savez_compressed(
            path,
            transforms=self.transforms,
            linear_velocities=self.linear_velocities,
            angular_velocities=self.angular_velocities,
            masses=self.masses,
            frictions=self.frictions,
            restitutions=self.restitutions,
        )
