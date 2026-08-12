from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from src.physics_ir.enums import ColliderShape

@dataclass
class ColliderRecord:
    collider_id: int
    object_id: int
    part_id: str
    shape_type: ColliderShape
    param_offset: int
    param_count: int
    local_transform: list

    def to_dict(self):
        return {
            "collider_id": self.collider_id,
            "object_id": self.object_id,
            "part_id": self.part_id,
            "shape_type": self.shape_type.value,
            "param_offset": self.param_offset,
            "param_count": self.param_count,
            "local_transform": self.local_transform,
        }

@dataclass
class ColliderBuffers:
    shape_params: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    records: List[ColliderRecord] = field(default_factory=list)

    def add_box(self, object_id: int, part_id: str, half_extents, local_transform=None):
        return self._add_shape(object_id, part_id, ColliderShape.BOX, half_extents, local_transform)

    def _add_shape(self, object_id: int, part_id: str, shape_type: ColliderShape, parameters, local_transform=None):
        local_transform = local_transform or [0, 0, 0, 0, 0, 0, 1, 1, 1]
        collider_id = len(self.records)
        offset = int(self.shape_params.shape[0])
        params = np.asarray(parameters, dtype=np.float32).reshape(-1)
        self.shape_params = np.concatenate([self.shape_params, params], axis=0)
        self.records.append(
            ColliderRecord(collider_id, object_id, part_id, shape_type, offset, int(params.size), local_transform)
        )
        return collider_id

    def add_sphere(self, object_id: int, part_id: str, radius: float, local_transform=None):
        return self._add_shape(object_id, part_id, ColliderShape.SPHERE, [radius], local_transform)

    def add_cylinder(self, object_id: int, part_id: str, radius: float, half_height: float, local_transform=None):
        return self._add_shape(object_id, part_id, ColliderShape.CYLINDER, [radius, half_height], local_transform)

    def add_capsule(self, object_id: int, part_id: str, radius: float, half_segment: float, local_transform=None):
        return self._add_shape(object_id, part_id, ColliderShape.CAPSULE, [radius, half_segment], local_transform)

    def add_cone(self, object_id: int, part_id: str, radius: float, half_height: float, local_transform=None):
        return self._add_shape(object_id, part_id, ColliderShape.CONE, [radius, half_height], local_transform)

    def to_list(self):
        return [r.to_dict() for r in self.records]

    def save_npz(self, path):
        np.savez_compressed(path, shape_params=self.shape_params)
