from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from src.blender.blender_runtime import require_bpy


def stable_debug_color(key: str) -> tuple[float, float, float, float]:
    """Generate a deterministic mid-range debug colour from an opaque object id."""
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return tuple(0.20 + 0.60 * (digest[index] / 255.0) for index in range(3)) + (1.0,)


def normalize_color(value: Sequence[float] | None, fallback_key: str) -> tuple[float, float, float, float]:
    if value is None:
        return stable_debug_color(fallback_key)
    values = [float(v) for v in value]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError("appearance.base_color must contain 3 or 4 numbers")
    return tuple(max(0.0, min(1.0, value)) for value in values)


def get_or_create_material(name: str, color_rgba, *, roughness: float = 0.72, metallic: float = 0.0):
    bpy = require_bpy()
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color_rgba
    try:
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color_rgba
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = float(roughness)
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = float(metallic)
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.25
    except Exception:
        pass
    return mat


def material_for_object_part(object_record: Mapping[str, Any], part: Mapping[str, Any]):
    object_appearance = dict(object_record.get("appearance", {}))
    part_appearance = dict(part.get("appearance", {}))
    merged = dict(object_appearance)
    merged.update(part_appearance)
    key = f"{object_record.get('object_id')}::{part.get('id')}"
    color = normalize_color(merged.get("base_color"), key)
    return get_or_create_material(
        "scaffold_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
        color,
        roughness=float(merged.get("roughness", 0.72)),
        metallic=float(merged.get("metallic", 0.0)),
    )


def assign_material(obj, mat):
    if hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
