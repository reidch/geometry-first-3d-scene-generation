from __future__ import annotations

import math
from pathlib import Path

from src.blender.blender_runtime import require_bpy
from src.blender.material_utils import assign_material, material_for_object_part
from src.blender.object_identity import set_object_identity
from src.blender.object_metadata import collect_metadata, set_metadata
from src.blender.primitive_builder import create_primitive_object
from src.blender.scene_setup import clear_scene, configure_preview_render, setup_preview_camera_and_lights, setup_units_and_world
from src.blender.uv_atlas import assign_stable_object_atlas_uvs
from src.io.json_io import load_json


def _deep_merge(base, override):
    result = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _matrix_from_transform(transform):
    from mathutils import Euler, Matrix, Vector

    transform = dict(transform or {})
    position = Vector([float(value) for value in transform.get("position", [0.0, 0.0, 0.0])])
    rotation_values = transform.get("rotation_deg", [0.0, 0.0, 0.0])
    rotation = Euler([math.radians(float(value)) for value in rotation_values], "XYZ")
    scale = [float(value) for value in transform.get("scale", [1.0, 1.0, 1.0])]
    return Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((*scale, 1.0))


def _scene_payload(document):
    scene = document.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("Blender scaffold builder requires canonical scene schema v2")
    return scene


def _object_id(record):
    return str(record.get("id", ""))


def _object_semantic(record):
    return str(record.get("semantic", ""))


def _generation(record):
    value = record.get("generation")
    if not isinstance(value, dict):
        raise ValueError(f"Object {_object_id(record)!r} has no explicit generation block")
    return dict(value)


def _physics(record):
    value = record.get("physics")
    if not isinstance(value, dict):
        raise ValueError(f"Object {_object_id(record)!r} has no explicit physics block")
    return dict(value)


def _parts(record):
    scaffold = record.get("scaffold", {})
    return list(scaffold.get("parts", [])) if isinstance(scaffold, dict) else []


def _registry_lookup(object_registry_json):
    return {str(item.get("name")): item for item in object_registry_json.get("objects", [])}


def _render_lookup(render_json):
    return {item.get("object_id"): item for item in render_json.get("render_objects", [])}


def build_scaffold_from_json(
    scene_path,
    object_registry_path,
    render_manifest_path,
    physics_manifest_path,
    binding_records_path,
    material_manifest_path,
    out_blend_path,
    preview_path=None,
):
    """Build the complete Blender scaffold from nested JSON objects and parts.

    Blender intentionally imports no NumPy or third-party package. Hierarchy and
    part transforms are composed with ``mathutils`` and all behaviour comes from
    explicit JSON fields.
    """
    bpy = require_bpy()
    from mathutils import Matrix

    document = load_json(scene_path)
    payload = _scene_payload(document)
    defaults = dict(payload.get("defaults", {}))
    object_lookup = _registry_lookup(load_json(object_registry_path))
    render_lookup = _render_lookup(load_json(render_manifest_path))

    clear_scene()
    setup_units_and_world()
    manifest = []

    def visit(record, parent_anchor, parent_id=None):
        oid = _object_id(record)
        local_matrix = _matrix_from_transform(record.get("transform", {}))
        generation = _deep_merge(defaults.get("generation", {}), _generation(record))
        physics = _deep_merge(defaults.get("physics", {}), _physics(record))
        appearance = _deep_merge(defaults.get("appearance", {}), record.get("appearance", {}))
        registry_record = object_lookup.get(oid, {})
        integer_id = registry_record.get("object_id")
        if integer_id is None:
            raise KeyError(f"World IR registry is missing runtime object ID for {oid!r}")
        render_id = registry_record.get("render_id")
        physics_id = registry_record.get("physics_id")
        binding_id = registry_record.get("binding_id")
        render_record = render_lookup.get(render_id, {}) if render_id is not None else {}
        material_id = render_record.get("material_id")

        # Publish one explicit object-level scaffold anchor. Root generated visuals
        # inherit this anchor's world matrix, while generated hierarchy children
        # retain the JSON local matrix relative to their generated parent.
        anchor = bpy.data.objects.new(f"PGW_SCAFFOLD_ROOT__{oid}", None)
        bpy.context.collection.objects.link(anchor)
        anchor.empty_display_type = "PLAIN_AXES"
        anchor.empty_display_size = 0.08
        anchor.hide_render = True
        anchor.parent = parent_anchor
        anchor.matrix_parent_inverse = Matrix.Identity(4)
        anchor.matrix_basis = local_matrix
        set_object_identity(
            anchor,
            world_object_id=oid,
            runtime_object_id=integer_id,
            semantic_owner_id=oid,
        )
        anchor["pgw_scaffold_owner_anchor"] = True
        anchor["pgw_visual_role"] = "scaffold_owner_anchor"
        anchor["parent_object_id"] = str(parent_id or "")
        anchor["generation_mode"] = str(generation.get("mode", "group"))

        manifest.append({
            "blender_object_name": anchor.name,
            "object_id": oid,
            "parent_object_id": str(parent_id or ""),
            "record_type": "scaffold_owner_anchor",
            "generation_mode": anchor["generation_mode"],
            "matrix_local": [[float(value) for value in row] for row in anchor.matrix_basis],
            "matrix_world": [[float(value) for value in row] for row in anchor.matrix_world],
        })

        for part in _parts(record):
            part_id = str(part.get("id", "part"))
            primitive = str(part.get("primitive"))
            part_local = _matrix_from_transform(part.get("transform", {}))
            blender_name = oid + "__" + part_id
            obj = create_primitive_object(
                blender_name,
                primitive,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            )
            obj.parent = anchor
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = part_local
            mat = material_for_object_part(
                {
                    "object_id": oid,
                    "appearance": appearance,
                },
                part,
            )
            assign_material(obj, mat)
            obj["uv_mode"] = str(dict(part.get("uv", {})).get("mode", "auto"))
            set_object_identity(
                obj,
                world_object_id=oid,
                runtime_object_id=integer_id,
                semantic_owner_id=oid,
            )
            set_metadata(
                obj,
                render_id=render_id,
                physics_id=physics_id,
                binding_id=binding_id,
                semantic_class=_object_semantic(record),
                physical_type=str(physics.get("body", "none")),
                part_id=part_id,
                primitive=primitive,
                material_id=material_id,
            )
            obj["generation_mode"] = str(generation.get("mode", "scaffold_only"))
            obj["display_name"] = str(record.get("name", oid))
            obj["parent_object_id"] = str(parent_id or "")
            manifest.append({
                "blender_object_name": obj.name,
                "object_id": oid,
                "parent_object_id": str(parent_id or ""),
                "record_type": "scaffold_part",
                "metadata": collect_metadata(obj),
                "generation_mode": obj["generation_mode"],
                "primitive": primitive,
                "matrix_local": [[float(value) for value in row] for row in obj.matrix_basis],
                "matrix_world": [[float(value) for value in row] for row in obj.matrix_world],
            })

        for child in record.get("children", []):
            if isinstance(child, dict):
                visit(child, anchor, oid)

    for root in payload.get("objects", []):
        if isinstance(root, dict):
            visit(root, None, None)

    uv_manifest = assign_stable_object_atlas_uvs()
    setup_preview_camera_and_lights()
    configure_preview_render()

    out_blend_path = Path(out_blend_path)
    out_blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend_path))
    if preview_path is not None:
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(preview_path)
        bpy.ops.render.render(write_still=True)
    return {"objects": manifest, "uv_atlas": uv_manifest}
