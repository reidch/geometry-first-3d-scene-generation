from __future__ import annotations

from collections import defaultdict

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import (
    get_runtime_object_id,
    get_semantic_owner_id,
    get_world_object_id,
)

# A 16 x 16 x 16 colour cube gives 4096 collision-free semantic owners.  Every
# channel stays away from black, which makes 8-bit PNG decoding robust across
# Blender versions and colour quantisation.
_PALETTE_LEVELS = tuple(32 + 14 * index for index in range(16))
_MAX_PALETTE_ENTRIES = len(_PALETTE_LEVELS) ** 3


def palette_index_to_color(palette_index: int):
    """Encode a positive palette index as a bright, collision-free RGBA colour."""

    if isinstance(palette_index, bool) or not isinstance(palette_index, int):
        raise TypeError("palette_index must be an integer allocated by the semantic renderer")
    if palette_index <= 0 or palette_index > _MAX_PALETTE_ENTRIES:
        raise ValueError(
            f"palette_index must be in [1, {_MAX_PALETTE_ENTRIES}], got {palette_index}"
        )
    # Odd multiplication is a permutation modulo 4096, spreading adjacent
    # palette indices across the cube instead of producing near-neighbour colours.
    code = ((palette_index - 1) * 0x9E5) & 0xFFF
    r = _PALETTE_LEVELS[code & 0xF]
    g = _PALETTE_LEVELS[(code >> 4) & 0xF]
    b = _PALETTE_LEVELS[(code >> 8) & 0xF]
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def object_id_to_color(_object_identifier, palette_index=None):
    """Compatibility wrapper with an intentionally strict integer second argument.

    Business IDs are not converted to integers.  Callers must first allocate a
    separate palette index from the complete set of semantic owners.
    """

    if palette_index is None:
        raise TypeError(
            "object_id_to_color no longer hashes or converts object IDs; pass an "
            "explicit integer palette_index"
        )
    return palette_index_to_color(palette_index)


def get_or_create_emission_material(name, color):
    bpy = require_bpy()
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    try:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in list(nodes):
            nodes.remove(node)
        out = nodes.new(type="ShaderNodeOutputMaterial")
        try:
            emission = nodes.new(type="ShaderNodeEmission")
            emission.inputs["Color"].default_value = color
            emission.inputs["Strength"].default_value = 1.0
            mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
        except Exception:
            shader = nodes.new(type="ShaderNodeBsdfPrincipled")
            shader.inputs["Emission Color"].default_value = color
            shader.inputs["Emission Strength"].default_value = 1.0
            mat.node_tree.links.new(shader.outputs["BSDF"], out.inputs["Surface"])
    except Exception:
        pass
    return mat


def collect_original_materials():
    bpy = require_bpy()
    originals = {}
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, "materials"):
            originals[obj.name] = {
                "materials": [material for material in obj.data.materials],
                "material_indices": [int(poly.material_index) for poly in getattr(obj.data, "polygons", [])],
            }
    return originals


def restore_materials(originals):
    bpy = require_bpy()
    for obj_name, record in originals.items():
        obj = bpy.data.objects.get(obj_name)
        if not obj or not hasattr(obj.data, "materials"):
            continue
        # Accept the list-only representation as a backward-compatible fallback.
        if isinstance(record, dict):
            materials = record.get("materials", [])
            material_indices = record.get("material_indices", [])
        else:
            materials = record
            material_indices = []
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
        for index, poly in enumerate(getattr(obj.data, "polygons", [])):
            if index < len(material_indices):
                poly.material_index = min(
                    int(material_indices[index]),
                    max(len(obj.data.materials) - 1, 0),
                )


def _semantic_meshes(bpy):
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not bool(obj.get("pgw_physics_proxy", False))
    ]


def build_semantic_palette(objects):
    """Build a stable palette keyed by object-owned semantic IDs.

    Multiple primitive parts or replacement meshes belonging to one JSON object
    intentionally receive one palette entry and one semantic colour.
    """

    grouped = defaultdict(list)
    for obj in objects:
        grouped[get_semantic_owner_id(obj)].append(obj)

    if len(grouped) > _MAX_PALETTE_ENTRIES:
        raise ValueError(
            f"Semantic render has {len(grouped)} owners; maximum supported is {_MAX_PALETTE_ENTRIES}."
        )

    palette = {}
    for palette_index, owner_id in enumerate(sorted(grouped), start=1):
        color = palette_index_to_color(palette_index)
        meshes = sorted(grouped[owner_id], key=lambda item: item.name)
        runtime_ids = sorted(
            {value for value in (get_runtime_object_id(obj) for obj in meshes) if value is not None}
        )
        world_ids = sorted({get_world_object_id(obj) for obj in meshes})
        palette[owner_id] = {
            "semantic_owner_id": owner_id,
            "palette_index": palette_index,
            "world_object_ids": world_ids,
            "runtime_object_ids": runtime_ids,
            "mesh_object_names": [obj.name for obj in meshes],
            "color_float_rgba": list(color),
            "color_uint8_rgb": [int(round(channel * 255.0)) for channel in color[:3]],
        }
    return palette


def apply_object_semantic_materials():
    bpy = require_bpy()
    objects = _semantic_meshes(bpy)
    palette = build_semantic_palette(objects)
    for obj in objects:
        owner_id = get_semantic_owner_id(obj)
        entry = palette[owner_id]
        color = tuple(entry["color_float_rgba"])
        mat = get_or_create_emission_material(
            f"__PGW_SEMANTIC_{entry['palette_index']:04d}",
            color,
        )
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
    return palette
