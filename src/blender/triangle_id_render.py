from __future__ import annotations

from pathlib import Path

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import get_semantic_owner_id
from src.blender.semantic_render import collect_original_materials, restore_materials

ATTRIBUTE_NAME = "PGW_TRIANGLE_ID_COLOR"


def encode_triangle_id(global_triangle_id: int):
    """Encode id+1 into linear RGB; black remains reserved for background."""
    code = int(global_triangle_id) + 1
    if code <= 0 or code >= (1 << 24):
        raise ValueError(f"Triangle id outside 24-bit render range: {global_triangle_id}")
    r = code & 255
    g = (code >> 8) & 255
    b = (code >> 16) & 255
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def assign_triangle_id_attribute(obj, base: int) -> int:
    """Assign encoded ids in one bulk transfer instead of millions of RNA writes."""
    from array import array

    mesh = obj.data
    if any(len(poly.vertices) != 3 for poly in mesh.polygons):
        raise RuntimeError(f"Mesh must be triangulated before triangle-id assignment: {obj.name}")
    old = mesh.color_attributes.get(ATTRIBUTE_NAME)
    if old is not None:
        mesh.color_attributes.remove(old)
    attr = mesh.color_attributes.new(name=ATTRIBUTE_NAME, type="FLOAT_COLOR", domain="CORNER")
    colors = array("f")
    for local_id in range(len(mesh.polygons)):
        color = encode_triangle_id(base + local_id)
        colors.extend(color)
        colors.extend(color)
        colors.extend(color)
    attr.data.foreach_set("color", colors)
    obj["pgw_triangle_base"] = int(base)
    obj["pgw_triangle_count"] = int(len(mesh.polygons))
    return len(mesh.polygons)


def _triangle_id_material():
    bpy = require_bpy()
    name = "__PGW_TRIANGLE_ID_MATERIAL__"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new("ShaderNodeOutputMaterial")
    try:
        attr = nodes.new("ShaderNodeVertexColor")
        attr.layer_name = ATTRIBUTE_NAME
        color_output = attr.outputs["Color"]
    except Exception:
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = ATTRIBUTE_NAME
        color_output = attr.outputs["Color"]
    try:
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        links.new(color_output, emission.inputs["Color"])
        links.new(emission.outputs["Emission"], out.inputs["Surface"])
    except Exception:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Emission Strength"].default_value = 1.0
        links.new(color_output, bsdf.inputs["Emission Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def render_triangle_id_png(camera, output_path, include_object=None):
    """Render exact per-face ids for all currently visible non-proxy meshes."""
    bpy = require_bpy()
    scene = bpy.context.scene
    visible = [
        obj for obj in scene.objects
        if obj.type == "MESH" and not obj.hide_render
        and not bool(obj.get("pgw_physics_proxy", False))
        and (include_object is None or get_semantic_owner_id(obj) == str(include_object))
    ]
    for obj in visible:
        if obj.data.color_attributes.get(ATTRIBUTE_NAME) is None:
            raise RuntimeError(f"Visible mesh lacks {ATTRIBUTE_NAME}: {obj.name}")
    originals = collect_original_materials()
    old = (
        scene.view_settings.view_transform,
        scene.view_settings.look,
        scene.view_settings.exposure,
        scene.view_settings.gamma,
        scene.render.film_transparent,
        scene.render.filter_size,
        float(scene.render.dither_intensity),
        tuple(scene.world.color) if scene.world is not None else None,
    )
    try:
        mat = _triangle_id_material()
        for obj in visible:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for poly in obj.data.polygons:
                poly.material_index = 0
        scene.camera = camera
        scene.use_nodes = False
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.render.film_transparent = False
        scene.render.filter_size = 0.01
        scene.render.dither_intensity = 0.0
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(output_path)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        bpy.ops.render.render(write_still=True)
    finally:
        restore_materials(originals)
        scene.view_settings.view_transform, scene.view_settings.look = old[0], old[1]
        scene.view_settings.exposure, scene.view_settings.gamma = old[2], old[3]
        scene.render.film_transparent = old[4]
        scene.render.filter_size = old[5]
        scene.render.dither_intensity = old[6]
        if scene.world is not None and old[7] is not None:
            scene.world.color = old[7]
