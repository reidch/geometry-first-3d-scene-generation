from __future__ import annotations

import json
from pathlib import Path

from src.blender.blender_runtime import require_bpy
from src.blender.semantic_render import collect_original_materials, restore_materials


def _clear(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    return tree


def _new_png_output(nodes, path: Path, *, color_depth: str, color_mode: str = "BW"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = nodes.new("CompositorNodeOutputFile")
    node.base_path = str(path.parent)
    node.file_slots[0].path = path.stem + "_"
    node.format.file_format = "PNG"
    node.format.color_mode = color_mode
    node.format.color_depth = str(color_depth)
    node.format.compression = 15
    return node


def _rename_latest_png(directory: Path, prefix: str, final: Path) -> Path:
    directory = Path(directory)
    final = Path(final)
    candidates = sorted(
        directory.glob(prefix + "*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"Blender did not produce the expected PNG: directory={directory}, prefix={prefix}"
        )
    if final.exists():
        final.unlink()
    candidates[0].rename(final)
    return final


def _uv_material():
    bpy = require_bpy()
    name = "__PGW_UV_DATA__"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    uv = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(uv.outputs["UV"], separate.inputs[0])
    tree.links.new(separate.outputs["X"], combine.inputs["X"])
    tree.links.new(separate.outputs["Y"], combine.inputs["Y"])
    combine.inputs["Z"].default_value = 0.0

    try:
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(combine.outputs[0], emission.inputs["Color"])
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
    except Exception:
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        if "Emission Color" in principled.inputs:
            tree.links.new(combine.outputs[0], principled.inputs["Emission Color"])
            principled.inputs["Emission Strength"].default_value = 1.0
        else:
            tree.links.new(combine.outputs[0], principled.inputs["Emission"])
        tree.links.new(principled.outputs[0], output.inputs["Surface"])
    return mat


def _separate_rgba(nodes, image_socket):
    """Return R, G and A sockets across Blender compositor API versions."""
    try:
        node = nodes.new("CompositorNodeSeparateColor")
        node.mode = "RGB"
        node.inputs[0].default_value = (0.0, 0.0, 0.0, 0.0)
        return node, node.outputs["Red"], node.outputs["Green"], node.outputs["Alpha"]
    except Exception:
        node = nodes.new("CompositorNodeSepRGBA")
        return node, node.outputs["R"], node.outputs["G"], node.outputs["A"]



def uv_png_bundle_paths(manifest_path):
    manifest_path = Path(manifest_path)
    stem = manifest_path.stem
    return (
        manifest_path.with_name(stem + "_u.png"),
        manifest_path.with_name(stem + "_v.png"),
        manifest_path.with_name(stem + "_valid.png"),
    )

def render_uv_png_bundle(camera, manifest_path):
    """Render atlas UVs as ordinary PNG images plus a JSON manifest.

    One Blender render writes three standard images:
      * U: 16-bit grayscale PNG
      * V: 16-bit grayscale PNG
      * validity: 8-bit grayscale PNG derived from render alpha

    This replaces the previous OpenEXR transport while retaining substantially
    more precision than an 8-bit packed RGB image.
    """
    bpy = require_bpy()
    scene = bpy.context.scene
    scene.camera = camera
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    u_path, v_path, valid_path = uv_png_bundle_paths(manifest_path)

    originals = collect_original_materials()
    material = _uv_material()
    old_state = {
        "film_transparent": bool(scene.render.film_transparent),
        "dither": float(scene.render.dither_intensity),
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": float(scene.view_settings.exposure),
        "gamma": float(scene.view_settings.gamma),
        "world_color": tuple(scene.world.color) if scene.world is not None else None,
    }

    try:
        for obj in scene.objects:
            if obj.type == "MESH" and hasattr(obj.data, "materials"):
                obj.data.materials.clear()
                obj.data.materials.append(material)

        scene.render.film_transparent = True
        scene.render.dither_intensity = 0.0
        try:
            scene.view_settings.view_transform = "Raw"
            scene.view_settings.look = "None"
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0
        except Exception:
            pass
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)

        tree = _clear(scene)
        nodes = tree.nodes
        links = tree.links
        render_layers = nodes.new("CompositorNodeRLayers")
        separate, red, green, alpha = _separate_rgba(nodes, render_layers.outputs["Image"])
        links.new(render_layers.outputs["Image"], separate.inputs[0])

        u_output = _new_png_output(nodes, u_path, color_depth="16")
        v_output = _new_png_output(nodes, v_path, color_depth="16")
        valid_output = _new_png_output(nodes, valid_path, color_depth="8")
        links.new(red, u_output.inputs[0])
        links.new(green, v_output.inputs[0])
        links.new(alpha, valid_output.inputs[0])

        bpy.ops.render.render(write_still=False)
        _rename_latest_png(u_path.parent, u_path.stem + "_", u_path)
        _rename_latest_png(v_path.parent, v_path.stem + "_", v_path)
        _rename_latest_png(valid_path.parent, valid_path.stem + "_", valid_path)
    finally:
        scene.use_nodes = False
        restore_materials(originals)
        scene.render.film_transparent = old_state["film_transparent"]
        scene.render.dither_intensity = old_state["dither"]
        try:
            scene.view_settings.view_transform = old_state["view_transform"]
            scene.view_settings.look = old_state["look"]
            scene.view_settings.exposure = old_state["exposure"]
            scene.view_settings.gamma = old_state["gamma"]
        except Exception:
            pass
        if scene.world is not None and old_state["world_color"] is not None:
            scene.world.color = old_state["world_color"]

    payload = {
        "schema_version": 1,
        "type": "uv_map_png_bundle",
        "image_size": [
            int(scene.render.resolution_x * scene.render.resolution_percentage / 100),
            int(scene.render.resolution_y * scene.render.resolution_percentage / 100),
        ],
        "u_image": u_path.name,
        "v_image": v_path.name,
        "valid_image": valid_path.name,
        "encoding": {
            "u": "uint16_normalized_0_1",
            "v": "uint16_normalized_0_1",
            "valid": "uint8_zero_background_nonzero_mesh",
            "bit_depth_uv": 16,
            "bit_depth_valid": 8,
            "color_management": "Raw",
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
