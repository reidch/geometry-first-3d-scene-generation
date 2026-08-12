from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import get_semantic_owner_id


def _fallback_color(identifier):
    digest = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    rgb = [0.28 + (value / 255.0) * 0.48 for value in digest[:3]]
    return (*rgb, 1.0)


def _load_texture_image(bpy, tex_path):
    tex_path = Path(tex_path)
    key = str(tex_path.resolve()) if tex_path.exists() else str(tex_path)
    old = bpy.data.images.get(key) or bpy.data.images.get(str(tex_path))
    if old is not None:
        try:
            bpy.data.images.remove(old, do_unlink=True)
        except Exception:
            try:
                old.reload()
                return old
            except Exception:
                pass
    image = bpy.data.images.load(str(tex_path.resolve() if tex_path.exists() else tex_path), check_existing=False)
    try:
        image.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    return image


def _lighting_config(config=None):
    return dict(config or {})


def _set_beauty_view_settings(scene, config=None):
    cfg = _lighting_config(config)
    requested_transform = str(cfg.get("view_transform", "AgX"))
    requested_look = str(cfg.get("look", "AgX - Medium Low Contrast"))
    applied_transform = requested_transform
    applied_look = requested_look
    try:
        scene.view_settings.view_transform = requested_transform
    except Exception:
        applied_transform = "Standard"
        scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = requested_look
    except Exception:
        applied_look = "None"
        try:
            scene.view_settings.look = "None"
        except Exception:
            pass
    scene.view_settings.exposure = float(cfg.get("exposure", -1.05))
    scene.view_settings.gamma = float(cfg.get("gamma", 1.0))
    return {
        "view_transform": applied_transform,
        "look": applied_look,
        "exposure": float(scene.view_settings.exposure),
        "gamma": float(scene.view_settings.gamma),
    }


def configure_beauty_color_management(config=None):
    bpy = require_bpy()
    scene = bpy.context.scene
    applied = {}
    try:
        applied = _set_beauty_view_settings(scene, config)
    except Exception:
        pass
    scene.render.film_transparent = False
    cfg = _lighting_config(config)
    world_color = tuple(float(value) for value in cfg.get("world_color", [0.020, 0.020, 0.020]))
    if scene.world is not None:
        try:
            scene.world.use_nodes = False
        except Exception:
            pass
        scene.world.color = world_color
    return {
        **applied,
        "world_color": list(world_color),
    }


def configure_worldmesh_flat_render(config=None):
    """Configure Blender to match WorldMesh's final ``--flat-lighting`` path.

    WorldMesh's production pipeline invokes render_multiview.py with
    ``--flat-lighting`` for both structure and final-with-objects renders. In
    that branch PyRender does not add the directional/spot/point helper lights;
    it renders with ``RenderFlags.FLAT``. Blender has no identical render flag,
    so Stage07 mirrors it by using emission/albedo materials and disabling all
    authored/helper lights. This makes output independent of camera-relative
    light direction and removes shadows/specular shading entirely.
    """
    bpy = require_bpy()
    scene = bpy.context.scene
    helper_names = {
        "PGW_STAGE07_KEY_DIRECTIONAL",
        "PGW_STAGE07_FILL_SPOT",
        "PGW_STAGE07_CENTER_POINT",
        "PGW_STAGE07_SOFT_KEY",
        "PGW_STAGE07_CAMERA_FILL",
        "PGW_STAGE07_TOP_FILL",
    }
    removed_helpers = []
    hidden_authored = []
    for obj in list(scene.objects):
        if obj.type != "LIGHT":
            continue
        if obj.name in helper_names:
            removed_helpers.append(obj.name)
            data = getattr(obj, "data", None)
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                obj.hide_render = True
                obj.hide_viewport = True
            if data is not None:
                try:
                    bpy.data.lights.remove(data, do_unlink=True)
                except Exception:
                    pass
        else:
            if not bool(getattr(obj, "hide_render", False)):
                hidden_authored.append(obj.name)
            obj.hide_render = True
    configure_albedo_color_management()
    try:
        scene.eevee.use_gtao = False
        scene.eevee.gtao_factor = 0.0
    except Exception:
        pass
    return {
        "policy": "worldmesh_final_flat_unlit",
        "worldmesh_equivalent": "pyrender.RenderFlags.FLAT",
        "dynamic_lights": False,
        "shadows": False,
        "specular_lighting": False,
        "gtao": False,
        "material_strategy": "emission_from_base_color_or_stage06_texture",
        "removed_helper_lights": removed_helpers,
        "hidden_authored_scene_lights": hidden_authored,
    }


def configure_albedo_color_management():
    bpy = require_bpy()
    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    scene.render.film_transparent = False
    if scene.world is not None:
        scene.world.color = (0.0, 0.0, 0.0)


def _material_image_paths(material):
    paths = []
    if material is None or not getattr(material, "use_nodes", False) or material.node_tree is None:
        return paths
    bpy = require_bpy()
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeTexImage" or node.image is None:
            continue
        raw = str(node.image.filepath or "")
        if not raw:
            continue
        try:
            paths.append(str(Path(bpy.path.abspath(raw)).resolve()))
        except Exception:
            paths.append(raw)
    return paths


def verify_texture_material_bindings(texture_root, required_owner_ids=None):
    bpy = require_bpy()
    root = Path(texture_root)
    required = None if required_owner_ids is None else {str(value) for value in required_owner_ids}
    meshes_by_owner = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or bool(obj.get("pgw_physics_proxy", False)):
            continue
        owner = str(get_semantic_owner_id(obj))
        meshes_by_owner.setdefault(owner, []).append(obj)
    owners = sorted(required if required is not None else meshes_by_owner)
    records = []
    for owner in owners:
        expected = (root / owner / "base_color.png").resolve()
        meshes = meshes_by_owner.get(owner, [])
        mesh_records = []
        for obj in meshes:
            materials = [slot.material for slot in obj.material_slots if slot.material is not None]
            image_paths = sorted({path for material in materials for path in _material_image_paths(material)})
            active_uv = obj.data.uv_layers.active.name if obj.data.uv_layers.active is not None else None
            valid = bool(materials) and active_uv is not None and str(expected) in image_paths
            mesh_records.append({
                "mesh_object_name": obj.name,
                "material_names": [material.name for material in materials],
                "image_paths": image_paths,
                "active_uv_layer": active_uv,
                "valid": bool(valid),
            })
        problems = []
        if not expected.exists() or expected.stat().st_size == 0:
            problems.append("missing_atlas_image")
        if not meshes:
            problems.append("missing_scene_mesh")
        if any(not record["valid"] for record in mesh_records):
            problems.append("mesh_material_or_uv_not_bound")
        records.append({
            "object_id": owner,
            "expected_atlas": str(expected),
            "mesh_count": len(meshes),
            "meshes": mesh_records,
            "problems": problems,
            "valid": not problems,
        })
    return {
        "status": "ok" if all(record["valid"] for record in records) else "failed",
        "texture_root": str(root.resolve()),
        "required_owner_ids": owners,
        "records": records,
    }


def apply_object_texture_materials(
    texture_root,
    render_mode="albedo",
    interpolation="Linear",
    *,
    required_owner_ids=None,
    binding_report_path=None,
    strict=False,
    lighting_config=None,
):
    bpy = require_bpy()
    root = Path(texture_root)
    mode = str(render_mode or "albedo").lower()
    cache = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or bool(obj.get("pgw_physics_proxy", False)):
            continue
        object_id = get_semantic_owner_id(obj)
        if object_id not in cache:
            material = bpy.data.materials.get(f"__PGW_ATLAS_{mode}_{object_id}") or bpy.data.materials.new(f"__PGW_ATLAS_{mode}_{object_id}")
            material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            for node in list(nodes):
                nodes.remove(node)
            output = nodes.new("ShaderNodeOutputMaterial")
            texture_path = root / object_id / "base_color.png"
            if texture_path.exists():
                texture = nodes.new("ShaderNodeTexImage")
                texture.interpolation = str(interpolation or "Linear")
                texture.extension = "EXTEND"
                texture.image = _load_texture_image(bpy, texture_path)
                color_output = texture.outputs["Color"]
            else:
                rgb = nodes.new("ShaderNodeRGB")
                rgb.outputs["Color"].default_value = _fallback_color(object_id)
                color_output = rgb.outputs["Color"]
            if mode == "beauty":
                shader = nodes.new("ShaderNodeBsdfPrincipled")
                shader.inputs["Roughness"].default_value = 0.68
                links.new(color_output, shader.inputs["Base Color"])
                links.new(shader.outputs["BSDF"], output.inputs["Surface"])
            else:
                shader = nodes.new("ShaderNodeEmission")
                shader.inputs["Strength"].default_value = 1.0
                links.new(color_output, shader.inputs["Color"])
                links.new(shader.outputs["Emission"], output.inputs["Surface"])
            cache[object_id] = material
        obj.data.materials.clear()
        obj.data.materials.append(cache[object_id])
    if mode == "albedo":
        configure_albedo_color_management()
    elif mode == "beauty":
        configure_beauty_color_management(lighting_config)
    report = verify_texture_material_bindings(root, required_owner_ids=required_owner_ids)
    if binding_report_path is not None:
        path = Path(binding_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if strict and report["status"] != "ok":
        invalid = [
            {"object_id": record["object_id"], "problems": record["problems"]}
            for record in report["records"]
            if not record["valid"]
        ]
        raise RuntimeError(f"Texture material binding validation failed: {invalid}")
    return list(cache)
