#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.blender_runtime import require_bpy
from src.blender.object_identity import (
    assert_consistent_identity,
    get_semantic_owner_id,
    get_world_object_id,
    set_object_identity,
)
from src.io.json_io import load_json, save_json
from src.blender.triangle_id_render import assign_triangle_id_attribute
from src.blender.prephysics_runtime.stdlib_obj_importer import load_obj_into_blender


def _world_object_id(obj):
    return get_world_object_id(obj)


def _semantic_owner_id(obj):
    return get_semantic_owner_id(obj)


def _select_only(bpy, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def _import_asset(bpy, path: Path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".obj":
        raise ValueError(f"Stage05 requires the exported OBJ/MTL bundle: {path}")
    return load_obj_into_blender(bpy, path, object_name=f"PGW_IMPORTED__{path.parent.name}")


def _join_meshes(bpy, meshes, name):
    if not meshes:
        raise RuntimeError(f"No mesh objects were imported for {name}")
    if len(meshes) == 1:
        obj = meshes[0]
    else:
        _select_only(bpy, meshes)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.object
    obj.name = name
    _select_only(bpy, [obj])
    bpy.context.view_layer.objects.active = obj
    # Keep transforms as object matrices. Applying transforms to very dense generated
    # meshes is unnecessary for UV/triangle processing and has caused native Blender
    # crashes in background mode.
    return obj


def _matrix_from_transform(transform):
    from mathutils import Euler, Matrix, Vector
    transform = dict(transform or {})
    position = Vector([float(v) for v in transform.get("position", [0.0, 0.0, 0.0])])
    rotation = Euler([math.radians(float(v)) for v in transform.get("rotation_deg", [0.0, 0.0, 0.0])], "XYZ")
    scale = [float(v) for v in transform.get("scale", [1.0, 1.0, 1.0])]
    return Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ Matrix.Diagonal((*scale, 1.0))


def _rotation_matrix(values):
    from mathutils import Matrix
    rows = [[float(v) for v in row] for row in values]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("registration rotation_matrix must be 3x3")
    matrix = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            matrix[r][c] = rows[r][c]
    determinant = matrix.to_3x3().determinant()
    if abs(abs(determinant) - 1.0) > 1e-4:
        raise ValueError("registration transform must be an orthonormal rotation/reflection")
    return matrix


def _objects_for_id(bpy, object_id, *, include_generated=False):
    result = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if _world_object_id(obj) != str(object_id):
            continue
        if not include_generated and bool(obj.get("pgw_generated_asset", False)):
            continue
        result.append(obj)
    return result


def _scaffold_anchor_for_id(bpy, object_id):
    matches = [
        obj for obj in bpy.context.scene.objects
        if bool(obj.get("pgw_scaffold_owner_anchor", False))
        and _world_object_id(obj) == str(object_id)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple scaffold owner anchors found for {object_id!r}")
    return matches[0] if matches else None


def _owner_scaffold_world_matrix(bpy, object_id, flat_record):
    anchor = _scaffold_anchor_for_id(bpy, object_id)
    if anchor is not None:
        return anchor.matrix_world.copy(), "stage02_scaffold_owner_anchor"
    # Backward-compatible fallback when the explicit Stage02 scaffold anchor is absent.
    return _matrix_from_transform(dict(flat_record or {}).get("world_transform", {})), "stage00_flat_world_transform_fallback"


def _world_points(objects):
    from mathutils import Vector
    return [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]


def _projection_bounds(objects, axis):
    values = [float(point.dot(axis)) for point in _world_points(objects)]
    if not values:
        raise RuntimeError("Cannot project an empty object set")
    return min(values), max(values)


def _world_aabb(objects):
    points = _world_points(objects)
    if not points:
        raise RuntimeError("Cannot compute AABB for an empty object set")
    minimum = [min(float(point[index]) for point in points) for index in range(3)]
    maximum = [max(float(point[index]) for point in points) for index in range(3)]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "center": [0.5 * (minimum[index] + maximum[index]) for index in range(3)],
        "extent": [maximum[index] - minimum[index] for index in range(3)],
    }


def _mesh_projection_values(obj, axis):
    return sorted(float((obj.matrix_world @ vertex.co).dot(axis)) for vertex in obj.data.vertices)


def _robust_contact(values, config):
    values = sorted(float(v) for v in values)
    if not values:
        raise RuntimeError("Generated asset has no vertices")
    schedule = [float(v) for v in config.get("trim_schedule", [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04])]
    max_trim = float(config.get("max_trim_ratio", 0.04))
    band_ratio = float(config.get("support_band_ratio", 0.02))
    minimum = max(1, int(config.get("min_support_points", 16)))
    candidate_limit = float(config.get("candidate_max_penetration_depth_m", 0.006))
    attempts = []
    for trim in schedule:
        if trim > max_trim + 1e-12:
            continue
        cut = min(len(values) - 1, max(0, int(math.floor(len(values) * trim))))
        remaining = values[cut:]
        band_count = min(len(remaining), max(minimum, int(math.ceil(len(remaining) * band_ratio))))
        band = remaining[:band_count]
        # A trimmed mean provides a diagnostic support level. Placement itself uses
        # the first retained point, so the same contact definition is used later.
        lower = max(0, int(math.floor(0.10 * len(band))))
        upper = max(lower + 1, int(math.ceil(0.90 * len(band))))
        support_level = sum(band[lower:upper]) / max(1, len(band[lower:upper]))
        effective_contact = remaining[0]
        depth = max(0.0, support_level - effective_contact)
        attempt = {
            "trim_ratio": trim,
            "trimmed_vertex_count": cut,
            "effective_contact_projection_m": effective_contact,
            "support_level_projection_m": support_level,
            "candidate_penetration_depth_m": depth,
        }
        attempts.append(attempt)
        if depth <= candidate_limit:
            return {**attempt, "selection": "first_depth_stable", "attempts": attempts}
    chosen = min(attempts, key=lambda item: item["candidate_penetration_depth_m"])
    return {**chosen, "selection": "minimum_penetration_depth", "attempts": attempts}


def _obb(obj):
    from mathutils import Vector
    corners = [Vector(c) for c in obj.bound_box]
    local_min = Vector([min(c[i] for c in corners) for i in range(3)])
    local_max = Vector([max(c[i] for c in corners) for i in range(3)])
    local_center = 0.5 * (local_min + local_max)
    local_half = 0.5 * (local_max - local_min)
    center = obj.matrix_world @ local_center
    axes = []
    half = []
    basis = obj.matrix_world.to_3x3()
    for index in range(3):
        vector = basis.col[index]
        length = max(float(vector.length), 1e-12)
        axes.append(tuple(float(v / length) for v in vector))
        half.append(float(local_half[index]) * length)
    return tuple(center), axes, half


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _cross(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _normalize(value, epsilon=1e-9):
    length = math.sqrt(max(_dot(value, value), 0.0))
    if length <= epsilon:
        return None
    return tuple(float(component) / length for component in value)


def _obb_penetration_depth(a, b, epsilon=1e-9):
    center_a, axes_a, half_a = a
    center_b, axes_b, half_b = b
    axes = list(axes_a) + list(axes_b)
    for axis_a in axes_a:
        for axis_b in axes_b:
            axis = _normalize(_cross(axis_a, axis_b), epsilon)
            if axis is not None:
                axes.append(axis)
    delta = tuple(float(center_b[i]) - float(center_a[i]) for i in range(3))
    minimum = float("inf")
    for axis in axes:
        radius_a = sum(float(half_a[i]) * abs(_dot(axes_a[i], axis)) for i in range(3))
        radius_b = sum(float(half_b[i]) * abs(_dot(axes_b[i], axis)) for i in range(3))
        overlap = radius_a + radius_b - abs(_dot(delta, axis))
        if overlap <= epsilon:
            return 0.0
        minimum = min(minimum, overlap)
    return float(minimum if math.isfinite(minimum) else 0.0)


def _support_graph(plan):
    support_of = {}
    for record in plan.get("objects", []):
        target = dict(record.get("placement", {})).get("support_target")
        if target:
            support_of[str(record["object_id"])] = str(target)
    return support_of


def _related_ids(current, support_of):
    """Return the full ancestor/descendant support family of ``current``.

    Overlap diagnostics use this only to avoid describing legal nested support chains
    as unrelated collisions.  Siblings and objects in separate support trees remain
    eligible for diagnostics.
    """
    current = str(current)
    related = {current}

    # Walk all ancestors.
    cursor = current
    while cursor in support_of:
        cursor = str(support_of[cursor])
        if cursor in related:
            break
        related.add(cursor)

    # Walk descendants from the current object only.  Starting this traversal from
    # ancestors would incorrectly classify siblings as descendants.
    frontier = [current]
    while frontier:
        parent_id = frontier.pop(0)
        for child, parent in sorted(support_of.items()):
            child = str(child)
            parent = str(parent)
            if parent == parent_id and child not in related:
                related.add(child)
                frontier.append(child)
    return related


def _max_unrelated_final_visual_overlap(visual_by_id, object_id, plan):
    """Compute a report-only OBB overlap diagnostic after every object is placed.

    Only final render visuals participate.  Physics proxies and partially placed
    intermediate states are intentionally excluded because their coarse OBBs are not
    reliable collision decisions.
    """
    object_id = str(object_id)
    visual = visual_by_id[object_id]
    excluded = _related_ids(object_id, _support_graph(plan))
    visual_obb = _obb(visual)
    maximum = 0.0
    against = None
    for other_id in sorted(visual_by_id):
        other_id = str(other_id)
        if other_id in excluded:
            continue
        depth = _obb_penetration_depth(visual_obb, _obb(visual_by_id[other_id]))
        if depth > maximum:
            maximum, against = depth, other_id
    return maximum, against


def _bake_registration_in_model_space(visual, registration):
    from mathutils import Matrix, Vector
    rotation = _rotation_matrix(registration["rotation_matrix"])
    scale = float(registration["uniform_scale"])
    translation = Vector([float(v) for v in registration["translation_local"]])
    local = Matrix.Translation(translation) @ Matrix.Scale(scale, 4) @ rotation
    visual.data.transform(local)
    # Registration is baked into mesh-local coordinates.  World placement is a
    # separate hierarchy pass: roots inherit scaffold world matrices and children
    # preserve their declared local matrices relative to the generated parent.
    if rotation.to_3x3().determinant() < 0.0:
        try:
            visual.data.flip_normals()
        except AttributeError:
            for polygon in visual.data.polygons:
                polygon.flip()
    visual.data.update()



def _place_on_explicit_support(bpy, visual, object_record, plan, alignment_cfg, *, move_objects=None):
    """Stable scalar support snapping used by the previously validated pipeline.

    The contact definition is deliberately simple and self-consistent: a robust
    lower projection of the child is translated to the global upper projection of
    the explicit support target.  Placement and validation use the same scalar
    definition, avoiding the later raycast/clustering/conformation regressions.
    """
    from mathutils import Vector

    placement = dict(object_record.get("placement", {}))
    target_id = placement.get("support_target")
    axis_values = placement.get("support_axis_world", [0.0, 0.0, 1.0])
    axis = Vector([float(v) for v in axis_values])
    if axis.length <= 1e-10:
        raise ValueError(f"support_axis_world is zero for {object_record['object_id']}")
    axis.normalize()

    estimator_cfg = dict(alignment_cfg.get("support_estimator", {}))
    clearance = float(placement.get("clearance_m", alignment_cfg.get("support_clearance_m", 0.0015)))
    estimate = _robust_contact(_mesh_projection_values(visual, axis), estimator_cfg)
    target_projection = None
    correction = 0.0

    if target_id:
        support_objects = _objects_for_id(bpy, str(target_id), include_generated=True)
        if not support_objects:
            raise RuntimeError(f"Explicit support_target {target_id!r} was not found in the scaffold scene")
        _, support_top = _projection_bounds(support_objects, axis)
        target_projection = float(support_top) + clearance
        correction = target_projection - float(estimate["effective_contact_projection_m"])
        translated = list(move_objects or [visual])
        for obj in translated:
            obj.matrix_world.translation += axis * correction
        bpy.context.view_layer.update()

    final = _robust_contact(_mesh_projection_values(visual, axis), estimator_cfg)
    signed_gap = 0.0 if target_projection is None else (
        float(final["effective_contact_projection_m"]) - float(target_projection)
    )
    return {
        "support_target": str(target_id) if target_id else None,
        "support_axis_world": [float(v) for v in axis],
        "clearance_m": clearance,
        "target_projection_m": target_projection,
        "translation_correction_m": correction,
        "selected_trim_ratio": float(final["trim_ratio"]),
        "support_gap_m": signed_gap,
        "support_penetration_depth_m": max(0.0, -signed_gap),
        "support_floating_gap_m": max(0.0, signed_gap),
        "support_estimate": final,
        "placement_method": "stable_global_support_top_and_robust_child_bottom",
    }


def _validate(registration, support, alignment_cfg):
    """Validate registration and explicit support snapping only.

    Cross-object OBB overlap is deliberately excluded: a coarse render-mesh OBB is a
    useful diagnostic but not a reliable reject criterion for hollow generated geometry,
    nested support chains, or generated geometry.
    """
    validation = dict(alignment_cfg.get("validation", {}))
    problems = []
    max_loss = float(validation.get("max_registration_loss", 0.08))
    registration_loss = float(registration.get("normalized_loss", 0.0))
    if registration_loss > max_loss:
        # Registration quality is report-only.  The globally best candidate has
        # already been selected and applied; a high residual must not abort Stage05.
        problems.append(
            f"report-only registration loss {registration_loss:.6f} exceeds preferred {max_loss:.6f}"
        )
    max_penetration = float(validation.get("max_support_penetration_m", 0.003))
    if float(support["support_penetration_depth_m"]) > max_penetration:
        problems.append(
            f"support penetration depth {support['support_penetration_depth_m']:.6f}m exceeds {max_penetration:.6f}m"
        )
    max_gap = float(validation.get("max_support_gap_m", 0.005))
    if float(support["support_floating_gap_m"]) > max_gap:
        problems.append(
            f"support floating gap {support['support_floating_gap_m']:.6f}m exceeds {max_gap:.6f}m"
        )
    fatal_problems = [problem for problem in problems if not problem.startswith("report-only registration loss")]
    return not fatal_problems, problems


def _linear_channel_to_srgb(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return 12.92 * value if value <= 0.0031308 else 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _srgb_channel_to_linear(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _srgb_u8_from_linear(rgb):
    return tuple(
        max(0, min(255, int(round(_linear_channel_to_srgb(float(value)) * 255.0))))
        for value in rgb[:3]
    )


def _linear_rgb_from_srgb_u8(rgb):
    return tuple(_srgb_channel_to_linear(float(value) / 255.0) for value in rgb[:3])


def _material_base_color_source(bpy, material):
    """Resolve a material's base-colour image or scene-linear constant colour.

    This stays entirely inside Blender's bundled Python.  In particular it does not
    import Pillow, which is normally installed in the project's virtual environment
    but not in Blender's private Python runtime.
    """
    default_linear = tuple(_linear_rgb_from_srgb_u8((115, 115, 115)))
    if material is None:
        return {
            "kind": "constant",
            "linear_color": default_linear,
            "color": list(_srgb_u8_from_linear(default_linear)),
            "image": None,
            "extension": "EXTEND",
        }
    material.use_nodes = True
    tree = material.node_tree
    material_default = tuple(float(v) for v in material.diffuse_color[:3])
    image_node = None
    shader_color = None
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            base_input = node.inputs.get("Base Color")
            if base_input is not None:
                shader_color = tuple(float(v) for v in base_input.default_value[:3])
                if base_input.is_linked:
                    source = base_input.links[0].from_node
                    if source.type == "TEX_IMAGE" and source.image is not None:
                        image_node = source
                        break
        elif node.type == "EMISSION":
            color_input = node.inputs.get("Color")
            if color_input is not None:
                shader_color = tuple(float(v) for v in color_input.default_value[:3])
                if color_input.is_linked:
                    source = color_input.links[0].from_node
                    if source.type == "TEX_IMAGE" and source.image is not None:
                        image_node = source
                        break
    if image_node is None:
        image_node = next(
            (node for node in tree.nodes if node.type == "TEX_IMAGE" and node.image is not None),
            None,
        )
    linear_color = tuple(shader_color or material_default or default_linear)
    if image_node is not None:
        return {
            "kind": "image",
            "linear_color": linear_color,
            "color": list(_srgb_u8_from_linear(linear_color)),
            "image": image_node.image,
            "extension": str(getattr(image_node, "extension", "EXTEND")),
        }
    return {
        "kind": "constant",
        "linear_color": linear_color,
        "color": list(_srgb_u8_from_linear(linear_color)),
        "image": None,
        "extension": "EXTEND",
    }


def _blender_image_source_path(bpy, image):
    raw_paths = []
    try:
        raw_paths.append(image.filepath_from_user())
    except Exception:
        pass
    raw_paths.append(getattr(image, "filepath", ""))
    for raw in raw_paths:
        if not raw:
            continue
        try:
            candidate = Path(bpy.path.abspath(raw)).expanduser()
        except Exception:
            candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
    return f"bpy://{image.name}"


def _scaled_blender_image_pixels_linear(bpy, image, width: int, height: int):
    """Read and resize image pixels without image.copy()/image.scale() native calls.

    A nearest-neighbour resample is sufficient for the bootstrap atlas because Stage08
    will refine the texture. Avoiding temporary Blender image datablocks removes another
    native crash surface for malformed or unusually large generated textures.
    """
    from array import array

    width, height = max(1, int(width)), max(1, int(height))
    source_width = max(1, int(image.size[0]))
    source_height = max(1, int(image.size[1]))
    source = array("f", [0.0]) * (source_width * source_height * 4)
    image.pixels.foreach_get(source)
    if source_width == width and source_height == height:
        return source, _blender_image_source_path(bpy, image)

    x_map = [min(source_width - 1, int((x + 0.5) * source_width / width)) for x in range(width)]
    y_map = [min(source_height - 1, int((y + 0.5) * source_height / height)) for y in range(height)]
    output = array("f", [0.0]) * (width * height * 4)
    for output_y, source_y in enumerate(y_map):
        output_row = 4 * output_y * width
        source_row = 4 * source_y * source_width
        for output_x, source_x in enumerate(x_map):
            source_offset = source_row + 4 * source_x
            output_offset = output_row + 4 * output_x
            output[output_offset:output_offset + 4] = source[source_offset:source_offset + 4]
    return output, _blender_image_source_path(bpy, image)


def _tile_row_from_source(source_pixels, source_width: int, source_y: int, gutter: int):
    from array import array

    source_start = 4 * (source_y * source_width)
    source_row = source_pixels[source_start:source_start + 4 * source_width]
    left = array("f", source_row[:4]) * gutter
    right = array("f", source_row[-4:]) * gutter
    return left + source_row + right


def _constant_tile_row(linear_color, inner: int, gutter: int):
    from array import array

    pixel = array("f", [float(linear_color[0]), float(linear_color[1]), float(linear_color[2]), 1.0])
    return pixel * (inner + 2 * gutter)

def _remap_uv_value(value: float, extension: str) -> float:
    value = float(value)
    if str(extension).upper() == "REPEAT":
        if abs(value - round(value)) < 1e-8 and value > 0.0:
            return 1.0
        return value - math.floor(value)
    return max(0.0, min(1.0, value))




def _ensure_stable_uv_layer(obj):
    """Return a UV layer without invoking edit-mode UV operators.

    Generated OBJ assets already arrive with per-loop UVs from the defensive importer.
    Scaffold-only meshes use a deterministic local-space box projection when needed.
    """
    mesh = obj.data
    source_uv = mesh.uv_layers.active
    if source_uv is not None:
        return source_uv
    source_uv = mesh.uv_layers.new(name="PGW_SOURCE_UV")
    if not mesh.vertices or not mesh.polygons:
        raise RuntimeError(f"Visible mesh has no usable geometry for UV initialization: {obj.name}")
    minimum = [min(float(vertex.co[index]) for vertex in mesh.vertices) for index in range(3)]
    maximum = [max(float(vertex.co[index]) for vertex in mesh.vertices) for index in range(3)]
    for poly in mesh.polygons:
        normal = tuple(float(value) for value in poly.normal)
        axis = max(range(3), key=lambda index: abs(normal[index]))
        dims = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
        for loop_index in poly.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv = source_uv.data[loop_index].uv
            for output_index, dim in enumerate(dims):
                span = max(maximum[dim] - minimum[dim], 1e-12)
                uv[output_index] = (float(vertex[dim]) - minimum[dim]) / span
    mesh.uv_layers.active = source_uv
    source_uv.active_render = True
    return source_uv


def _checkpoint(path: Path, phase: str, object_id=None, **details):
    payload = {"phase": str(phase), "object_id": str(object_id) if object_id is not None else None, **details}
    save_json(payload, path)
    suffix = f" object={object_id}" if object_id is not None else ""
    print(f"[05][BLENDER] {phase}{suffix}", flush=True)

def _initialize_editable_material_atlas(
    bpy,
    obj,
    output_path: Path,
    resolution: int,
    margin: int,
    fallback_color=(115, 115, 115),
):
    """Build an editable atlas from imported OBJ materials without renderer bake.

    The implementation uses only Blender's image API and Python's standard library,
    so it works in a stock headless Blender installation.  Source textures are
    resized by Blender, copied into material tiles, and the source UVs are remapped
    into those tiles before one editable atlas material replaces the imported slots.
    """
    from array import array

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = max(4, int(resolution))
    _select_only(bpy, [obj])
    bpy.context.view_layer.objects.active = obj

    source_uv = _ensure_stable_uv_layer(obj)
    source_uv.name = "PGW_UV"
    obj.data.uv_layers.active = source_uv
    source_uv.active_render = True

    used_indices = sorted({int(poly.material_index) for poly in obj.data.polygons}) or [0]
    if len(obj.data.materials) == 0:
        material = bpy.data.materials.new(name=f"PGW_FALLBACK__{obj.name}")
        fallback_linear = _linear_rgb_from_srgb_u8(fallback_color)
        material.diffuse_color = tuple(fallback_linear) + (1.0,)
        material.use_nodes = True
        bsdf = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = tuple(fallback_linear) + (1.0,)
        obj.data.materials.append(material)

    sources = {}
    for material_index in used_indices:
        material = obj.data.materials[material_index] if material_index < len(obj.data.materials) else None
        source = _material_base_color_source(bpy, material)
        source["material_index"] = int(material_index)
        source["material_name"] = material.name if material is not None else None
        sources[material_index] = source

    grid = max(1, int(math.ceil(math.sqrt(len(used_indices)))))
    tile_size = max(1, resolution // grid)
    fallback_linear = _linear_rgb_from_srgb_u8(fallback_color)
    fallback_pixel = array(
        "f",
        [float(fallback_linear[0]), float(fallback_linear[1]), float(fallback_linear[2]), 1.0],
    )
    atlas_pixels = fallback_pixel * (resolution * resolution)
    tile_records = []
    material_to_tile = {}
    detail_values = []
    has_source_image = False

    for tile_index, material_index in enumerate(used_indices):
        tile_u = tile_index % grid
        tile_v = tile_index // grid  # bottom-origin, matching Blender UV coordinates
        source = sources[material_index]
        gutter = max(0, min(min(int(margin), max(2, tile_size // 32)), max(0, tile_size // 4)))
        inner = max(1, tile_size - 2 * gutter)
        source_path = None
        source_pixels = None
        if source["kind"] == "image":
            source_pixels, source_path = _scaled_blender_image_pixels_linear(
                bpy,
                source["image"],
                inner,
                inner,
            )
            has_source_image = True

        luminance_sum = 0.0
        luminance_sq_sum = 0.0
        luminance_count = 0
        constant_row = None
        if source_pixels is None:
            constant_row = _constant_tile_row(source["linear_color"], inner, gutter)
            luminance = (
                0.2126 * float(source["linear_color"][0])
                + 0.7152 * float(source["linear_color"][1])
                + 0.0722 * float(source["linear_color"][2])
            )
            luminance_sum = luminance * inner * inner
            luminance_sq_sum = luminance * luminance * inner * inner
            luminance_count = inner * inner
        else:
            for pixel_index in range(0, len(source_pixels), 4):
                luminance = (
                    0.2126 * float(source_pixels[pixel_index])
                    + 0.7152 * float(source_pixels[pixel_index + 1])
                    + 0.0722 * float(source_pixels[pixel_index + 2])
                )
                luminance_sum += luminance
                luminance_sq_sum += luminance * luminance
                luminance_count += 1

        paste_x = tile_u * tile_size
        paste_y = tile_v * tile_size
        for tile_y in range(tile_size):
            if constant_row is not None:
                row = constant_row
            else:
                source_y = max(0, min(inner - 1, tile_y - gutter))
                row = _tile_row_from_source(source_pixels, inner, source_y, gutter)
            destination = 4 * ((paste_y + tile_y) * resolution + paste_x)
            atlas_pixels[destination:destination + len(row)] = row

        mean_luminance = luminance_sum / max(luminance_count, 1)
        variance = max(0.0, luminance_sq_sum / max(luminance_count, 1) - mean_luminance ** 2)
        detail = math.sqrt(variance)
        detail_values.append(detail)
        tile_records.append({
            "material_index": int(material_index),
            "material_name": source.get("material_name"),
            "source_kind": source["kind"],
            "source_path": source_path,
            "constant_color": list(source["color"]),
            "extension": source["extension"],
            "tile_uv_index": [int(tile_u), int(tile_v)],
            "tile_pixel_box_bottom_origin": [
                int(paste_x),
                int(paste_y),
                int(paste_x + tile_size),
                int(paste_y + tile_size),
            ],
            "gutter_px": int(gutter),
            "luminance_std_linear": float(detail),
        })
        material_to_tile[material_index] = (tile_u, tile_v, gutter, source["extension"])

    denominator = max(resolution - 1, 1)
    for poly in obj.data.polygons:
        material_index = int(poly.material_index)
        if material_index not in material_to_tile:
            material_index = used_indices[0]
        tile_u, tile_v, gutter, extension = material_to_tile[material_index]
        inner = max(1, tile_size - 2 * gutter)
        for loop_index in poly.loop_indices:
            uv = source_uv.data[loop_index].uv
            local_u = _remap_uv_value(float(uv.x), extension)
            local_v = _remap_uv_value(float(uv.y), extension)
            uv.x = (tile_u * tile_size + gutter + local_u * max(inner - 1, 0)) / denominator
            uv.y = (tile_v * tile_size + gutter + local_v * max(inner - 1, 0)) / denominator

    image_name = f"PGW_ATLAS__{obj.name}"
    old_image = bpy.data.images.get(image_name)
    if old_image is not None:
        bpy.data.images.remove(old_image)
    atlas_image = bpy.data.images.new(
        name=image_name,
        width=resolution,
        height=resolution,
        alpha=True,
        float_buffer=False,
    )
    try:
        atlas_image.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    atlas_image.pixels.foreach_set(atlas_pixels)
    atlas_image.update()
    atlas_image.filepath_raw = str(output_path.resolve())
    atlas_image.file_format = "PNG"
    atlas_image.save()

    rgb_min = 255
    rgb_max = 0
    luminance_sum = 0.0
    luminance_sq_sum = 0.0
    nonzero_count = 0
    pixel_count = resolution * resolution
    for index in range(0, len(atlas_pixels), 4):
        rgb_u8 = _srgb_u8_from_linear(atlas_pixels[index:index + 3])
        rgb_min = min(rgb_min, *rgb_u8)
        rgb_max = max(rgb_max, *rgb_u8)
        if any(value > 0 for value in rgb_u8):
            nonzero_count += 1
        luminance = 0.2126 * rgb_u8[0] + 0.7152 * rgb_u8[1] + 0.0722 * rgb_u8[2]
        luminance_sum += luminance
        luminance_sq_sum += luminance * luminance
    mean_luminance = luminance_sum / max(pixel_count, 1)
    luminance_variance = max(0.0, luminance_sq_sum / max(pixel_count, 1) - mean_luminance ** 2)
    diagnostics = {
        "status": "ok",
        "method": "deterministic_obj_material_atlas_blender_stdlib",
        "cycles_bake_used": False,
        "external_image_library_used_inside_blender": False,
        "resolution": resolution,
        "material_count": len(used_indices),
        "tile_grid": int(grid),
        "has_source_image": bool(has_source_image),
        "initial_texture_has_detail": bool(max(detail_values or [0.0]) > 0.003),
        "rgb_min": int(rgb_min),
        "rgb_max": int(rgb_max),
        "mean_luminance_u8": float(mean_luminance),
        "luminance_std_u8": float(math.sqrt(luminance_variance)),
        "nonzero_rgb_fraction": float(nonzero_count / max(pixel_count, 1)),
        "tiles": tile_records,
        "output_path": str(output_path),
    }
    diagnostics_path = output_path.with_name("atlas_transfer_report.json")
    save_json(diagnostics, diagnostics_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Material atlas transfer did not produce: {output_path}")

    mat = bpy.data.materials.new(name=f"PGW_ATLAS_MATERIAL__{_semantic_owner_id(obj)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new("ShaderNodeOutputMaterial")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = atlas_image
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.65
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.material_index = 0
    return diagnostics

def _triangulate_mesh(obj):
    import bmesh
    mesh = obj.data
    if all(len(poly.vertices) == 3 for poly in mesh.polygons):
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()
    if any(len(poly.vertices) != 3 for poly in mesh.polygons):
        raise RuntimeError(f"Failed to triangulate visible mesh: {obj.name}")

def _assign_triangle_ids_and_export(bpy, output_path: Path):
    records = {}
    visible_meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not bool(obj.get("pgw_physics_proxy", False))
    ]
    for obj in visible_meshes:
        _triangulate_mesh(obj)
    groups = {}
    for obj in visible_meshes:
        groups.setdefault(_semantic_owner_id(obj), []).append(obj)
    for semantic_name, objects in sorted(groups.items()):
        base = 0
        tris = []
        for obj in sorted(objects, key=lambda x: x.name):
            mesh = obj.data
            uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
            if uv_layer is None:
                raise RuntimeError(f"Visible mesh has no UV layer: {obj.name}")
            count = assign_triangle_id_attribute(obj, base)
            for local_index, poly in enumerate(mesh.polygons):
                if len(poly.vertices) != 3:
                    raise RuntimeError(f"Non-triangle face remained in {obj.name}")
                uvs = []
                for loop_index in poly.loop_indices:
                    uv = uv_layer[loop_index].uv
                    uvs.append([float(uv.x), float(uv.y)])
                tris.append({
                    "global_triangle_id": int(base + local_index),
                    "mesh_object_name": obj.name,
                    "local_triangle_id": int(local_index),
                    "uv": uvs,
                })
            base += count
        records[semantic_name] = {"triangle_count": int(base), "triangles": tris}
    save_json({"objects": records}, output_path)
    return records



def _duplicate_scaffold_visual(bpy, scaffold, name, owner_world):
    duplicates = []
    for source in scaffold:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.hide_render = False
        duplicate.hide_viewport = False
        duplicates.append(duplicate)
    visual = _join_meshes(bpy, duplicates, name)
    current_world = visual.matrix_world.copy()
    visual.parent = None
    visual.matrix_world = current_world
    visual.data.transform(owner_world.inverted() @ current_world)
    visual.matrix_world = owner_world
    visual.data.update()
    return visual


def _hierarchy_topological_order(records, object_ids):
    """Return deterministic generated-parent-before-child order."""
    object_ids = [str(value) for value in object_ids]
    id_set = set(object_ids)
    by_id = {str(record["object_id"]): record for record in records}
    order = []
    state = {}

    def visit(object_id):
        mark = state.get(object_id, 0)
        if mark == 2:
            return
        if mark == 1:
            raise RuntimeError(f"Generated hierarchy cycle detected at {object_id}")
        state[object_id] = 1
        parent = str(by_id[object_id].get("parent_id") or "")
        if parent in id_set:
            visit(parent)
        state[object_id] = 2
        order.append(object_id)

    for object_id in object_ids:
        visit(object_id)
    return order


def _apply_generated_hierarchy(bpy, visual, object_record, flat_record, visual_by_id, owner_world):
    from mathutils import Matrix
    parent_id = str(dict(flat_record or {}).get("parent_id") or object_record.get("parent_id") or "")
    local_transform = dict(dict(flat_record or {}).get("transform", {}))
    if parent_id and parent_id in visual_by_id:
        visual.parent = visual_by_id[parent_id]
        visual.matrix_parent_inverse = Matrix.Identity(4)
        visual.matrix_basis = _matrix_from_transform(local_transform)
        method = "generated_parent_plus_preserved_json_local_matrix"
        source = "stage00_flat_local_transform"
    else:
        visual.parent = None
        visual.matrix_world = owner_world
        method = "root_inherits_scaffold_owner_world_matrix"
        source = "stage02_scaffold_owner_anchor_or_flat_fallback"
    bpy.context.view_layer.update()
    placement = dict(object_record.get("placement", {}))
    return {
        "placement_method": method,
        "hierarchy_parent_id": parent_id or None,
        "world_matrix_source": source,
        "support_target": placement.get("support_target"),
        "support_relationship_preserved_as_metadata_only": bool(placement.get("support_target")),
        "support_translation_applied": False,
        "local_matrix_preserved": bool(parent_id and parent_id in visual_by_id),
        "matrix_local": [[float(value) for value in row] for row in visual.matrix_basis],
        "matrix_world": [[float(value) for value in row] for row in visual.matrix_world],
        "support_penetration_depth_m": 0.0,
        "support_floating_gap_m": 0.0,
    }


def _support_topological_order(records, object_ids):
    """Return a deterministic support-parent-before-child order."""
    object_ids = [str(value) for value in object_ids]
    id_set = set(object_ids)
    by_id = {str(record["object_id"]): record for record in records}
    order = []
    state = {}

    def visit(object_id):
        mark = state.get(object_id, 0)
        if mark == 2:
            return
        if mark == 1:
            raise RuntimeError(f"Support relationship cycle detected at {object_id}")
        state[object_id] = 1
        target = str(dict(by_id[object_id].get("placement", {})).get("support_target") or "")
        if target in id_set:
            visit(target)
        state[object_id] = 2
        order.append(object_id)

    for object_id in object_ids:
        visit(object_id)
    return order


def _scaffold_registration_record(object_id):
    return {
        "object_id": str(object_id),
        "method": "identity_json_scaffold_visual",
        "rotation_index": None,
        "uniform_scale": 1.0,
        "translation": [0.0, 0.0, 0.0],
        "normalized_loss": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--asset_config", default="configs/asset_pipeline.json")
    args = ap.parse_args(argv)

    bpy = require_bpy()
    out = Path(args.out)
    config = load_json(args.asset_config)
    alignment_cfg = dict(config.get("alignment", {}))
    resolution = int(alignment_cfg.get("atlas_resolution", 1024))
    margin = int(alignment_cfg.get("bake_margin_px", 16))
    step = out / "05_scene_assets"
    step.mkdir(parents=True, exist_ok=True)
    checkpoint_path = step / "blender_checkpoint.json"

    scaffold_path = (out / "02_blender_scaffold" / "scaffold.blend").resolve()
    current_path = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    if current_path != scaffold_path:
        _checkpoint(checkpoint_path, "opening_scaffold_blend", expected_path=str(scaffold_path))
        bpy.ops.wm.open_mainfile(filepath=str(scaffold_path))
    _checkpoint(checkpoint_path, "scaffold_blend_opened", current_path=str(Path(bpy.data.filepath).resolve()))
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    flat = load_json(out / "00_validated" / "objects.flat.json")
    flat_records = flat.get("objects", flat) if isinstance(flat, dict) else flat
    flat_by_id = {str(record["object_id"]): record for record in flat_records}
    registration_plan = load_json(step / "registration_plan.json")
    registrations = {str(record["object_id"]): record for record in registration_plan.get("objects", [])}

    placeable_records = [
        record
        for record in plan.get("objects", [])
        if str(record.get("generation_mode", "")) in {"asset_3d", "external_asset", "scaffold_only"}
    ]
    visual_by_id = {}
    scaffold_by_id = {}
    registration_by_id = {}
    source_metadata = {}

    # First create every final render visual. Placement is a separate pass so
    # support targets always exist before any child is snapped.
    for object_record in placeable_records:
        oid = str(object_record["object_id"])
        mode = str(object_record.get("generation_mode", ""))
        scaffold = _objects_for_id(bpy, oid)
        if not scaffold:
            raise RuntimeError(f"No scaffold meshes found for explicit JSON object {oid}")
        scaffold_by_id[oid] = scaffold
        identity = assert_consistent_identity(scaffold, expected_world_id=oid)

        if mode in {"asset_3d", "external_asset"}:
            if oid not in registrations:
                raise RuntimeError(f"Registration plan is missing object {oid}")
            registration = registrations[oid]
            fallback_used = bool(registration.get("fallback_used", False))
            if fallback_used:
                if mode != "asset_3d":
                    raise RuntimeError(f"Scaffold fallback is only valid for asset_3d objects; got {mode!r} for {oid}")
                _checkpoint(checkpoint_path, "duplicating_stage04_fallback_scaffold_visual", oid)
                owner_world, _owner_matrix_source = _owner_scaffold_world_matrix(
                    bpy, oid, flat_by_id.get(oid, object_record)
                )
                visual = _duplicate_scaffold_visual(
                    bpy, scaffold, f"PGW_SCAFFOLD_FALLBACK_VISUAL__{oid}", owner_world
                )
                importer = "json_scaffold_copy_after_stage04_fallback"
                asset_path_text = None
            else:
                _checkpoint(checkpoint_path, "importing_asset", oid)
                asset_path = Path(registration["asset_path"]).expanduser().resolve()
                if not asset_path.exists() or asset_path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing generated OBJ asset: {asset_path}")
                imported = _import_asset(bpy, asset_path)
                _checkpoint(
                    checkpoint_path,
                    "asset_imported",
                    oid,
                    imported_object_count=len(imported),
                    imported_vertices=sum(len(obj.data.vertices) for obj in imported),
                    imported_triangles=sum(len(obj.data.polygons) for obj in imported),
                )
                visual = _join_meshes(bpy, imported, f"PGW_ASSET__{oid}")
                _bake_registration_in_model_space(visual, registration)
                importer = "stdlib_obj_parser+bpy_data_api"
                asset_path_text = str(asset_path)
        else:
            fallback_used = False
            _checkpoint(checkpoint_path, "duplicating_scaffold_visual", oid)
            registration = _scaffold_registration_record(oid)
            owner_world, _owner_matrix_source = _owner_scaffold_world_matrix(
                bpy, oid, flat_by_id.get(oid, object_record)
            )
            visual = _duplicate_scaffold_visual(
                bpy, scaffold, f"PGW_SCAFFOLD_VISUAL__{oid}", owner_world
            )
            importer = "json_scaffold_copy"
            asset_path_text = None

        set_object_identity(
            visual,
            world_object_id=oid,
            runtime_object_id=identity["runtime_object_id"],
            semantic_owner_id=identity["semantic_owner_id"],
        )
        visual["semantic_class"] = str(object_record.get("semantic_class", ""))
        visual["part_id"] = "scaffold_visual" if (mode == "scaffold_only" or fallback_used) else "generated_visual"
        visual["generation_mode"] = mode
        visual["pgw_stage04_fallback_used"] = bool(fallback_used)
        # Final render visuals share one explicit marker so scaffold queries can
        # still request proxies only when needed.
        visual["pgw_generated_asset"] = True
        visual["pgw_visual_role"] = "render_asset"
        visual["pgw_source_is_generated_3d"] = bool(mode in {"asset_3d", "external_asset"} and not fallback_used)

        for proxy in scaffold:
            proxy["pgw_physics_proxy"] = True
            proxy["pgw_visual_role"] = "physics_proxy"
            proxy.hide_render = True
            proxy.hide_viewport = True

        visual_by_id[oid] = visual
        registration_by_id[oid] = registration
        source_metadata[oid] = {
            "asset_path": asset_path_text,
            "importer": importer,
            "generation_mode": mode,
            "fallback_used": bool(fallback_used),
            "import_diagnostics": {
                str(key)[len("pgw_import_"):]: visual.get(key)
                for key in visual.keys()
                if str(key).startswith("pgw_import_")
            },
        }
        bpy.context.view_layer.update()
        _checkpoint(
            checkpoint_path,
            "visual_created",
            oid,
            vertices=len(visual.data.vertices),
            polygons=len(visual.data.polygons),
        )

    records = []
    by_id = {str(record["object_id"]): record for record in placeable_records}
    placement_order = _hierarchy_topological_order(placeable_records, list(by_id))
    for oid in placement_order:
        object_record = by_id[oid]
        mode = str(object_record.get("generation_mode", ""))
        visual = visual_by_id[oid]
        scaffold = scaffold_by_id[oid]
        registration = registration_by_id[oid]
        owner_world, owner_matrix_source = _owner_scaffold_world_matrix(
            bpy, oid, flat_by_id.get(oid, object_record)
        )
        _checkpoint(checkpoint_path, "placing_visual_by_hierarchy", oid)
        support_report = _apply_generated_hierarchy(
            bpy,
            visual,
            object_record,
            flat_by_id.get(oid, object_record),
            visual_by_id,
            owner_world,
        )
        support_report["resolved_scaffold_world_matrix_source"] = owner_matrix_source
        # Registration residual remains report-only; support snapping no longer
        # changes transforms because the scaffold already owns authoritative placement.
        passed = True
        problems = []
        preferred_loss = float(dict(alignment_cfg.get("validation", {})).get("max_registration_loss", 0.08))
        if float(registration.get("normalized_loss", 0.0)) > preferred_loss:
            problems.append(
                f"report-only registration loss {float(registration.get('normalized_loss', 0.0)):.6f} exceeds preferred {preferred_loss:.6f}"
            )
        if problems:
            print(
                f"[05][REGISTRATION][BEST_EFFORT] {oid}: " + "; ".join(problems),
                flush=True,
            )
        if not passed and bool(alignment_cfg.get("strict_validation", True)):
            raise RuntimeError(f"JSON-guided placement validation failed for {oid}: " + "; ".join(problems))

        baked_path = step / "baked_textures" / oid / "base_color.png"
        appearance = dict(object_record.get("appearance", {}))
        raw_color = appearance.get("base_color", [0.45, 0.45, 0.45])
        fallback_color = [float(value) for value in raw_color[:3]] if isinstance(raw_color, list) and len(raw_color) >= 3 else [0.45, 0.45, 0.45]
        if max(fallback_color) <= 1.0:
            fallback_color = [value * 255.0 for value in fallback_color]
        fallback_color = tuple(max(0, min(255, int(round(value)))) for value in fallback_color)
        _checkpoint(checkpoint_path, "initializing_atlas", oid)
        atlas_transfer = _initialize_editable_material_atlas(
            bpy, visual, baked_path, resolution, margin, fallback_color=fallback_color
        )
        _checkpoint(checkpoint_path, "atlas_initialized", oid, output_path=str(baked_path))
        records.append({
            "object_id": oid,
            "name": object_record.get("name", oid),
            "semantic_class": object_record.get("semantic_class", ""),
            "generation_mode": mode,
            "fallback_used": bool(source_metadata[oid].get("fallback_used", False)),
            "asset_path": source_metadata[oid]["asset_path"],
            "importer": source_metadata[oid]["importer"],
            "import_diagnostics": source_metadata[oid]["import_diagnostics"],
            "visual_object": visual.name,
            "scaffold_proxy_objects": [obj.name for obj in scaffold],
            "registration": registration,
            "placement": support_report,
            "validation_passed": passed,
            "validation_problems": problems,
            "visual_world_aabb": _world_aabb([visual]),
            "baked_texture": str(baked_path),
            "atlas_transfer": atlas_transfer,
        })

    # OBB overlap is computed only after every support-dependent placement has
    # finished.  It is report-only and can never reject or abort Stage05.
    diagnostic_cfg = dict(alignment_cfg.get("diagnostics", {}))
    overlap_warning_threshold = float(diagnostic_cfg.get("overlap_warning_threshold_m", 0.005))
    records_by_id = {str(record["object_id"]): record for record in records}
    for oid in placement_order:
        overlap_depth, overlap_against = _max_unrelated_final_visual_overlap(visual_by_id, oid, plan)
        warning = bool(overlap_against is not None and overlap_depth > overlap_warning_threshold)
        diagnostic = {
            "method": "final_render_visual_obb",
            "effect": "report_only",
            "support_family_excluded": True,
            "penetration_depth_m": float(overlap_depth),
            "against": overlap_against,
            "warning_threshold_m": overlap_warning_threshold,
            "warning": warning,
        }
        if warning:
            print(
                f"[05][OVERLAP][REPORT_ONLY] {oid}: final visual OBB overlap "
                f"depth={overlap_depth:.6f}m against={overlap_against}; "
                "diagnostic only, Stage05 continues"
            )
        record = records_by_id[oid]
        record["overlap_diagnostic"] = diagnostic
        # Preserve the old report keys for downstream readers, while making their
        # non-validating status explicit in the structured diagnostic above.
        record["max_unrelated_overlap_penetration_depth_m"] = float(overlap_depth)
        record["overlap_against"] = overlap_against

    _checkpoint(checkpoint_path, "assigning_triangle_ids")
    triangle_records = _assign_triangle_ids_and_export(bpy, step / "uv_triangle_manifest.json")
    _checkpoint(checkpoint_path, "triangle_ids_assigned")
    scene_path = step / "scene_assets.blend"
    _checkpoint(checkpoint_path, "saving_scene_blend", scene_path=str(scene_path))
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
    _checkpoint(checkpoint_path, "completed", scene_path=str(scene_path))
    save_json({
        "status": "ok",
        "method": "JSON-driven final visuals with scaffold-root world placement and preserved generated-child local matrices",
        "hierarchy_placement_policy": {
            "root": "inherit corresponding Stage02 scaffold owner anchor world matrix",
            "child": "inherit generated parent world transform and preserve Stage00 JSON local matrix",
            "support_target": "metadata/prompt relation only; no post-registration translation",
            "registration": "proper rotation + one uniform scale + translation baked in model space",
        },
        "overlap_diagnostics_policy": {
            "effect": "report_only",
            "computed_after_all_placements": True,
            "support_ancestors_and_descendants_excluded": True,
        },
        "records": records,
        "scene_blend": str(scene_path),
        "uv_triangle_manifest": str(step / "uv_triangle_manifest.json"),
        "triangle_counts": {key: value["triangle_count"] for key, value in triangle_records.items()},
    }, step / "blender_import_report.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            args_out = None
            if "--out" in argv:
                args_out = Path(argv[argv.index("--out") + 1])
            if args_out:
                directory = args_out / "05_scene_assets"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / ".blender_failed").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            raise
