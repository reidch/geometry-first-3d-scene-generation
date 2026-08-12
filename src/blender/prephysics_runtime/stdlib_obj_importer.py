from __future__ import annotations

"""Dependency-free, defensive Wavefront OBJ/MTL importer for Blender.

The importer intentionally avoids Blender's OBJ add-on and avoids vertex expansion by
normal/UV corner tuples. Blender stores UVs per loop, so the original OBJ position
indices can be preserved directly. This keeps generated Pixal3D meshes much smaller
inside Blender and removes a major source of headless Blender SIGSEGVs.
"""

from array import array
from dataclasses import dataclass, field
import math
from pathlib import Path
import shlex
import struct
from typing import Dict, List, Optional, Tuple


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Corner = Tuple[int, Optional[int], Optional[int]]


@dataclass
class MaterialSpec:
    name: str
    diffuse_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    diffuse_texture: Optional[Path] = None


@dataclass
class ParsedObj:
    source_path: Path
    vertices: List[Vec3] = field(default_factory=list)
    texcoords: List[Vec2] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    faces: List[List[Corner]] = field(default_factory=list)
    face_materials: List[Optional[str]] = field(default_factory=list)
    mtllibs: List[Path] = field(default_factory=list)


class ObjParseError(RuntimeError):
    pass


def _parse_float(raw: str, kind: str, line_number: int) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ObjParseError(f"Invalid {kind} value {raw!r} on OBJ line {line_number}") from exc


def _resolve_obj_index(raw: str, count: int, kind: str, line_number: int) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ObjParseError(f"Invalid {kind} index {raw!r} on OBJ line {line_number}") from exc
    if value == 0:
        raise ObjParseError(f"OBJ {kind} index 0 is invalid on line {line_number}")
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ObjParseError(
            f"OBJ {kind} index {value} is outside 1..{count} on line {line_number}"
        )
    return index


def _parse_corner(token: str, parsed: ParsedObj, line_number: int) -> Corner:
    parts = token.split("/")
    if not parts or not parts[0]:
        raise ObjParseError(f"Face corner lacks a vertex index on line {line_number}")
    vertex = _resolve_obj_index(parts[0], len(parsed.vertices), "vertex", line_number)
    texcoord = None
    normal = None
    if len(parts) >= 2 and parts[1]:
        texcoord = _resolve_obj_index(parts[1], len(parsed.texcoords), "texture", line_number)
    if len(parts) >= 3 and parts[2]:
        normal = _resolve_obj_index(parts[2], len(parsed.normals), "normal", line_number)
    if len(parts) > 3:
        raise ObjParseError(f"Invalid face corner {token!r} on line {line_number}")
    return vertex, texcoord, normal


def parse_obj(path: Path | str) -> ParsedObj:
    source = Path(path).expanduser().resolve()
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"OBJ file is missing or empty: {source}")

    parsed = ParsedObj(source_path=source)
    current_material: Optional[str] = None
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            keyword = fields[0]
            values = fields[1:]
            if keyword == "v":
                if len(values) < 3:
                    raise ObjParseError(f"Vertex needs three coordinates on line {line_number}")
                parsed.vertices.append(tuple(_parse_float(x, "vertex", line_number) for x in values[:3]))
            elif keyword == "vt":
                if len(values) < 2:
                    raise ObjParseError(f"Texture coordinate needs two values on line {line_number}")
                parsed.texcoords.append(
                    (_parse_float(values[0], "texture", line_number), _parse_float(values[1], "texture", line_number))
                )
            elif keyword == "vn":
                if len(values) < 3:
                    raise ObjParseError(f"Normal needs three coordinates on line {line_number}")
                parsed.normals.append(tuple(_parse_float(x, "normal", line_number) for x in values[:3]))
            elif keyword == "f":
                if len(values) < 3:
                    raise ObjParseError(f"Face needs at least three corners on line {line_number}")
                parsed.faces.append([_parse_corner(token, parsed, line_number) for token in values])
                parsed.face_materials.append(current_material)
            elif keyword == "usemtl":
                current_material = " ".join(values).strip() or None
            elif keyword == "mtllib":
                try:
                    names = shlex.split(stripped[len("mtllib") :].strip())
                except ValueError:
                    names = values
                for name in names:
                    candidate = (source.parent / name).resolve()
                    if candidate not in parsed.mtllibs:
                        parsed.mtllibs.append(candidate)

    if not parsed.vertices:
        raise ObjParseError(f"OBJ contains no vertices: {source}")
    if not parsed.faces:
        raise ObjParseError(f"OBJ contains no faces: {source}")
    return parsed


def _parse_map_path(raw_value: str) -> Optional[str]:
    try:
        tokens = shlex.split(raw_value)
    except ValueError:
        tokens = raw_value.split()
    return tokens[-1] if tokens else None


def parse_mtl(path: Path | str) -> Dict[str, MaterialSpec]:
    source = Path(path).expanduser().resolve()
    if not source.exists() or source.stat().st_size == 0:
        return {}
    materials: Dict[str, MaterialSpec] = {}
    current: Optional[MaterialSpec] = None
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            keyword = fields[0]
            values = fields[1:]
            if keyword == "newmtl":
                name = " ".join(values).strip() or "material"
                current = MaterialSpec(name=name)
                materials[name] = current
            elif current is None:
                continue
            elif keyword == "Kd" and len(values) >= 3:
                alpha = current.diffuse_color[3]
                current.diffuse_color = (float(values[0]), float(values[1]), float(values[2]), alpha)
            elif keyword == "d" and values:
                alpha = max(0.0, min(1.0, float(values[0])))
                current.diffuse_color = (*current.diffuse_color[:3], alpha)
            elif keyword == "Tr" and values:
                alpha = 1.0 - max(0.0, min(1.0, float(values[0])))
                current.diffuse_color = (*current.diffuse_color[:3], alpha)
            elif keyword.lower() == "map_kd":
                texture_name = _parse_map_path(stripped[len(fields[0]) :].strip())
                if texture_name:
                    current.diffuse_texture = (source.parent / texture_name).resolve()
    return materials


def load_material_specs(parsed: ParsedObj) -> Dict[str, MaterialSpec]:
    materials: Dict[str, MaterialSpec] = {}
    for library in parsed.mtllibs:
        materials.update(parse_mtl(library))
    return materials


def _image_header_is_safe(path: Path) -> bool:
    """Cheap decoder guard before passing generated files into Blender's C image API."""
    try:
        size = path.stat().st_size
        if size <= 0 or size > 512 * 1024 * 1024:
            return False
        with path.open("rb") as handle:
            header = handle.read(32)
        suffix = path.suffix.lower()
        if suffix == ".png":
            if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                return False
            width, height = struct.unpack(">II", header[16:24])
            return 0 < width <= 32768 and 0 < height <= 32768
        if suffix in {".jpg", ".jpeg"}:
            return header[:2] == b"\xff\xd8"
        if suffix == ".webp":
            return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
        if suffix == ".bmp":
            return header[:2] == b"BM"
        if suffix == ".tga":
            return len(header) >= 18
        return False
    except OSError:
        return False


def _create_blender_material(bpy, spec: MaterialSpec):
    material = bpy.data.materials.new(name=f"PGW_OBJ_MAT__{spec.name}")
    material.use_nodes = True
    material.diffuse_color = spec.diffuse_color
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = spec.diffuse_color
    bsdf.inputs["Roughness"].default_value = 0.65
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if spec.diffuse_texture is not None and _image_header_is_safe(spec.diffuse_texture):
        image = bpy.data.images.load(str(spec.diffuse_texture), check_existing=True)
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "EXTEND"
        links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        if "Alpha" in texture.outputs and "Alpha" in bsdf.inputs:
            links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    return material


def _is_finite_vector(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _cross_sq(a: Vec3, b: Vec3, c: Vec3) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return cx * cx + cy * cy + cz * cz


def sanitize_obj_triangles(parsed: ParsedObj):
    finite_vertices = [vertex for vertex in parsed.vertices if _is_finite_vector(vertex)]
    if not finite_vertices:
        raise ObjParseError(f"OBJ contains no finite vertices: {parsed.source_path}")
    minimum = [min(vertex[i] for vertex in finite_vertices) for i in range(3)]
    maximum = [max(vertex[i] for vertex in finite_vertices) for i in range(3)]
    diagonal_sq = sum((maximum[i] - minimum[i]) ** 2 for i in range(3))
    area_sq_epsilon = max(1e-30, diagonal_sq * diagonal_sq * 1e-24)

    triangles: List[Tuple[int, int, int]] = []
    triangle_uvs: List[Tuple[Optional[Vec2], Optional[Vec2], Optional[Vec2]]] = []
    triangle_materials: List[Optional[str]] = []
    dropped_nonfinite = 0
    dropped_degenerate = 0
    triangulated_ngons = 0

    for face, material_name in zip(parsed.faces, parsed.face_materials):
        cleaned: List[Corner] = []
        for corner in face:
            if cleaned and cleaned[-1][0] == corner[0]:
                continue
            cleaned.append(corner)
        if len(cleaned) >= 2 and cleaned[0][0] == cleaned[-1][0]:
            cleaned.pop()
        if len(cleaned) < 3:
            dropped_degenerate += 1
            continue
        if len(cleaned) > 3:
            triangulated_ngons += 1
        for fan_index in range(1, len(cleaned) - 1):
            corners = (cleaned[0], cleaned[fan_index], cleaned[fan_index + 1])
            indices = (corners[0][0], corners[1][0], corners[2][0])
            if len(set(indices)) != 3:
                dropped_degenerate += 1
                continue
            coords = tuple(parsed.vertices[index] for index in indices)
            if not all(_is_finite_vector(coord) for coord in coords):
                dropped_nonfinite += 1
                continue
            if _cross_sq(coords[0], coords[1], coords[2]) <= area_sq_epsilon:
                dropped_degenerate += 1
                continue
            uvs = []
            for corner in corners:
                uv_index = corner[1]
                uv = parsed.texcoords[uv_index] if uv_index is not None else None
                uvs.append(uv if uv is not None and _is_finite_vector(uv) else None)
            triangles.append(indices)
            triangle_uvs.append((uvs[0], uvs[1], uvs[2]))
            triangle_materials.append(material_name)

    if not triangles:
        raise ObjParseError(f"OBJ contains no finite positive-area triangles: {parsed.source_path}")

    used_vertex_indices = sorted({index for triangle in triangles for index in triangle})
    vertex_remap = {old_index: new_index for new_index, old_index in enumerate(used_vertex_indices)}
    compact_vertices = [parsed.vertices[index] for index in used_vertex_indices]
    compact_triangles = [tuple(vertex_remap[index] for index in triangle) for triangle in triangles]
    diagnostics = {
        "source_vertex_count": len(parsed.vertices),
        "imported_vertex_count": len(compact_vertices),
        "unused_or_invalid_vertex_count": len(parsed.vertices) - len(compact_vertices),
        "source_face_count": len(parsed.faces),
        "triangle_count": len(compact_triangles),
        "triangulated_ngon_count": triangulated_ngons,
        "dropped_nonfinite_triangle_count": dropped_nonfinite,
        "dropped_degenerate_triangle_count": dropped_degenerate,
        "vertex_expansion_by_corner_tuple": False,
        "normal_indices_ignored_and_recomputed": True,
    }
    return compact_vertices, compact_triangles, triangle_uvs, triangle_materials, diagnostics


def _box_project_uv(vertex: Vec3, triangle_vertices: Tuple[Vec3, Vec3, Vec3], minimum, maximum) -> Vec2:
    a, b, c = triangle_vertices
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    normal = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    axis = max(range(3), key=lambda index: abs(normal[index]))
    if axis == 0:
        dims = (1, 2)
    elif axis == 1:
        dims = (0, 2)
    else:
        dims = (0, 1)
    result = []
    for dim in dims:
        span = max(maximum[dim] - minimum[dim], 1e-12)
        result.append((vertex[dim] - minimum[dim]) / span)
    return float(result[0]), float(result[1])


def load_obj_into_blender(bpy, path: Path | str, object_name: Optional[str] = None):
    """Create one triangulated Blender mesh without add-ons or corner vertex explosion."""
    parsed = parse_obj(path)
    material_specs = load_material_specs(parsed)
    vertices, triangles, triangle_uvs, triangle_materials, diagnostics = sanitize_obj_triangles(parsed)

    mesh_name = object_name or parsed.source_path.stem
    mesh = bpy.data.meshes.new(name=f"{mesh_name}__MESH")

    coordinates = array("f", (float(value) for vertex in vertices for value in vertex))
    loop_vertex_indices = array("i", (int(index) for triangle in triangles for index in triangle))
    loop_starts = array("i", range(0, len(triangles) * 3, 3))
    loop_totals = array("i", [3]) * len(triangles)

    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", coordinates)
    mesh.loops.add(len(loop_vertex_indices))
    mesh.loops.foreach_set("vertex_index", loop_vertex_indices)
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(obj)

    used_names: List[str] = []
    for name in triangle_materials:
        canonical = name or "__default__"
        if canonical not in used_names:
            used_names.append(canonical)
    if not used_names:
        used_names = ["__default__"]
    material_index: Dict[str, int] = {}
    for name in used_names:
        spec = material_specs.get(name) or MaterialSpec(name=name)
        obj.data.materials.append(_create_blender_material(bpy, spec))
        material_index[name] = len(obj.data.materials) - 1
    material_values = array("i", [material_index[name or "__default__"] for name in triangle_materials])
    mesh.polygons.foreach_set("material_index", material_values)
    mesh.polygons.foreach_set("use_smooth", [True] * len(triangles))

    minimum = [min(vertex[i] for vertex in vertices) for i in range(3)]
    maximum = [max(vertex[i] for vertex in vertices) for i in range(3)]
    flat_uvs = array("f")
    source_uv_count = 0
    for triangle, uvs in zip(triangles, triangle_uvs):
        tri_vertices = tuple(vertices[index] for index in triangle)
        for vertex_index, uv in zip(triangle, uvs):
            if uv is None:
                uv = _box_project_uv(vertices[vertex_index], tri_vertices, minimum, maximum)
            else:
                source_uv_count += 1
            flat_uvs.extend((float(uv[0]), float(uv[1])))
    uv_layer = mesh.uv_layers.new(name="PGW_SOURCE_UV")
    uv_layer.data.foreach_set("uv", flat_uvs)
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True
    mesh.update(calc_edges=True)

    diagnostics["source_uv_loop_count"] = source_uv_count
    diagnostics["fallback_projected_uv_loop_count"] = len(loop_vertex_indices) - source_uv_count
    for key, value in diagnostics.items():
        obj[f"pgw_import_{key}"] = value
    return [obj]
