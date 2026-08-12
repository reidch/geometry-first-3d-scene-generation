#!/usr/bin/env python
from __future__ import annotations

import argparse
import array
import sys
import traceback
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.blender.blender_runtime import require_bpy
from src.blender.active_owner_filter import apply_active_owner_filter
from src.blender.object_identity import get_semantic_owner_id
from src.io.json_io import save_json


def _visible_meshes(bpy):
    return sorted(
        [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH"
            and not obj.hide_render
            and not bool(obj.get("pgw_physics_proxy", False))
        ],
        key=lambda obj: (str(get_semantic_owner_id(obj)), obj.name),
    )


def _write_array(path: Path, typecode: str, values) -> dict:
    payload = array.array(typecode, values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        payload.tofile(handle)
    return {
        "path": str(path),
        "byte_count": int(path.stat().st_size),
        "endianness": "little",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True, help="Output JSON catalog manifest")
    parser.add_argument("--texture_root", required=True)
    parser.add_argument("--active_owner_manifest")
    args = parser.parse_args(argv)

    bpy = require_bpy()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.scene).resolve()))
    active_owner_filter = apply_active_owner_filter(bpy, args.active_owner_manifest)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    texture_root = Path(args.texture_root)
    stem = output.with_suffix("")

    vertices = array.array("f")
    normals = array.array("f")
    uvs = array.array("f")
    owner_indices = array.array("i")
    scene_triangle_ids = array.array("q")
    owners = []
    owner_to_index = {}
    objects = []
    scene_triangle_id = 0

    for obj in _visible_meshes(bpy):
        if any(len(poly.vertices) != 3 for poly in obj.data.polygons):
            raise RuntimeError(f"Gaussian mesh export requires triangulated mesh: {obj.name}")
        owner = str(get_semantic_owner_id(obj))
        if owner not in owner_to_index:
            owner_to_index[owner] = len(owners)
            texture = texture_root / owner / "base_color.png"
            owners.append(
                {
                    "owner_id": owner,
                    "texture_path": str(texture),
                    "texture_exists": bool(texture.exists() and texture.stat().st_size > 0),
                }
            )
        owner_index = owner_to_index[owner]
        uv_layer = obj.data.uv_layers.active.data if obj.data.uv_layers.active else None
        if uv_layer is None:
            raise RuntimeError(f"Gaussian mesh export requires active UVs: {obj.name}")
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        object_start = scene_triangle_id
        for poly in obj.data.polygons:
            world_vertices = [obj.matrix_world @ obj.data.vertices[index].co for index in poly.vertices]
            world_normal = (normal_matrix @ poly.normal).normalized()
            triangle_uvs = [uv_layer[loop_index].uv for loop_index in poly.loop_indices]
            for value in world_vertices:
                vertices.extend((float(value.x), float(value.y), float(value.z)))
            normals.extend((float(world_normal.x), float(world_normal.y), float(world_normal.z)))
            for value in triangle_uvs:
                uvs.extend((float(value.x), float(value.y)))
            owner_indices.append(int(owner_index))
            scene_triangle_ids.append(int(scene_triangle_id))
            scene_triangle_id += 1
        objects.append(
            {
                "mesh_object_name": obj.name,
                "owner_id": owner,
                "scene_triangle_start": object_start,
                "scene_triangle_end_exclusive": scene_triangle_id,
            }
        )

    paths = {
        "triangle_vertices_world": stem.with_name(stem.name + ".vertices.f32"),
        "triangle_normals_world": stem.with_name(stem.name + ".normals.f32"),
        "triangle_uvs": stem.with_name(stem.name + ".uvs.f32"),
        "owner_indices": stem.with_name(stem.name + ".owner_indices.i32"),
        "scene_triangle_ids": stem.with_name(stem.name + ".scene_triangle_ids.i64"),
    }
    arrays = {
        "triangle_vertices_world": {
            **_write_array(paths["triangle_vertices_world"], "f", vertices),
            "dtype": "<f4",
            "shape": [scene_triangle_id, 3, 3],
        },
        "triangle_normals_world": {
            **_write_array(paths["triangle_normals_world"], "f", normals),
            "dtype": "<f4",
            "shape": [scene_triangle_id, 3],
        },
        "triangle_uvs": {
            **_write_array(paths["triangle_uvs"], "f", uvs),
            "dtype": "<f4",
            "shape": [scene_triangle_id, 3, 2],
        },
        "owner_indices": {
            **_write_array(paths["owner_indices"], "i", owner_indices),
            "dtype": "<i4",
            "shape": [scene_triangle_id],
        },
        "scene_triangle_ids": {
            **_write_array(paths["scene_triangle_ids"], "q", scene_triangle_ids),
            "dtype": "<i8",
            "shape": [scene_triangle_id],
        },
    }
    report = {
        "schema_version": 2,
        "scene": str(Path(args.scene)),
        "catalog": str(output),
        "storage": "stdlib_raw_binary_arrays",
        "numpy_required_inside_blender": False,
        "triangle_count": int(scene_triangle_id),
        "owner_count": len(owners),
        "owners": owners,
        "objects": objects,
        "arrays": arrays,
        "triangle_id_contract": "same sorted renderable mesh/object polygon order as Stage07 shared buffers",
        "active_owner_filter": active_owner_filter,
    }
    save_json(report, output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise
