from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from src.binding.binding_record import BindingRecord
from src.binding.binding_types import BindingType
from src.physics_ir.enums import DynamicMode, PhysicsKind
from src.physics_ir.physics_object import PhysicsObject
from src.render_ir.primitive_meshes import primitive_mesh
from src.render_ir.render_object import RenderMeshView
from src.scene_ir.json_scene import flat_objects, scene_id, scene_prompt
from src.scene_ir.transforms import euler_xyz_matrix_deg, matrix_from_transform
from src.scene_ir.world import World


def _appearance(record: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(record.get("appearance", {}))
    color = value.get("base_color", [0.7, 0.7, 0.7, 1.0])
    if len(color) == 3:
        color = list(color) + [1.0]
    value["base_color"] = [float(v) for v in color]
    return value


def _part_geometry(part: Mapping[str, Any]):
    transform = dict(part.get("transform", {}))
    scale = [float(v) for v in transform.get("scale", [1.0, 1.0, 1.0])]
    positions, normals, tangents, uvs, indices = primitive_mesh(str(part["primitive"]), scale)
    rotation = euler_xyz_matrix_deg(transform.get("rotation_deg", [0.0, 0.0, 0.0]))
    translation = np.asarray(transform.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    positions = (positions @ rotation.T.astype(np.float32)) + translation[None, :]
    normals = normals @ rotation.T.astype(np.float32)
    tangents_xyz = tangents[:, :3] @ rotation.T.astype(np.float32)
    tangents = np.concatenate([tangents_xyz, tangents[:, 3:4]], axis=1)
    return positions, normals, tangents, uvs, indices


def _add_render_object(world: World, object_id: int, record: Mapping[str, Any]) -> int:
    appearance = _appearance(record)
    generation = dict(record["generation"])
    material_id = world.materials.add(
        name=f"mat_{record['object_id']}",
        albedo_texture=appearance.get("texture_path"),
        base_color=appearance["base_color"],
        prompt=str(generation.get("prompt", "")),
    )
    world_matrix = np.asarray(record["world_transform"]["matrix"], dtype=np.float32)
    render_id = world.render_world.add_object(
        object_id=object_id,
        material_id=material_id,
        local_to_world=world_matrix,
    )
    for part in record.get("scaffold", {}).get("parts", []):
        positions, normals, tangents, uvs, indices = _part_geometry(part)
        vo, vc, io, ic = world.render_world.buffers.append_mesh(
            positions, normals, tangents, uvs, indices
        )
        world.render_world.add_part(
            render_id,
            str(part["id"]),
            str(part["primitive"]),
            RenderMeshView(vo, vc, io, ic),
        )
    return render_id


def _dynamic_mode(mode: str) -> DynamicMode:
    if mode == "dynamic":
        return DynamicMode.DYNAMIC
    if mode == "kinematic":
        return DynamicMode.KINEMATIC
    return DynamicMode.STATIC


def _add_rigid_physics(world: World, object_id: int, record: Mapping[str, Any]):
    physics = dict(record["physics"])
    mode = _dynamic_mode(str(physics["mode"]))
    parameters = dict(physics.get("parameters", {}))
    mass_default = 0.0 if mode == DynamicMode.STATIC else 1.0
    rb_idx = world.physics_world.rigid.add(
        np.asarray(record["world_transform"]["matrix"], dtype=np.float32),
        mass=float(parameters.get("mass", mass_default)),
        friction=float(parameters.get("friction", 0.5)),
        restitution=float(parameters.get("restitution", 0.1)),
    )

    collider_ids: list[int] = []
    collider_cfg = dict(physics.get("colliders", {}))
    if collider_cfg.get("mode", "from_scaffold") == "from_scaffold":
        collider_parts = list(record.get("scaffold", {}).get("parts", []))
    else:
        collider_parts = list(collider_cfg.get("parts", []))
    for part in collider_parts:
        transform = dict(part.get("transform", {}))
        scale = [float(v) for v in transform.get("scale", [1.0, 1.0, 1.0])]
        local_transform = (
            [float(v) for v in transform.get("position", [0.0, 0.0, 0.0])]
            + [float(v) for v in transform.get("rotation_deg", [0.0, 0.0, 0.0])]
            + [1.0, 1.0, 1.0]
        )
        primitive = str(part["primitive"])
        part_id = str(part["id"])
        colliders = world.physics_world.colliders
        if primitive == "box":
            collider_id = colliders.add_box(
                object_id, part_id, [0.5 * value for value in scale], local_transform=local_transform
            )
        elif primitive == "sphere":
            collider_id = colliders.add_sphere(
                object_id, part_id, 0.5 * max(scale), local_transform=local_transform
            )
        elif primitive == "cylinder":
            collider_id = colliders.add_cylinder(
                object_id, part_id, 0.5 * max(scale[0], scale[1]), 0.5 * scale[2], local_transform=local_transform
            )
        elif primitive == "capsule":
            radius = 0.5 * max(scale[0], scale[1])
            half_segment = max(0.0, 0.5 * scale[2] - radius)
            collider_id = colliders.add_capsule(
                object_id, part_id, radius, half_segment, local_transform=local_transform
            )
        elif primitive == "cone":
            collider_id = colliders.add_cone(
                object_id, part_id, 0.5 * max(scale[0], scale[1]), 0.5 * scale[2], local_transform=local_transform
            )
        else:
            raise ValueError(f"Unsupported collider primitive: {primitive!r}")
        collider_ids.append(collider_id)

    physics_id = len(world.physics_world.objects)
    kind = PhysicsKind.STATIC if mode == DynamicMode.STATIC else PhysicsKind.RIGID
    world.physics_world.add_object(
        PhysicsObject(
            physics_id=physics_id,
            object_id=object_id,
            kind=kind,
            dynamic_mode=mode,
            collider_ids=collider_ids,
            rigid_body_index=rb_idx,
        )
    )
    return physics_id, rb_idx, mode


def _add_elastic_physics(world: World, object_id: int, record: Mapping[str, Any], render_id: int):
    physics = dict(record["physics"])
    parameters = dict(physics.get("parameters", {}))
    topology = dict(physics.get("topology", {}))
    if topology.get("generator") != "grid":
        raise NotImplementedError(
            f"Object {record['object_id']}: this runtime currently compiles grid topology; "
            "explicit particle/spring topology remains preserved in normalized JSON for the backend adapter."
        )
    nx, ny = [int(v) for v in topology["resolution"]]
    size = [float(v) for v in topology["size"]]
    part = world.render_world.objects[render_id].parts[0]
    deformable_id = world.physics_world.deformable.add_grid_cloth(
        object_id=object_id,
        nx=nx,
        ny=ny,
        size=size,
        mass=float(parameters.get("mass_per_particle", 0.03)),
        k=float(parameters.get("stiffness", 0.7)),
        d=float(parameters.get("damping", 0.15)),
        render_vertex_offset=part.mesh_view.vertex_offset,
        render_vertex_count=part.mesh_view.vertex_count,
    )
    view = world.physics_world.deformable.views[deformable_id]
    # Optional explicit fixed-point indices override the default generated row.
    if isinstance(topology.get("fixed_points"), list):
        start = view.particle_offset
        end = start + view.particle_count
        world.physics_world.deformable.fixed[start:end] = False
        for index in topology["fixed_points"]:
            local_index = int(index)
            if 0 <= local_index < view.particle_count:
                world.physics_world.deformable.fixed[start + local_index] = True
    physics_id = len(world.physics_world.objects)
    world.physics_world.add_object(
        PhysicsObject(
            physics_id=physics_id,
            object_id=object_id,
            kind=PhysicsKind.DEFORMABLE,
            dynamic_mode=DynamicMode.DYNAMIC,
            deformable_id=deformable_id,
        )
    )
    return physics_id, view


def build_world_from_scene(scene_dict: Mapping[str, Any], registry=None) -> World:
    """Compile the explicit JSON scene graph into render, physics, and binding IR.

    Runtime decisions are based only on explicit generic fields. The semantic label
    is copied into the registry and never interpreted by this function.
    """
    world = World(scene_id=scene_id(scene_dict), prompt=scene_prompt(scene_dict))
    records = flat_objects(scene_dict, include_groups=True)
    registry_ids: Dict[str, int] = {}

    # Register the hierarchy first so parent links are stable even for group nodes.
    for record in records:
        parent_name = record.get("parent_id")
        parent_id = registry_ids.get(str(parent_name)) if parent_name else None
        oid = world.objects.add(
            record["object_id"],
            record.get("semantic_class", ""),
            display_name=record.get("name", record["object_id"]),
            parent_id=parent_id,
            generation_mode=str(record["generation"]["mode"]),
        )
        registry_ids[record["object_id"]] = oid

    for record in records:
        generation_mode = str(record["generation"]["mode"])
        if generation_mode == "group":
            continue
        object_id = registry_ids[record["object_id"]]
        render_id = _add_render_object(world, object_id, record)
        physics = dict(record["physics"])
        body = str(physics["body"])
        mode = str(physics["mode"])

        if mode == "visual_only" or body == "none":
            physics_id = None
            binding = BindingRecord(
                binding_id=len(world.binding_world.records),
                object_id=object_id,
                render_id=render_id,
                physics_id=None,
                binding_type=BindingType.VISUAL_ONLY,
            )
        elif body == "rigid":
            physics_id, rb_idx, dynamic_mode = _add_rigid_physics(world, object_id, record)
            binding = BindingRecord(
                binding_id=len(world.binding_world.records),
                object_id=object_id,
                render_id=render_id,
                physics_id=physics_id,
                binding_type=(
                    BindingType.STATIC
                    if dynamic_mode == DynamicMode.STATIC
                    else BindingType.RIGID_TRANSFORM
                ),
                rigid_body_index=rb_idx,
            )
        elif body == "elastic":
            physics_id, view = _add_elastic_physics(world, object_id, record, render_id)
            binding = BindingRecord(
                binding_id=len(world.binding_world.records),
                object_id=object_id,
                render_id=render_id,
                physics_id=physics_id,
                binding_type=BindingType.DEFORMABLE_VERTEX,
                render_vertex_offset=view.render_vertex_offset,
                render_vertex_count=view.render_vertex_count,
                physics_particle_offset=view.particle_offset,
                physics_particle_count=view.particle_count,
            )
        elif body == "fluid":
            physics_id = None
            binding = BindingRecord(
                binding_id=len(world.binding_world.records),
                object_id=object_id,
                render_id=render_id,
                physics_id=None,
                binding_type=BindingType.FLUID_PARTICLE,
            )
        else:
            raise ValueError(f"Unsupported physics.body: {body!r}")

        binding_id = world.binding_world.add(binding)
        rec = world.objects.get(object_id)
        rec.render_id = render_id
        rec.physics_id = physics_id
        rec.binding_id = binding_id

    return world
