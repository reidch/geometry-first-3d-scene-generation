#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assets.eligibility import build_generation_plan
from src.core.logging_utils import setup_step_logger, status
from src.io.json_io import load_json, save_json
from src.io.world_io import save_world
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
from src.scene_ir.build_world import build_world_from_scene
from src.scene_ir.json_scene import scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    scene_path = Path(args.scene) if args.scene else out / "00_validated" / "scene.normalized.json"
    step_dir = out / "01_world_ir"
    buffer_dir = step_dir / "buffers"
    logger = setup_step_logger(Path("logs") / out.name / "01_build_world_ir.log")
    status("[01] Compiling JSON hierarchy into World IR...")
    if not scene_path.exists():
        raise FileNotFoundError(f"Normalized scene not found: {scene_path}")
    step_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)

    scene_dict = load_json(scene_path)
    world = build_world_from_scene(scene_dict)
    save_world(world, step_dir / "world.pkl")
    world.render_world.buffers.save_npz(buffer_dir / "render_buffers.npz")
    world.physics_world.rigid.save_npz(buffer_dir / "rigid_buffers.npz")
    world.physics_world.colliders.save_npz(buffer_dir / "collider_buffers.npz")
    world.physics_world.deformable.save_npz(buffer_dir / "deformable_buffers.npz")
    world.physics_world.fluid.save_npz(buffer_dir / "fluid_buffers.npz")

    save_json({"objects": world.objects.to_list()}, step_dir / "object_registry.json")
    save_json({"render_objects": world.render_world.to_list()}, step_dir / "render_manifest.json")
    save_json({
        "physics_objects": world.physics_world.to_list(),
        "colliders": world.physics_world.colliders.to_list(),
        "deformables": world.physics_world.deformable.to_list(),
        "fluids": world.physics_world.fluid.to_list(),
    }, step_dir / "physics_manifest.json")
    save_json({"bindings": world.binding_world.to_list()}, step_dir / "binding_records.json")
    save_json({"materials": world.materials.to_list()}, step_dir / "material_manifest.json")
    plan = build_generation_plan(scene_dict)
    save_json(plan, step_dir / "generation_plan.json")

    summary = {
        "scene_id": scene_id(scene_dict),
        "object_count_including_groups": len(world.objects.records),
        "render_object_count": len(world.render_world.objects),
        "material_count": len(world.materials.materials),
        "physics_object_count": len(world.physics_world.objects),
        "binding_count": len(world.binding_world.records),
        "render_vertex_count": int(world.render_world.buffers.positions.shape[0]),
        "render_index_count": int(world.render_world.buffers.indices.shape[0]),
        "rigid_body_count": int(world.physics_world.rigid.transforms.shape[0]),
        "collider_count": len(world.physics_world.colliders.records),
        "deformable_particle_count": int(world.physics_world.deformable.positions.shape[0]),
        "spring_count": int(world.physics_world.deformable.spring_pairs.shape[0]),
        "generation_counts": plan["counts"],
    }
    save_json(summary, step_dir / "world_summary.json")

    index = ArtifactIndex(scene_id=out.name, step="01_build_world_ir")
    for name, path in {
        "world": step_dir / "world.pkl",
        "summary": step_dir / "world_summary.json",
        "object_registry": step_dir / "object_registry.json",
        "render_manifest": step_dir / "render_manifest.json",
        "physics_manifest": step_dir / "physics_manifest.json",
        "binding_records": step_dir / "binding_records.json",
        "material_manifest": step_dir / "material_manifest.json",
        "generation_plan": step_dir / "generation_plan.json",
    }.items():
        index.add(name, path)
    index.save(step_dir / "artifact_index.json")
    logger.info("World summary: %s", summary)
    mark_done(step_dir)
    status("[01] Done. World IR and explicit routing plan built.")


if __name__ == "__main__":
    main()
