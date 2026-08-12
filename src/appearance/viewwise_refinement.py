from __future__ import annotations

from pathlib import Path
from typing import Dict
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageFilter

from src.appearance.atlas_fusion import fuse_view
from src.appearance.atlas_maps import decode_objects, load_uv, valid_uv_mask
from src.appearance.atlas_state import load_registry
from src.appearance.backend_factory import create_backend
from src.appearance.model_cache import ensure_backend_models
from src.appearance.pattern_guard import periodicity_score
from src.appearance.prompt_builder import build_negative_prompt
from src.coverage.triangle_coverage import update_triangle_seen, weighted_seen_ratio
from src.io.json_io import load_json, save_json


def _run(cmd):
    subprocess.run(cmd, check=True)


def render_current(
    blender: str,
    out: Path,
    camera_file: Path,
    camera_id: str,
    output: Path,
    *,
    width: int,
    height: int,
):
    _run([
        blender,
        "--background",
        "--python",
        "src/blender/prephysics_runtime/render_textured_view.py",
        "--",
        "--out",
        str(out),
        "--camera_file",
        str(camera_file),
        "--camera_id",
        camera_id,
        "--output",
        str(output),
        "--texture_root",
        str(out / "05_texture_state"),
        "--render_mode",
        "albedo",
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--binding_report",
        str(output.with_suffix(".material_binding.json")),
    ])


def _shared_buffers(evaluation: Dict) -> Dict:
    buffers = dict(evaluation.get("shared_buffers", {}))
    transport = str(buffers.get("transport", ""))
    if transport and transport != "standard_images_and_json":
        raise RuntimeError(
            f"Unsupported Step08 shared-buffer transport: {transport}. "
            "Re-run Stage07 with v44."
        )
    required = ["depth", "semantic", "palette", "uv", "triangle_id", "owner_manifests"]
    missing_keys = [key for key in required if not buffers.get(key)]
    if missing_keys:
        raise RuntimeError(
            f"Stage07 evaluation lacks shared Step08 buffers: {missing_keys}"
        )
    depth_path = Path(str(buffers.get("depth", "")))
    uv_path = Path(str(buffers.get("uv", "")))
    if depth_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise RuntimeError(
            f"Step08 depth must be a standard image, got: {depth_path}. Re-run Stage07 with v44."
        )
    if uv_path.suffix.lower() != ".json":
        raise RuntimeError(
            f"Step08 UV must be a JSON manifest referencing PNG channels, got: {uv_path}. "
            "Re-run Stage07 with v44."
        )
    missing_files = []
    for key in ["depth", "semantic", "palette", "uv", "triangle_id"]:
        path = Path(buffers[key])
        if not path.exists() or path.stat().st_size == 0:
            missing_files.append(str(path))
    if missing_files:
        raise RuntimeError(f"Stage07 shared Step08 buffers are missing: {missing_files}")
    resolution = list(buffers.get("resolution", [768, 432]))
    if len(resolution) != 2:
        raise ValueError(f"Invalid shared-buffer resolution: {resolution}")
    buffers["resolution"] = [int(resolution[0]), int(resolution[1])]
    return buffers


def _bool_image(mask):
    return Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), "L")


def build_masks(semantic_path: Path, uv_path: Path, palette_path: Path, object_id: str, directory: Path, dilation_px: int):
    directory.mkdir(parents=True, exist_ok=True)
    width, height, u, v = load_uv(uv_path)
    decoded, names = decode_objects(semantic_path, palette_path)
    if object_id not in names:
        return None
    semantic = decoded == names.index(object_id)
    writable = semantic & valid_uv_mask(u, v)
    if int(writable.sum()) == 0:
        return None
    exact = _bool_image(semantic)
    generation = exact.filter(ImageFilter.MaxFilter(max(1, 2 * int(dilation_px) + 1))) if dilation_px > 0 else exact
    write = _bool_image(writable)
    exact_path = directory / "exact_semantic.png"
    generation_path = directory / "generation_mask.png"
    write_path = directory / "write_mask.png"
    exact.save(exact_path)
    generation.save(generation_path)
    write.save(write_path)
    return {
        "exact": exact_path,
        "generation": generation_path,
        "write": write_path,
        "semantic_pixels": int(semantic.sum()),
        "write_pixels": int(writable.sum()),
        "semantic_array": semantic,
    }


def composite_locked(init_path: Path, generated_path: Path, exact_mask: Path, output: Path):
    init = Image.open(init_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB").resize(init.size, Image.Resampling.LANCZOS)
    mask = Image.open(exact_mask).convert("L").resize(init.size, Image.Resampling.NEAREST)
    Image.composite(generated, init, mask).save(output)


def _clean_prompt(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def refinement_prompt(scene_prompt: str, object_record: Dict, prompts_cfg: Dict | None = None) -> str:
    """Build an explicit, visible style-refinement prompt without semantic-name routing."""
    prompts_cfg = prompts_cfg or {}
    generation = dict(object_record.get("generation", {}))
    appearance = dict(object_record.get("appearance", {}))
    generation_mode = str(object_record.get("generation_mode") or generation.get("mode") or "")
    intended = _clean_prompt(generation.get("prompt") or appearance.get("prompt") or "preserve the current appearance")
    style_prompt = _clean_prompt(generation.get("style_prompt"))
    appearance_prompt = _clean_prompt(
        generation.get("appearance_prompt")
        or generation.get("material_prompt")
        or appearance.get("prompt")
    )
    suffix = _clean_prompt(
        prompts_cfg.get(
            "global_suffix",
            "preserve geometry, object identity, semantic boundary, material identity, and cross-view consistency",
        )
    )
    clauses = [
        f"Explicit intended identity and appearance: {intended}.",
    ]
    if style_prompt:
        clauses.append(f"Explicit style direction: {style_prompt}.")
    if appearance_prompt:
        clauses.append(f"Explicit material and finish direction: {appearance_prompt}.")
    clauses.extend([
        f"Scene design context: {_clean_prompt(scene_prompt)}.",
        "Edit only the active semantic region and keep every other semantic region exactly unchanged.",
    ])
    if generation_mode == "surface_texture":
        clauses.extend([
            "Perform a clearly visible full-material architectural refinement across the complete visible part of this installed surface, not a barely perceptible touch-up.",
            "Carry the requested design language continuously across the active surface using coherent material variation, panel or trim rhythm, relief, motifs, seams, or grain only as explicitly requested.",
            "Do not reduce a rich surface design to one tiny centered emblem, isolated ornament, picture, sticker, or decal surrounded by blank color.",
            "Preserve the existing geometry, UV correspondence, shared boundaries, perspective, and architectural continuity.",
        ])
    else:
        clauses.extend([
            "Perform a visible photorealistic material and ornamental refinement, strong enough to express the requested design style while preserving the current object geometry.",
            "Improve surface material, finish, craftsmanship, decorative detail, and coherent style identity without changing the silhouette, proportions, part layout, or support relationship.",
        ])
    clauses.extend([
        suffix + "." if suffix else "",
        "Do not introduce a new object, duplicate part, background element, text, watermark, or flat decal.",
    ])
    return " ".join(_clean_prompt(clause) for clause in clauses if _clean_prompt(clause))


def _visible_surface_owners(
    buffers: Dict,
    object_info: Dict[str, Dict],
    allowed_owners: set[str],
) -> list[str]:
    """Return visible surface_texture owners directly from the cached semantic image."""
    decoded, names = decode_objects(Path(buffers["semantic"]), Path(buffers["palette"]))
    visible_indices = set(int(value) for value in np.unique(decoded))
    owners: list[str] = []
    for index, owner in enumerate(names):
        if index not in visible_indices or owner not in allowed_owners:
            continue
        if str(object_info.get(owner, {}).get("generation_mode", "")) == "surface_texture":
            owners.append(owner)
    return owners


def _current_candidate_context(
    evaluation: Dict,
    texture_root: Path,
    allowed_owners: set[str] | None = None,
) -> Dict:
    # Camera selection uses all actually visible owners in the configured Step08
    # object domain. The cached >=50% frustum ratio remains generation-only.
    visible = dict(evaluation.get("visible_triangles", {}))
    if allowed_owners is not None:
        visible = {owner: records for owner, records in visible.items() if owner in allowed_owners}
    return weighted_seen_ratio(visible, texture_root)


def _frustum_gate_records(evaluation: Dict) -> Dict:
    path_value = evaluation.get("semantic_frustum_ratios_file") or dict(
        evaluation.get("shared_buffers", {})
    ).get("semantic_frustum_ratios")
    if not path_value:
        raise RuntimeError(
            f"Camera {evaluation.get('camera', {}).get('camera_id')} lacks semantic_frustum_ratios.json"
        )
    path = Path(path_value)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing cached semantic frustum ratio file: {path}")
    payload = load_json(path)
    semantics = dict(payload.get("semantics", {}))
    threshold = float(payload.get("minimum_generation_ratio", 0.50))
    return {
        "path": str(path),
        "threshold": threshold,
        "semantics": semantics,
        "metric": payload.get("metric"),
        "includes_back_facing_triangles": bool(payload.get("includes_back_facing_triangles", True)),
        "ignores_occlusion": bool(payload.get("ignores_occlusion", True)),
    }


def run_viewwise_refinement(out: str | Path, config: Dict, scene_json: str | Path, prompts_json: str | Path) -> Dict:
    out = Path(out)
    stage = out / "08_viewwise_refinement"
    stage.mkdir(parents=True, exist_ok=True)
    stage07 = load_json(out / "07_refinement_cameras" / "stage_report.json")
    if stage07.get("evaluation_files"):
        evaluations = [load_json(path) for path in stage07["evaluation_files"]]
    else:
        evaluations = stage07["evaluations"]
    camera_file = Path(stage07["cameras_file"])
    scene = load_json(scene_json)
    prompts_cfg = load_json(prompts_json) if Path(prompts_json).exists() else {}
    scene_payload = scene.get("scene", scene)
    scene_prompt = str(scene_payload.get("prompt", "a coherent scene"))
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    object_info = {str(o["object_id"]): o for o in plan.get("objects", [])}
    step08_target_modes = set(
        str(value)
        for value in config.get(
            "step08_target_generation_modes",
            ["surface_texture", "asset_3d", "external_asset", "scaffold_only"],
        )
    )
    step08_object_ids = {
        object_id
        for object_id, record in object_info.items()
        if str(record.get("generation_mode", "")) in step08_target_modes
    }
    atlases = load_registry(out / "05_texture_state")

    backend_name = config.get("diffusion_backend", "flux1_depth_control_inpaint_nf4_16gb")
    params = load_json("configs/parameters.json")
    backend_cfg = params["backend"]["profiles"][backend_name]
    auth_cfg = params["backend"].get("authentication", {})
    model_preparation = ensure_backend_models(backend_name, backend_cfg, auth_cfg)
    backend = create_backend(backend_name, backend_cfg, auth_config=auth_cfg)

    evaluations = sorted(
        [evaluation for evaluation in evaluations if evaluation.get("valid_geometry") and evaluation.get("shared_buffers")],
        key=lambda evaluation: int(dict(evaluation.get("camera", {})).get("deterministic_order", 0)),
    )
    target_count = len(evaluations)
    fusion_cfg = config.get("refinement_fusion", {})
    update_weight_by_mode = {
        str(key): float(value)
        for key, value in dict(config.get("update_weight_by_generation_mode", {})).items()
    }
    uniform_strength = float(fusion_cfg.get("strength", 0.20))
    fusion_strength_by_mode = {
        str(key): float(value)
        for key, value in dict(fusion_cfg.get("strength_by_generation_mode", {})).items()
    }
    generation_strength = float(config.get("generation_strength", 0.48))
    generation_strength_by_mode = {
        str(key): float(value)
        for key, value in dict(config.get("generation_strength_by_generation_mode", {})).items()
    }
    max_periodicity = float(config.get("max_periodicity_score", 0.72))
    max_periodicity_by_mode = {
        str(key): float(value)
        for key, value in dict(config.get("max_periodicity_score_by_generation_mode", {})).items()
    }
    dilation_px = int(config.get("mask_dilation_px", 5))
    blender = os.environ.get("BLENDER_BIN", "blender")

    used = set()
    iterations = []
    decision_trace = []
    selection_round = 0
    for evaluation in evaluations:
        selection_round += 1
        camera = evaluation["camera"]
        camera_id = camera["camera_id"]
        context_before = _current_candidate_context(
            evaluation,
            out / "05_texture_state",
            step08_object_ids,
        )
        score = 0.0
        view_diversity = 0.0
        decision_trace.append({
            "round": selection_round,
            "camera_id": camera_id,
            "event": "deterministic_valid_camera",
            "policy": "per-target-quota room-interior cameras in stored acceptance order; no camera ranking or greedy coverage scoring",
        })
        idir = stage / "iterations" / f"{len(iterations):02d}_{camera_id}"
        idir.mkdir(parents=True, exist_ok=True)
        frustum_gate = _frustum_gate_records(evaluation)
        target_owner = str(camera.get("target_object_id") or camera.get("target_source") or "")
        buffers = _shared_buffers(evaluation)
        visible_objects = [
            str(owner)
            for owner, record in frustum_gate["semantics"].items()
            if owner in step08_object_ids
            and (
                owner == target_owner
                or float(record.get("frustum_triangle_ratio", 0.0)) >= float(frustum_gate["threshold"])
            )
        ]
        # Stage07 already accepted this camera specifically for target_owner using
        # rendered target visibility. The main target therefore bypasses the generic
        # 50% all-facing frustum gate; other semantics keep the cached gate.
        target_screen_pixels = int(
            dict(evaluation.get("semantic_visibility", {}))
            .get(target_owner, {})
            .get("screen_pixels", 0)
        )
        if target_owner in step08_object_ids and target_screen_pixels > 0 and target_owner not in visible_objects:
            visible_objects.append(target_owner)
        # Room-shell surfaces are admitted by actual cached semantic pixels. Large
        # walls/floors/ceilings must not be excluded merely because less than 50%
        # of all their triangles fit in a object-target camera frustum.
        for surface_owner in _visible_surface_owners(buffers, object_info, step08_object_ids):
            if surface_owner not in visible_objects:
                visible_objects.append(surface_owner)
        visible_objects.sort(
            key=lambda owner: (
                int(owner == target_owner),
                float(frustum_gate["semantics"].get(owner, {}).get("frustum_triangle_ratio", 0.0)),
            ),
            reverse=True,
        )
        save_json({
            "chosen_camera": camera,
            "selection_score": score,
            "context_before": context_before,
            "view_diversity": view_diversity,
            "selection_policy": "all Stage07 accepted room-interior cameras execute in deterministic acceptance order; no camera ranking score",
            "step08_target_generation_modes": sorted(step08_target_modes),
            "semantic_frustum_gate": frustum_gate,
            "eligible_semantic_owners": visible_objects,
        }, idir / "selection.json")

        if not visible_objects:
            used.add(camera_id)
            decision_trace.append({
                "round": selection_round,
                "camera_id": camera_id,
                "event": "selected_camera_has_no_semantic_at_or_above_frustum_ratio_threshold",
                "threshold": frustum_gate["threshold"],
                "semantic_frustum_ratios_file": frustum_gate["path"],
            })
            continue

        frame_width, frame_height = buffers["resolution"]
        current = idir / "00_current_render.png"
        # The only Blender render for this camera by default. Geometry buffers
        # are immutable and were cached once in Stage07.
        render_current(
            blender,
            out,
            camera_file,
            camera_id,
            current,
            width=frame_width,
            height=frame_height,
        )
        subpasses = []
        successful_subpasses = 0
        owner_manifests = dict(buffers.get("owner_manifests", {}))
        for object_index, object_id in enumerate(visible_objects):
            visibility = dict(evaluation.get("semantic_visibility", {}).get(object_id, {}))
            ratio_record = dict(frustum_gate["semantics"].get(object_id, {}))
            frustum_ratio = float(ratio_record.get("frustum_triangle_ratio", 0.0))
            object_mode = str(object_info.get(object_id, {}).get("generation_mode", ""))
            # The assigned target bypasses the generic gate. Visible architectural
            # surfaces also bypass it because their actual semantic pixels are the
            # relevant signal; large surfaces often have <50% of all triangles in
            # a object-target camera despite occupying a major image region.
            if (
                object_id != target_owner
                and object_mode != "surface_texture"
                and frustum_ratio < float(frustum_gate["threshold"])
            ):
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_cached_frustum_ratio_below_threshold",
                    "frustum_triangle_ratio": frustum_ratio,
                    "minimum_frustum_triangle_ratio": frustum_gate["threshold"],
                    "semantic_frustum_ratios_file": frustum_gate["path"],
                })
                continue
            if object_id not in atlases or object_id not in object_info:
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_missing_atlas_or_generation_record",
                })
                continue
            manifest_path = owner_manifests.get(object_id)
            if not manifest_path or not Path(manifest_path).exists():
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_missing_cached_owner_manifest",
                })
                continue
            d = idir / "object_subpasses" / f"{object_index:02d}_{object_id}"
            masks = build_masks(
                Path(buffers["semantic"]),
                Path(buffers["uv"]),
                Path(buffers["palette"]),
                object_id,
                d / "masks",
                dilation_px,
            )
            if masks is None:
                subpasses.append({"object_id": object_id, "status": "skipped_no_writable_pixels"})
                continue
            edit_fraction = masks["semantic_pixels"] / float(max(frame_width * frame_height, 1))
            if edit_fraction < float(config.get("minimum_object_edit_fraction", 0.00015)):
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_too_small",
                    "edit_fraction": edit_fraction,
                })
                continue

            current_object_context = weighted_seen_ratio(
                {object_id: evaluation.get("visible_triangles", {}).get(object_id, [])},
                out / "05_texture_state",
            )
            minimum_object_gain = float(config.get("minimum_object_new_surface_ratio", 0.0))
            if float(current_object_context.get("new_surface_ratio", 0.0)) < minimum_object_gain:
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_no_current_uv_gain",
                    "current_object_context": current_object_context,
                })
                continue

            semantic_class = str(object_info[object_id].get("semantic_class", ""))
            generation_mode = object_mode
            refinement_record = dict(object_info[object_id].get("refinement", {}))
            object_generation_strength = float(np.clip(
                float(refinement_record.get(
                    "generation_strength",
                    generation_strength_by_mode.get(generation_mode, generation_strength),
                )),
                0.0,
                1.0,
            ))
            update_weight = float(
                refinement_record.get(
                    "update_weight",
                    update_weight_by_mode.get(generation_mode, 1.0),
                )
            )
            if update_weight <= 0.0:
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_nonpositive_update_weight",
                    "update_weight": update_weight,
                })
                continue
            base_fusion_strength = float(np.clip(
                float(refinement_record.get(
                    "fusion_strength",
                    fusion_strength_by_mode.get(generation_mode, uniform_strength),
                )),
                0.0,
                1.0,
            ))
            object_uniform_strength = min(1.0, base_fusion_strength * update_weight)
            object_max_periodicity = float(np.clip(
                float(refinement_record.get(
                    "max_periodicity_score",
                    max_periodicity_by_mode.get(generation_mode, max_periodicity),
                )),
                0.0,
                1.0,
            ))
            prompt = refinement_prompt(
                scene_prompt, object_info[object_id], prompts_cfg
            )
            raw = d / "generated_raw.png"
            rolling_input = Path(current)
            generation = backend.generate({
                "prompt": prompt,
                "negative_prompt": build_negative_prompt(
                    object_info[object_id],
                    str(prompts_cfg.get(
                        "global_negative",
                        "new object, duplicate object, changed geometry, changed silhouette, leakage outside active mask, text, watermark",
                    )),
                ),
                "init_image_path": str(rolling_input),
                "generation_mask_path": str(masks["generation"]),
                "depth_image_path": str(buffers["depth"]),
                "output_path": str(raw),
                "control_preview_path": str(d / "depth_control.png"),
                "object_name": object_id,
                "semantic_class": semantic_class,
                "region_policy": {"masked_object_only": True, "preserve_geometry": True},
                "seed": int(config.get("seed", 9100)) + (selection_round - 1) * 100 + object_index,
                "strength": object_generation_strength,
                "guidance_scale": float(
                    config.get("guidance_scale", backend_cfg.get("guidance_scale", 10.0))
                ),
                "num_inference_steps": int(
                    config.get("num_inference_steps", backend_cfg.get("num_inference_steps", 24))
                ),
                "width": frame_width,
                "height": frame_height,
            })
            locked = d / "generated_locked.png"
            # The exact semantic owner mask locks every previously generated owner.
            composite_locked(rolling_input, Path(generation["output_path"]), masks["exact"], locked)
            pscore = periodicity_score(locked, masks["write"])
            if pscore > object_max_periodicity:
                subpasses.append({
                    "object_id": object_id,
                    "status": "rejected_periodicity",
                    "periodicity_score": pscore,
                    "maximum_periodicity_score": object_max_periodicity,
                    "rolling_input": str(rolling_input),
                })
                continue

            # Write directly into the existing complete base texture. Triangle
            # visibility remains separate scheduling state.
            fusion = fuse_view(
                locked,
                buffers["semantic"],
                buffers["palette"],
                manifest_path,
                {object_id: atlases[object_id]},
                valid_mask_path=masks["write"],
                supersample_radius=float(config.get("supersample_radius", 0.35)),
                alpha_override=object_uniform_strength,
                conservative_barycentric_epsilon=float(
                    config.get("conservative_barycentric_epsilon", 0.0025)
                ),
                triangle_id_path=buffers["triangle_id"],
                uv_map_path=buffers["uv"],
                screen_uv_gap_fill_iterations=int(config.get("screen_uv_gap_fill_iterations", 2)),
            )
            fusion_report = fusion.get(object_id, {})
            observed_ids = set(
                int(value) for value in fusion_report.get("observed_triangle_ids", [])
            )
            visible_records = [
                record
                for record in evaluation.get("visible_triangles", {}).get(object_id, [])
                if int(record.get("global_triangle_id", -1)) in observed_ids
            ]
            if int(fusion_report.get("unique_observed_texels", 0)) <= 0:
                subpasses.append({
                    "object_id": object_id,
                    "status": "skipped_no_reachable_fusion_texels",
                    "fusion": fusion,
                    "rolling_input": str(rolling_input),
                })
                continue
            if not visible_records:
                # The authoritative Blender Triangle-ID + UV writeback already
                # proved which owner triangles were observed. Use those ids to
                # advance coverage even when an older Stage07 evaluation lacks
                # per-owner visible-triangle records (common for room surfaces).
                visible_records = [
                    {"global_triangle_id": int(triangle_id)}
                    for triangle_id in sorted(observed_ids)
                ]
            triangle_update = update_triangle_seen(
                atlases[object_id].dir / "triangle_seen.npy", visible_records
            )
            # Only a successfully fused owner advances the rolling image.
            current = locked
            successful_subpasses += 1
            subpasses.append({
                "object_id": object_id,
                "semantic_class": semantic_class,
                "status": "accepted",
                "edit_fraction": edit_fraction,
                "semantic_visibility": visibility,
                "frustum_triangle_ratio": frustum_ratio,
                "minimum_frustum_triangle_ratio": frustum_gate["threshold"],
                "semantic_frustum_ratios_file": frustum_gate["path"],
                "current_object_context": current_object_context,
                "generation_strength": object_generation_strength,
                "generation_mode": generation_mode,
                "update_weight": update_weight,
                "base_fusion_strength": base_fusion_strength,
                "uniform_fusion_strength": object_uniform_strength,
                "maximum_periodicity_score": object_max_periodicity,
                "initial_texture_source": "Stage06/Stage05 committed object-owned base texture",
                "periodicity_score": pscore,
                "generation": generation,
                "fusion": fusion,
                "triangle_coverage_update": triangle_update,
                "rolling_input": str(rolling_input),
                "rolling_output": str(locked),
            })

        rolling_final = idir / "rolling_final.png"
        shutil.copy2(current, rolling_final)
        validation_render = None
        if bool(config.get("render_validation_after_camera", False)):
            validation_render = idir / "final_validation_render.png"
            render_current(
                blender,
                out,
                camera_file,
                camera_id,
                validation_render,
                width=frame_width,
                height=frame_height,
            )

        if successful_subpasses == 0:
            used.add(camera_id)
            failed_iteration = {
                "camera_id": camera_id,
                "selection_score": score,
                "context_before": context_before,
                "subpasses": subpasses,
                "successful_subpasses": 0,
                "status": "skipped_no_successful_subpasses",
                "initial_render": str(idir / "00_current_render.png"),
                "rolling_final": str(idir / "rolling_final.png"),
                "validation_render": None if validation_render is None else str(validation_render),
            }
            save_json(failed_iteration, idir / "iteration_report.json")
            decision_trace.append({
                "round": selection_round,
                "camera_id": camera_id,
                "event": "selected_camera_produced_no_successful_subpasses",
            })
            continue
        threshold_update = {"policy": "no dynamic camera threshold; all Stage07 accepted quota cameras are processed"}
        used.add(camera_id)
        context_after = _current_candidate_context(
            evaluation,
            out / "05_texture_state",
            step08_object_ids,
        )
        iteration_report = {
            "camera_id": camera_id,
            "selection_score": score,
            "context_before": context_before,
            "context_after": context_after,
            "threshold_update": threshold_update,
            "subpasses": subpasses,
            "successful_subpasses": successful_subpasses,
            "semantic_frustum_gate": frustum_gate,
            "initial_render": str(idir / "00_current_render.png"),
            "rolling_final": str(rolling_final),
            "validation_render": None if validation_render is None else str(validation_render),
            "blender_renders_for_camera": 1 + int(validation_render is not None),
        }
        save_json(iteration_report, idir / "iteration_report.json")
        iterations.append(iteration_report)

    final_blend = stage / "final_textured_scene.blend"
    subprocess.run([
        blender, "--background", "--python", "src/blender/prephysics_runtime/save_final_textured_scene.py", "--",
        "--out", str(out), "--output", str(final_blend),
    ], check=True)
    if not final_blend.exists() or final_blend.stat().st_size == 0:
        raise RuntimeError(f"Final textured Blender scene was not saved: {final_blend}")
    status = "completed_all_valid_cameras" if len(used) >= target_count else "completed_with_skipped_cameras"
    report = {
        "status": status,
        "stage": "08_run_viewwise_refinement",
        "completed_views": len(iterations),
        "target_views": target_count,
        "used_cameras": sorted(used),
        "camera_iteration_policy": "per_target_quota_in_deterministic_acceptance_order",
        "processed_camera_count": len(used),
        "uniform_fusion_strength": uniform_strength,
        "initial_texture_policy": "Stage08 only refines committed object-owned base textures from Stage06/Stage05",
        "semantic_participation": "The assigned main target updates after Stage07 acceptance; visible surface_texture owners are admitted from actual cached semantic pixels; other configured non-main owners retain the cached all-facing frustum ratio gate.",
        "step08_target_generation_modes": sorted(step08_target_modes),
        "render_validation_after_camera": bool(config.get("render_validation_after_camera", False)),
        "iterations": iterations,
        "decision_trace": decision_trace,
        "model_preparation": model_preparation,
        "final_textured_scene": str(final_blend),
        "runtime": "Independent room-interior AABB-distance probability cameras; deterministic acceptance-order execution; main-target guaranteed update; actual-screen-pixel admission for room surfaces; cached frustum gate for other non-main owners; one current RGB render per camera; sequential rolling-RGB FLUX; JSON-driven per-mode and per-object generation/fusion strengths; reachable-only owner atlas fuse_view writeback",
    }
    save_json(report, stage / "stage_report.json")
    return report
