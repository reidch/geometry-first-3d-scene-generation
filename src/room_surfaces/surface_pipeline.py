from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.appearance.atlas_fusion import fuse_view
from src.appearance.atlas_state import load_registry
from src.appearance.backend_factory import create_backend
from src.appearance.model_cache import ensure_backend_models
from src.appearance.triangle_id_map import load_triangle_id_map, visible_records_from_id_map
from src.coverage.triangle_coverage import update_triangle_seen
from src.io.json_io import load_json, save_json
from src.room_surfaces.surface_prompting import build_surface_negative_prompt, build_surface_prompt
from src.room_surfaces.surface_commit import sha256_file
from src.room_surfaces.exact_surface_uv_commit import commit_rectified_surface_by_exact_triangles
from src.scene_ir.json_scene import scene_payload


_MAX_SURFACE_QUALITY_ATTEMPTS = 3


def _resolve_surface_candidate_selection(
    attempt_reports: List[Dict],
    minimum_detail_score: float,
) -> Dict:
    """Select the first threshold-passing candidate, else the highest-scoring candidate.

    Attempt order is significant: generation must stop immediately when a candidate
    reaches the configured threshold. When no attempt reaches it, score is a ranking
    signal only and must never block Stage06 publication. Ties keep the earlier attempt.
    """
    if not attempt_reports:
        raise ValueError("At least one surface candidate is required")
    threshold = float(minimum_detail_score)
    for report in attempt_reports:
        score = float(report["detail"]["score"])
        if score >= threshold:
            return {
                "candidate": report,
                "threshold_met": True,
                "selection_reason": "first_threshold_pass",
            }
    best = max(attempt_reports, key=lambda report: float(report["detail"]["score"]))
    return {
        "candidate": best,
        "threshold_met": False,
        "selection_reason": "highest_score_after_max_attempts",
    }


def _run_canonical_capture(out: Path, object_id: str, directory: Path, scene_json: str) -> None:
    blender = os.environ.get("BLENDER_BIN", "blender")
    script = Path("src/blender/prephysics_runtime/render_canonical_surface.py")
    command = [
        blender,
        "--background",
        "--python",
        str(script),
        "--",
        "--out",
        str(out),
        "--object_id",
        object_id,
        "--output_dir",
        str(directory),
        "--scene_json",
        str(scene_json),
    ]
    subprocess.run(command, check=True)
    required = [
        directory / "rgb.png",
        directory / "depth_control.png",
        directory / "mask_rgba.png",
        directory / "uv_map.json",
        directory / "uv_map_u.png",
        directory / "uv_map_v.png",
        directory / "uv_map_valid.png",
        directory / "triangle_id.png",
        directory / "triangles.json",
        directory / "capture_report.json",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        detail = (directory / ".failed").read_text(encoding="utf-8") if (directory / ".failed").exists() else ""
        raise RuntimeError(f"Canonical surface capture incomplete for {object_id}: {missing}\n{detail[-6000:]}")



def _surface_base_color(record: Dict) -> tuple[int, int, int]:
    values = list(dict(record.get("appearance", {})).get("base_color", [0.72, 0.70, 0.66, 1.0]))
    while len(values) < 3:
        values.append(0.7)
    return tuple(int(round(max(0.0, min(1.0, float(value))) * 255.0)) for value in values[:3])


def _prepare_rectified_surface_inputs(
    record: Dict,
    surface_spec: Dict,
    capture_report: Dict,
    directory: Path,
) -> Dict:
    """Create a complete planar texture-design canvas for a regular room surface.

    Blender remains the authority for aspect ratio and UV correspondence, but the
    diffusion model no longer receives a photographed room view. It receives one
    edge-to-edge rectified surface, a full generation mask, and constant planar
    depth. This makes Stage06 a direct surface-texture design task.
    """
    resolution = list(dict(capture_report.get("camera", {})).get("capture_resolution", []))
    if len(resolution) != 2:
        raise RuntimeError("Canonical capture did not report a two-dimensional surface resolution")
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid rectified surface resolution: {resolution}")

    base = np.empty((height, width, 3), dtype=np.uint8)
    base[:] = _surface_base_color(record)
    # A tiny broad gradient prevents a completely singular starting image without
    # inserting a local motif or perspective cue.
    axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    modulation = np.rint(axis * 3.0).astype(np.int16)
    base = np.clip(base.astype(np.int16) + modulation, 0, 255).astype(np.uint8)

    init_path = directory / "rectified_surface_init.png"
    mask_path = directory / "rectified_surface_mask.png"
    depth_path = directory / "rectified_surface_depth_control.png"
    Image.fromarray(base, "RGB").save(init_path)
    Image.new("L", (width, height), 255).save(mask_path)
    # Constant, non-zero 16-bit depth describes a planar surface and is legal for
    # FLUX depth control. No EXR or perspective-derived depth is involved.
    depth = np.full((height, width), 32768, dtype=np.uint16)
    Image.fromarray(depth).save(depth_path)
    report = {
        "generation_space": "rectified_surface",
        "resolution": [width, height],
        "aspect_ratio": float(width / max(height, 1)),
        "base_color_u8": list(_surface_base_color(record)),
        "mask_policy": "entire_rectified_surface",
        "depth_policy": "constant_planar_16bit_png",
        "uv_commit_policy": "authoritative_blender_uv_and_triangle_id",
        "surface_layout_type": str(surface_spec.get("layout_type", "")),
    }
    save_json(report, directory / "rectified_surface_input.json")
    return {
        "init_path": init_path,
        "mask_path": mask_path,
        "depth_path": depth_path,
        "mask_image": Image.open(mask_path).convert("L"),
        "report": report,
    }

def _mask_from_rgba(path: Path, output: Path) -> Image.Image:
    mask = Image.open(path).convert("RGBA").getchannel("A")
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    return mask

def _commit_rectified_surface_direct(
    atlas,
    generated_image_path: Path,
    object_id: str,
    directory: Path,
) -> Dict:
    """Compatibility wrapper for the v63 exact per-triangle surface commit."""
    return commit_rectified_surface_by_exact_triangles(
        atlas=atlas,
        generated_image_path=generated_image_path,
        object_id=object_id,
        directory=directory,
    )


def _synthetic_semantic(mask: Image.Image, object_id: str, directory: Path) -> Tuple[Path, Path]:
    color = (73, 151, 211)
    active = np.asarray(mask.convert("L")) > 0
    rgb = np.zeros((active.shape[0], active.shape[1], 3), dtype=np.uint8)
    rgb[active] = color
    semantic = directory / "semantic.png"
    palette = directory / "semantic.palette.json"
    Image.fromarray(rgb, "RGB").save(semantic)
    save_json(
        {
            object_id: {
                "object_id": object_id,
                "color_float_rgba": [component / 255.0 for component in color] + [1.0],
                "color_uint8_rgb": list(color),
            }
        },
        palette,
    )
    return semantic, palette


_VALID_REFERENCE_EDGES = {"left", "right", "top", "bottom"}


def _active_bbox(mask: Image.Image, *, label: str) -> tuple[int, int, int, int]:
    active = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    ys, xs = np.where(active)
    if len(xs) == 0:
        raise RuntimeError(f"Surface mask is empty: {label}")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _edge_crop_box(
    bbox: tuple[int, int, int, int],
    edge: str,
    fraction: float,
) -> tuple[int, int, int, int]:
    if edge not in _VALID_REFERENCE_EDGES:
        raise ValueError(f"Unsupported surface reference edge {edge!r}; expected one of {sorted(_VALID_REFERENCE_EDGES)}")
    x0, y0, x1, y1 = bbox
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    if edge in {"left", "right"}:
        extent = max(8, min(max(8, width // 3), int(round(width * float(fraction)))))
        return (x0, y0, min(x1, x0 + extent), y1) if edge == "left" else (max(x0, x1 - extent), y0, x1, y1)
    extent = max(8, min(max(8, height // 3), int(round(height * float(fraction)))))
    return (x0, y0, x1, min(y1, y0 + extent)) if edge == "top" else (x0, max(y0, y1 - extent), x1, y1)


def _paste_reference_edge(
    canvas: Image.Image,
    generation_mask: Image.Image,
    target_mask: Image.Image,
    *,
    reference_image: Path,
    reference_mask: Path,
    reference_object_id: str,
    source_edge: str,
    target_edge: str,
    fraction: float,
) -> Dict:
    """Embed one generated neighbouring-surface edge into the current target.

    The copied strip is protected from regeneration and therefore becomes an exact
    shared-boundary condition.  Everything outside the target semantic mask remains
    untouched, and the current surface remains the only generated semantic owner.
    """
    target_bbox = _active_bbox(target_mask, label="current target")
    source = Image.open(reference_image).convert("RGB")
    source_mask_image = Image.open(reference_mask).convert("L")
    source_bbox = _active_bbox(source_mask_image, label=str(reference_mask))
    source_box = _edge_crop_box(source_bbox, source_edge, fraction)
    target_box = _edge_crop_box(target_bbox, target_edge, fraction)
    target_width = max(1, target_box[2] - target_box[0])
    target_height = max(1, target_box[3] - target_box[1])

    patch = source.crop(source_box).resize((target_width, target_height), Image.Resampling.LANCZOS)
    patch_mask = source_mask_image.crop(source_box).resize((target_width, target_height), Image.Resampling.NEAREST)
    target_active = target_mask.crop(target_box).convert("L")
    paste_mask = Image.fromarray(
        np.minimum(
            np.asarray(patch_mask, dtype=np.uint8),
            np.asarray(target_active, dtype=np.uint8),
        ),
        "L",
    )
    canvas.paste(patch, (target_box[0], target_box[1]), paste_mask)

    generation = np.asarray(generation_mask.convert("L"), dtype=np.uint8).copy()
    protected = np.asarray(paste_mask, dtype=np.uint8) > 0
    region = generation[target_box[1] : target_box[3], target_box[0] : target_box[2]]
    region[protected] = 0
    generation[target_box[1] : target_box[3], target_box[0] : target_box[2]] = region
    generation_mask.paste(Image.fromarray(generation, "L"))
    return {
        "reference_object_id": str(reference_object_id),
        "reference_image": str(reference_image),
        "source_edge": source_edge,
        "target_edge": target_edge,
        "fraction": float(fraction),
        "source_box": list(source_box),
        "target_box": list(target_box),
        "protected_pixel_count": int(protected.sum()),
    }


def _ordered_surface_records(plan: Dict) -> List[Dict]:
    surfaces = [record for record in plan.get("objects", []) if record.get("generation_mode") == "surface_texture"]
    if surfaces and all(
        "generation_order" in dict(record.get("generation", {})).get("surface", {})
        for record in surfaces
    ):
        return sorted(
            surfaces,
            key=lambda record: (
                float(dict(record.get("generation", {})).get("surface", {}).get("generation_order", 0)),
                str(record["object_id"]),
            ),
        )

    groups: Dict[str, List[Dict]] = {}
    for record in surfaces:
        surface = dict(record.get("generation", {})).get("surface", {})
        group = str(surface.get("continuity_group", record["object_id"]))
        groups.setdefault(group, []).append(record)
    for values in groups.values():
        values.sort(
            key=lambda record: (
                float(dict(record.get("generation", {})).get("surface", {}).get("order", 0)),
                str(record["object_id"]),
            )
        )
    independent_groups = sorted(
        (values for values in groups.values() if len(values) == 1),
        key=lambda values: (
            float(dict(values[0].get("generation", {})).get("surface", {}).get("order", 0)),
            str(values[0]["object_id"]),
        ),
    )
    sequential_groups = sorted(
        (values for values in groups.values() if len(values) > 1),
        key=lambda values: (
            float(dict(values[0].get("generation", {})).get("surface", {}).get("order", 0)),
            str(dict(values[0].get("generation", {})).get("surface", {}).get("continuity_group", "")),
        ),
    )
    return [record for group in [*independent_groups, *sequential_groups] for record in group]


def _surface_reference_policy(surface: Dict) -> str:
    policy = str(surface.get("reference_policy", "independent")).strip().lower()
    supported = {"independent", "previous_in_continuity_group"}
    if policy not in supported:
        raise ValueError(f"Unsupported generation.surface.reference_policy={policy!r}; expected one of {sorted(supported)}")
    return policy


def _validate_surface_sequence(records: List[Dict], *, require_explicit: bool = True) -> Dict:
    seen_ids: set[str] = set()
    seen_groups: Dict[str, List[str]] = {}
    report_records = []
    for record in records:
        object_id = str(record["object_id"])
        surface = dict(record.get("generation", {})).get("surface", {})
        if require_explicit:
            missing = [key for key in ("generation_order", "reference_policy") if key not in surface]
            if missing:
                raise ValueError(
                    f"surface_texture object {object_id} is missing explicit generation.surface fields: {missing}"
                )
        group = str(surface.get("continuity_group", object_id))
        if "reference_policy" in surface:
            policy = _surface_reference_policy(surface)
        else:
            policy = "previous_in_continuity_group" if seen_groups.get(group) else "independent"
        previous_id = None
        if policy == "previous_in_continuity_group":
            previous = seen_groups.get(group, [])
            if not previous:
                raise ValueError(
                    f"surface_texture object {object_id} requests previous_in_continuity_group, but no earlier "
                    f"object in continuity_group={group!r} exists in explicit generation order"
                )
            previous_id = previous[-1]
            if require_explicit:
                missing = [key for key in ("reference_edge", "previous_reference_edge") if key not in surface]
                if missing:
                    raise ValueError(
                        f"sequential surface_texture object {object_id} is missing explicit shared-edge fields: {missing}"
                    )
        cycle_reference_id = None
        if bool(surface.get("close_cycle", False)):
            cycle_reference_id = str(surface.get("cycle_reference_object_id", "")).strip()
            if require_explicit:
                missing = [
                    key for key in ("cycle_reference_object_id", "cycle_reference_edge", "cycle_target_edge")
                    if key not in surface
                ]
                if missing:
                    raise ValueError(
                        f"cycle-closing surface_texture object {object_id} is missing explicit fields: {missing}"
                    )
            if not cycle_reference_id:
                earlier = seen_groups.get(group, [])
                cycle_reference_id = earlier[0] if earlier else ""
            if cycle_reference_id not in seen_ids:
                raise ValueError(
                    f"cycle-closing surface_texture object {object_id} references {cycle_reference_id!r}, "
                    "which must appear earlier in explicit generation order"
                )
        seen_ids.add(object_id)
        seen_groups.setdefault(group, []).append(object_id)
        report_records.append(
            {
                "object_id": object_id,
                "generation_order": float(surface.get("generation_order", len(report_records))),
                "continuity_group": group,
                "reference_policy": policy,
                "previous_reference_object_id": previous_id,
                "cycle_reference_object_id": cycle_reference_id,
            }
        )
    return {
        "status": "ok",
        "require_explicit_surface_sequence": bool(require_explicit),
        "records": report_records,
    }




def _masked_surface_detail_score(image_path: Path, mask_path: Path) -> Dict[str, float]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if not np.any(mask):
        return {"score": 0.0, "luminance_std": 0.0, "gradient_mean": 0.0, "color_std": 0.0}
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    values = luminance[mask]
    luminance_std = float(values.std())
    color_std = float(np.mean([image[..., channel][mask].std() for channel in range(3)]))
    gx = np.abs(luminance[:, 1:] - luminance[:, :-1])
    gy = np.abs(luminance[1:, :] - luminance[:-1, :])
    mx = mask[:, 1:] & mask[:, :-1]
    my = mask[1:, :] & mask[:-1, :]
    gradient_values = []
    if np.any(mx):
        gradient_values.append(float(gx[mx].mean()))
    if np.any(my):
        gradient_values.append(float(gy[my].mean()))
    gradient_mean = float(sum(gradient_values) / max(len(gradient_values), 1))
    score = float(0.45 * min(luminance_std / 0.12, 1.0) + 0.35 * min(gradient_mean / 0.06, 1.0) + 0.20 * min(color_std / 0.10, 1.0))
    return {
        "score": score,
        "luminance_std": luminance_std,
        "gradient_mean": gradient_mean,
        "color_std": color_std,
    }


def generate_surface_textures(
    out: str | Path,
    config: Dict,
    scene_json: str | Path,
    *,
    generation_only: bool = False,
) -> Dict:
    out = Path(out)
    step = out / "06_surface_textures"
    step.mkdir(parents=True, exist_ok=True)
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    scene_document = load_json(scene_json)
    scene_prompt = str(scene_payload(scene_document).get("prompt", ""))
    ordered = _ordered_surface_records(plan)
    if not ordered:
        report = {
            "status": "ok",
            "stage": "06a_generate_surface_images" if generation_only else "06_generate_surface_textures",
            "records": [],
            "reason": "no surface_texture objects",
        }
        save_json(
            report,
            step / ("stage06a_report.json" if generation_only else "stage_report.json"),
        )
        return report

    sequence_validation = _validate_surface_sequence(
        ordered,
        require_explicit=bool(config.get("require_explicit_surface_sequence", False)),
    )
    sequence_by_id = {
        str(item["object_id"]): item for item in sequence_validation.get("records", [])
    }

    backend_name = str(config.get("diffusion_backend", "flux1_depth_control_inpaint_nf4_16gb"))
    parameters = load_json("configs/parameters.json")
    backend_config = parameters["backend"]["profiles"][backend_name]
    authentication = parameters["backend"].get("authentication", {})
    model_preparation = ensure_backend_models(backend_name, backend_config, authentication)
    backend = create_backend(backend_name, backend_config, auth_config=authentication)
    atlases = load_registry(out / "05_texture_state")
    generated_by_group: Dict[str, List[Dict]] = {}
    generated_by_id: Dict[str, Dict] = {}
    records = []
    for index, record in enumerate(ordered):
        object_id = str(record["object_id"])
        if object_id not in atlases:
            raise RuntimeError(f"Texture state is missing explicit surface object {object_id}")
        generation_spec = dict(record.get("generation", {}))
        surface_spec = dict(generation_spec.get("surface", {}))
        negative_prompt = build_surface_negative_prompt(record, str(config.get("negative_prompt", "")))
        directory = step / object_id
        capture = directory / "capture"
        directory.mkdir(parents=True, exist_ok=True)
        _run_canonical_capture(out, object_id, capture, str(scene_json))
        exact_mask = _mask_from_rgba(capture / "mask_rgba.png", directory / "exact_mask.png")
        capture_report = load_json(capture / "capture_report.json")
        generation_space = str(
            surface_spec.get("generation_space", config.get("surface_generation_space_default", "canonical_view"))
        ).strip().lower()
        if generation_space not in {"canonical_view", "rectified_surface"}:
            raise ValueError(
                f"Unsupported generation.surface.generation_space={generation_space!r} for {object_id}"
            )
        if generation_space == "rectified_surface":
            surface_inputs = _prepare_rectified_surface_inputs(record, surface_spec, capture_report, directory)
            model_init_path = Path(surface_inputs["init_path"])
            model_depth_path = Path(surface_inputs["depth_path"])
            model_target_mask = Image.open(surface_inputs["mask_path"]).convert("L")
            model_target_mask_path = Path(surface_inputs["mask_path"])
        else:
            model_init_path = capture / "rgb.png"
            model_depth_path = capture / "depth_control.png"
            model_target_mask = exact_mask.copy().convert("L")
            model_target_mask_path = directory / "exact_mask.png"
        exact_array = np.asarray(exact_mask, dtype=np.uint8) > 0
        if not np.any(exact_array):
            raise RuntimeError(f"Canonical surface mask is empty for {object_id}")
        ys, xs = np.where(exact_array)
        frame_width, frame_height = exact_mask.size
        bbox_width_fraction = float((int(xs.max()) - int(xs.min()) + 1) / max(frame_width, 1))
        bbox_height_fraction = float((int(ys.max()) - int(ys.min()) + 1) / max(frame_height, 1))
        framing_cfg = dict(config.get("surface_framing_validation", {}))
        minimum_bbox_width = float(framing_cfg.get("minimum_target_bbox_width_fraction", 0.80))
        minimum_bbox_height = float(framing_cfg.get("minimum_target_bbox_height_fraction", 0.80))
        if bbox_width_fraction < minimum_bbox_width or bbox_height_fraction < minimum_bbox_height:
            raise RuntimeError(
                f"Canonical camera did not frame {object_id} tightly enough: "
                f"bbox={bbox_width_fraction:.3%}x{bbox_height_fraction:.3%}, "
                f"required={minimum_bbox_width:.3%}x{minimum_bbox_height:.3%}."
            )
        init_path = model_init_path
        generation_mask = directory / "generation_mask.png"
        group = str(surface_spec.get("continuity_group", object_id))
        reference_policy = str(sequence_by_id[object_id]["reference_policy"])
        reference_fraction = float(
            surface_spec.get("reference_strip_fraction", config.get("reference_strip_fraction", 0.12))
        )
        reference_bindings: List[Dict] = []
        canvas = Image.open(model_init_path).convert("RGB")
        generation_mask_image = model_target_mask.copy().convert("L")

        if reference_policy == "previous_in_continuity_group":
            previous_records = generated_by_group.get(group, [])
            if not previous_records:
                raise RuntimeError(
                    f"Surface {object_id} requests previous_in_continuity_group, but no earlier generated "
                    f"surface exists in continuity_group={group!r}. Fix generation.surface.generation_order/order."
                )
            previous = previous_records[-1]
            reference_bindings.append(
                _paste_reference_edge(
                    canvas,
                    generation_mask_image,
                    model_target_mask,
                    reference_image=Path(previous["image"]),
                    reference_mask=Path(previous["mask"]),
                    reference_object_id=str(previous["object_id"]),
                    source_edge=str(surface_spec.get("previous_reference_edge", "right")),
                    target_edge=str(surface_spec.get("reference_edge", "left")),
                    fraction=reference_fraction,
                )
            )

        cycle_reference_id = str(surface_spec.get("cycle_reference_object_id", "")).strip()
        if bool(surface_spec.get("close_cycle", False)):
            if not cycle_reference_id:
                group_records = generated_by_group.get(group, [])
                if not group_records:
                    raise RuntimeError(
                        f"Surface {object_id} requests close_cycle but has no earlier continuity-group surface."
                    )
                cycle_reference_id = str(group_records[0]["object_id"])
            if cycle_reference_id not in generated_by_id:
                raise RuntimeError(
                    f"Surface {object_id} requests cycle reference {cycle_reference_id!r}, but it has not been generated yet."
                )
            cycle_reference = generated_by_id[cycle_reference_id]
            reference_bindings.append(
                _paste_reference_edge(
                    canvas,
                    generation_mask_image,
                    model_target_mask,
                    reference_image=Path(cycle_reference["image"]),
                    reference_mask=Path(cycle_reference["mask"]),
                    reference_object_id=cycle_reference_id,
                    source_edge=str(surface_spec.get("cycle_reference_edge", "left")),
                    target_edge=str(surface_spec.get("cycle_target_edge", "right")),
                    fraction=float(surface_spec.get("cycle_reference_strip_fraction", reference_fraction)),
                )
            )

        chain = {
            "reference_policy": reference_policy,
            "continuity_group": group,
            "reference_bindings": reference_bindings,
            "current_surface_is_only_generation_target": True,
        }
        save_json(chain, directory / "reference_context.json")
        if reference_bindings:
            init_path = directory / "reference_init.png"
            canvas.save(init_path)
        generation_mask_image.save(generation_mask)
        prompt = build_surface_prompt(record, scene_prompt, reference_context=chain)

        quality_cfg = dict(config.get("surface_quality_validation", {}))
        requested_max_attempts = max(1, int(quality_cfg.get("max_attempts", _MAX_SURFACE_QUALITY_ATTEMPTS)))
        max_attempts = min(requested_max_attempts, _MAX_SURFACE_QUALITY_ATTEMPTS)
        minimum_detail_score = float(quality_cfg.get("minimum_detail_score", 0.32))
        attempt_reports = []
        initial = Image.open(init_path).convert("RGB")
        generation_width, generation_height = initial.size
        for attempt_index in range(max_attempts):
            attempt_prompt = build_surface_prompt(
                record,
                scene_prompt,
                retry_for_detail=attempt_index > 0,
                reference_context=chain,
            )
            raw = directory / f"generated_raw_attempt_{attempt_index + 1:02d}.png"
            result = backend.generate(
                {
                    "prompt": attempt_prompt,
                    "negative_prompt": negative_prompt,
                    "init_image_path": str(init_path),
                    "generation_mask_path": str(generation_mask),
                    "depth_image_path": str(model_depth_path),
                    "output_path": str(raw),
                    "control_preview_path": str(directory / f"depth_control_attempt_{attempt_index + 1:02d}.png"),
                    "object_name": object_id,
                    "semantic_class": str(record.get("semantic_class", "")),
                    "region_policy": {
                        "masked_object_only": True,
                        "continuous_surface": True,
                        "preserve_geometry": generation_space != "rectified_surface",
                        "rectified_surface_texture": generation_space == "rectified_surface",
                    },
                    "seed": int(generation_spec.get("seed", config.get("seed", 8100) + index)) + attempt_index * int(quality_cfg.get("seed_stride", 131)),
                    "strength": float(generation_spec.get("strength", config.get("generation_strength", 0.78))) + attempt_index * float(quality_cfg.get("retry_strength_increment", 0.04)),
                    "guidance_scale": float(generation_spec.get("guidance_scale", config.get("guidance_scale", backend_config.get("guidance_scale", 10.0)))),
                    "num_inference_steps": int(generation_spec.get("num_inference_steps", config.get("num_inference_steps", 30))) + attempt_index * int(quality_cfg.get("retry_step_increment", 4)),
                    "width": int(generation_width),
                    "height": int(generation_height),
                }
            )
            generated = Image.open(result["output_path"]).convert("RGB").resize(initial.size, Image.Resampling.LANCZOS)
            candidate_locked = directory / f"generated_locked_attempt_{attempt_index + 1:02d}.png"
            Image.composite(generated, initial, Image.open(generation_mask).convert("L")).save(candidate_locked)
            detail = _masked_surface_detail_score(candidate_locked, model_target_mask_path)
            attempt_report = {
                "attempt": attempt_index + 1,
                "prompt": attempt_prompt,
                "generation": result,
                "locked_image": str(candidate_locked),
                "detail": detail,
            }
            attempt_reports.append(attempt_report)
            score = float(detail["score"])
            passed = score >= minimum_detail_score
            print(
                f"[06][SURFACE QUALITY] object={object_id} attempt={attempt_index + 1}/{max_attempts} "
                f"detail={score:.4f} threshold={minimum_detail_score:.4f} "
                f"decision={'accept' if passed else ('retry' if attempt_index + 1 < max_attempts else 'select_best')}",
                flush=True,
            )
            if passed:
                break
        selection = _resolve_surface_candidate_selection(attempt_reports, minimum_detail_score)
        selected = selection["candidate"]
        selected_score = float(selected["detail"]["score"])
        if bool(selection["threshold_met"]):
            print(
                f"[06][SURFACE SELECT] object={object_id} selected_attempt={selected['attempt']} "
                f"detail={selected_score:.4f} reason=first_threshold_pass",
                flush=True,
            )
        else:
            print(
                f"[06][SURFACE SELECT] object={object_id} no candidate reached "
                f"{minimum_detail_score:.4f} after {len(attempt_reports)} attempts; "
                f"using best attempt={selected['attempt']} detail={selected_score:.4f}",
                flush=True,
            )
        prompt = str(selected["prompt"])
        result = dict(selected["generation"])
        locked = directory / "generated_locked.png"
        Image.open(selected["locked_image"]).save(locked)
        Image.open(selected["locked_image"]).save(directory / "complete_surface_view.png")
        generated_record = {
            "object_id": object_id,
            "image": str(locked),
            "mask": str(model_target_mask_path),
            "generation_space": generation_space,
            "continuity_group": group,
            "reference_policy": reference_policy,
        }
        generated_by_group.setdefault(group, []).append(generated_record)
        generated_by_id[object_id] = generated_record

        generation_manifest = {
            "schema_version": 1,
            "status": "ok",
            "object_id": object_id,
            "generation_space": generation_space,
            "generated_image": str(locked),
            "exact_mask": str(directory / "exact_mask.png"),
            "capture_directory": str(capture),
            "capture_report": str(capture / "capture_report.json"),
            "triangle_manifest": str(capture / "triangles.json"),
            "triangle_id": str(capture / "triangle_id.png"),
            "continuity_group": group,
            "reference_policy": reference_policy,
            "target_bbox_width_fraction": bbox_width_fraction,
            "target_bbox_height_fraction": bbox_height_fraction,
        }
        save_json(generation_manifest, directory / "generation_manifest.json")

        if generation_only:
            item_report = {
                "status": "ok",
                "phase": "06a_generation_only",
                "object_id": object_id,
                "continuity_group": group,
                "chain_reference": chain,
                "prompt": prompt,
                "generation": result,
                "surface_generation_attempts": attempt_reports,
                "surface_candidate_selection": {
                    "policy": "first candidate at or above threshold; otherwise highest score after at most three attempts",
                    "requested_max_attempts": requested_max_attempts,
                    "effective_max_attempts": max_attempts,
                    "attempts_executed": len(attempt_reports),
                    "minimum_detail_score": minimum_detail_score,
                    "threshold_met": bool(selection["threshold_met"]),
                    "selection_reason": str(selection["selection_reason"]),
                    "selected_attempt": int(selected["attempt"]),
                    "selected_score": selected_score,
                    "score_is_not_a_publication_gate": True,
                },
                "selected_detail_score": selected["detail"],
                "minimum_detail_score": minimum_detail_score,
                "canonical_capture": str(capture / "capture_report.json"),
                "generation_space": generation_space,
                "generation_manifest": str(directory / "generation_manifest.json"),
                "rectified_surface_input": (
                    str(directory / "rectified_surface_input.json")
                    if generation_space == "rectified_surface" else None
                ),
                "atlas_write_deferred_to_stage06b": True,
            }
            save_json(item_report, directory / "surface_generation_report.json")
            records.append(item_report)
            continue

        if generation_space == "rectified_surface":
            direct_commit = _commit_rectified_surface_direct(
                atlases[object_id],
                locked,
                object_id,
                directory,
            )
            fusion = direct_commit["fusion"]
            atlas_commit = direct_commit["atlas_commit"]
            atlases[object_id].update_metadata({"stage06_surface_commit": atlas_commit})
            triangle_update = direct_commit["triangle_update"]
        else:
            semantic, palette = _synthetic_semantic(exact_mask, object_id, directory)
            manifest = load_json(capture / "triangles.json")
            metadata = {
                int(item["global_triangle_id"]): item
                for item in manifest.get("triangles", [])
            }
            decoded = load_triangle_id_map(capture / "triangle_id.png")
            exact_array = np.asarray(exact_mask, dtype=np.uint8) > 0
            canonical_visible_ids = {
                int(value)
                for value in np.unique(decoded[exact_array & (decoded >= 0)]).tolist()
                if int(value) in metadata
            }
            if not canonical_visible_ids:
                raise RuntimeError(
                    f"Canonical isolated render for {object_id} contains no target triangle IDs; "
                    "the camera is not looking at the intended room-facing surface."
                )
            write_config = dict(config.get("texture_write", {}))
            observed_uv_mask_path = directory / "writeback_observed_uv_mask.png"
            atlas_before = np.asarray(Image.open(atlases[object_id].color_path).convert("RGB"), dtype=np.int16)
            atlas_before_sha256 = sha256_file(atlases[object_id].color_path)
            fusion = fuse_view(
                locked,
                semantic,
                palette,
                capture / "triangles.json",
                {object_id: atlases[object_id]},
                valid_mask_path=directory / "exact_mask.png",
                supersample_radius=float(write_config.get("supersample_radius", 0.35)),
                alpha_override=1.0,
                conservative_barycentric_epsilon=float(write_config.get("conservative_barycentric_epsilon", 0.0025)),
                triangle_id_path=capture / "triangle_id.png",
                uv_map_path=capture / "uv_map.json",
                screen_uv_gap_fill_iterations=int(write_config.get("screen_uv_gap_fill_iterations", 2)),
                observation_mask_output_path=observed_uv_mask_path,
            )
            fusion_report = dict(fusion.get(object_id, {}) or {})
            unique_observed_texels = int(fusion_report.get("unique_observed_texels", 0))
            if unique_observed_texels <= 0:
                failure_report = {
                    "object_id": object_id,
                    "reason": "no_atlas_texels_written",
                    "fusion": fusion_report,
                    "canonical_visible_triangle_ids": sorted(canonical_visible_ids),
                    "uv_map": str(capture / "uv_map.json"),
                    "triangle_id": str(capture / "triangle_id.png"),
                    "triangles": str(capture / "triangles.json"),
                    "exact_mask_pixels": int(exact_array.sum()),
                }
                failure_path = directory / "writeback_failure.json"
                save_json(failure_report, failure_path)
                raise RuntimeError(
                    f"Stage06 generated an image for {object_id}, but neither the authoritative "
                    "Blender screen-UV writeback nor the compatibility clip-space fallback wrote "
                    f"any atlas texels. Diagnostics: {failure_path}"
                )
            observed_ids = {
                int(value) for value in fusion_report.get("observed_triangle_ids", [])
            }
            visible_world_area = float(
                sum(float(metadata[value].get("world_area", 0.0)) for value in canonical_visible_ids)
            )
            observed_world_area = float(
                sum(
                    float(metadata[value].get("world_area", 0.0))
                    for value in canonical_visible_ids & observed_ids
                )
            )
            world_area_coverage = float(
                observed_world_area / max(visible_world_area, 1e-12)
            )
            atlas_after = np.asarray(Image.open(atlases[object_id].color_path).convert("RGB"), dtype=np.int16)
            changed_mask = np.any(atlas_after != atlas_before, axis=2)
            changed_texels = int(changed_mask.sum())
            if changed_texels <= 0:
                raise RuntimeError(
                    f"Stage06 surface atlas for {object_id} did not change after fusion; refusing to publish a scene "
                    "whose material would still show the Stage05 placeholder."
                )
            atlas_after_sha256 = sha256_file(atlases[object_id].color_path)
            committed_preview_path = directory / "committed_base_color.png"
            Image.open(atlases[object_id].color_path).convert("RGB").save(committed_preview_path)
            atlas_commit = {
                "committed": True,
                "object_id": object_id,
                "before_sha256": atlas_before_sha256,
                "after_sha256": atlas_after_sha256,
                "unique_observed_texels": unique_observed_texels,
                "changed_texels": changed_texels,
                "changed_texel_fraction": float(changed_texels / max(changed_mask.size, 1)),
                "maximum_channel_delta_u8": int(np.abs(atlas_after - atlas_before).max()),
                "writeback_core": "src.appearance.atlas_fusion.fuse_view",
                "writeback_policy": "triangle_id_grouped_blender_screen_uv_projection_into_object_owned_base_color",
                "canonical_view_direction_source": capture_report.get("camera", {}).get("view_direction_source"),
                "canonical_visible_triangle_count": len(canonical_visible_ids),
                "canonical_observed_triangle_count": len(canonical_visible_ids & observed_ids),
                "canonical_visible_world_area": visible_world_area,
                "canonical_observed_world_area": observed_world_area,
                "canonical_observed_world_area_ratio_diagnostic": world_area_coverage,
                "target_bbox_width_fraction": bbox_width_fraction,
                "target_bbox_height_fraction": bbox_height_fraction,
                "target_mask_pixel_fraction": float(exact_array.sum() / max(exact_array.size, 1)),
                "non_target_meshes_hidden": bool(
                    capture_report.get("visibility", {}).get("non_target_meshes_hidden", False)
                ),
                "observed_uv_mask": str(observed_uv_mask_path),
                "coverage_is_not_an_acceptance_gate": True,
                "committed_base_color_preview": str(committed_preview_path),
            }
            atlases[object_id].update_metadata({"stage06_surface_commit": atlas_commit})
            visible = [
                record for record in visible_records_from_id_map(
                    decoded, np.asarray(exact_mask) > 0, metadata
                )
                if int(record.get("global_triangle_id", -1)) in observed_ids
            ]
            triangle_update = update_triangle_seen(atlases[object_id].dir / "triangle_seen.npy", visible)
        item_report = {
            "status": "ok",
            "object_id": object_id,
            "continuity_group": group,
            "chain_reference": chain,
            "prompt": prompt,
            "generation": result,
            "surface_generation_attempts": attempt_reports,
            "surface_candidate_selection": {
                "policy": "first candidate at or above threshold; otherwise highest score after at most three attempts",
                "requested_max_attempts": requested_max_attempts,
                "effective_max_attempts": max_attempts,
                "attempts_executed": len(attempt_reports),
                "minimum_detail_score": minimum_detail_score,
                "threshold_met": bool(selection["threshold_met"]),
                "selection_reason": str(selection["selection_reason"]),
                "selected_attempt": int(selected["attempt"]),
                "selected_score": selected_score,
                "score_is_not_a_publication_gate": True,
            },
            "selected_detail_score": selected["detail"],
            "minimum_detail_score": minimum_detail_score,
            "fusion": fusion,
            "atlas_commit": atlas_commit,
            "triangle_visibility_update": triangle_update,
            "canonical_capture": str(capture / "capture_report.json"),
            "generation_space": generation_space,
            "rectified_surface_input": (
                str(directory / "rectified_surface_input.json")
                if generation_space == "rectified_surface" else None
            ),
        }
        save_json(item_report, directory / "surface_report.json")
        records.append(item_report)

    report = {
        "status": "ok",
        "stage": "06a_generate_surface_images" if generation_only else "06_generate_surface_textures",
        "records": records,
        "surface_sequence": sequence_validation,
        "model_preparation": model_preparation,
        "runtime": "explicit JSON surface routing and generation order; rectified full-surface texture generation for regular walls/floors/ceilings; regular room surfaces now commit their generated rectified image directly as the authoritative bound texture so wall/floor/ceiling coverage is edge-to-edge with no screen-space re-projection gaps; independent floor/ceiling passes; first wall independent; later walls conditioned by protected previous/cycle edge strips; non-surface objects still use triangle-id grouped UV projection; best-of-three non-blocking candidate selection; strict Blender material publication",
    }
    save_json(
        report,
        step / ("stage06a_report.json" if generation_only else "stage_report.json"),
    )
    return report


def _commit_existing_canonical_surface(
    *,
    atlas,
    object_id: str,
    directory: Path,
    config: Dict,
) -> Dict:
    """Compatibility path for JSON surfaces generated in canonical-view space."""
    capture = directory / "capture"
    locked = directory / "generated_locked.png"
    exact_mask_path = directory / "exact_mask.png"
    exact_mask = Image.open(exact_mask_path).convert("L")
    semantic, palette = _synthetic_semantic(exact_mask, object_id, directory)
    manifest = load_json(capture / "triangles.json")
    metadata = {
        int(item["global_triangle_id"]): item
        for item in manifest.get("triangles", [])
    }
    decoded = load_triangle_id_map(capture / "triangle_id.png")
    exact_array = np.asarray(exact_mask, dtype=np.uint8) > 0
    canonical_visible_ids = {
        int(value)
        for value in np.unique(decoded[exact_array & (decoded >= 0)]).tolist()
        if int(value) in metadata
    }
    if not canonical_visible_ids:
        raise RuntimeError(
            f"Canonical isolated render for {object_id} contains no target triangle IDs"
        )
    write_config = dict(config.get("texture_write", {}))
    observed_uv_mask_path = directory / "writeback_observed_uv_mask.png"
    atlas_before = np.asarray(Image.open(atlas.color_path).convert("RGB"), dtype=np.int16)
    atlas_before_sha256 = sha256_file(atlas.color_path)
    fusion = fuse_view(
        locked,
        semantic,
        palette,
        capture / "triangles.json",
        {object_id: atlas},
        valid_mask_path=exact_mask_path,
        supersample_radius=float(write_config.get("supersample_radius", 0.35)),
        alpha_override=1.0,
        conservative_barycentric_epsilon=float(
            write_config.get("conservative_barycentric_epsilon", 0.0025)
        ),
        triangle_id_path=capture / "triangle_id.png",
        uv_map_path=capture / "uv_map.json",
        screen_uv_gap_fill_iterations=int(
            write_config.get("screen_uv_gap_fill_iterations", 2)
        ),
        observation_mask_output_path=observed_uv_mask_path,
    )
    fusion_report = dict(fusion.get(object_id, {}) or {})
    unique_observed_texels = int(fusion_report.get("unique_observed_texels", 0))
    if unique_observed_texels <= 0:
        raise RuntimeError(f"Canonical-view writeback produced no atlas texels for {object_id}")
    observed_ids = {int(value) for value in fusion_report.get("observed_triangle_ids", [])}
    visible_world_area = float(
        sum(float(metadata[value].get("world_area", 0.0)) for value in canonical_visible_ids)
    )
    observed_world_area = float(
        sum(
            float(metadata[value].get("world_area", 0.0))
            for value in canonical_visible_ids & observed_ids
        )
    )
    atlas_after = np.asarray(Image.open(atlas.color_path).convert("RGB"), dtype=np.int16)
    changed_mask = np.any(atlas_after != atlas_before, axis=2)
    changed_texels = int(changed_mask.sum())
    if changed_texels <= 0:
        raise RuntimeError(f"Canonical-view writeback did not change atlas for {object_id}")
    committed_preview_path = directory / "committed_base_color.png"
    Image.open(atlas.color_path).convert("RGB").save(committed_preview_path)
    atlas_commit = {
        "committed": True,
        "object_id": object_id,
        "before_sha256": atlas_before_sha256,
        "after_sha256": sha256_file(atlas.color_path),
        "unique_observed_texels": unique_observed_texels,
        "changed_texels": changed_texels,
        "changed_texel_fraction": float(changed_texels / max(changed_mask.size, 1)),
        "maximum_channel_delta_u8": int(np.abs(atlas_after - atlas_before).max()),
        "writeback_core": "src.appearance.atlas_fusion.fuse_view",
        "writeback_policy": "canonical_view_triangle_id_screen_uv_projection",
        "canonical_visible_triangle_count": len(canonical_visible_ids),
        "canonical_observed_triangle_count": len(canonical_visible_ids & observed_ids),
        "canonical_visible_world_area": visible_world_area,
        "canonical_observed_world_area": observed_world_area,
        "canonical_observed_world_area_ratio_diagnostic": float(
            observed_world_area / max(visible_world_area, 1e-12)
        ),
        "target_bbox_width_fraction": 1.0,
        "target_bbox_height_fraction": 1.0,
        "target_mask_pixel_fraction": float(exact_array.mean()),
        "non_target_meshes_hidden": True,
        "observed_uv_mask": str(observed_uv_mask_path),
        "coverage_is_not_an_acceptance_gate": True,
        "committed_base_color_preview": str(committed_preview_path),
    }
    visible = [
        record for record in visible_records_from_id_map(decoded, exact_array, metadata)
        if int(record.get("global_triangle_id", -1)) in observed_ids
    ]
    triangle_update = update_triangle_seen(atlas.dir / "triangle_seen.npy", visible)
    return {
        "fusion": fusion,
        "atlas_commit": atlas_commit,
        "triangle_update": triangle_update,
    }


def commit_generated_surface_textures(
    out: str | Path,
    config: Dict,
    scene_json: str | Path,
) -> Dict:
    """Stage06b: commit existing Stage06a images without loading a diffusion model."""
    del scene_json  # mapping authority is the already captured Blender geometry/UV data
    out = Path(out)
    step = out / "06_surface_textures"
    plan = load_json(out / "01_world_ir" / "generation_plan.json")
    ordered = _ordered_surface_records(plan)
    atlases = load_registry(out / "05_texture_state")
    records = []
    for record in ordered:
        object_id = str(record["object_id"])
        if object_id not in atlases:
            raise RuntimeError(f"Texture state is missing explicit surface object {object_id}")
        directory = step / object_id
        manifest_path = directory / "generation_manifest.json"
        if manifest_path.exists():
            generation_manifest = load_json(manifest_path)
        else:
            # Migration path for Stage06 outputs produced before the 06a/06b split.
            # Existing generated images and canonical captures remain authoritative;
            # only the writeback/publication phase is rebuilt.
            surface_spec = dict(dict(record.get("generation", {})).get("surface", {}))
            inferred_generation_space = str(
                surface_spec.get(
                    "generation_space",
                    "rectified_surface" if (directory / "rectified_surface_input.json").exists() else "canonical_view",
                )
            ).strip().lower()
            generation_manifest = {
                "schema_version": 1,
                "status": "migrated_existing_stage06_output",
                "object_id": object_id,
                "generation_space": inferred_generation_space,
                "generated_image": str(directory / "generated_locked.png"),
                "exact_mask": str(directory / "exact_mask.png"),
                "capture_directory": str(directory / "capture"),
                "capture_report": str(directory / "capture" / "capture_report.json"),
                "triangle_manifest": str(directory / "capture" / "triangles.json"),
                "triangle_id": str(directory / "capture" / "triangle_id.png"),
                "migrated_without_regeneration": True,
            }
            save_json(generation_manifest, manifest_path)
        generation_space = str(generation_manifest.get("generation_space", ""))
        locked = Path(str(generation_manifest.get("generated_image", directory / "generated_locked.png")))
        if not locked.exists() or locked.stat().st_size == 0:
            raise RuntimeError(f"Stage06a generated image is missing for {object_id}: {locked}")

        if generation_space == "rectified_surface":
            commit_result = commit_rectified_surface_by_exact_triangles(
                atlas=atlases[object_id],
                generated_image_path=locked,
                object_id=object_id,
                directory=directory,
            )
        elif generation_space == "canonical_view":
            commit_result = _commit_existing_canonical_surface(
                atlas=atlases[object_id],
                object_id=object_id,
                directory=directory,
                config=config,
            )
        else:
            raise RuntimeError(
                f"Unsupported Stage06a generation space for Stage06b: {generation_space!r}"
            )

        atlas_commit = dict(commit_result["atlas_commit"])
        atlases[object_id].update_metadata({"stage06_surface_commit": atlas_commit})
        generation_report_path = directory / "surface_generation_report.json"
        generation_report = load_json(generation_report_path) if generation_report_path.exists() else {}
        item_report = {
            **generation_report,
            "status": "ok",
            "phase": "06b_exact_texture_commit",
            "object_id": object_id,
            "generation_space": generation_space,
            "generation_manifest": str(manifest_path),
            "fusion": commit_result["fusion"],
            "atlas_commit": atlas_commit,
            "triangle_visibility_update": commit_result["triangle_update"],
        }
        save_json(item_report, directory / "surface_report.json")
        records.append(item_report)

    report = {
        "status": "ok",
        "stage": "06b_commit_generated_surface_textures",
        "records": records,
        "generation_report": str(step / "stage06a_report.json"),
        "runtime": (
            "existing Stage06a images only; original mesh loop UVs preserved; "
            "rectified surfaces are split by the actual room-facing mesh triangles "
            "and rasterized exactly into each triangle's recorded UV footprint"
        ),
    }
    save_json(report, step / "stage06b_report.json")
    return report


# Compatibility name for downstream imports; behavior is fully generic.
generate_room_surfaces = generate_surface_textures
