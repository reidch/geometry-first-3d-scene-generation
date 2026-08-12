from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from src.cameras.reconstruction_sampler import _build_context, _render_final_buffers
from src.cameras.room_pair_sampling import build_room_surface_model
from src.cameras.scene_geometry import scaffold_points
from src.cameras.worldmesh_coverage_sampling import (
    _make_camera,
    _point_distance_to_collider,
    _point_in_polygon,
    _sample_perimeter_positions,
    floor_height,
    room_center_xy,
    room_floor_polygon,
)
from src.io.json_io import load_json, save_json
from src.scene_ir.json_scene import flat_objects


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _pitch_deg(position: Sequence[float], target: Sequence[float]) -> float:
    d = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    horizontal = float(np.linalg.norm(d[:2]))
    return math.degrees(math.atan2(float(d[2]), max(horizontal, 1e-12)))


def _stage07_pitch_limit(accepted: Sequence[Mapping[str, Any]]) -> float:
    overhead = [c for c in accepted if str(c.get("camera_role", "")) == "perimeter_overhead"]
    source = overhead if overhead else list(accepted)
    values = [abs(_pitch_deg(c["position"], c["target"])) for c in source]
    if not values:
        raise RuntimeError("Evaluation camera sampling requires Stage07 accepted cameras")
    return max(float(max(values)), 1.0)


def _pose_far_from_training(camera: Mapping[str, Any], training: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    p = np.asarray(camera["position"], dtype=np.float64)
    d = _unit(np.asarray(camera["target"], dtype=np.float64) - p)
    min_pos = float(cfg.get("minimum_training_camera_position_separation_m", 0.10))
    min_ang = math.radians(float(cfg.get("minimum_training_camera_direction_separation_degrees", 8.0)))
    for other in training:
        q = np.asarray(other["position"], dtype=np.float64)
        if float(np.linalg.norm(p - q)) >= min_pos:
            continue
        od = _unit(np.asarray(other["target"], dtype=np.float64) - q)
        angle = math.acos(float(np.clip(np.dot(d, od), -1.0, 1.0)))
        if angle < min_ang:
            return False
    return True


def _camera_clear(camera: Mapping[str, Any], polygon: np.ndarray, colliders: Sequence[Mapping[str, Any]], clearance: float) -> bool:
    p = np.asarray(camera["position"], dtype=np.float64)
    if not _point_in_polygon(p[:2], polygon):
        return False
    return all(float(_point_distance_to_collider(p, body)) >= clearance for body in colliders)


def _clamp_target_pitch(position: Sequence[float], target: Sequence[float], pitch_limit_deg: float) -> list[float]:
    p = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64).copy()
    horizontal = float(np.linalg.norm((t - p)[:2]))
    max_dz = math.tan(math.radians(pitch_limit_deg)) * horizontal
    t[2] = float(np.clip(t[2], p[2] - max_dz, p[2] + max_dz))
    return t.astype(float).tolist()


def _shift_xy_toward_room_center(position: Sequence[float], center: Sequence[float], fraction: float) -> np.ndarray:
    """Move an XY camera sample inward by a deterministic fraction of its center vector."""
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("inward fraction must be in [0, 1)")
    p = np.asarray(position[:2], dtype=np.float64)
    c = np.asarray(center[:2], dtype=np.float64)
    return p + float(fraction) * (c - p)


def _characteristic_object_records(scene: Mapping[str, Any], minimum_size: float) -> list[Dict[str, Any]]:
    points_by_id = scaffold_points(scene)
    # Generic object/room-surface distinction comes only from declarative generation.mode.
    # Never route on names or semantic classes. Room surface_texture and group records are not
    # valid close-up/orbit targets; asset_3d/scaffold_only/external_asset objects are.
    allowed_ids = {
        str(record["object_id"])
        for record in flat_objects(scene)
        if str(dict(record.get("generation", {})).get("mode", "")) not in {"", "group", "surface_texture"}
    }
    result: list[Dict[str, Any]] = []
    for object_id, points in points_by_id.items():
        if str(object_id) not in allowed_ids:
            continue
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            continue
        minimum = pts.min(axis=0)
        maximum = pts.max(axis=0)
        extent = maximum - minimum
        characteristic = float(np.linalg.norm(extent))
        if characteristic < minimum_size:
            continue
        result.append({
            "object_id": str(object_id),
            "center": (0.5 * (minimum + maximum)).astype(float),
            "extent": extent.astype(float),
            "characteristic_size": characteristic,
            "volume_proxy": float(np.prod(np.maximum(extent, 1e-3))),
        })
    return sorted(result, key=lambda x: (-x["volume_proxy"], -x["characteristic_size"], x["object_id"]))


def _feasible_object_arc(
    record: Mapping[str, Any], *, room_center: np.ndarray, polygon: np.ndarray,
    colliders: Sequence[Mapping[str, Any]], camera_config: Mapping[str, Any], height: float,
    pitch_limit: float, clearance: float, radius: float, half_arcs: Sequence[float],
    training: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], prefix: str, frames: int,
) -> tuple[list[Dict[str, Any]], float] | None:
    center3 = np.asarray(record["center"], dtype=np.float64)
    inward = room_center - center3[:2]
    if float(np.linalg.norm(inward)) < 1e-9:
        inward = np.array([1.0, 0.0], dtype=np.float64)
    base = math.atan2(float(inward[1]), float(inward[0]))
    for half_arc in half_arcs:
        angles = np.linspace(-float(half_arc), float(half_arc), max(int(frames), 2))
        cameras: list[Dict[str, Any]] = []
        good = True
        for idx, angle_deg in enumerate(angles):
            a = base + math.radians(float(angle_deg))
            pos = [float(center3[0] + radius * math.cos(a)), float(center3[1] + radius * math.sin(a)), float(height)]
            target = _clamp_target_pitch(pos, center3.tolist(), pitch_limit)
            camera = _make_camera(f"{prefix}_{idx:02d}", "evaluation_object_rotation", pos, target, camera_config, source="stage09e_object_rotation")
            camera.update({
                "evaluation_type": "object_rotation",
                "target_object_id": str(record["object_id"]),
                "target_object_center_world": center3.astype(float).tolist(),
                "orbit_angle_deg": float(angle_deg),
                "frame_index": idx,
            })
            if abs(_pitch_deg(pos, target)) > pitch_limit + 1e-6 or not _camera_clear(camera, polygon, colliders, clearance) or not _pose_far_from_training(camera, training, protocol):
                good = False
                break
            cameras.append(camera)
        if good:
            return cameras, float(half_arc)
    return None


def _contact_sheet(paths: Sequence[tuple[str, Path]], output: Path, thumb=(320, 180), columns=4) -> None:
    if not paths:
        return
    rows = int(math.ceil(len(paths) / columns))
    label_h = 26
    canvas = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(paths):
        try:
            image = Image.open(path).convert("RGB")
            image = ImageOps.fit(image, thumb)
        except Exception:
            image = Image.new("RGB", thumb, "black")
        x = (i % columns) * thumb[0]
        y = (i // columns) * (thumb[1] + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 5, y + thumb[1] + 4), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def collect_evaluation_candidates(
    out: Path, scene: Mapping[str, Any], camera_config: Mapping[str, Any], refinement_config: Mapping[str, Any], evaluation_config: Mapping[str, Any]
) -> Dict[str, Any]:
    stage = out / str(evaluation_config.get("stage_root", "09e_evaluation"))
    candidates_root = stage / "A_candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    protocol = dict(evaluation_config.get("camera_protocol", {}))
    stage07_payload = load_json(out / "07_refinement_cameras" / "cameras.accepted.json")
    training_cameras = list(stage07_payload.get("cameras", []))
    if not training_cameras:
        raise RuntimeError("Stage09E-A requires Stage07 accepted cameras")
    context = _build_context(out, scene, camera_config)
    cfg = dict(camera_config.get("reconstruction_sampling", {}))
    room_model = build_room_surface_model(scene, cfg)
    polygon = room_floor_polygon(room_model)
    center = room_center_xy(room_model)
    floor_z = floor_height(room_model)
    layout = dict(cfg.get("worldmesh_base_layout", {}))
    eye = floor_z + float(layout.get("camera_height_m", 1.6))
    overhead = eye + float(layout.get("overhead_height_offset_m", 0.8))
    pitch_limit = _stage07_pitch_limit(training_cameras)
    clearance = float(protocol.get("minimum_camera_clearance_m", layout.get("collision_padding_m", 0.3)))
    wall_offset = float(layout.get("perimeter_wall_offset_m", 0.4))
    candidates: list[Dict[str, Any]] = []

    # 1) Short trajectories: keep the deterministic phase structure of the Stage07-style
    # perimeter path, but move the complete path inward before evaluation.  A raw perimeter
    # path over-emphasises room corners / near-wall foreground and can turn a normal navigation
    # metric into a boundary stress test.  The inward shift is geometry-only, scene-generic and
    # preserves a smooth, repeatable novel trajectory without selecting on Gaussian quality.
    trajectory_count = int(protocol.get("short_trajectory_count", 3))
    trajectory_frames = int(protocol.get("short_trajectory_frames", 5))
    dense_count = max(int(protocol.get("perimeter_dense_sample_count", 72)), trajectory_count * trajectory_frames * 2)
    height_fracs = list(protocol.get("height_fractions_between_eye_and_overhead", [0.125, 0.3125, 0.5, 0.6875, 0.875]))
    short_inward_fraction = float(protocol.get("short_trajectory_inward_fraction", 0.35))
    if not 0.0 <= short_inward_fraction < 1.0:
        raise ValueError("short_trajectory_inward_fraction must be in [0, 1)")
    short_clearance = max(
        clearance,
        float(protocol.get("short_trajectory_minimum_object_clearance_m", 0.60)),
    )
    dense = _sample_perimeter_positions(polygon, dense_count, wall_offset, eye, center, eye)
    preferred_centers = [int(round((k + 0.5) * dense_count / trajectory_count)) % dense_count for k in range(trajectory_count)]
    half = trajectory_frames // 2
    used_centers: list[int] = []
    for tidx, preferred in enumerate(preferred_centers):
        selected_clip = None
        # Deterministically search near each phase-offset anchor until a complete safe 5-frame
        # trajectory is found. Never publish fragmented trajectories.
        offsets = [0]
        for delta in range(1, dense_count // 2 + 1): offsets.extend([delta, -delta])
        for offset in offsets:
            cidx = (preferred + offset) % dense_count
            if any(min((cidx-u)%dense_count,(u-cidx)%dense_count) < trajectory_frames for u in used_centers):
                continue
            clip=[]
            for local in range(trajectory_frames):
                index = (cidx + local - half) % dense_count
                base_pos, base_target = dense[index]
                frac = float(height_fracs[local % len(height_fracs)])
                z = eye + frac * (overhead - eye)
                interior_xy = _shift_xy_toward_room_center(base_pos, center, short_inward_fraction)
                pos = [float(interior_xy[0]), float(interior_xy[1]), float(z)]
                target = _clamp_target_pitch(pos, base_target, pitch_limit)
                cam = _make_camera(f"eval_short_{tidx:02d}_{local:02d}", "evaluation_short_trajectory", pos, target, cfg, source="stage09e_short_trajectory")
                cam.update({"evaluation_type":"short_trajectory", "trajectory_id":f"short_trajectory_{tidx:02d}", "frame_index":local})
                if not (_camera_clear(cam, polygon, context["object_boxes"], short_clearance) and _pose_far_from_training(cam, training_cameras, protocol)):
                    clip=[]; break
                clip.append(cam)
            if len(clip)==trajectory_frames:
                selected_clip=clip; used_centers.append(cidx); break
        if selected_clip:
            candidates.extend(selected_clip)

    # Generic object records used by close-up and rotation; no semantic-class routing.
    object_records = _characteristic_object_records(scene, float(protocol.get("object_minimum_characteristic_size_m", 0.15)))
    room_diag = float(np.linalg.norm(polygon.max(axis=0) - polygon.min(axis=0)))
    radius_max = float(protocol.get("object_radius_max_fraction_room_diagonal", 0.45)) * room_diag
    radius_min = float(protocol.get("object_radius_min_m", 0.65))
    v_fov = math.radians(float(dict(cfg.get("camera_model", {})).get("vertical_fov_degrees", 60.0)))
    framing_margin = float(protocol.get("object_closeup_framing_margin", 1.25))
    chosen_closeup: list[str] = []
    chosen_rotation: list[str] = []

    # 2) Close-up views: two safe views around each of the first feasible generic objects.
    closeup_needed = int(protocol.get("closeup_object_count", 3))
    closeup_views = int(protocol.get("closeup_views_per_object", 2))
    closeup_z = eye + float(protocol.get("closeup_height_fraction", 0.35)) * (overhead - eye)
    for record in object_records:
        if len(chosen_closeup) >= closeup_needed:
            break
        center3 = np.asarray(record["center"], dtype=np.float64)
        radius = np.clip((0.5 * float(record["characteristic_size"]) / max(math.tan(v_fov/2), 1e-6)) * framing_margin, radius_min, radius_max)
        inward = center - center3[:2]
        if np.linalg.norm(inward) < 1e-9: inward = np.array([1.0,0.0])
        base = math.atan2(float(inward[1]), float(inward[0]))
        cams=[]
        for idx, angle_deg in enumerate(np.linspace(-18.0, 18.0, closeup_views)):
            a=base+math.radians(float(angle_deg))
            pos=[float(center3[0]+radius*math.cos(a)),float(center3[1]+radius*math.sin(a)),float(closeup_z)]
            target=_clamp_target_pitch(pos, center3.tolist(), pitch_limit)
            cam=_make_camera(f"eval_closeup_{len(chosen_closeup):02d}_{idx:02d}","evaluation_closeup",pos,target,cfg,source="stage09e_closeup")
            cam.update({"evaluation_type":"close_up","target_object_id":record["object_id"],"target_object_center_world":center3.tolist(),"frame_index":idx})
            if not (_camera_clear(cam, polygon, context["object_boxes"], clearance) and _pose_far_from_training(cam, training_cameras, protocol)):
                cams=[]; break
            cams.append(cam)
        if cams:
            chosen_closeup.append(str(record["object_id"])); candidates.extend(cams)

    # 3) Object rotations: explicit target_object_id and safe partial arcs, never forced 360 degrees.
    rotation_needed = int(protocol.get("rotation_object_count", 3))
    rotation_frames = int(protocol.get("rotation_frames", 5))
    rotation_z = eye + float(protocol.get("rotation_height_fraction", 0.35)) * (overhead - eye)
    half_arcs = [float(protocol.get("rotation_nominal_half_arc_degrees", 40.0)), *[float(v) for v in protocol.get("rotation_fallback_half_arcs_degrees", [30.0,20.0])]]
    for record in object_records:
        if len(chosen_rotation) >= rotation_needed:
            break
        center3=np.asarray(record["center"],dtype=np.float64)
        radius=np.clip((0.5*float(record["characteristic_size"])/max(math.tan(v_fov/2),1e-6))*1.55,radius_min,radius_max)
        prefix=f"eval_rotation_{len(chosen_rotation):02d}"
        feasible=_feasible_object_arc(record,room_center=center,polygon=polygon,colliders=context["object_boxes"],camera_config=cfg,height=rotation_z,pitch_limit=pitch_limit,clearance=clearance,radius=float(radius),half_arcs=half_arcs,training=training_cameras,protocol=protocol,prefix=prefix,frames=rotation_frames)
        if feasible is None: continue
        cams, half_arc=feasible
        trajectory_id=f"object_rotation_{len(chosen_rotation):02d}"
        for cam in cams:
            cam["trajectory_id"]=trajectory_id; cam["selected_half_arc_deg"]=half_arc
        chosen_rotation.append(str(record["object_id"])); candidates.extend(cams)

    # Render geometry-only previews using the exact Stage07 shared-buffer renderer.
    active_owner_manifest = out / "07_refinement_cameras" / "active_owner_ids.json"
    render_report = _render_final_buffers(out, candidates, candidates_root, refinement_config, active_owner_manifest)
    shared_root = candidates_root / "shared_buffers"
    for cam in candidates:
        cid=str(cam["camera_id"])
        report_path=shared_root/cid/"camera.json"
        if report_path.exists(): cam["rendered_camera"]=str(report_path)
        cam["mesh_rgb"]=str(shared_root/cid/"rgb.png")
        cam["mesh_depth_control"]=str(shared_root/cid/"depth_control.png")
        cam["mesh_normal_world"]=str(shared_root/cid/"normal_world.png")
        cam["triangle_id"]=str(shared_root/cid/"triangle_id.png")

    training_manifest = load_json(out / "08_viewwise_refinement" / "stage09_training_manifest.json")
    replay = [{"camera_id":f["camera_id"],"camera_role":f.get("camera_role"),"target_rgb":f["target_rgb"],"camera":f["camera"],"evaluation_type":"training_replay"} for f in training_manifest.get("frames",[])]
    manifest={
        "schema_version":1,"scene_id":out.name,"geometry_only_candidate_generation":True,"gaussian_output_used_for_curation":False,
        "camera_height_range_m":[eye,overhead],"maximum_absolute_pitch_deg":pitch_limit,
        "short_trajectory_sampling_policy":"inward_shifted_perimeter_phase",
        "short_trajectory_inward_fraction":short_inward_fraction,
        "short_trajectory_minimum_object_clearance_m":short_clearance,
        "novel_candidate_count":len(candidates),"novel_candidates":candidates,
        "training_replay_count":len(replay),"training_replay":replay,
        "selected_closeup_object_ids":chosen_closeup,"selected_rotation_object_ids":chosen_rotation,
        "render_report":render_report,
    }
    save_json(manifest,candidates_root/"candidate_manifest.json")
    with (stage/"selection.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["camera_id","evaluation_type","trajectory_id","target_object_id","frame_index","keep","reject_reason"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for cam in candidates:
            w.writerow({"camera_id":cam["camera_id"],"evaluation_type":cam.get("evaluation_type",""),"trajectory_id":cam.get("trajectory_id",""),"target_object_id":cam.get("target_object_id",""),"frame_index":cam.get("frame_index",""),"keep":1,"reject_reason":""})
    _contact_sheet([(str(c["camera_id"]),Path(c["mesh_rgb"])) for c in candidates],stage/"candidate_contact_sheet_rgb.png")
    _contact_sheet([(str(c["camera_id"]),Path(c["mesh_depth_control"])) for c in candidates],stage/"candidate_contact_sheet_depth.png")
    save_json({"status":"ok","candidate_manifest":str(candidates_root/"candidate_manifest.json"),"selection_csv":str(stage/"selection.csv"),"candidate_count":len(candidates),"training_replay_count":len(replay)},stage/"A_stage_report.json")
    return manifest


def freeze_evaluation_manifest(stage: Path) -> Dict[str, Any]:
    source=load_json(stage/"A_candidates"/"candidate_manifest.json")
    rows={}
    with (stage/"selection.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f): rows[str(row["camera_id"])]=row
    frozen=[]; rejected=[]
    for cam in source.get("novel_candidates",[]):
        row=rows.get(str(cam["camera_id"]))
        if row is None: raise RuntimeError(f"selection.csv is missing {cam['camera_id']}")
        keep=str(row.get("keep","1")).strip().lower() in {"1","true","yes","y","keep"}
        if keep:
            frozen.append(cam)
        else:
            rejected.append({"camera_id":str(cam["camera_id"]),"reject_reason":str(row.get("reject_reason","")).strip()})
    # Pairs only between originally consecutive retained frames; never bridge a manually removed gap.
    by_traj: Dict[str,list[Dict[str,Any]]]={}
    for cam in frozen:
        tid=str(cam.get("trajectory_id", ""))
        if tid and cam.get("evaluation_type") in {"short_trajectory","object_rotation"}: by_traj.setdefault(tid,[]).append(cam)
    pairs=[]
    for tid, cams in sorted(by_traj.items()):
        cams=sorted(cams,key=lambda c:int(c.get("frame_index",0)))
        for a,b in zip(cams,cams[1:]):
            if int(b.get("frame_index",0))-int(a.get("frame_index",0))==1:
                pairs.append({"pair_id":f"{a['camera_id']}__{b['camera_id']}","trajectory_id":tid,"evaluation_type":a.get("evaluation_type"),"source_camera_id":a["camera_id"],"target_camera_id":b["camera_id"]})
    selection_bytes=(stage/"selection.csv").read_bytes()
    frozen_manifest={"schema_version":1,"scene_id":source.get("scene_id"),"novel_views":frozen,"novel_view_count":len(frozen),"rejected_novel_views":rejected,"rejected_novel_view_count":len(rejected),"training_replay":source.get("training_replay",[]),"training_replay_count":source.get("training_replay_count",0),"reprojection_pairs":pairs,"reprojection_pair_count":len(pairs),"curation_policy":"geometry_only_manual_validity_screening_before_any_gaussian_render","selection_csv_sha256":hashlib.sha256(selection_bytes).hexdigest()}
    out=stage/"B_frozen_evaluation_manifest.json"; save_json(frozen_manifest,out)
    digest=hashlib.sha256(out.read_bytes()).hexdigest(); (stage/"B_frozen_evaluation_manifest.sha256").write_text(digest+"\n",encoding="utf-8")
    return frozen_manifest
