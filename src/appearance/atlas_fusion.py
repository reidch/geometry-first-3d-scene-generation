from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.appearance.atlas_maps import decode_objects, load_uv, valid_uv_mask
from src.appearance.triangle_id_map import load_triangle_id_map


def _srgb_to_linear(rgb):
    x=np.clip(np.asarray(rgb,np.float32)/255.0,0.0,1.0)
    return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4).astype(np.float32)


def _linear_to_srgb(rgb):
    x=np.clip(np.asarray(rgb,np.float32),0.0,1.0)
    y=np.where(x<=0.0031308,12.92*x,1.055*np.power(x,1.0/2.4)-0.055)
    return np.clip(y*255.0,0.0,255.0).astype(np.float32)


def _luminance(rgb):
    x=np.asarray(rgb,np.float32)
    return 0.2126*x[...,0]+0.7152*x[...,1]+0.0722*x[...,2]


def _masked_box_blur(values,mask,radius=2):
    v=np.asarray(values,np.float32);m=np.asarray(mask,bool)
    h,w=v.shape
    total=np.zeros((h,w),np.float32);count=np.zeros((h,w),np.float32)
    r=max(1,int(radius))
    for dy in range(-r,r+1):
        ys0=max(0,-dy);ys1=min(h,h-dy);yd0=max(0,dy);yd1=min(h,h+dy)
        for dx in range(-r,r+1):
            xs0=max(0,-dx);xs1=min(w,w-dx);xd0=max(0,dx);xd1=min(w,w+dx)
            valid=m[ys0:ys1,xs0:xs1]
            total[yd0:yd1,xd0:xd1]+=v[ys0:ys1,xs0:xs1]*valid
            count[yd0:yd1,xd0:xd1]+=valid
    return np.where(count>0,total/np.maximum(count,1e-6),v).astype(np.float32)


def _normalized_hf_variance(rgb_linear,mask,blur_radius=2):
    mask=np.asarray(mask,bool)
    if int(mask.sum())<16:
        return 0.0, {'valid_pixels':int(mask.sum()),'reason':'too_few_pixels'}
    y=_luminance(rgb_linear)
    low=_masked_box_blur(y,mask,radius=blur_radius)
    residual=(y-low)[mask]
    clip=float(np.percentile(np.abs(residual),95.0)) if residual.size else 0.0
    if clip>0:
        residual=np.clip(residual,-clip,clip)
    mean_y=float(np.mean(y[mask]))
    variance=float(np.var(residual)/(mean_y*mean_y+1e-6))
    return variance,{
        'valid_pixels':int(mask.sum()),
        'mean_luminance_linear':mean_y,
        'residual_clip_p95':clip,
        'normalized_high_frequency_variance':variance,
        'blur_radius':int(blur_radius),
    }


def _triangle_area(points):
    p=np.asarray(points,np.float32)
    return 0.5*abs(float((p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[2,0]-p[0,0])*(p[1,1]-p[0,1])))


def _barycentric(px,py,tri):
    x0,y0=tri[0];x1,y1=tri[1];x2,y2=tri[2]
    den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
    if abs(float(den))<1e-12:
        return None
    l0=((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den
    l1=((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den
    return l0,l1,1.0-l0-l1


def rasterize_uv_triangle_mask(
    triangles,
    resolution,
    *,
    triangle_ids=None,
    conservative_barycentric_epsilon=0.0025,
):
    """Rasterize selected manifest triangles into atlas UV space.

    This uses the exact same texel-center and barycentric convention as
    :func:`fuse_view`, so Stage06 can prove that the complete canonical
    room-facing surface was written rather than merely checking that some texels
    changed.
    """
    res = int(resolution)
    mask = np.zeros((res, res), dtype=bool)
    selected = None if triangle_ids is None else {int(value) for value in triangle_ids}
    eps = float(max(0.0, conservative_barycentric_epsilon))
    for tri in triangles:
        triangle_id = int(tri.get('global_triangle_id', -1))
        if selected is not None and triangle_id not in selected:
            continue
        uv = np.asarray(tri['uv'], np.float32)
        uv_px = np.stack([uv[:, 0] * (res - 1), (1.0 - uv[:, 1]) * (res - 1)], axis=1)
        if _triangle_area(uv_px) < 1e-5:
            continue
        x0 = max(0, int(np.floor(uv_px[:, 0].min())))
        x1 = min(res - 1, int(np.ceil(uv_px[:, 0].max())))
        y0 = max(0, int(np.floor(uv_px[:, 1].min())))
        y1 = min(res - 1, int(np.ceil(uv_px[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        bary = _barycentric(gx, gy, uv_px)
        if bary is None:
            continue
        l0, l1, l2 = bary
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if inside.any():
            mask[y0 : y1 + 1, x0 : x1 + 1] |= inside
    return mask


def _bilinear_sample(image,x,y):
    h,w=image.shape[:2]
    x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
    x0=np.floor(x).astype(np.int32);y0=np.floor(y).astype(np.int32)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    fx=(x-x0).astype(np.float32);fy=(y-y0).astype(np.float32)
    c00=image[y0,x0];c10=image[y0,x1];c01=image[y1,x0];c11=image[y1,x1]
    return ((1-fx)*(1-fy))[:,None]*c00+(fx*(1-fy))[:,None]*c10+((1-fx)*fy)[:,None]*c01+(fx*fy)[:,None]*c11


def _supersampled_color(image,x,y,radius=0.35):
    offsets=((-radius,-radius),(radius,-radius),(-radius,radius),(radius,radius))
    return np.mean([_bilinear_sample(image,x+dx,y+dy) for dx,dy in offsets],axis=0)


def _variance_support(new_variance,current_variance,relative_change_saturation=0.5,denominator_floor=1e-5):
    vn=max(0.0,float(new_variance));vc=max(0.0,float(current_variance))
    sat=max(float(relative_change_saturation),1e-8)
    ratio=(vn-vc)/max(vc,float(denominator_floor)) if not (vc<=float(denominator_floor) and vn<=float(denominator_floor)) else 0.0
    if ratio<=-sat:
        support=0.0
    elif ratio>=sat:
        support=1.0
    else:
        support=0.5*(1.0+ratio/sat)
    return float(np.clip(support,0.0,1.0)),{
        'new_variance':vn,
        'current_variance':vc,
        'relative_variance_change':float(ratio),
        'relative_change_saturation':sat,
        'support':float(np.clip(support,0.0,1.0)),
        'formula':'piecewise linear support: 0 at r<=-sat, 0.5 at r=0, 1 at r>=sat',
    }


def _threshold_support(ratio,threshold):
    thr=max(float(threshold),1e-8)
    support=float(np.clip(float(ratio)/thr,0.0,1.0))
    return support,{'ratio':float(ratio),'threshold':thr,'support':support,'formula':'clip(ratio/threshold,0,1)'}


def _weighted_geometric_mean(values,weights,eps=1e-8):
    vals=np.asarray(values,np.float64); w=np.asarray(weights,np.float64)
    sw=float(w.sum())
    if sw<=0: return 0.0
    w=w/sw
    if np.any(vals<=0):
        return 0.0
    return float(np.exp(np.sum(w*np.log(np.clip(vals,eps,1.0)))))



def _manifest_reachable_mask(manifest, resolution, conservative_barycentric_epsilon):
    return rasterize_uv_triangle_mask(
        manifest.get("triangles", []),
        int(resolution),
        conservative_barycentric_epsilon=float(conservative_barycentric_epsilon),
    )


def _fill_small_resampling_gaps(observation, hit, allowed, iterations):
    """Fill only tiny holes caused by screen/atlas sampling-rate differences.

    This is a local resampling operation, not an atlas-coverage completion policy.
    It never expands outside UV triangles that were actually visible in the
    Triangle-ID pass.
    """
    observation = np.asarray(observation, dtype=np.float32).copy()
    hit = np.asarray(hit, dtype=bool).copy()
    allowed = np.asarray(allowed, dtype=bool)
    filled_total = 0
    for _ in range(max(0, int(iterations))):
        accum = np.zeros_like(observation)
        count = np.zeros(hit.shape, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            src_y0 = max(0, -dy)
            src_y1 = min(hit.shape[0], hit.shape[0] - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(hit.shape[1], hit.shape[1] - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(hit.shape[0], hit.shape[0] + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(hit.shape[1], hit.shape[1] + dx)
            source_hit = hit[src_y0:src_y1, src_x0:src_x1]
            accum[dst_y0:dst_y1, dst_x0:dst_x1] += (
                observation[src_y0:src_y1, src_x0:src_x1] * source_hit[..., None]
            )
            count[dst_y0:dst_y1, dst_x0:dst_x1] += source_hit
        fill = allowed & (~hit) & (count > 0)
        if not fill.any():
            break
        observation[fill] = accum[fill] / count[fill, None]
        hit[fill] = True
        filled_total += int(fill.sum())
    return observation, hit, filled_total


def _screen_uv_triangle_observation(
    rgb,
    semantic_mask,
    triangle_id_map,
    uv_map_path,
    manifest,
    resolution,
    runtime_reachable,
    *,
    gap_fill_iterations=2,
):
    """Bake visible pixels through Blender's exact UV and Triangle-ID buffers.

    Every accepted source pixel must simultaneously belong to the target semantic,
    carry a valid active UV, and carry a Triangle ID listed in the target manifest.
    This avoids reconstructing the screen projection a second time on the host.
    """
    if triangle_id_map is None:
        raise ValueError("Triangle-ID buffer is required for screen-UV triangle writeback")
    width, height, u, v = load_uv(uv_map_path)
    if (height, width) != semantic_mask.shape:
        raise RuntimeError(
            f"UV map shape {(height, width)} does not match generated image {semantic_mask.shape}"
        )
    if triangle_id_map.shape != semantic_mask.shape:
        raise RuntimeError(
            f"Triangle-ID map shape {triangle_id_map.shape} does not match generated image {semantic_mask.shape}"
        )

    triangles = list(manifest.get("triangles", []))
    render_to_global = {
        int(tri.get("render_triangle_id", tri.get("global_triangle_id", -1))):
        int(tri.get("global_triangle_id", -1))
        for tri in triangles
        if int(tri.get("global_triangle_id", -1)) >= 0
    }
    render_ids = np.asarray(sorted(render_to_global), dtype=np.int32)
    valid = np.asarray(semantic_mask, dtype=bool) & valid_uv_mask(u, v) & (triangle_id_map >= 0)
    if render_ids.size:
        valid &= np.isin(triangle_id_map, render_ids)
    else:
        valid &= False

    source_y, source_x = np.where(valid)
    res = int(resolution)
    obs_sum = np.zeros((res, res, 3), dtype=np.float32)
    obs_weight = np.zeros((res, res), dtype=np.float32)
    if source_x.size:
        atlas_x = np.clip(u[source_y, source_x] * (res - 1), 0.0, res - 1.0)
        atlas_y = np.clip((1.0 - v[source_y, source_x]) * (res - 1), 0.0, res - 1.0)
        x0 = np.floor(atlas_x).astype(np.int32)
        y0 = np.floor(atlas_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, res - 1)
        y1 = np.minimum(y0 + 1, res - 1)
        fx = (atlas_x - x0).astype(np.float32)
        fy = (atlas_y - y0).astype(np.float32)
        samples = rgb[source_y, source_x]
        for dst_x, dst_y, weight in (
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x1, y0, fx * (1.0 - fy)),
            (x0, y1, (1.0 - fx) * fy),
            (x1, y1, fx * fy),
        ):
            positive = weight > 1e-8
            if not positive.any():
                continue
            np.add.at(obs_weight, (dst_y[positive], dst_x[positive]), weight[positive])
            for channel in range(3):
                np.add.at(
                    obs_sum[..., channel],
                    (dst_y[positive], dst_x[positive]),
                    samples[positive, channel] * weight[positive],
                )

    raw_hit = obs_weight > 1e-8
    observed_render_ids = set(int(value) for value in np.unique(triangle_id_map[valid]).tolist())
    observed_global_ids = {
        int(render_to_global[value]) for value in observed_render_ids if value in render_to_global
    }
    visible_triangle_mask = rasterize_uv_triangle_mask(
        triangles,
        res,
        triangle_ids=observed_global_ids,
    )
    # Exact Blender UV samples are authoritative even when a texel-center
    # raster of a very small UV triangle would miss that sample.  The runtime
    # triangle mask is used to constrain only local resampling-gap expansion.
    allowed = visible_triangle_mask | raw_hit
    discarded_outside_runtime = int((raw_hit & (~runtime_reachable)).sum())
    hit = raw_hit.copy()
    observation = np.zeros_like(obs_sum)
    observation[hit] = obs_sum[hit] / np.maximum(obs_weight[hit, None], 1e-8)
    observation, hit, gap_filled = _fill_small_resampling_gaps(
        observation,
        hit,
        allowed,
        gap_fill_iterations,
    )
    obs_weight[hit & (obs_weight <= 1e-8)] = 1.0

    world_area_by_global = {
        int(tri.get("global_triangle_id", -1)): max(0.0, float(tri.get("world_area", 0.0)))
        for tri in triangles
    }
    projectable_surface_area = float(sum(world_area_by_global.values()))
    visible_surface_area = float(sum(world_area_by_global.get(value, 0.0) for value in observed_global_ids))
    return {
        "observation": observation,
        "hit": hit,
        "sample_weight": obs_weight,
        "tested": int(valid.sum()),
        "visible": int(valid.sum()),
        "projectable_surface_area": projectable_surface_area,
        "visible_surface_area": visible_surface_area,
        "observed_triangle_ids": observed_global_ids,
        "diagnostics": {
            "mode": "triangle_id_grouped_screen_uv",
            "source_semantic_uv_triangle_pixels": int(valid.sum()),
            "raw_splatted_texels": int(raw_hit.sum()),
            "visible_triangle_uv_texels": int(visible_triangle_mask.sum()),
            "gap_filled_resampling_texels": int(gap_filled),
            "discarded_outside_runtime_manifest_uv": discarded_outside_runtime,
            "gap_fill_iterations": int(max(0, gap_fill_iterations)),
        },
    }


def _clip_space_triangle_observation(
    rgb,
    semantic_mask,
    triangle_id_map,
    manifest,
    resolution,
    runtime_reachable,
    *,
    supersample_radius,
    conservative_barycentric_epsilon,
):
    """Legacy host-side clip reconstruction retained as a compatibility path."""
    h, w = rgb.shape[:2]
    res = int(resolution)
    obs_sum = np.zeros((res, res, 3), np.float32)
    obs_count = np.zeros((res, res), np.float32)
    tested = visible = 0
    projectable_surface_area = 0.0
    visible_surface_area = 0.0
    observed_triangle_ids = set()
    for tri in manifest.get("triangles", []):
        uv = np.asarray(tri["uv"], np.float32)
        clip = np.asarray(tri["clip"], np.float32)
        world_area = float(tri.get("world_area", 0.0))
        uv_px = np.stack([uv[:, 0] * (res - 1), (1.0 - uv[:, 1]) * (res - 1)], axis=1)
        if _triangle_area(uv_px) < 1e-5:
            continue
        x0 = max(0, int(np.floor(uv_px[:, 0].min())))
        x1 = min(res - 1, int(np.ceil(uv_px[:, 0].max())))
        y0 = max(0, int(np.floor(uv_px[:, 1].min())))
        y1 = min(res - 1, int(np.ceil(uv_px[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        bary = _barycentric(gx, gy, uv_px)
        if bary is None:
            continue
        l0, l1, l2 = bary
        eps = float(max(0.0, conservative_barycentric_epsilon))
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not inside.any():
            continue
        yy, xx = np.where(inside)
        lam = np.stack([l0[inside], l1[inside], l2[inside]], axis=1)
        c = lam @ clip
        good = np.abs(c[:, 3]) > 1e-8
        if not good.any():
            continue
        yy = yy[good]
        xx = xx[good]
        c = c[good]
        ndc = c[:, :2] / c[:, 3:4]
        sx = (ndc[:, 0] * 0.5 + 0.5) * (w - 1)
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)
        in_frame = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        if not in_frame.any():
            continue
        projectable_surface_area += max(world_area, 0.0)
        yy = yy[in_frame]
        xx = xx[in_frame]
        sx = sx[in_frame]
        sy = sy[in_frame]
        ix = np.clip(np.rint(sx).astype(np.int32), 0, w - 1)
        iy = np.clip(np.rint(sy).astype(np.int32), 0, h - 1)
        vis = semantic_mask[iy, ix]
        if triangle_id_map is not None:
            render_triangle_id = int(tri.get('render_triangle_id', tri.get('global_triangle_id', -1)))
            vis &= triangle_id_map[iy, ix] == render_triangle_id
        tested += len(vis)
        if not vis.any():
            continue
        observed_triangle_ids.add(int(tri.get('global_triangle_id', -1)))
        visible_surface_area += max(world_area, 0.0)
        yy = yy[vis] + y0
        xx = xx[vis] + x0
        sx = sx[vis]
        sy = sy[vis]
        visible += len(xx)
        sampled = _supersampled_color(rgb, sx, sy, float(supersample_radius))
        np.add.at(obs_count, (yy, xx), 1.0)
        for channel in range(3):
            np.add.at(obs_sum[..., channel], (yy, xx), sampled[:, channel])
    hit = (obs_count > 0) & runtime_reachable
    observation = np.zeros_like(obs_sum)
    observation[hit] = obs_sum[hit] / np.maximum(obs_count[hit, None], 1e-8)
    return {
        "observation": observation,
        "hit": hit,
        "sample_weight": obs_count,
        "tested": int(tested),
        "visible": int(visible),
        "projectable_surface_area": float(projectable_surface_area),
        "visible_surface_area": float(visible_surface_area),
        "observed_triangle_ids": observed_triangle_ids,
        "diagnostics": {
            "mode": "host_clip_space_triangle_projection",
            "raw_projected_texels": int((obs_count > 0).sum()),
            "discarded_outside_runtime_manifest_uv": int(((obs_count > 0) & (~runtime_reachable)).sum()),
        },
    }


def fuse_view(
    image_path,
    semantic_path,
    palette_path,
    triangles_path,
    atlases,
    valid_mask_path=None,
    supersample_radius=0.35,
    variance_blur_radius=2,
    variance_relative_change_saturation=0.5,
    variance_denominator_floor=1e-5,
    visible_area_saturation_threshold=0.5,
    screen_area_saturation_threshold=0.05,
    weight_variance=0.4,
    weight_visible_area=0.35,
    weight_screen_area=0.25,
    alpha_override=None,
    conservative_barycentric_epsilon=0.0025,
    triangle_id_path=None,
    uv_map_path=None,
    screen_uv_gap_fill_iterations=2,
    observation_mask_output_path=None,
):
    """Write a generated view into an object-owned atlas.

    When Blender UV and Triangle-ID buffers are available, they are the
    authoritative projection. Pixels are grouped by Triangle ID and mapped using
    the exact active UV rendered by Blender. The older host clip reconstruction is
    retained only for caches that do not provide a UV buffer.
    """
    rgb = _srgb_to_linear(np.asarray(Image.open(image_path).convert("RGB"), np.float32))
    h, w = rgb.shape[:2]
    obj_idx, names = decode_objects(semantic_path, palette_path)
    write_mask = np.ones((h, w), bool) if not valid_mask_path else np.asarray(
        Image.open(valid_mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    ) > 0
    triangle_id_map = load_triangle_id_map(triangle_id_path) if triangle_id_path else None
    if triangle_id_map is not None and triangle_id_map.shape != (h, w):
        raise RuntimeError(
            f"Triangle-id map shape {triangle_id_map.shape} does not match generated image {(h, w)}"
        )
    manifest = json.loads(Path(triangles_path).read_text(encoding="utf-8"))
    target = str(manifest["target_object"])
    if target not in atlases or target not in names:
        return {}
    semantic_mask = (obj_idx == names.index(target)) & write_mask
    atlas = atlases[target]
    color_u8 = atlas.load()
    color = _srgb_to_linear(color_u8)
    color_before = color.copy()
    stored_reachable = (
        np.asarray(Image.open(atlas.reachable_path).convert("L")) > 0
        if atlas.reachable_path.exists()
        else None
    )
    res = int(atlas.resolution)
    runtime_reachable = _manifest_reachable_mask(
        manifest,
        res,
        conservative_barycentric_epsilon,
    )
    if not runtime_reachable.any():
        raise RuntimeError(
            f"Target manifest for {target} contains no rasterizable active-UV triangles"
        )
    stored_overlap = None
    if stored_reachable is not None and stored_reachable.shape == runtime_reachable.shape:
        stored_overlap = int((stored_reachable & runtime_reachable).sum())

    if uv_map_path is not None:
        reachable_source = "current_runtime_triangle_manifest"
        projection = _screen_uv_triangle_observation(
            rgb,
            semantic_mask,
            triangle_id_map,
            uv_map_path,
            manifest,
            res,
            runtime_reachable,
            gap_fill_iterations=screen_uv_gap_fill_iterations,
        )
        if not projection["hit"].any():
            fallback = _clip_space_triangle_observation(
                rgb,
                semantic_mask,
                triangle_id_map,
                manifest,
                res,
                runtime_reachable,
                supersample_radius=supersample_radius,
                conservative_barycentric_epsilon=conservative_barycentric_epsilon,
            )
            fallback["diagnostics"]["screen_uv_attempt"] = projection["diagnostics"]
            projection = fallback
    else:
        # Legacy callers without a Blender UV buffer keep the historical stored
        # reachable-mask contract. New Stage06/08 calls always provide uv_map_path
        # and therefore use the current runtime manifest instead.
        use_stored_reachable = (
            stored_reachable is not None and stored_reachable.shape == runtime_reachable.shape
        )
        legacy_reachable = stored_reachable if use_stored_reachable else runtime_reachable
        reachable_source = (
            "stored_stage05_reachable" if use_stored_reachable
            else "current_runtime_triangle_manifest"
        )
        projection = _clip_space_triangle_observation(
            rgb,
            semantic_mask,
            triangle_id_map,
            manifest,
            res,
            legacy_reachable,
            supersample_radius=supersample_radius,
            conservative_barycentric_epsilon=conservative_barycentric_epsilon,
        )

    observation = projection["observation"]
    hit = projection["hit"]
    sample_weight = projection["sample_weight"]
    tested = int(projection["tested"])
    visible = int(projection["visible"])
    projectable_surface_area = float(projection["projectable_surface_area"])
    visible_surface_area = float(projection["visible_surface_area"])
    observed_triangle_ids = set(projection["observed_triangle_ids"])

    if observation_mask_output_path:
        output = Path(observation_mask_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((hit.astype(np.uint8) * 255), "L").save(output)

    current_variance, current_var_report = _normalized_hf_variance(
        color_before, hit, blur_radius=variance_blur_radius
    )
    new_variance, new_var_report = _normalized_hf_variance(
        observation, hit, blur_radius=variance_blur_radius
    )
    variance_support, variance_report = _variance_support(
        new_variance,
        current_variance,
        relative_change_saturation=variance_relative_change_saturation,
        denominator_floor=variance_denominator_floor,
    )
    visible_area_ratio = (
        float(visible_surface_area / max(projectable_surface_area, 1e-8))
        if projectable_surface_area > 0
        else 0.0
    )
    visible_area_support, visible_area_report = _threshold_support(
        visible_area_ratio, visible_area_saturation_threshold
    )
    screen_area_ratio = float(int(semantic_mask.sum()) / max(h * w, 1))
    screen_area_support, screen_area_report = _threshold_support(
        screen_area_ratio, screen_area_saturation_threshold
    )
    computed_alpha = _weighted_geometric_mean(
        [variance_support, visible_area_support, screen_area_support],
        [weight_variance, weight_visible_area, weight_screen_area],
    )
    alpha = float(
        np.clip(computed_alpha if alpha_override is None else float(alpha_override), 0.0, 1.0)
    )
    alpha_map = np.full((res, res), alpha, dtype=np.float32)
    if hit.any():
        a = alpha_map[hit, None]
        color[hit] = (1.0 - a) * color[hit] + a * observation[hit]
    atlas.save(_linear_to_srgb(color))

    support_report = {
        "variance_support": variance_report,
        "visible_area_support": visible_area_report,
        "screen_area_support": screen_area_report,
        "weights": {
            "variance": float(weight_variance),
            "visible_area": float(weight_visible_area),
            "screen_area": float(weight_screen_area),
        },
        "weighted_geometric_mean_alpha": float(computed_alpha),
        "alpha_override": None if alpha_override is None else float(alpha_override),
        "applied_alpha": alpha,
        "effective_alpha_mean": float(alpha_map[hit].mean()) if hit.any() else 0.0,
        "projectable_surface_area_world": projectable_surface_area,
        "visible_surface_area_world": visible_surface_area,
        "screen_object_pixel_ratio": screen_area_ratio,
        "formula": "alpha = weighted geometric mean of variance, visible-area, and screen-area supports",
    }
    outside_stored = None
    if stored_reachable is not None and stored_reachable.shape == runtime_reachable.shape:
        outside_stored = int((hit & (~stored_reachable)).sum())
    return {
        target: {
            "triangle_count": int(len(manifest.get("triangles", []))),
            "tested_uv_texels": tested,
            "visible_uv_texels_before_dedup": visible,
            "unique_observed_texels": int(hit.sum()),
            "samples_per_texel_mean": float(sample_weight[hit].mean()) if hit.any() else 0.0,
            "supersample_radius": float(supersample_radius),
            "conservative_barycentric_epsilon": float(conservative_barycentric_epsilon),
            "triangle_id_visibility_test": bool(triangle_id_map is not None),
            "observed_triangle_ids": sorted(value for value in observed_triangle_ids if value >= 0),
            "writeback_projection": projection["diagnostics"],
            "writeback_projection_mode": projection["diagnostics"].get("mode"),
            "current_texture_variance": current_var_report,
            "new_observation_variance": new_var_report,
            "alpha_supports": support_report,
            "pass_alpha": float(alpha),
            "stored_reachable_texels": int(stored_reachable.sum()) if stored_reachable is not None else None,
            "runtime_manifest_reachable_texels": int(runtime_reachable.sum()),
            "stored_runtime_reachable_overlap_texels": stored_overlap,
            "observed_outside_stored_reachable_texels": outside_stored,
            "observed_outside_reachable_discarded": int(
                projection["diagnostics"].get("discarded_outside_runtime_manifest_uv", 0)
            ),
            "reachable_source": reachable_source,
            "fusion_color_space": "linear_sRGB",
            "fusion_rule": "Triangle-ID-grouped Blender screen UV bake into the current runtime manifest UV triangles; direct rolling alpha blend",
            "persistent_observation_state": False,
            "removed_weight_factors": [
                "frontality",
                "projected_density",
                "boundary_confidence",
                "minimum_observation_weight",
                "first_hit_override",
                "accumulated_fusion_weight",
                "luminance_anchor",
            ],
        }
    }
