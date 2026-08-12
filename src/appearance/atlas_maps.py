from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image


def _load_normalized_scalar_png(path: Path, bit_depth: int) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if not np.issubdtype(array.dtype, np.integer):
        values = np.asarray(array, dtype=np.float32)
        if values.size and float(np.nanmax(values)) > 1.0:
            values /= float((1 << int(bit_depth)) - 1)
        return np.clip(values, 0.0, 1.0)
    denominator = float((1 << int(bit_depth)) - 1)
    return np.clip(array.astype(np.float32) / denominator, 0.0, 1.0)


def load_uv(path):
    """Load a text+PNG UV bundle without OpenEXR.

    The manifest contains ordinary 16-bit grayscale U/V images and an 8-bit
    validity image. Old EXR caches are rejected with an explicit rerun message
    rather than importing an optional binary package.
    """
    path = Path(path)
    if path.suffix.lower() == ".exr":
        raise RuntimeError(
            f"Legacy EXR UV buffer is no longer supported: {path}. "
            "Re-run Stage06/Stage07 with v44 to generate uv_map.json and PNG channels."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "uv_map_png_bundle":
        raise ValueError(f"Unsupported UV manifest type in {path}: {payload.get('type')}")
    encoding = dict(payload.get("encoding", {}))
    bit_depth = int(encoding.get("bit_depth_uv", 16))
    u_path = path.parent / str(payload["u_image"])
    v_path = path.parent / str(payload["v_image"])
    valid_path = path.parent / str(payload["valid_image"])
    for required in (u_path, v_path, valid_path):
        if not required.exists() or required.stat().st_size == 0:
            raise RuntimeError(f"UV bundle component is missing: {required}")
    u = _load_normalized_scalar_png(u_path, bit_depth)
    v = _load_normalized_scalar_png(v_path, bit_depth)
    valid = np.asarray(Image.open(valid_path).convert("L"), dtype=np.uint8) > 0
    if u.shape != v.shape or u.shape != valid.shape:
        raise RuntimeError(
            f"UV bundle shape mismatch: u={u.shape}, v={v.shape}, valid={valid.shape}, manifest={path}"
        )
    u = np.where(valid, u, np.nan).astype(np.float32)
    v = np.where(valid, v, np.nan).astype(np.float32)
    h, w = u.shape
    expected = list(payload.get("image_size", [w, h]))
    if expected != [w, h]:
        raise RuntimeError(
            f"UV manifest size {expected} does not match PNG size {[w, h]}: {path}"
        )
    return w, h, u, v


def load_palette(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    return {name:tuple(int(x) for x in info['color_uint8_rgb']) for name,info in data.items()}


def decode_objects(semantic_path,palette_path,max_distance=96.0,return_report=False):
    """Decode the object segmentation robustly.

    Stage 04 renders semantic colours with Raw colour management. We still use nearest-
    palette decoding because PNG quantisation and Blender versions may move values by a
    few levels. Black pixels are always background and never assigned to an object.
    """
    im=np.asarray(Image.open(semantic_path).convert('RGB'),np.int16)
    pal=load_palette(palette_path);names=list(pal)
    if not names:
        raise ValueError(f'Empty semantic palette: {palette_path}')
    colors=np.asarray([pal[n] for n in names],np.int16)
    h,w=im.shape[:2];flat=im.reshape(-1,3);result=np.full(flat.shape[0],-1,np.int32)
    best_dist=np.full(flat.shape[0],np.inf,np.float32)
    nonblack=flat.max(axis=1)>=6
    chunk=200000
    for s in range(0,len(flat),chunk):
        block=flat[s:s+chunk]
        d=((block[:,None,:]-colors[None,:,:]).astype(np.int32)**2).sum(2)
        idx=d.argmin(1);best=np.sqrt(d[np.arange(len(block)),idx]).astype(np.float32)
        valid=(best<=float(max_distance)) & nonblack[s:s+len(block)]
        result[s:s+len(block)]=np.where(valid,idx,-1)
        best_dist[s:s+len(block)]=best
    decoded=result.reshape(h,w)
    if not return_report:
        return decoded,names
    counts={name:int((decoded==i).sum()) for i,name in enumerate(names)}
    report={
        'image_size':[w,h],
        'decoded_pixels':int((decoded>=0).sum()),
        'background_or_rejected_pixels':int((decoded<0).sum()),
        'max_distance':float(max_distance),
        'median_palette_distance':float(np.median(best_dist[nonblack])) if nonblack.any() else None,
        'counts':counts,
    }
    return decoded,names,report


def valid_uv_mask(u,v):
    return np.isfinite(u)&np.isfinite(v)&(u>=-1e-4)&(u<=1.0001)&(v>=-1e-4)&(v<=1.0001)


def uv_to_texel(u,v,res):
    x=np.clip(np.rint(np.asarray(u)*(res-1)).astype(np.int32),0,res-1)
    y=np.clip(np.rint((1.0-np.asarray(v))*(res-1)).astype(np.int32),0,res-1)
    return x,y
