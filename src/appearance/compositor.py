from __future__ import annotations
from pathlib import Path
from PIL import Image
from src.appearance.pil_utils import ensure_rgb


def apply_masked_composite(base_image_path, overlay_image_path, mask_path, out_path):
    return apply_guided_masked_composite(base_image_path, overlay_image_path, mask_path, out_path)


def build_reference_canvas(scaffold_image_path, warped_reference_image_path=None, warped_reference_mask_path=None, out_path=None):
    base = ensure_rgb(Image.open(scaffold_image_path))
    if warped_reference_image_path and Path(warped_reference_image_path).exists() and warped_reference_mask_path and Path(warped_reference_mask_path).exists():
        guide = ensure_rgb(Image.open(warped_reference_image_path)).resize(base.size)
        mask = Image.open(warped_reference_mask_path).convert('L').resize(base.size)
        bp = base.load(); gp = guide.load(); mp = mask.load()
        w, h = base.size
        for y in range(h):
            for x in range(w):
                if mp[x, y] > 0:
                    bp[x, y] = gp[x, y]
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(out_path)
        return str(out_path)
    return base


def apply_guided_masked_composite(base_image_path, overlay_image_path, mask_path, out_path,
                                  guide_image_path=None, guide_mask_path=None, guide_weight=0.35):
    base = ensure_rgb(Image.open(base_image_path))
    overlay = ensure_rgb(Image.open(overlay_image_path)).resize(base.size)
    mask = Image.open(mask_path).convert('L').resize(base.size)
    guide = ensure_rgb(Image.open(guide_image_path)).resize(base.size) if guide_image_path and Path(guide_image_path).exists() else None
    guide_mask = Image.open(guide_mask_path).convert('L').resize(base.size) if guide_mask_path and Path(guide_mask_path).exists() else None

    out = Image.new('RGB', base.size)
    bp = base.load(); op = overlay.load(); mp = mask.load(); xp = out.load()
    gp = guide.load() if guide else None
    gmp = guide_mask.load() if guide_mask else None
    w, h = base.size
    for y in range(h):
        for x in range(w):
            a = mp[x,y] / 255.0
            if a <= 0.0:
                xp[x,y] = bp[x,y]
            else:
                rr, rg, rb = op[x,y]
                if gp is not None and gmp is not None and gmp[x,y] > 0:
                    rr = int(round((1-guide_weight)*rr + guide_weight*gp[x,y][0]))
                    rg = int(round((1-guide_weight)*rg + guide_weight*gp[x,y][1]))
                    rb = int(round((1-guide_weight)*rb + guide_weight*gp[x,y][2]))
                r = int(round((1-a)*bp[x,y][0] + a*rr))
                g = int(round((1-a)*bp[x,y][1] + a*rg))
                b = int(round((1-a)*bp[x,y][2] + a*rb))
                xp[x,y] = (r,g,b)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)
