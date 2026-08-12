from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageChops
import json


def validate_region(before_path, after_path, edit_mask_path, region_mask_path, semantic_name, out_path,
                    min_change=4.0, max_outside_change=2.5):
    before=Image.open(before_path).convert('RGB');after=Image.open(after_path).convert('RGB').resize(before.size)
    edit=Image.open(edit_mask_path).convert('L').resize(before.size);region=Image.open(region_mask_path).convert('L').resize(before.size)
    bp=before.load();ap=after.load();ep=edit.load();rp=region.load();w,h=before.size
    inside=outside=0.0;ni=no=0
    for y in range(h):
        for x in range(w):
            d=sum(abs(bp[x,y][c]-ap[x,y][c]) for c in range(3))/3.0
            if ep[x,y]>0:inside+=d;ni+=1
            elif rp[x,y]==0:outside+=d;no+=1
    inside/=max(1,ni);outside/=max(1,no)
    ok=inside>=min_change and outside<=max_outside_change
    rec={'semantic_name':semantic_name,'ok':ok,'mean_change_inside_edit':inside,'mean_change_outside_region':outside,
         'min_change':min_change,'max_outside_change':max_outside_change,'edit_pixels':ni}
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(rec,indent=2),encoding='utf-8')
    return rec
