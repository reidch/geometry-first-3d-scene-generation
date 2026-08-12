from __future__ import annotations
from pathlib import Path
from PIL import Image
import math, json


def load_palette_map(palette_json):
    data=json.loads(Path(palette_json).read_text(encoding='utf-8'))
    entries=[]
    for name,info in data.items():
        c=tuple(int(v) for v in info.get('color_uint8_rgb',[0,0,0]))
        entries.append((name,c,info))
    return data,entries


def _nearest_palette(rgb, entries):
    best=None; best_d=10**18
    r,g,b=rgb
    for name,(pr,pg,pb),info in entries:
        d=(r-pr)**2+(g-pg)**2+(b-pb)**2
        if d<best_d: best_d=d; best=(name,math.sqrt(d))
    return best


def build_object_masks(semantic_png_path,palette_json_path,out_dir,max_color_distance=90.0):
    semantic=Image.open(semantic_png_path).convert('RGB')
    palette,entries=load_palette_map(palette_json_path)
    w,h=semantic.size; src=semantic.load()
    masks={name:Image.new('L',(w,h),0) for name in palette}
    dst={name:masks[name].load() for name in masks}
    assigned={name:0 for name in masks}; rejected=0
    for y in range(h):
        for x in range(w):
            rgb=src[x,y]
            # Background/black must not be forced to the nearest object.
            if max(rgb)<8:
                rejected+=1; continue
            nearest=_nearest_palette(rgb,entries)
            if nearest is None or nearest[1]>max_color_distance:
                rejected+=1; continue
            name,_=nearest; dst[name][x,y]=255; assigned[name]+=1
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    out={}
    for name,img in masks.items():
        p=out_dir/f'{name}.png'; img.save(p); out[name]=str(p)
    report={'image_size':[w,h],'assigned_pixels':assigned,'rejected_pixels':rejected,'max_color_distance':max_color_distance}
    (out_dir/'decode_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return out


def merge_masks(mask_paths,out_path):
    imgs=[Image.open(p).convert('L') for p in mask_paths if p and Path(p).exists()]
    if not imgs:return None
    base=Image.new('L',imgs[0].size,0)
    from PIL import ImageChops
    for img in imgs: base=ImageChops.lighter(base,img)
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True);base.save(out_path);return str(out_path)


def invert_mask(mask_path,out_path):
    from PIL import ImageChops
    img=Image.open(mask_path).convert('L');out=ImageChops.invert(img)
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True);out.save(out_path);return str(out_path)


def bbox_from_mask(mask_path):
    img=Image.open(mask_path).convert('L'); bbox=img.getbbox()
    if bbox is None:return None
    hist=img.histogram(); count=sum(hist[1:]); w,h=img.size
    x0,y0,x1,y1=bbox
    return {'x_min':x0,'y_min':y0,'x_max':x1-1,'y_max':y1-1,'pixel_count':count,'width':w,'height':h,'fraction':count/float(w*h)}
