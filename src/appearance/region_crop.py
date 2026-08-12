from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter, ImageOps


def mask_bbox(mask_path, padding=48, min_side=192):
    mask=Image.open(mask_path).convert('L')
    box=mask.getbbox()
    if box is None:return None
    x0,y0,x1,y1=box
    cx=(x0+x1)/2.0;cy=(y0+y1)/2.0
    side=max(min_side,x1-x0,y1-y0)+2*padding
    x0=int(round(cx-side/2));y0=int(round(cy-side/2));x1=x0+int(side);y1=y0+int(side)
    # Clamp while preserving as much square area as possible.
    if x0<0:x1-=x0;x0=0
    if y0<0:y1-=y0;y0=0
    if x1>mask.width:x0-=x1-mask.width;x1=mask.width
    if y1>mask.height:y0-=y1-mask.height;y1=mask.height
    return max(0,x0),max(0,y0),min(mask.width,x1),min(mask.height,y1)


def prepare_region_crop(current_path, edit_mask_path, structure_plate_path, depth_preview_path,
                        out_dir, padding=48, model_size=768, structure_mix=1.0):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    box=mask_bbox(edit_mask_path,padding=padding)
    if box is None:return None
    current=Image.open(current_path).convert('RGB')
    mask=Image.open(edit_mask_path).convert('L')
    structure=Image.open(structure_plate_path).convert('RGB').resize(current.size)
    depth=Image.open(depth_preview_path).convert('RGB').resize(current.size)
    # structure_mix=1.0 means fully neutralized geometry plate inside the editable area.
    # Lower values preserve more projected reference appearance for cross-view consistency.
    mixed_inside=Image.blend(current, structure, max(0.0, min(1.0, float(structure_mix))))
    init=Image.composite(mixed_inside,current,mask)
    init_crop=init.crop(box).resize((model_size,model_size),Image.Resampling.LANCZOS)
    mask_crop=mask.crop(box).resize((model_size,model_size),Image.Resampling.NEAREST)
    depth_crop=depth.crop(box).resize((model_size,model_size),Image.Resampling.LANCZOS)
    paths={'init':out/'init_crop.png','mask':out/'mask_crop.png','depth':out/'depth_crop.png'}
    init_crop.save(paths['init']);mask_crop.save(paths['mask']);depth_crop.save(paths['depth'])
    return {'bbox':list(box),'model_size':model_size,**{k:str(v) for k,v in paths.items()}}


def composite_crop_back(current_path, generated_crop_path, edit_mask_path, bbox, out_path):
    current=Image.open(current_path).convert('RGB')
    generated=Image.open(generated_crop_path).convert('RGB')
    mask=Image.open(edit_mask_path).convert('L')
    box=tuple(int(v) for v in bbox);size=(box[2]-box[0],box[3]-box[1])
    generated=generated.resize(size,Image.Resampling.LANCZOS)
    local_mask=mask.crop(box).resize(size,Image.Resampling.NEAREST)
    region=current.crop(box)
    region=Image.composite(generated,region,local_mask)
    current.paste(region,(box[0],box[1]))
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True);current.save(out_path)
    return str(out_path)
