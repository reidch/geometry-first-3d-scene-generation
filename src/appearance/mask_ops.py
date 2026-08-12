from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageFilter, ImageChops


def load_l(path, size=None):
    img=Image.open(path).convert('L')
    return img.resize(size,Image.Resampling.NEAREST) if size and img.size!=size else img


def invert(mask): return ImageChops.invert(mask.convert('L'))
def intersect(a,b): return ImageChops.multiply(a.convert('L'),b.convert('L'))
def union(a,b): return ImageChops.lighter(a.convert('L'),b.convert('L'))
def dilate(mask,radius=3): return mask.convert('L').filter(ImageFilter.MaxFilter(radius*2+1))
def erode(mask,radius=2): return mask.convert('L').filter(ImageFilter.MinFilter(radius*2+1))

def semantic_boundary_mask(semantic_path, radius=2):
    img=Image.open(semantic_path).convert('RGB')
    edge=img.filter(ImageFilter.FIND_EDGES).convert('L').point(lambda x:255 if x>8 else 0)
    return dilate(edge,radius)

def normal_boundary_mask(normal_preview_path, radius=2):
    img=Image.open(normal_preview_path).convert('RGB')
    edge=img.filter(ImageFilter.FIND_EDGES).convert('L').point(lambda x:255 if x>24 else 0)
    return dilate(edge,radius)

def save(mask,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); mask.save(path); return str(path)
