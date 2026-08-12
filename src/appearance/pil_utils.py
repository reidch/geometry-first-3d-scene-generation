from __future__ import annotations
from pathlib import Path
from PIL import Image

def ensure_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def ensure_rgba(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
