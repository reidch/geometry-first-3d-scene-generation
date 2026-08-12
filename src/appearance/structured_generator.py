from __future__ import annotations
from pathlib import Path
from PIL import Image
from src.appearance.pil_utils import ensure_rgb, save_image


class StructuredSparseGenerator:
    def __init__(self, backend):
        self.backend = backend

    def generate_background(self, request, background_mask_path, output_path):
        result = self.backend.generate(request)
        generated = ensure_rgb(Image.open(result['output_path']))
        mask = Image.open(background_mask_path).convert('L') if background_mask_path else Image.new('L', generated.size, 255)
        output_path = Path(output_path)
        save_image(generated, output_path)
        return {
            'result': result,
            'image_path': str(output_path),
            'mask_path': str(background_mask_path) if background_mask_path else None,
        }

    def generate_object_pass(self, request, object_mask_path, output_path):
        result = self.backend.generate(request)
        generated = ensure_rgb(Image.open(result['output_path']))
        output_path = Path(output_path)
        save_image(generated, output_path)
        return {
            'result': result,
            'image_path': str(output_path),
            'mask_path': str(object_mask_path),
        }
