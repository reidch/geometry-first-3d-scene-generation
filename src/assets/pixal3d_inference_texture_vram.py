from __future__ import annotations

"""Pixal3D inference wrapper with texture-only NAF memory reduction.

This keeps Pixal3D's 3D cascade resolution unchanged (normally 1024), but
reduces only the NAF high-resolution feature map used by the texture
conditioning model. Later pipeline atlases and refinement textures are not
controlled by this wrapper.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def _load_official_inference(repo: Path):
    inference_path = repo / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"Pixal3D inference.py not found: {inference_path}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("pixal3d_official_inference", inference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Pixal3D inference module: {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_texture_naf_override(module, target_size: int) -> None:
    if target_size <= 0:
        raise ValueError("texture NAF target size must be positive")
    original_init_pipeline = module.init_pipeline

    def init_pipeline_with_texture_override(*args, **kwargs):
        pipeline = original_init_pipeline(*args, **kwargs)
        texture_model = getattr(pipeline, "image_cond_model_tex_1024", None)
        if texture_model is None:
            raise RuntimeError("Pixal3D pipeline has no image_cond_model_tex_1024")
        if not getattr(texture_model, "use_naf_upsample", False):
            print("[Texture-VRAM] Texture condition model does not use NAF; override is unnecessary.", flush=True)
            return pipeline
        old_size = tuple(getattr(texture_model, "naf_target_size", ()))
        texture_model.naf_target_size = (int(target_size), int(target_size))
        print(
            f"[Texture-VRAM] Texture NAF target changed {old_size} -> "
            f"{texture_model.naf_target_size}; 3D cascade resolution remains unchanged.",
            flush=True,
        )
        return pipeline

    module.init_pipeline = init_pipeline_with_texture_override


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official Pixal3D with texture-conditioning-only NAF memory reduction"
    )
    parser.add_argument("--repo", required=True, help="Path to the official Pixal3D repository")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0, help="Manual FOV in radians")
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--texture_naf_target_size",
        type=int,
        default=64,
        help="Texture conditioning NAF feature-map side length; official default is commonly 128",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    module = _load_official_inference(repo)
    _install_texture_naf_override(module, args.texture_naf_target_size)
    module.run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
