from __future__ import annotations

from pathlib import Path


def stage05_scene_path(out: str | Path) -> Path:
    return Path(out) / "05_scene_assets" / "scene_assets.blend"


def stage06_surface_scene_path(out: str | Path) -> Path:
    return Path(out) / "06_surface_textures" / "scene_surface_textured.blend"


def resolve_scene_for_textured_downstream(out: str | Path) -> Path:
    """Return the newest completed scene suitable for Stage07/08 rendering.

    Stage06 publishes a scene whose material nodes explicitly reference the
    canonical ``05_texture_state`` atlas files.  Downstream renderers prefer
    that scene so room-surface textures are never silently replaced by the
    Stage05 placeholder materials.  Stage05 remains a compatibility fallback.
    """
    out = Path(out)
    published = stage06_surface_scene_path(out)
    if published.exists() and published.stat().st_size > 0:
        return published
    fallback = stage05_scene_path(out)
    if fallback.exists() and fallback.stat().st_size > 0:
        return fallback
    raise FileNotFoundError(
        "No usable Blender scene was found. Expected either "
        f"{published} or {fallback}."
    )
