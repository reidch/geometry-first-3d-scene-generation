from __future__ import annotations

def require_bpy():
    try:
        import bpy  # type: ignore
        return bpy
    except Exception as exc:
        raise RuntimeError(
            "This operation requires Blender's Python runtime. "
            "Run with: blender --background --python <script.py> -- <args>"
        ) from exc
