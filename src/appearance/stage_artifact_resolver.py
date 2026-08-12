from __future__ import annotations

from pathlib import Path
import json


class ArtifactResolutionError(FileNotFoundError):
    pass


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_stage01_object_registry(scene_out_dir):
    """
    Resolve the JSON object registry produced by Stage 01.

    The canonical current path is:
      <out>/01_world_ir/object_registry.json

    The artifact index is consulted first so the resolver remains compatible
    with future output-layout changes.
    """
    scene_out_dir = Path(scene_out_dir)
    stage_dir = scene_out_dir / "01_world_ir"
    checked = []

    artifact_index = stage_dir / "artifact_index.json"
    checked.append(artifact_index)

    if artifact_index.exists():
        try:
            data = _load_json(artifact_index)
            artifacts = data.get("artifacts", {})
            for key in ("object_registry", "objects", "world_ir_json"):
                for value in artifacts.get(key, []):
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        # Artifact paths may be relative to the project root.
                        project_relative = Path.cwd() / candidate
                        stage_relative = stage_dir / candidate.name
                        for resolved in (project_relative, stage_relative):
                            checked.append(resolved)
                            if resolved.exists():
                                return resolved
                    else:
                        checked.append(candidate)
                        if candidate.exists():
                            return candidate
        except Exception:
            # Fall through to explicit candidates with a clearer final error.
            pass

    candidates = [
        stage_dir / "object_registry.json",
        stage_dir / "world.ir.json",
        stage_dir / "world_ir.json",
        stage_dir / "objects.json",
    ]

    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n".join(f"- {p}" for p in checked)
    raise ArtifactResolutionError(
        "Could not locate Stage 01 object metadata JSON.\n"
        "Run Stage 01 first and verify its artifact index.\n"
        "Checked paths:\n"
        f"{checked_text}"
    )


def load_stage01_objects(scene_out_dir):
    path = resolve_stage01_object_registry(scene_out_dir)
    data = _load_json(path)

    if isinstance(data, dict):
        objects = data.get("objects")
        if isinstance(objects, list):
            return objects, path

    if isinstance(data, list):
        return data, path

    raise ValueError(
        f"Stage 01 object registry has an unsupported structure: {path}"
    )
