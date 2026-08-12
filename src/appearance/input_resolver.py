from __future__ import annotations

from pathlib import Path
from PIL import Image
import json


class MissingInputError(FileNotFoundError):
    pass


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_from_artifact_value(value, stage_dir, camera_id):
    """
    Convert a Stage 04 artifact-index value into a candidate RGB path.

    The artifact may point to either:
    - a directory such as rgb_scaffold/
    - a specific file
    - a project-relative path
    """
    raw = Path(value)
    candidates = []

    if raw.is_absolute():
        resolved = raw
    else:
        project_relative = Path.cwd() / raw
        stage_relative = stage_dir / raw
        local_name_relative = stage_dir / raw.name

        for base in (
            project_relative,
            stage_relative,
            local_name_relative,
        ):
            candidates.append(base)

        resolved = None

    if resolved is not None:
        candidates.append(resolved)

    expanded = []
    for candidate in candidates:
        if candidate.suffix.lower() == ".png":
            expanded.append(candidate)
        else:
            expanded.append(candidate / f"{camera_id}.png")

    return expanded


def resolve_scaffold_rgb(scene_out_dir, camera_id):
    """
    Resolve the scaffold RGB rendered by Stage 04.

    Current canonical path:
      <out>/04_sparse_conditions/rgb_scaffold/<camera_id>.png

    The Stage 04 artifact index is checked first so future layout changes
    do not require another hardcoded path update.
    """
    scene_out_dir = Path(scene_out_dir)
    stage_dir = scene_out_dir / "04_sparse_conditions"

    checked = []
    artifact_notes = []

    artifact_index = stage_dir / "artifact_index.json"
    checked.append(artifact_index)

    if artifact_index.exists():
        try:
            data = _load_json(artifact_index)
            artifacts = data.get("artifacts", {})

            preferred_keys = (
                "rgb_scaffold",
                "scaffold_rgb",
                "rgb",
                "beauty",
                "preview",
            )

            for key in preferred_keys:
                values = artifacts.get(key, [])
                if isinstance(values, str):
                    values = [values]

                if values:
                    artifact_notes.append(
                        f"{key}: {values}"
                    )

                for value in values:
                    for candidate in _candidate_from_artifact_value(
                        value,
                        stage_dir,
                        camera_id,
                    ):
                        checked.append(candidate)
                        if candidate.exists():
                            return candidate
        except Exception as exc:
            artifact_notes.append(
                f"artifact index parse error: {exc}"
            )
    else:
        artifact_notes.append(
            "artifact index does not exist"
        )

    candidates = [
        # Canonical current Stage 04 output.
        stage_dir / "rgb_scaffold" / f"{camera_id}.png",

        # Compatibility paths used by earlier iterations.
        stage_dir / "scaffold_rgb" / f"{camera_id}.png",
        stage_dir / "rgb" / f"{camera_id}.png",
        stage_dir / "preview" / f"{camera_id}.png",
        stage_dir / "beauty" / f"{camera_id}.png",
    ]

    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_text = "\n".join(
        f"- {path}" for path in checked
    )
    artifact_text = "\n".join(
        f"- {note}" for note in artifact_notes
    )

    raise MissingInputError(
        f"Could not resolve scaffold RGB for camera {camera_id}.\n"
        f"Stage 04 directory: {stage_dir}\n\n"
        "Artifact-index diagnostics:\n"
        f"{artifact_text}\n\n"
        "Checked paths:\n"
        f"{checked_text}"
    )


def blank_rgb_from_mask(
    mask_path,
    color=(240, 240, 240),
):
    mask = Image.open(mask_path).convert("L")
    return Image.new("RGB", mask.size, color)
