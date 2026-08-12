from __future__ import annotations

from collections.abc import Mapping


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _layout_clauses(surface: Mapping, *, retry_for_detail: bool = False) -> list[str]:
    layout = _clean(surface.get("layout_type")).lower()
    material = _clean(surface.get("material_family")).lower()
    clauses: list[str] = []
    if layout == "parallel_strips":
        width = surface.get("nominal_strip_width_m")
        width_text = f" around {float(width):.2f} meters wide" if width not in (None, "") else ""
        element_name = "boards" if material == "wood" else "strips"
        clauses.extend([
            f"Render the surface as an installed assembly of many long parallel {element_name}{width_text}, not as one monolithic slab or one giant material patch.",
            "Make seams between adjacent strips clearly readable but not exaggerated, with subtle board-to-board variation and coherent strip direction across the full surface.",
        ])
        if retry_for_detail:
            clauses.append("Strengthen the mid-scale strip structure so the repeated installed boards remain clearly legible across the room-scale view.")
    elif layout == "panel_rhythm":
        clauses.extend([
            "Build a continuous architectural composition across the whole surface using a readable hierarchy of panels, mouldings, trims, fields, or repeated motifs as explicitly requested by the JSON prompt.",
            "Distribute the design language across the full installed surface rather than leaving a mostly blank field with one isolated central symbol.",
            "Keep broad tonal continuity while ensuring the mid-scale panel, trim, relief, or motif structure remains legible at room scale.",
        ])
        if retry_for_detail:
            clauses.append("Make the panel, relief, trim, or motif hierarchy clearer while preserving coherent real-world scale and architectural plausibility.")
    elif retry_for_detail:
        clauses.append("Increase room-scale material structure and mid-frequency detail while avoiding close-up swatch appearance.")
    return clauses


def build_surface_prompt(
    record: Mapping,
    scene_prompt: str = "",
    *,
    retry_for_detail: bool = False,
    reference_context: Mapping | None = None,
) -> str:
    """Compile a room-scale architectural surface prompt from explicit JSON fields.

    Semantic labels and object names remain opaque. Style comes only from explicit
    scene/generation prompt fields and generic installation/layout metadata.
    """
    generation = dict(record.get("generation", {}))
    surface = dict(generation.get("surface", {}))
    base = _clean(generation.get("prompt"))
    if not base:
        raise ValueError(f"surface_texture object {record.get('object_id')} requires generation.prompt")

    clauses = [base]
    style_prompt = _clean(generation.get("style_prompt"))
    appearance_prompt = _clean(generation.get("appearance_prompt"))
    if style_prompt:
        clauses.append(f"Explicit design style: {style_prompt}.")
    if appearance_prompt:
        clauses.append(f"Explicit material and finish direction: {appearance_prompt}.")
    if _clean(scene_prompt):
        clauses.append(f"Keep the surface coherent with the enclosing scene: {_clean(scene_prompt)}.")
    if _clean(surface.get("generation_space")).lower() == "rectified_surface":
        clauses.extend([
            "Generate one rectified orthographic texture design for the complete physical surface at its exact width-to-height ratio, from edge to edge, with no room perspective, camera view, foreground object, horizon, vignette, or cropped composition.",
            "Every part of the image is usable installed material: distribute color, material grain, relief, moulding, seams, and ornament across the complete rectangle rather than composing a decorative object in the centre.",
            "Treat the image borders as the true physical borders of the wall, floor, or ceiling and keep the design readable at room scale when directly mapped to regular UV coordinates.",
        ])
    clauses.extend([
        "Treat this as a complete room-scale installed architectural surface rather than a close-up material swatch, poster, decal, or isolated decorative object.",
        "Translate the requested style into a continuous material-and-architecture system covering the full active region, with clearly visible but coherent design character.",
        "Use a deliberate hierarchy of low-frequency tonal structure, mid-frequency installation or construction structure, ornamental structure where explicitly requested, and fine material detail.",
        "Include physically plausible and clearly visible material identity: elegant color harmony, natural grain or weave, plaster or stone microtexture, seams, mouldings, relief, joints, controlled roughness variation, and decorative pattern where requested; never leave the material as a plain colored block.",
        "Maintain consistent real-world scale across the entire active region, with no blank, flat, featureless, or uniformly filled areas unless the explicit JSON prompt asks for them.",
        "Keep all repeated patterns continuous, perspective-free, non-stamped, and suitable for projection onto the full surface atlas.",
        "Do not collapse a rich full-surface design into one tiny centered emblem, medallion, icon, picture, or unrelated object surrounded by empty color.",
    ])
    clauses.extend(_layout_clauses(surface, retry_for_detail=retry_for_detail))
    if _clean(surface.get("continuity_group")):
        clauses.append(
            "Match the material family, feature scale, trim elevation, motif scale, and design language of the linked continuity group while preserving orientation-appropriate variation."
        )
    if _clean(surface.get("style_group")):
        clauses.append(
            "Remain visually compatible with the shared room style group, while keeping this surface's own material family and complete standalone identity."
        )
    references = list(dict(reference_context or {}).get("reference_bindings", []))
    if references:
        clauses.extend([
            "Continue the already generated neighbouring surface shown by the protected reference edge strips in the input image.",
            "Match its material family, color balance, real-world feature scale, trim elevation, panel rhythm, motif rhythm, and broad tonal movement exactly at every protected shared boundary.",
            "Generate only the remaining active region of the current surface; do not reinterpret, repaint, or move the protected neighbour-derived edge strips.",
            "Allow subtle non-repeating variation away from the shared boundaries while preserving one coherent room-wide surface system.",
        ])
    if retry_for_detail:
        clauses.append(
            "The previous candidate was too visually simple: make the requested style unmistakably visible through clearer multi-scale material and ornamental structure, while preserving realistic installation scale and avoiding visual noise."
        )
    return " ".join(_clean(clause) for clause in clauses if _clean(clause))


def build_surface_negative_prompt(record: Mapping, global_negative: str = "") -> str:
    generation = dict(record.get("generation", {}))
    surface = dict(generation.get("surface", {}))
    own = _clean(generation.get("negative_prompt"))
    generic = _clean(global_negative)
    additions = (
        "flat uniform color, featureless surface, blank region, blurry low-detail fill, "
        "single tiny centered emblem, isolated central ornament, poster-like composition, decal-like decoration, "
        "oversized repeated motif, identical stamped repetition, giant seams, inconsistent material scale, abrupt discontinuity, "
        "isolated object, close-up swatch, text, watermark"
    )
    layout = _clean(surface.get("layout_type")).lower()
    if layout == "parallel_strips":
        additions += ", monolithic wood slab, single giant board, missing plank seams, unreadable strip layout"
    elif layout == "panel_rhythm":
        additions += ", featureless wall field, missing panel rhythm, missing trim rhythm, empty field around one symbol"
    values = [value for value in (own, generic, additions) if value]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        for term in (part.strip(" ,.;") for part in value.split(",")):
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                ordered.append(term)
    return ", ".join(ordered)
