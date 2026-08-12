from __future__ import annotations

from typing import Mapping

from src.appearance.relation_constraints import explicit_relations


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def build_object_prompt(record: Mapping, scene_prompt: str = "", global_suffix: str = "") -> str:
    generation = dict(record.get("generation", {}))
    prompt = _clean(generation.get("prompt"))
    if generation.get("mode") in {"asset_3d", "surface_texture"} and not prompt:
        raise ValueError(f"Object {record.get('object_id')} requires generation.prompt")
    style = _clean(generation.get("style_prompt"))
    appearance = _clean(
        generation.get("appearance_prompt")
        or generation.get("material_prompt")
        or dict(record.get("appearance", {})).get("prompt")
    )
    relations = explicit_relations(record)
    relation_clause = ""
    if relations["support_target"]:
        relation_clause = f"Preserve the explicit support relationship to object {relations['support_target']}."
    values = [
        prompt,
        f"Explicit style direction: {style}." if style else "",
        f"Explicit material and finish direction: {appearance}." if appearance else "",
        f"Scene design context: {_clean(scene_prompt)}." if _clean(scene_prompt) else "",
        relation_clause,
        _clean(global_suffix),
    ]
    return " ".join(value for value in values if value)


def build_negative_prompt(record: Mapping, global_negative: str = "") -> str:
    own = _clean(dict(record.get("generation", {})).get("negative_prompt"))
    values = [value for value in (own, _clean(global_negative)) if value]
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        for term in (part.strip(" ,.;") for part in value.split(",")):
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                terms.append(term)
    return ", ".join(terms)
