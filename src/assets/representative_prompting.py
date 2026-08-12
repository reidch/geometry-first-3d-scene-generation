from __future__ import annotations

from typing import Mapping, Sequence


def _clean(text: str) -> str:
    return ' '.join(str(text or '').replace('\n', ' ').split())


def _sentence(text: str) -> str:
    text = _clean(text).strip(' ,.;')
    if not text:
        return ''
    if text[-1] not in '.!?':
        text += '.'
    return text


def _view_phrase(view: Mapping) -> str:
    hint = _clean(str(view.get('text_hint', '')))
    if hint:
        return hint
    azimuth = int(round(float(view.get('azimuth_deg', 0.0)))) % 360
    elevation = float(view.get('elevation_deg', 0.0))
    if elevation > 1e-6:
        elevation_text = f'elevated by {abs(int(round(elevation)))} degrees'
    elif elevation < -1e-6:
        elevation_text = f'viewing from below by {abs(int(round(elevation)))} degrees'
    else:
        elevation_text = 'level with the object center'
    return f'isolated orbit view at azimuth {azimuth} degrees, {elevation_text}'


def _object_title(record: Mapping) -> str:
    return _clean(str(record.get('name') or record.get('object_id') or 'object'))


def _generation(record: Mapping) -> Mapping:
    return dict(record.get('generation', {}))


def _appearance_guidance(record: Mapping) -> str:
    generation = _generation(record)
    # Optional, JSON-driven extra field. No hardcoded object names.
    extras = [
        generation.get('representative_prompt'),
        generation.get('appearance_prompt'),
        generation.get('material_prompt'),
        generation.get('style_prompt'),
    ]
    extras = [_clean(str(item)) for item in extras if item is not None and _clean(str(item))]
    return '; '.join(extras)



def _scaffold_structure_guidance(record: Mapping) -> str:
    scaffold = dict(record.get('scaffold', {}))
    parts = [dict(part) for part in scaffold.get('parts', []) if isinstance(part, Mapping)]
    if not parts:
        return ''
    clause = (
        f'The scaffold defines the main object structure; keep the object complete, fully assembled, and structurally coherent without obvious missing sections'
    )
    return _sentence(clause)


def build_representative_prompt(
    record: Mapping,
    view: Mapping,
    *,
    role: str = 'hero',
    generic_suffix: str = '',
    hero_summary: str = '',
) -> str:
    generation = _generation(record)
    base = _clean(str(generation.get('prompt', '')))
    if not base:
        raise ValueError(f"Object {record.get('object_id')}: generation.prompt is empty")

    appearance = _appearance_guidance(record)
    view_clause = _sentence(f'Rendered as a photorealistic {_view_phrase(view)}')
    structure_clause = _sentence('Preserve the overall structure, proportions, and part layout from the scaffold')
    component_clause = _scaffold_structure_guidance(record)
    isolation_clause = _sentence('Show one complete isolated object only')
    detail_clause = _sentence('Use realistic material texture, visible surface detail, and a distinctive but coherent product design')
    anti_flat_clause = _sentence('Avoid flat clay-like surfaces, plain block colors, and featureless placeholder appearance')
    hero_match_clause = _sentence(
        f"Match the established appearance of the hero anchor: {_clean(hero_summary)}"
    ) if _clean(hero_summary) else _sentence('Keep one consistent material design and visual identity across anchor views')

    ordered: list[str] = []
    ordered.append(_sentence(base))
    if appearance:
        ordered.append(_sentence(appearance))
    if role == 'hero':
        ordered.extend([
            detail_clause,
            anti_flat_clause,
            view_clause,
            structure_clause,
            component_clause,
            isolation_clause,
        ])
    else:
        ordered.extend([
            hero_match_clause,
            view_clause,
            structure_clause,
            component_clause,
            isolation_clause,
            detail_clause,
        ])
    suffix = _sentence(generic_suffix) if _clean(generic_suffix) else ''
    if suffix:
        ordered.append(suffix)
    return ' '.join(item for item in ordered if item).strip()


def build_representative_negative_prompt(record: Mapping, *, role: str = 'hero') -> str:
    generation = _generation(record)
    base_negative = _clean(str(generation.get('negative_prompt', '')))
    common = [
        'extra objects',
        'duplicate parts',
        'broken proportions',
        'heavy deformation',
        'missing component',
        'incomplete object',
        'truncated geometry',
        'fused-away parts',
        'detached components',
        'text',
        'watermark',
    ]
    if role == 'hero':
        common.extend(['flat decal', 'featureless clay render'])
    seen = set()
    ordered: list[str] = []
    for term in ([base_negative] if base_negative else []) + common:
        term = _clean(term).strip(' ,.;')
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(term)
    return ', '.join(ordered)


def build_hero_summary(prompt: str) -> str:
    """Return a concise appearance summary suitable for later consistency prompts.

    The summary intentionally keeps the first few high-priority clauses because the
    prompt compiler orders appearance-defining clauses first.
    """
    prompt = _clean(prompt)
    if not prompt:
        return ''
    clauses = [chunk.strip(' ,.;') for chunk in prompt.split('.') if chunk.strip()]
    kept: list[str] = []
    for clause in clauses:
        kept.append(clause)
        if len(kept) >= 3:
            break
    return '; '.join(kept)
