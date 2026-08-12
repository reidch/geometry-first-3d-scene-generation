from __future__ import annotations

from typing import Dict, Mapping

from src.appearance.prompt_builder import build_negative_prompt, build_object_prompt
from src.appearance.relation_constraints import explicit_relations


def build_object_constraints(record: Mapping, scene_prompt: str = "", prompt_config: Mapping | None = None) -> Dict:
    prompt_config = dict(prompt_config or {})
    return {
        "object_id": str(record["object_id"]),
        "semantic_label": str(record.get("semantic_class", "")),
        "generation_mode": record.get("generation_mode", dict(record.get("generation", {})).get("mode")),
        "prompt": build_object_prompt(record, scene_prompt, str(prompt_config.get("global_suffix", ""))),
        "negative_prompt": build_negative_prompt(record, str(prompt_config.get("global_negative", ""))),
        "relations": explicit_relations(record),
        "scaffold": dict(record.get("scaffold", {})),
        "physics": dict(record.get("physics", {})),
    }
