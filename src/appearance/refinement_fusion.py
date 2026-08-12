from __future__ import annotations


def refinement_fusion_policy(strength: float = 0.06):
    return {
        "strength": float(strength),
        "scope": "all JSON objects with editable atlases",
        "selection": "measured camera visibility and uncovered triangles",
    }
