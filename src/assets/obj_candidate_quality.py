from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict
import math


def inspect_obj(path: str | Path) -> Dict:
    """Cheap, dependency-free structural diagnostics for candidate ranking.

    This is deliberately generic: it does not know object categories.  It favors
    candidates with finite non-degenerate geometry and one dominant connected
    body, while retaining all candidates for diagnostics instead of imposing a
    semantic hard-coded reject rule.
    """
    path = Path(path)
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append(tuple(float(v) for v in parts[1:4]))
                    except ValueError:
                        pass
            elif line.startswith("f "):
                ids = []
                for token in line.split()[1:]:
                    try:
                        value = int(token.split("/", 1)[0])
                    except ValueError:
                        continue
                    if value < 0:
                        value = len(vertices) + value + 1
                    ids.append(value - 1)
                if len(ids) >= 3:
                    faces.append(tuple(ids))

    finite_vertices = [v for v in vertices if all(math.isfinite(x) for x in v)]
    invalid_vertex_fraction = 1.0 - len(finite_vertices) / max(len(vertices), 1)
    degenerate = 0
    vertex_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        if len(set(face)) < 3 or any(i < 0 or i >= len(vertices) for i in face):
            degenerate += 1
            continue
        for vi in set(face):
            vertex_to_faces[vi].append(fi)

    adjacency = defaultdict(set)
    for incident in vertex_to_faces.values():
        for fi in incident:
            adjacency[fi].update(incident)
            adjacency[fi].discard(fi)
    remaining = set(range(len(faces)))
    component_sizes = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for nxt in adjacency.get(current, ()):
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        component_sizes.append(size)
    component_sizes.sort(reverse=True)
    largest_ratio = component_sizes[0] / max(len(faces), 1) if component_sizes else 0.0
    meaningful_components = sum(1 for n in component_sizes if n >= max(16, int(0.005 * max(len(faces), 1))))
    degenerate_fraction = degenerate / max(len(faces), 1)

    if finite_vertices:
        mins = [min(v[i] for v in finite_vertices) for i in range(3)]
        maxs = [max(v[i] for v in finite_vertices) for i in range(3)]
        extents = [maxs[i] - mins[i] for i in range(3)]
    else:
        extents = [0.0, 0.0, 0.0]
    positive = [x for x in extents if x > 1e-8]
    collapsed_axis = len(positive) < 3

    # Ranking score, not a category-specific pass/fail gate.
    score = (
        3.0 * largest_ratio
        - 0.20 * max(0, meaningful_components - 1)
        - 4.0 * degenerate_fraction
        - 5.0 * invalid_vertex_fraction
        - (2.0 if collapsed_axis else 0.0)
    )
    return {
        "path": str(path),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "connected_component_count": len(component_sizes),
        "meaningful_component_count": meaningful_components,
        "largest_component_face_fraction": float(largest_ratio),
        "degenerate_face_fraction": float(degenerate_fraction),
        "invalid_vertex_fraction": float(invalid_vertex_fraction),
        "bbox_extents": [float(x) for x in extents],
        "collapsed_axis": bool(collapsed_axis),
        "generic_quality_score": float(score),
    }
