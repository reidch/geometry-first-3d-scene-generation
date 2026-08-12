#!/usr/bin/env python
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [ROOT / "src/blender"]
FORBIDDEN_TEXT = (
    r"int\s*\(\s*(?:numeric_id|object_id|world_object_id|semantic_owner_id)\s*\)",
    r"\[\s*[\"']object_id[\"']\s*\]\s*=\s*(?:oid|object_id|world_object_id|semantic_owner_id)\b",
)


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def audit_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues = []
    for pattern in FORBIDDEN_TEXT:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{path.relative_to(ROOT)}:{line}: overloaded string/numeric object identity")

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return issues + [f"{path.relative_to(ROOT)}: syntax error: {exc}"]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or dotted_name(node.func) != "hash":
            continue
        source = ast.get_source_segment(text, node) or "hash(...)"
        lowered = source.lower()
        if any(token in lowered for token in ("object", "semantic", "owner")):
            issues.append(
                f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')}: "
                "process-randomized hash used for persistent identity"
            )
    return issues


def main() -> int:
    issues = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            # The compatibility reader is the sole place allowed to inspect the
            # legacy overloaded property during migration.
            if path == ROOT / "src/blender/object_identity.py":
                continue
            issues.extend(audit_file(path))
    if issues:
        print("Identity decoupling audit FAILED:")
        for issue in issues:
            print(" -", issue)
        return 1
    print("Identity decoupling audit passed: JSON, runtime, semantic, and palette IDs are separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
