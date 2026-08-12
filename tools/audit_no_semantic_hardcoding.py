#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [ROOT / "src", ROOT / "scripts", ROOT / "configs"]
# These scene-category tokens must not appear in executable source/configuration.
# They are legal in input scene JSON and historical documentation.
FORBIDDEN_CATEGORY_TOKENS = {
    "bed", "table", "desk", "chair", "sofa", "cabinet", "shelf",
    "nightstand", "dresser", "lamp", "blanket", "furniture",
}
SEMANTIC_NAMES = {"semantic_class", "semantic_label", "category", "class_name"}


def source_files():
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".json"}:
                yield path


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def contains_semantic_reference(node: ast.AST) -> bool:
    for item in ast.walk(node):
        name = dotted_name(item)
        if name and name.split(".")[-1].lower() in SEMANTIC_NAMES:
            return True
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "get":
            if item.args and isinstance(item.args[0], ast.Constant):
                if str(item.args[0].value).lower() in {"semantic", "semantic_class"}:
                    return True
    return False


def literal_strings(node: ast.AST) -> set[str]:
    values = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value.lower())
    return values


def audit_python(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues = []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{path.relative_to(ROOT)}: syntax error: {exc}"]
    for node in ast.walk(tree):
        condition = None
        if isinstance(node, ast.If):
            condition = node.test
        elif isinstance(node, ast.IfExp):
            condition = node.test
        elif isinstance(node, ast.Compare):
            condition = node
        elif isinstance(node, ast.Match):
            condition = node.subject
        if condition is not None and contains_semantic_reference(condition):
            issues.append(
                f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')}: "
                "semantic identity is used in executable control flow"
            )
    # Strong lexical guard catches category dictionaries and substring rules even
    # when variable names are obscure. Ignore comments/docstrings only by parsing
    # actual string tokens conservatively: any forbidden token in executable files
    # must be removed or moved to input scene data.
    for token in sorted(FORBIDDEN_CATEGORY_TOKENS):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
            issues.append(f"{path.relative_to(ROOT)}: forbidden executable category token {token!r}")
    return issues


def audit_json(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"{path.relative_to(ROOT)}: invalid JSON: {exc}"]
    issues = []
    for token in sorted(FORBIDDEN_CATEGORY_TOKENS):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
            issues.append(f"{path.relative_to(ROOT)}: forbidden executable category token {token!r}")
    return issues


def main() -> int:
    issues = []
    for path in source_files():
        issues.extend(audit_python(path) if path.suffix == ".py" else audit_json(path))
    if issues:
        print("Semantic hardcoding audit FAILED:")
        for issue in issues:
            print(" -", issue)
        return 1
    print("Semantic hardcoding audit passed: routing is driven by explicit JSON fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
