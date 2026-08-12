#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

BLENDER_BUILTIN_TOP_LEVEL = {
    "bpy",
    "bmesh",
    "mathutils",
    "gpu",
    "gpu_extras",
    "blf",
    "aud",
    "imbuf",
    "idprop",
    "bgl",
}

# Every script currently launched through ``blender --background --python``.
BLENDER_ENTRYPOINTS = (
    "scripts/02_build_blender_scaffold.py",
    "src/blender/prephysics_runtime/capture_object_multiview_inputs.py",
    "src/blender/prephysics_runtime/import_align_prepare_assets.py",
    "src/blender/prephysics_runtime/render_canonical_surface.py",
    "src/blender/prephysics_runtime/render_refinement_candidates_batch.py",
    "src/blender/prephysics_runtime/render_refinement_shared_buffers_batch.py",
    "src/blender/prephysics_runtime/render_textured_view.py",
    "src/blender/prephysics_runtime/render_object_subpass_buffers.py",
    "src/blender/prephysics_runtime/save_final_textured_scene.py",
    "src/blender/prephysics_runtime/export_gaussian_mesh_catalog.py",
    # Kept in the runtime directory even though it is not in the default stage path.
    "src/blender/prephysics_runtime/render_semantic_probe.py",
)


@dataclass(frozen=True)
class ImportFinding:
    entrypoint: str
    importer: str
    imported_module: str
    line: int


class ProjectIndex:
    def __init__(self, root: Path):
        self.root = root
        self.module_to_path: Dict[str, Path] = {}
        self.path_to_module: Dict[Path, str] = {}
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            parts = list(relative.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            module = ".".join(parts)
            self.module_to_path[module] = path
            self.path_to_module[path.resolve()] = module

    def module_for_path(self, path: Path) -> str:
        resolved = path.resolve()
        module = self.path_to_module.get(resolved)
        if module is None:
            raise KeyError(f"Python path is outside the indexed project: {path}")
        return module

    def resolve_project_module(self, imported_module: str) -> Optional[str]:
        candidate = imported_module
        while candidate:
            if candidate in self.module_to_path:
                return candidate
            if "." not in candidate:
                break
            candidate = candidate.rsplit(".", 1)[0]
        return None


def _absolute_from_relative(current_module: str, level: int, imported_module: str) -> str:
    package_parts = current_module.split(".")[:-1]
    if level > len(package_parts) + 1:
        return imported_module
    base = package_parts[: len(package_parts) - level + 1]
    if imported_module:
        base.extend(imported_module.split("."))
    return ".".join(base)


def _imports(path: Path, current_module: str) -> List[Tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, int(node.lineno)) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = _absolute_from_relative(current_module, int(node.level), module)
            if module:
                imports.append((module, int(node.lineno)))
    return imports


def audit(root: Path = ROOT, entrypoints: Sequence[str] = BLENDER_ENTRYPOINTS) -> Dict:
    root = root.resolve()
    index = ProjectIndex(root)
    stdlib: Set[str] = set(getattr(sys, "stdlib_module_names", ()))
    findings: List[ImportFinding] = []
    closures: Dict[str, List[str]] = {}

    for entrypoint in entrypoints:
        entry_path = (root / entrypoint).resolve()
        if not entry_path.exists():
            findings.append(ImportFinding(entrypoint, entrypoint, "<missing entrypoint>", 0))
            continue
        entry_module = index.module_for_path(entry_path)
        queue = [entry_module]
        visited: Set[str] = set()
        while queue:
            module = queue.pop()
            if module in visited:
                continue
            visited.add(module)
            module_path = index.module_to_path[module]
            for imported, line in _imports(module_path, module):
                top = imported.split(".", 1)[0]
                project_module = index.resolve_project_module(imported)
                if project_module is not None:
                    queue.append(project_module)
                    continue
                if top in stdlib or top in BLENDER_BUILTIN_TOP_LEVEL or top == "__future__":
                    continue
                findings.append(
                    ImportFinding(
                        entrypoint=entrypoint,
                        importer=str(module_path.relative_to(root)),
                        imported_module=imported,
                        line=line,
                    )
                )
        closures[entrypoint] = sorted(visited)

    report = {
        "status": "ok" if not findings else "failed",
        "root": str(root),
        "entrypoints": list(entrypoints),
        "blender_builtin_allowlist": sorted(BLENDER_BUILTIN_TOP_LEVEL),
        "closures": closures,
        "findings": [asdict(item) for item in findings],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Blender dependency boundary: {report['status']}")
        for entrypoint, closure in report["closures"].items():
            print(f"  {entrypoint}: {len(closure)} project modules checked")
        for finding in report["findings"]:
            print(
                "  FORBIDDEN "
                f"{finding['imported_module']} imported by {finding['importer']}:{finding['line']} "
                f"(entrypoint {finding['entrypoint']})"
            )
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
