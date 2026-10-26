#!/usr/bin/env python3
"""Script to calculate API and signature compliance against the Grain snapshot."""

from __future__ import annotations

import ast
from collections.abc import Sequence
import json
from pathlib import Path
import sys
import zero_grain
import zero_grain.python


def _extract_module_exports(init_file: Path) -> set[str]:
    """Parse ``init_file`` via AST to extract all imported, assigned, and ``__all__`` symbol names.

    Args:
        init_file (Path): Path to a Python ``__init__.py`` file.

    Returns:
        set[str]: Exported symbol names defined in ``init_file``.
    """
    names: set[str] = set()
    if not init_file.is_file():
        return names
    tree = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname if alias.asname is not None else alias.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if isinstance(node.value, (ast.List, ast.Tuple)):
                names.update(
                    elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
    return names


def _collect_exported_names() -> set[str]:
    """Collect all public symbol names exported by ``zero_grain`` and ``zero_grain.python``.

    Returns:
        set[str]: Set of exported symbol names.
    """
    names: set[str] = set(dir(zero_grain)) | set(dir(zero_grain.python))
    root_dir = Path("src/zero_grain")
    for mod_path in sorted(root_dir.rglob("*.py")):
        names |= _extract_module_exports(mod_path)
    return names


def calculate_compliance(
    snapshot_path: Path = Path("snapshots/pygrain_v0.2.16.dev20260112.json"),
) -> tuple[int, int, list[str]]:
    """Compare exported ``zero_grain`` symbols against ``snapshot_path``.

    Args:
        snapshot_path (Path): Path to the JSON snapshot file.

    Returns:
        tuple[int, int, list[str]]: ``(covered_count, total_count, missing_api_names)``.
    """
    raw_text = snapshot_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    categories = data.get("categories", {})

    expected: set[str] = {
        str(item["name"])
        for items in categories.values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
    }

    exported = _collect_exported_names()
    covered = expected & exported
    missing = sorted(expected - exported)
    return len(covered), len(expected), missing


def compute_compliance(
    snapshot_path: Path = Path("snapshots/pygrain_v0.2.16.dev20260112.json"),
) -> tuple[int, int, list[str]]:
    """Compute ``(total_count, covered_count, missing_api_names)`` against ``snapshot_path``.

    Args:
        snapshot_path (Path): Path to the JSON snapshot file.

    Returns:
        tuple[int, int, list[str]]: ``(total_count, covered_count, missing_api_names)``.
    """
    covered_count, total_count, missing = calculate_compliance(snapshot_path)
    return total_count, covered_count, missing


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for API compliance calculation.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments (first argument overrides snapshot path).

    Returns:
        int: Zero if 100% compliant, 1 otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    snap = Path(args[0]) if args else Path("snapshots/pygrain_v0.2.16.dev20260112.json")
    total, covered_count, missing = compute_compliance(snap)
    pct = (covered_count / total * 100.0) if total > 0 else 100.0

    print(f"Total APIs: {total}")
    print(f"Covered APIs: {covered_count}")
    print(f"Missing APIs: {missing}")
    print(f"Compliance: {pct:.2f}%")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
