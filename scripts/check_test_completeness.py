#!/usr/bin/env python3
"""Script to verify that every public symbol exported by ``zero_grain`` is tested inside test bodies."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys


def get_exported_classes(init_path: Path = Path("src/zero_grain/__init__.py")) -> set[str]:
    """Extract exported public symbol names from ``init_path``.

    Args:
        init_path (Path): Path to ``src/zero_grain/__init__.py``.

    Returns:
        set[str]: Set of public symbol names exported in ``init_path``.
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname is not None else alias.name
                if name and not name.startswith("_") and name != "annotations":
                    exported.add(name)
    return exported


def get_tested_names(tests_dir: Path = Path("tests")) -> set[str]:
    """Collect all identifier names referenced inside function/class bodies under ``tests_dir``.

    Args:
        tests_dir (Path): Root tests directory.

    Returns:
        set[str]: Set of identifier names appearing inside test function or class bodies.
    """
    tested: set[str] = set()
    for test_file in sorted(tests_dir.rglob("*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for top_node in tree.body:
            if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for node in ast.walk(top_node):
                    if isinstance(node, ast.Name):
                        tested.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        tested.add(node.attr)
    return tested


def main(argv: Sequence[str] | None = None) -> int:
    """Verify that all exported symbols in ``zero_grain`` are referenced in ``tests/`` bodies.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments ``[init_path, tests_dir]``.

    Returns:
        int: Zero if all exported symbols are tested, or 1 if any are missing.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    init_path = Path(args[0]) if len(args) >= 1 else Path("src/zero_grain/__init__.py")
    tests_dir = Path(args[1]) if len(args) >= 2 else Path("tests")

    expected = get_exported_classes(init_path)
    tested = get_tested_names(tests_dir)
    missing = expected - tested
    if missing:
        print(f"Missing tests for classes: {sorted(missing)}")
        return 1
    print("All operations have tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
