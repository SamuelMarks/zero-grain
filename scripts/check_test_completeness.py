#!/usr/bin/env python3
"""Script to verify that every public symbol exported by ``zero_grain`` is behaviorally tested."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys


def _get_explicit_all(filepath: Path) -> set[str]:
    """Extract string entries from ``__all__`` in ``filepath``.

    Args:
        filepath (Path): Python file to inspect.

    Returns:
        set[str]: Public symbol names listed in ``__all__``.
    """
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and not elt.value.startswith("_"):
                    names.add(elt.value)
    return names


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
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and not elt.value.startswith("_"):
                    exported.add(elt.value)
    if init_path == Path("src/zero_grain/__init__.py"):
        for extra_file in (Path("src/zero_grain/experimental.py"), Path("src/zero_grain/python/__init__.py")):
            exported |= _get_explicit_all(extra_file)
    return exported


def _is_tautological_expr(expr: ast.AST) -> bool:
    """Check whether ``expr`` is a trivially tautological boolean expression (e.g. ``True`` or ``x == x``).

    Args:
        expr (ast.AST): Expression node to inspect.

    Returns:
        bool: True if ``expr`` is a static tautology.
    """
    if isinstance(expr, ast.Constant) and expr.value is True:
        return True
    if (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and isinstance(expr.ops[0], (ast.Eq, ast.Is))
        and len(expr.comparators) == 1
    ):
        left_dump = ast.dump(expr.left)
        right_dump = ast.dump(expr.comparators[0])
        if left_dump == right_dump:
            return True
    return False


def find_tautological_assertions(tests_dir: Path = Path("tests")) -> list[str]:
    """Scan ``tests_dir`` for tautological assertions (``assert True``, ``assert x == x``, ``assertEqual(x, x)``).

    Args:
        tests_dir (Path): Root tests directory.

    Returns:
        list[str]: Formatted violation messages for any tautological assertions found.
    """
    violations: list[str] = []
    for test_file in sorted(tests_dir.rglob("*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and _is_tautological_expr(node.test):
                violations.append(f"{test_file}:{node.lineno}: Tautological assertion")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "assertEqual"
                    and len(node.args) >= 2
                    and ast.dump(node.args[0]) == ast.dump(node.args[1])
                ):
                    violations.append(f"{test_file}:{node.lineno}: Tautological assertEqual(x, x)")
                elif node.func.attr == "assertTrue" and len(node.args) >= 1 and _is_tautological_expr(node.args[0]):
                    violations.append(f"{test_file}:{node.lineno}: Tautological assertTrue(True)")
    return violations


def _function_has_assertion(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether ``fn_node`` contains a non-tautological ``assert`` or ``self.assert*`` / ``pytest.raises`` call.

    Args:
        fn_node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if at least one non-tautological assertion or exception check is present in ``fn_node``.
    """
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assert) and not _is_tautological_expr(node.test):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and (
                node.func.attr.startswith("assert") or node.func.attr in ("raises", "fail", "compare_output")
            ):
                if (
                    node.func.attr == "assertEqual"
                    and len(node.args) >= 2
                    and ast.dump(node.args[0]) == ast.dump(node.args[1])
                ):
                    continue
                if node.func.attr == "assertTrue" and len(node.args) >= 1 and _is_tautological_expr(node.args[0]):
                    continue
                return True
    return False


def _collect_symbols_in_verified_fn(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Collect symbols invoked, instantiated, or inspected inside an assertion-bearing function.

    Args:
        fn_node (ast.FunctionDef | ast.AsyncFunctionDef): Verified test function AST node.

    Returns:
        set[str]: Set of symbol names exercised inside ``fn_node``.
    """
    if not _function_has_assertion(fn_node):
        return set()
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name):
                    names.add(arg.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
            names.add(node.attr)
    return names


def get_tested_names(tests_dir: Path = Path("tests")) -> set[str]:
    """Collect all identifier names behaviorally exercised inside test functions under ``tests_dir``.

    Args:
        tests_dir (Path): Root tests directory.

    Returns:
        set[str]: Set of identifier names exercised inside assertion-bearing test functions or subclasses.
    """
    tested: set[str] = set()
    for test_file in sorted(tests_dir.rglob("*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for top_node in tree.body:
            if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tested |= _collect_symbols_in_verified_fn(top_node)
            elif isinstance(top_node, ast.ClassDef):
                for base in top_node.bases:
                    target_base = base.value if isinstance(base, ast.Subscript) else base
                    if isinstance(target_base, ast.Name):
                        tested.add(target_base.id)
                    elif isinstance(target_base, ast.Attribute):
                        tested.add(target_base.attr)
                for member in top_node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        tested |= _collect_symbols_in_verified_fn(member)
    return tested


def main(argv: Sequence[str] | None = None) -> int:
    """Verify that all exported symbols in ``zero_grain`` are behaviorally tested in ``tests/``.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments ``[init_path, tests_dir]``.

    Returns:
        int: Zero if all exported symbols are tested, or 1 if any are missing.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    init_path = Path(args[0]) if len(args) >= 1 else Path("src/zero_grain/__init__.py")
    tests_dir = Path(args[1]) if len(args) >= 2 else Path("tests")

    tautologies = find_tautological_assertions(tests_dir)
    if tautologies:
        for t_err in tautologies:
            print(t_err)
        return 1

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
