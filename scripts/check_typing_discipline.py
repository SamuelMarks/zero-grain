#!/usr/bin/env python3
"""AST-based typing and code discipline verifier for zero-grain."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys
import tokenize

FORBIDDEN_BUILTINS: frozenset[str] = frozenset({"getattr", "setattr", "hasattr", "delattr", "cast"})
FORBIDDEN_TYPES: frozenset[str] = frozenset({"Any", "object"})


def _check_comments_for_type_ignore(filepath: Path) -> list[str]:
    """Scan ``filepath`` tokens for forbidden ``# type: ignore`` comments.

    Args:
        filepath (Path): Python source file to inspect.

    Returns:
        list[str]: List of violation descriptions.
    """
    violations: list[str] = []
    with filepath.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
                violations.append(f"{filepath}:{tok.start[0]}: Forbidden '# type: ignore' comment")
    return violations


def check_file_typing(filepath: Path) -> list[str]:
    """Check ``filepath`` for forbidden typing shortcuts, reflection, or dunder tricks.

    Args:
        filepath (Path): Python source file to inspect.

    Returns:
        list[str]: List of violation descriptions.
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        return [f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"]

    violations: list[str] = _check_comments_for_type_ignore(filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name in ("Any", "cast"):
                    violations.append(f"{filepath}:{node.lineno}: Forbidden import 'typing.{alias.name}'")

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_TYPES:
                violations.append(f"{filepath}:{node.lineno}: Forbidden type reference '{node.id}'")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__name__":
                    violations.append(f"{filepath}:{node.lineno}: Forbidden dunder '__name__' reassignment")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                violations.append(f"{filepath}:{node.lineno}: Forbidden reflection/cast call '{node.func.id}()'")
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cast"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "typing"
            ):
                violations.append(f"{filepath}:{node.lineno}: Forbidden call 'typing.cast()'")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is None:
                violations.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' is missing a return type annotation"
                )
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            for arg in all_args:
                if arg.arg not in ("self", "cls") and arg.annotation is None:
                    violations.append(
                        f"{filepath}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' is missing a type annotation"
                    )
            if node.args.vararg is not None and node.args.vararg.annotation is None:
                violations.append(f"{filepath}:{node.lineno}: '*args' in '{node.name}' is missing a type annotation")
            if node.args.kwarg is not None and node.args.kwarg.annotation is None:
                violations.append(f"{filepath}:{node.lineno}: '**kwargs' in '{node.name}' is missing a type annotation")

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run typing discipline checks across target directories.

    Args:
        argv (Sequence[str] | None): Optional target directories or files.

    Returns:
        int: Zero if compliant, or 1 if any violations are detected.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    targets = [Path(a) for a in args] if args else [Path("src"), Path("scripts"), Path("tests")]
    all_violations: list[str] = []

    for target in targets:
        if not target.exists():
            all_violations.append(f"{target}: Path does not exist")
            continue
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for py_file in files:
            all_violations.extend(check_file_typing(py_file))

    for violation in all_violations:
        print(violation, file=sys.stderr)
    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
