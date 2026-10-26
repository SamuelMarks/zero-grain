#!/usr/bin/env python3
"""AST-based docstring verifier enforcing 100% module, class, function, arg, and return coverage."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import re
import sys

BANNED_PLACEHOLDERS: tuple[str, ...] = (
    "Module docstring.",
    "Execute implementation detail.",
    "The computed result.",
    "The return value.",
    "The yielded value.",
    "Docstring for ",
)

_ARGS_SECTION_RE = re.compile(r"^\s*Args:\s*$", re.MULTILINE)
_RETURNS_SECTION_RE = re.compile(r"^\s*(Returns|Yields):\s*$", re.MULTILINE)


def _extract_documented_params(docstring: str) -> set[str]:
    """Extract parameter names documented under the ``Args:`` section of ``docstring``.

    Args:
        docstring (str): Raw docstring text.

    Returns:
        set[str]: Set of parameter names stripped of leading ``*`` characters.
    """
    lines = docstring.splitlines()
    in_args = False
    args_indent: int | None = None
    documented: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "Args:":
            in_args = True
            args_indent = indent
            continue
        if in_args:
            if args_indent is not None and indent <= args_indent and stripped.endswith(":"):
                in_args = False
                continue
            match = re.match(r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def _is_non_none_return(annotation: ast.expr | None) -> bool:
    """Check whether a return type annotation represents a non-``None`` return value.

    Args:
        annotation (ast.expr | None): Return annotation AST node.

    Returns:
        bool: True if the function has a non-None return annotation.
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and annotation.value in (None, "None", "NoReturn"):
        return False
    if isinstance(annotation, ast.Attribute) and annotation.attr in ("None", "NoReturn"):
        return False
    return not (isinstance(annotation, ast.Name) and annotation.id in ("None", "NoReturn"))


def check_file_docstrings(filepath: Path) -> list[str]:
    """Verify module, class, function, argument, and return docstrings in ``filepath``.

    Args:
        filepath (Path): Python file to inspect.

    Returns:
        list[str]: List of docstring violation strings.
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        return [f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"]

    violations: list[str] = []
    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{filepath}:1: Missing module docstring")
    else:
        for banned in BANNED_PLACEHOLDERS:
            if banned in mod_doc:
                violations.append(f"{filepath}:1: Banned placeholder in module docstring: '{banned}'")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(f"{filepath}:{node.lineno}: Class '{node.name}' is missing a docstring")
            else:
                for banned in BANNED_PLACEHOLDERS:
                    if banned in cls_doc:
                        violations.append(
                            f"{filepath}:{node.lineno}: Class '{node.name}' contains banned placeholder '{banned}'"
                        )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_doc = ast.get_docstring(node)
            if not fn_doc or not fn_doc.strip():
                violations.append(f"{filepath}:{node.lineno}: Function '{node.name}' is missing a docstring")
                continue

            for banned in BANNED_PLACEHOLDERS:
                if banned in fn_doc:
                    violations.append(
                        f"{filepath}:{node.lineno}: Function '{node.name}' contains banned placeholder '{banned}'"
                    )

            all_args: list[str] = []
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.arg not in ("self", "cls"):
                    all_args.append(arg.arg)
            if node.args.vararg is not None:
                all_args.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                all_args.append(node.args.kwarg.arg)

            if all_args:
                if not _ARGS_SECTION_RE.search(fn_doc):
                    violations.append(
                        f"{filepath}:{node.lineno}: Function '{node.name}' is missing 'Args:' section for {all_args}"
                    )
                else:
                    documented_params = _extract_documented_params(fn_doc)
                    for param in all_args:
                        if param not in documented_params:
                            violations.append(
                                f"{filepath}:{node.lineno}: Function '{node.name}' is missing docstring entry for parameter '{param}'"
                            )

            if node.name != "__init__" and _is_non_none_return(node.returns) and not _RETURNS_SECTION_RE.search(fn_doc):
                violations.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' has non-None return annotation "
                    "but is missing 'Returns:' or 'Yields:' section"
                )

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run docstring verification across target paths.

    Args:
        argv (Sequence[str] | None): Target files or directories to check.

    Returns:
        int: Zero if no violations exist, or 1 if any violations are found.
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
            all_violations.extend(check_file_docstrings(py_file))

    for violation in all_violations:
        print(violation, file=sys.stderr)
    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
