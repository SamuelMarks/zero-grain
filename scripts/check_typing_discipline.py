#!/usr/bin/env python3
"""AST-based typing and code discipline verifier for zero-grain."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys
import tokenize

BANNED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "object",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
    }
)
BANNED_TYPING_ATTRS: frozenset[str] = frozenset({"Any", "cast"})
BANNED_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})
BANNED_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__name__",
        "__dict__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
    }
)


def _check_function_typing(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    filepath: Path,
) -> list[str]:
    """Verify explicit parameter and return type annotations on ``node``.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.
        filepath (Path): Source file path for error reporting.

    Returns:
        list[str]: List of typing violation messages.
    """
    violations: list[str] = []
    if node.returns is None:
        violations.append(f"{filepath}:{node.lineno} Function '{node.name}' missing return type annotation.")

    all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for idx, arg in enumerate(all_args):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            violations.append(
                f"{filepath}:{node.lineno} Parameter '{arg.arg}' in '{node.name}' missing type annotation."
            )

    if node.args.vararg is not None and node.args.vararg.annotation is None:
        violations.append(f"{filepath}:{node.lineno} Variadic '*args' in '{node.name}' missing type annotation.")
    if node.args.kwarg is not None and node.args.kwarg.annotation is None:
        violations.append(f"{filepath}:{node.lineno} Keyword '**kwargs' in '{node.name}' missing type annotation.")
    return violations


def _check_ast_node_discipline(node: ast.AST, filepath: Path) -> list[str]:
    """Check a single AST node for banned symbols, casts, reflection, and dunder hacks.

    Args:
        node (ast.AST): AST node to inspect.
        filepath (Path): Source file path for error reporting.

    Returns:
        list[str]: List of violation messages for ``node``.
    """
    violations: list[str] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        violations.extend(_check_function_typing(node, filepath))
    elif isinstance(node, ast.Name):
        if node.id in ("Any", "object"):
            violations.append(f"{filepath}:{node.lineno} Banned type/symbol '{node.id}' used.")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES | {"cast"}:
            violations.append(f"{filepath}:{node.lineno} Banned call '{node.func.id}()' used.")
        elif isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in BANNED_TYPING_MODULES
                and node.func.attr == "cast"
            ):
                violations.append(f"{filepath}:{node.lineno} Banned call '{node.func.value.id}.cast()' used.")
            elif (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
                and node.func.attr in BANNED_CALL_NAMES
            ):
                violations.append(f"{filepath}:{node.lineno} Banned call 'builtins.{node.func.attr}()' used.")
    elif isinstance(node, ast.ImportFrom) and node.module in BANNED_TYPING_MODULES:
        for alias in node.names:
            if alias.name in BANNED_TYPING_ATTRS:
                violations.append(f"{filepath}:{node.lineno} Banned import '{node.module}.{alias.name}'.")
    elif isinstance(node, ast.Attribute):
        if node.attr in BANNED_DUNDER_ATTRS:
            violations.append(f"{filepath}:{node.lineno} Banned dunder attribute access '.{node.attr}'.")
        elif (
            isinstance(node.value, ast.Name)
            and node.value.id in BANNED_TYPING_MODULES
            and node.attr in BANNED_TYPING_ATTRS
        ):
            violations.append(f"{filepath}:{node.lineno} Banned attribute '{node.value.id}.{node.attr}'.")
        elif isinstance(node.value, ast.Name) and node.value.id == "builtins" and node.attr in BANNED_CALL_NAMES:
            violations.append(f"{filepath}:{node.lineno} Banned attribute 'builtins.{node.attr}'.")
    elif isinstance(node, ast.ExceptHandler):
        if node.type is None:
            violations.append(f"{filepath}:{node.lineno} Bare 'except:' handler is banned.")
        elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
            violations.append(
                f"{filepath}:{node.lineno} Broad 'except {node.type.id}:' handler is banned; "
                "catch specific exception types."
            )
    return violations


def check_file_typing(filepath: Path) -> list[str]:
    """Check ``filepath`` for typing and reflection discipline violations.

    Args:
        filepath (Path): Python source file to inspect.

    Returns:
        list[str]: List of violation descriptions.
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        return [f"{filepath}:{exc.lineno} SyntaxError: {exc.msg}"]

    violations: list[str] = []
    with filepath.open("rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
                violations.append(f"{filepath}:{tok.start[0]} Banned '# type: ignore' comment.")

    for node in ast.walk(tree):
        violations.extend(_check_ast_node_discipline(node, filepath))

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for typing and code discipline verification.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments specifying paths to check.

    Returns:
        int: Zero if all files pass, 1 otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    paths = [Path(p) for p in args] if args else [Path("src"), Path("scripts"), Path("tests")]
    all_violations: list[str] = []

    for path in paths:
        if not path.exists():
            print(f"Error: {path} does not exist.", file=sys.stderr)
            return 1
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for py_file in files:
            all_violations.extend(check_file_typing(py_file))

    for violation in all_violations:
        print(violation, file=sys.stderr)
    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
