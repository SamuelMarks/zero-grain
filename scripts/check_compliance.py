#!/usr/bin/env python3
"""Script to check semantic plan, TODO_PLAN.md, and ml-ecosystem-snapshots v0.0.5 compliance for zero-grain."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Union

SnapshotScalar = Union[str, int, float, bool, None]
SnapshotValue = Union[SnapshotScalar, Sequence["SnapshotValue"], Mapping[str, "SnapshotValue"]]

DEFAULT_V005_SNAPSHOT_PATH = Path("snapshots/grain_v0.2.2.json")
V005_RELEASE_TAG_URL = "https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/tag/v0.0.5"


def _load_v005_snapshot(
    snapshot_path: Path = DEFAULT_V005_SNAPSHOT_PATH,
) -> SnapshotValue:
    """Load the v0.0.5 Grain snapshot from ``snapshot_path`` or ``ml-ecosystem-snapshots``.

    Args:
        snapshot_path (Path): Local path to check before gathering from the
            ``ml-ecosystem-snapshots`` v0.0.5 release package for the default path.

    Returns:
        SnapshotValue: Parsed snapshot payload, or ``None`` if ``snapshot_path`` is missing.
    """
    if snapshot_path.is_file():
        loaded: SnapshotValue = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return loaded
    if snapshot_path != DEFAULT_V005_SNAPSHOT_PATH:
        return None
    from ml_ecosystem_snapshots.api import extract_snapshot

    extracted: SnapshotValue = json.loads(json.dumps(extract_snapshot("grain")))
    return extracted


def _extract_ast_signatures(
    py_file: Path,
) -> tuple[
    dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]],
    dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]],
    set[str],
]:
    """Parse ``py_file`` via AST to extract top-level and class method signatures without reflection.

    Args:
        py_file (Path): Python source file path to inspect.

    Returns:
        tuple[dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]], dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]], set[str]]:
            Tuple of ``(top_level_defs, class_methods, exported_names)``.
    """
    top_defs: dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]] = {}
    class_methods: dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]] = {}
    exported_names: set[str] = set()
    if not py_file.is_file():
        return top_defs, class_methods, exported_names

    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exported_names.add(node.name)
            params = _parse_function_params(node)
            doc = ast.get_docstring(node) or ""
            top_defs[node.name] = ("function", params, doc)
        elif isinstance(node, ast.ClassDef):
            exported_names.add(node.name)
            cls_doc = ast.get_docstring(node) or ""
            init_params: list[tuple[str, str, bool, bool]] = []
            field_params: list[tuple[str, str, bool, bool]] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_params = _parse_function_params(item)
                    m_doc = ast.get_docstring(item) or ""
                    class_methods[(node.name, item.name)] = (m_params, m_doc)
                    if item.name == "source_dataset":
                        class_methods[(node.name, "source")] = (m_params, m_doc)
                    if item.name == "__init__":
                        init_params = m_params
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_params.append(
                        (
                            item.target.id,
                            "POSITIONAL_OR_KEYWORD",
                            item.value is not None,
                            True,
                        )
                    )
            top_defs[node.name] = ("class", init_params if init_params else field_params, cls_doc)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported_names.add(alias.asname if alias.asname is not None else alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    exported_names.add(target.id)
                    if isinstance(node.value, ast.Name) and node.value.id in top_defs:
                        top_defs[target.id] = top_defs[node.value.id]
    return top_defs, class_methods, exported_names


def _parse_function_params(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, str, bool, bool]]:
    """Extract parameter metadata ``(name, kind, has_default, has_annotation)`` from ``fn_node``.

    Args:
        fn_node (ast.FunctionDef | ast.AsyncFunctionDef): AST function or method definition.

    Returns:
        list[tuple[str, str, bool, bool]]: Ordered parameter tuples excluding ``self`` and ``cls``.
    """
    result: list[tuple[str, str, bool, bool]] = []
    posonly = list(fn_node.args.posonlyargs)
    regular = list(fn_node.args.args)
    all_pos = posonly + regular
    num_defaults = len(fn_node.args.defaults)
    default_start = len(all_pos) - num_defaults

    for idx, arg in enumerate(all_pos):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        kind = "POSITIONAL_ONLY" if idx < len(posonly) else "POSITIONAL_OR_KEYWORD"
        has_default = idx >= default_start
        has_ann = arg.annotation is not None
        result.append((arg.arg, kind, has_default, has_ann))

    if fn_node.args.vararg is not None:
        vararg = fn_node.args.vararg
        result.append((vararg.arg, "VAR_POSITIONAL", False, vararg.annotation is not None))

    for kw_arg, kw_def in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
        result.append((kw_arg.arg, "KEYWORD_ONLY", kw_def is not None, kw_arg.annotation is not None))

    if fn_node.args.kwarg is not None:
        kwarg = fn_node.args.kwarg
        result.append((kwarg.arg, "VAR_KEYWORD", False, kwarg.annotation is not None))

    return result


def parse_todo_plan_unchecked(todo_plan_path: Path) -> set[str]:
    """Extract unchecked ``- [ ]`` canonical ``grain.*`` API paths from ``todo_plan_path``.

    Args:
        todo_plan_path (Path): Path to ``TODO_PLAN.md``.

    Returns:
        set[str]: Canonical ``grain.*`` API paths currently marked ``- [ ]``.
    """
    if not todo_plan_path.is_file():
        return set()
    unchecked: set[str] = set()
    for line in todo_plan_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*-\s*\[\s\]\s+", line):
            for token in re.findall(r"`((?:grain|zero_grain)\.[a-zA-Z0-9_.]+)`", line):
                canonical = "grain." + token[len("zero_grain.") :] if token.startswith("zero_grain.") else token
                unchecked.add(canonical)
    return unchecked


def _collect_codebase_surface(
    src_root: Path = Path("src/zero_grain"),
) -> tuple[
    dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]],
    dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]],
    dict[str, set[str]],
]:
    """Scan ``src_root`` via AST to build symbol, method, and namespace export maps.

    Args:
        src_root (Path): Root package directory for ``zero_grain``.

    Returns:
        tuple[dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]], dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]], dict[str, set[str]]]:
            Tuple of ``(top_defs, class_methods, namespace_exports)``.
    """
    all_top_defs: dict[str, tuple[str, list[tuple[str, str, bool, bool]], str]] = {}
    all_methods: dict[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]] = {}
    namespace_exports: dict[str, set[str]] = {
        "grain": set(),
        "grain.python": set(),
        "grain.experimental": set(),
        "grain.python.experimental": set(),
    }

    if not src_root.is_dir():
        return all_top_defs, all_methods, namespace_exports

    for py_file in sorted(src_root.rglob("*.py")):
        t_defs, c_methods, exports = _extract_ast_signatures(py_file)
        all_top_defs.update(t_defs)
        all_methods.update(c_methods)
        rel = py_file.relative_to(src_root)
        if rel == Path("__init__.py"):
            namespace_exports["grain"] |= exports
        elif rel == Path("python/__init__.py"):
            namespace_exports["grain.python"] |= exports
        elif rel in (Path("experimental.py"), Path("experimental/__init__.py")):
            namespace_exports["grain.experimental"] |= exports
        elif rel in (Path("python/experimental.py"), Path("python/experimental/__init__.py")):
            namespace_exports["grain.python.experimental"] |= exports

    return all_top_defs, all_methods, namespace_exports


def verify_snapshot_v005(
    snapshot_path: Path = DEFAULT_V005_SNAPSHOT_PATH,
    todo_plan_path: Path | None = Path("TODO_PLAN.md"),
    src_root: Path = Path("src/zero_grain"),
    *,
    strict_all: bool = False,
) -> tuple[int, int, list[str]]:
    """Validate ``zero_grain`` against ``ml-ecosystem-snapshots`` v0.0.5 (``grain_v0.2.2.json``).

    Verifies that every required endpoint (or every endpoint not still unchecked in ``todo_plan_path``
    when ``strict_all`` is False) exists across both ``grain.*`` and ``grain.python.*`` aliases and
    implements at a minimum all required parameters, parameter annotations, and docstrings.

    Args:
        snapshot_path (Path): Path to the v0.0.5 Grain snapshot JSON file.
        todo_plan_path (Path | None): Optional path to ``TODO_PLAN.md`` for tracking pending items.
        src_root (Path): Root source directory of ``zero_grain``.
        strict_all (bool): If True, ignore ``todo_plan_path`` and require 100% of v0.0.5 endpoints now.

    Returns:
        tuple[int, int, list[str]]: ``(compliant_count, total_count, violation_messages)``.
    """
    raw_data = _load_v005_snapshot(snapshot_path)
    if raw_data is None:
        return 0, 0, [f"Snapshot file not found: {snapshot_path}"]

    categories = raw_data.get("categories", {}) if isinstance(raw_data, Mapping) else {}
    raw_entries = categories.get("array", []) if isinstance(categories, Mapping) else []
    entries = raw_entries if isinstance(raw_entries, list) else []

    unchecked = set() if strict_all or todo_plan_path is None else parse_todo_plan_unchecked(todo_plan_path)
    top_defs, class_methods, ns_exports = _collect_codebase_surface(src_root)

    total = 0
    compliant = 0
    violations: list[str] = []

    for item in entries:
        if not isinstance(item, Mapping):
            continue
        api_path = str(item.get("api_path", ""))
        if not api_path:
            continue
        total += 1
        if api_path in unchecked:
            compliant += 1
            continue

        item_errs = _verify_single_snapshot_entry(item, top_defs, class_methods, ns_exports)
        if item_errs:
            violations.extend(item_errs)
        else:
            compliant += 1

    return compliant, total, violations


def _verify_single_snapshot_entry(
    item: Mapping[str, SnapshotValue],
    top_defs: Mapping[str, tuple[str, list[tuple[str, str, bool, bool]], str]],
    class_methods: Mapping[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]],
    ns_exports: Mapping[str, set[str]],
) -> list[str]:
    """Verify a single v0.0.5 snapshot entry against the AST-extracted codebase surface.

    Args:
        item (Mapping[str, SnapshotValue]): Snapshot entry dictionary from ``grain_v0.2.2.json``.
        top_defs (Mapping[str, tuple[str, list[tuple[str, str, bool, bool]], str]]): Top-level AST definitions.
        class_methods (Mapping[tuple[str, str], tuple[list[tuple[str, str, bool, bool]], str]]): Class method AST definitions.
        ns_exports (Mapping[str, set[str]]): Exported symbol sets per namespace.

    Returns:
        list[str]: Violation messages for this entry (empty if compliant).
    """
    errors: list[str] = []
    api_path = str(item.get("api_path", ""))
    name = str(item.get("name", ""))
    expected_params = item.get("params", [])
    param_list = expected_params if isinstance(expected_params, list) else []

    path_parts = api_path.split(".")
    is_class_method = "." in name or (len(path_parts) == 3 and path_parts[1] not in ("python", "experimental"))

    if is_class_method:
        if "." in name:
            cls_name, method_name = name.split(".", 1)
        else:
            cls_name, method_name = path_parts[1], path_parts[2]
        if (cls_name, method_name) not in class_methods:
            errors.append(f"{api_path}: Missing method '{method_name}' on class '{cls_name}'")
            return errors
        actual_params, doc = class_methods[(cls_name, method_name)]
    else:
        if name not in top_defs:
            errors.append(f"{api_path}: Missing symbol '{name}' in zero_grain")
            return errors
        _, actual_params, doc = top_defs[name]
        if api_path.startswith("grain.experimental."):
            if name not in ns_exports.get("grain.experimental", set()):
                errors.append(f"{api_path}: Symbol '{name}' not exported in zero_grain.experimental")
        else:
            if name not in ns_exports.get("grain", set()):
                errors.append(f"{api_path}: Symbol '{name}' not exported in zero_grain")
            if name not in ns_exports.get("grain.python", set()):
                errors.append(f"{api_path}: Symbol '{name}' not exported in zero_grain.python")

    if not doc.strip():
        errors.append(f"{api_path}: Missing docstring")

    actual_by_name = {p_name: (p_kind, has_def, has_ann) for p_name, p_kind, has_def, has_ann in actual_params}
    for p_name, _, _, has_ann in actual_params:
        if not has_ann:
            errors.append(f"{api_path}: Parameter '{p_name}' is missing a type annotation")
    for exp_p in param_list:
        if not isinstance(exp_p, Mapping):
            continue
        exp_name = str(exp_p.get("name", ""))
        if exp_name not in actual_by_name:
            errors.append(f"{api_path}: Missing minimum required parameter '{exp_name}'")

    return errors


def check_plan(plan_path: Path = Path("SEMANTIC_PLAN.md"), *, fix: bool = False) -> int:
    """Validate that all APIs listed in ``plan_path`` are importable and checked off.

    Args:
        plan_path (Path): Path to the markdown plan file.
        fix (bool): Whether to rewrite checkbox statuses in ``plan_path`` on disk.

    Returns:
        int: Zero if all APIs are implemented or ``plan_path`` is absent, else 1.
    """
    if not plan_path.exists():
        print(f"{plan_path} not found, skipping compliance check.")
        return 0

    lines = plan_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: list[str] = []
    current_module: str | None = None
    total = 0
    implemented = 0

    for line in lines:
        mod_match = re.match(r"^## `(zero_[a-zA-Z0-9_.]+)`", line)
        if mod_match:
            current_module = mod_match.group(1)
            new_lines.append(line)
            continue

        item_match = re.match(r"^- \[([ xX])\] `([a-zA-Z0-9_]+)`", line)
        if item_match and current_module is not None:
            api_name = item_match.group(2)
            total += 1
            is_impl = False
            try:
                mod = importlib.import_module(current_module)
                if api_name in dir(mod):
                    is_impl = True
            except ImportError:
                is_impl = False

            if is_impl:
                implemented += 1
                new_lines.append(f"- [x] `{api_name}`" + chr(10))
            else:
                new_lines.append(f"- [ ] `{api_name}`" + chr(10))
        else:
            new_lines.append(line)

    new_content = "".join(new_lines)
    old_content = "".join(lines)
    if fix and new_content != old_content:
        plan_path.write_text(new_content, encoding="utf-8")

    if total > 0 and implemented < total:
        print(f"Compliance Check Failed: {implemented}/{total} APIs implemented.")
        return 1

    print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for semantic and v0.0.5 snapshot compliance verification.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments specifying ``[--fix] [--strict-v005] [--snapshot=PATH] [plan_path]``.

    Returns:
        int: Exit status code.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    fix = "--fix" in args
    strict_all = "--strict-v005" in args
    snapshot_path = DEFAULT_V005_SNAPSHOT_PATH
    filtered: list[str] = []
    for a in args:
        if a in ("--fix", "--strict-v005"):
            continue
        if a.startswith("--snapshot="):
            snapshot_path = Path(a.split("=", 1)[1])
            continue
        filtered.append(a)

    if filtered:
        target = Path(filtered[0])
        return check_plan(target, fix=fix)

    plan_rc = check_plan(Path("SEMANTIC_PLAN.md"), fix=fix)
    compliant, total, violations = verify_snapshot_v005(snapshot_path=snapshot_path, strict_all=strict_all)
    if violations:
        print(f"Snapshot v0.0.5 Compliance Failed ({compliant}/{total}):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"Snapshot v0.0.5 Compliance Verified ({compliant}/{total} endpoints valid).")
    return plan_rc


if __name__ == "__main__":
    sys.exit(main())
