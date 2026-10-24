#!/usr/bin/env python3
"""Script to check semantic plan and module compliance for zero-grain."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import re
import sys


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
            except Exception:
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
    """CLI entry point for semantic compliance verification.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments specifying ``[--fix] [plan_path]``.

    Returns:
        int: Exit status code.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    fix = "--fix" in args
    filtered = [a for a in args if a != "--fix"]
    target = Path(filtered[0]) if filtered else Path("SEMANTIC_PLAN.md")
    return check_plan(target, fix=fix)


if __name__ == "__main__":
    sys.exit(main())
