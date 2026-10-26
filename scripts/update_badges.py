#!/usr/bin/env python3
"""Script to update test, doc, and API compliance badges in ``README.md``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import calc_compliance, check_docstrings


def get_color(pct: float) -> str:
    """Select a shields.io badge color for a percentage value.

    Args:
        pct (float): Coverage or compliance percentage in ``[0.0, 100.0]``.

    Returns:
        str: Badge color name.
    """
    if pct >= 95.0:
        return "brightgreen"
    if pct >= 80.0:
        return "green"
    if pct >= 60.0:
        return "yellow"
    return "red"


def format_badge_block(
    test_pct: float = 100.0,
    doc_pct: float = 100.0,
    api_pct: float = 100.0,
) -> str:
    """Format the three markdown badge lines for test, doc, and API compliance.

    Args:
        test_pct (float): Test coverage percentage.
        doc_pct (float): Documentation coverage percentage.
        api_pct (float): API compliance percentage.

    Returns:
        str: Multi-line markdown badge string.
    """
    test_str = f"{test_pct:.1f}".rstrip("0").rstrip(".")
    doc_str = f"{doc_pct:.1f}".rstrip("0").rstrip(".")
    api_str = f"{api_pct:.1f}".rstrip("0").rstrip(".")

    test_color = get_color(test_pct)
    doc_color = get_color(doc_pct)
    api_color = get_color(api_pct)

    return chr(10).join(
        [
            f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
            f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
            f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)",
        ]
    )


def compute_doc_pct(targets: Sequence[Path] = (Path("src"), Path("scripts"), Path("tests"))) -> float:
    """Compute documentation compliance percentage across ``targets``.

    Args:
        targets (Sequence[Path]): Directories or files to inspect.

    Returns:
        float: Percentage of files with zero docstring violations.
    """
    files: list[Path] = []
    for target in targets:
        if target.is_file():
            files.append(target)
        else:
            files.extend(sorted(target.rglob("*.py")))
    if not files:
        return 100.0
    clean = sum(1 for py_file in files if not check_docstrings.check_file_docstrings(py_file))
    return (clean / len(files)) * 100.0


def update_readme(
    readme_path: Path = Path("README.md"),
    test_pct: float = 100.0,
    doc_pct: float | None = None,
    api_pct: float | None = None,
) -> bool:
    """Update the coverage and compliance badges in ``readme_path``.

    Args:
        readme_path (Path): Path to ``README.md``.
        test_pct (float): Test coverage percentage.
        doc_pct (float | None): Documentation coverage percentage (computed dynamically if None).
        api_pct (float | None): API compliance percentage (computed dynamically if None).

    Returns:
        bool: True if ``readme_path`` existed and was processed, False otherwise.
    """
    if not readme_path.exists():
        return False

    effective_doc_pct = compute_doc_pct() if doc_pct is None else doc_pct
    if api_pct is None:
        total, covered, _ = calc_compliance.compute_compliance()
        effective_api_pct = (covered / total * 100.0) if total > 0 else 100.0
    else:
        effective_api_pct = api_pct

    content = readme_path.read_text(encoding="utf-8")
    badges = format_badge_block(
        test_pct=test_pct,
        doc_pct=effective_doc_pct,
        api_pct=effective_api_pct,
    )
    lines = badges.splitlines()

    content = re.sub(
        r"\[!\[Test Coverage\]\(https://img\.shields\.io/badge/test_coverage-.*?\)\]\(#\)",
        lines[0],
        content,
    )
    content = re.sub(
        r"\[!\[Doc Coverage\]\(https://img\.shields\.io/badge/doc_coverage-.*?\)\]\(#\)",
        lines[1],
        content,
    )
    if "[![API Compliance]" in content:
        content = re.sub(
            r"\[!\[API Compliance\]\(https://img\.shields\.io/badge/api_compliance-.*?\)\]\(#\)",
            lines[2],
            content,
        )
    elif "[![Doc Coverage]" in content:
        content = content.replace(lines[1], lines[1] + chr(10) + lines[2])

    readme_path.write_text(content, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for updating ``README.md`` badges.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments ``[readme_path]``.

    Returns:
        int: Zero on success.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    target = Path(args[0]) if args else Path("README.md")
    update_readme(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
