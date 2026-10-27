#!/usr/bin/env python3
"""AST-based static typing verifier entry point delegating to ``check_typing_discipline``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import check_typing_discipline  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AST-based static typing discipline check.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments specifying target paths.

    Returns:
        int: Zero if compliant, or 1 if any typing discipline violations are found.
    """
    return check_typing_discipline.main(argv)


if __name__ == "__main__":
    sys.exit(main())
