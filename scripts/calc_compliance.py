#!/usr/bin/env python3
"""Script to calculate API and signature compliance against the Grain snapshot."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys


def compute_compliance(
    snapshot_path: Path = Path("snapshots/pygrain_v0.2.16.dev20260112.json"),
) -> tuple[int, int, set[str]]:
    """Compare exported symbols in ``zero_grain`` against ``snapshot_path``.

    Args:
        snapshot_path (Path): Path to the Grain snapshot JSON file.

    Returns:
        tuple[int, int, set[str]]: Tuple of ``(expected_count, covered_count, missing_names)``.
    """
    raw_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    categories = raw_data.get("categories", {})
    apis = categories.get("extras", [])
    expected: set[str] = {str(api["name"]) for api in apis if "name" in api}

    import zero_grain
    import zero_grain.python

    exported: set[str] = set()
    for sym_name in dir(zero_grain.python):
        if sym_name.startswith("_"):
            continue
        exported.add(sym_name)
        sym_obj = zero_grain.python.__dict__.get(sym_name)
        if isinstance(sym_obj, type):
            exported.add(sym_obj.__name__)

    for sym_name in dir(zero_grain):
        if not sym_name.startswith("_"):
            exported.add(sym_name)

    missing = expected - exported
    covered = expected.intersection(exported)
    return len(expected), len(covered), missing


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API compliance calculation and print summary metrics.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments (first argument overrides snapshot path).

    Returns:
        int: Zero if 100% compliant, or 1 if any APIs are missing.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    snap_path = Path(args[0]) if args else Path("snapshots/pygrain_v0.2.16.dev20260112.json")
    total, covered_count, missing = compute_compliance(snap_path)
    pct = (covered_count / total * 100.0) if total > 0 else 100.0

    print(f"Total APIs: {total}")
    print(f"Covered APIs: {covered_count}")
    print(f"Missing APIs: {missing}")
    print(f"Compliance: {pct:.2f}%")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
