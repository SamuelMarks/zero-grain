"""Pytest configuration for the zero_grain test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import shutil
import pytest
import zero_grain


@pytest.fixture(autouse=True)
def zero_grain_test_env() -> Iterator[None]:
    """Provide a clean test execution context and remove ephemeral testdata after each test.

    Yields:
        None: Execution control to the running test.
    """
    pkg_root = Path(zero_grain.__file__).resolve().parent
    try:
        yield
    finally:
        shutil.rmtree(pkg_root / "_src", ignore_errors=True)
        shutil.rmtree(pkg_root / "python" / "_src", ignore_errors=True)
