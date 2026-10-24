"""Pytest configuration for the zero_grain test suite."""

from __future__ import annotations

from collections.abc import Iterator
import pytest


@pytest.fixture(autouse=True)
def zero_grain_test_env() -> Iterator[None]:
    """Provide a clean test execution context for each test function.

    Yields:
        None: Execution control to the running test.
    """
    yield
