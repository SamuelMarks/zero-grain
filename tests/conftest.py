"""Module docstring."""

import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler as ml_switcheroo


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    """Docstring for switcheroo_config."""
    with ml_switcheroo.EagerMode():
        yield
