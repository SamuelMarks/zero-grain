"""Module docstring."""

import sys
from unittest.mock import MagicMock

sys.modules["zero_jax"] = MagicMock()
sys.modules["zero_jax.tree_util"] = MagicMock()
