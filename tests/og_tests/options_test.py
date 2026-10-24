"""Official Grain tests for ``ReadOptions``."""

from __future__ import annotations

from unittest import mock

from absl import logging
from absl.testing import absltest
from zero_grain import python as options


class ReadOptionsTest(absltest.TestCase):
    """Test suite for ``ReadOptions`` validation and logging."""

    def test_defaults(self) -> None:
        """Verify default ``num_threads`` and ``prefetch_buffer_size`` values."""
        ro = options.ReadOptions()
        self.assertEqual(ro.num_threads, 16)
        self.assertEqual(ro.prefetch_buffer_size, 500)

    def test_num_threads_negative_raises_value_error(self) -> None:
        """Verify negative ``num_threads`` raises ``ValueError``."""
        with self.assertRaisesRegex(ValueError, "num_threads must be non-negative"):
            options.ReadOptions(num_threads=-1)

    def test_prefetch_buffer_size_negative_raises_value_error(self) -> None:
        """Verify negative ``prefetch_buffer_size`` raises ``ValueError``."""
        with self.assertRaisesRegex(ValueError, "prefetch_buffer_size must be non-negative"):
            options.ReadOptions(prefetch_buffer_size=-1)

    def test_prefetch_buffer_size_less_than_num_threads_logs_warning(self) -> None:
        """Verify warning is logged when ``0 < prefetch_buffer_size < num_threads``."""
        with self.assertLogs(level="WARNING") as logs:
            options.ReadOptions(num_threads=10, prefetch_buffer_size=5)
        self.assertIn("prefetch_buffer_size=5 is smaller than num_threads=10", logs.output[0])

    def test_prefetch_buffer_size_zero(self) -> None:
        """Verify no warning is logged when ``prefetch_buffer_size == 0``."""
        with mock.patch.object(logging, "warning") as mock_warning:
            options.ReadOptions(num_threads=10, prefetch_buffer_size=0)
            mock_warning.assert_not_called()


if __name__ == "__main__":
    absltest.main()
