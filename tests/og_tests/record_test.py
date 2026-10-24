"""Official Grain tests for ``Record`` and ``RecordMetadata``."""

from __future__ import annotations

from absl.testing import absltest
import numpy as np
from zero_grain import python as record


class RecordTest(absltest.TestCase):
    """Test suite for ``RecordMetadata`` string representations."""

    def test_RecordMetadata_str(self) -> None:
        """Verify ``RecordMetadata.__str__`` with a populated NumPy generator."""
        record_metadata = record.RecordMetadata(index=0, record_key=0, rng=np.random.default_rng())
        self.assertEqual(
            str(record_metadata),
            "RecordMetadata(index=0, record_key=0, rng=Generator(PCG64))",
        )

    def test_RecordMetadata_str_none_rng(self) -> None:
        """Verify ``RecordMetadata.__str__`` when ``rng`` is None."""
        record_metadata = record.RecordMetadata(index=0, record_key=0)
        self.assertStartsWith(
            str(record_metadata),
            "RecordMetadata(index=0, record_key=0, rng=None",
        )


if __name__ == "__main__":
    absltest.main()
