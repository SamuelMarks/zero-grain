"""Official Grain tests for ``RangeDataSource``, ``SharedMemoryDataSource``, and ``ArrayRecordDataSource``."""

from __future__ import annotations

from collections.abc import Sequence
import multiprocessing as grain_multiprocessing
import pathlib
import platform

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from zero_grain import python as data_sources
from zero_grain import python as dataset_base

FLAGS = flags.FLAGS


class DataSourceTest(parameterized.TestCase):
    """Base test class for data sources."""

    def setUp(self) -> None:
        """Set up the test directory path."""
        super().setUp()
        self.testdata_dir = pathlib.Path(".")


class RangeDataSourceTest(DataSourceTest):
    """Test suite for ``RangeDataSource``."""

    @parameterized.parameters(
        [
            (0, 10, 2),
            (10, 0, -2),
            (0, 0, 1),
        ]
    )
    def test_range_data_source(self, start: int, stop: int, step: int) -> None:
        """Verify ``RangeDataSource`` matches Python ``range(start, stop, step)``.

        Args:
            start (int): Range start value.
            stop (int): Range stop value.
            step (int): Range step value.
        """
        expected_output = list(range(start, stop, step))
        range_ds = data_sources.RangeDataSource(start, stop, step)
        actual_output = [range_ds[i] for i in range(len(range_ds))]
        self.assertEqual(expected_output, actual_output)


class InMemoryDataSourceTest(DataSourceTest):
    """Test suite for ``SharedMemoryDataSource``."""

    def test_single_process(self) -> None:
        """Verify single-process indexing and iteration."""
        sequence = list(range(12))
        in_memory_ds = data_sources.SharedMemoryDataSource(sequence)

        output_by_index = [in_memory_ds[i] for i in range(len(in_memory_ds))]
        self.assertEqual(sequence, output_by_index)

        output_by_list = list(in_memory_ds)
        self.assertEqual(sequence, output_by_list)

        in_memory_ds.close()
        in_memory_ds.unlink()

    @staticmethod
    def read_elements(
        in_memory_ds: data_sources.SharedMemoryDataSource[int],
        indices: Sequence[int],
    ) -> Sequence[int]:
        """Read elements at ``indices`` from ``in_memory_ds``.

        Args:
            in_memory_ds (data_sources.SharedMemoryDataSource[int]): Shared data source.
            indices (Sequence[int]): Indices to read.

        Returns:
            Sequence[int]: List of elements at ``indices``.
        """
        return [in_memory_ds[i] for i in indices]

    def test_multi_processes_co_read(self) -> None:
        """Verify concurrent reads across spawned worker processes."""
        sequence = list(range(12))
        in_memory_ds = data_sources.SharedMemoryDataSource(sequence, name="DataSourceTestingCoRead")

        num_processes = 3
        indices_for_processes = [[1, 3, 5], [2, 3, 4], [5, 2, 3]]
        expected_elements_read = [[in_memory_ds[i] for i in indices] for indices in indices_for_processes]

        mp_context = grain_multiprocessing.get_context("spawn")
        with mp_context.Pool(processes=num_processes) as pool:
            elements_read = pool.starmap(
                InMemoryDataSourceTest.read_elements,
                zip([in_memory_ds] * num_processes, indices_for_processes),
            )
            pool.close()
            pool.join()

        self.assertEqual(elements_read, expected_elements_read)

        in_memory_ds.close()
        in_memory_ds.unlink()

    def test_empty_sequence(self) -> None:
        """Verify ``SharedMemoryDataSource([])`` is empty."""
        in_memory_ds: data_sources.SharedMemoryDataSource[int] = data_sources.SharedMemoryDataSource([])
        self.assertEmpty(in_memory_ds)

        in_memory_ds.close()
        in_memory_ds.unlink()

    def test_str(self) -> None:
        """Verify ``SharedMemoryDataSource.__str__`` output."""
        sequence = list(range(12))
        name = "DataSourceTestingStr"
        in_memory_ds = data_sources.SharedMemoryDataSource(sequence, name=name)
        actual_str = str(in_memory_ds)
        self.assertEqual(
            actual_str,
            f"SharedMemoryDataSource(name={name}, len={len(sequence)})",
        )

        in_memory_ds.close()
        in_memory_ds.unlink()


@absltest.skipIf(platform.system() == "Windows", "ArrayRecord isn't supported on Windows.")
class ArrayRecordDataSourceTest(DataSourceTest):
    """Test suite for ``ArrayRecordDataSource``."""

    def test_array_record_data_implements_random_access(self) -> None:
        """Verify ``ArrayRecordDataSource`` is a ``RandomAccessDataSource`` subclass."""
        assert issubclass(data_sources.ArrayRecordDataSource, dataset_base.RandomAccessDataSource)

    def test_array_record_source_empty_sequence(self) -> None:
        """Verify ``ArrayRecordDataSource([])`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            data_sources.ArrayRecordDataSource([])

    def test_array_record_file_instructions_and_bounds(self) -> None:
        """Verify ``FileInstruction`` skip/take windowing and out-of-bounds ``IndexError``."""
        inst1 = data_sources.FileInstruction(filename="shard_0", skip=1, take=3, examples_in_shard=5)
        inst2 = data_sources.FileInstruction(filename="shard_1", skip=2, take=-1, examples_in_shard=5)
        ar_ds = data_sources.ArrayRecordDataSource([inst1, inst2])
        self.assertEqual(len(ar_ds), 6)
        self.assertEqual([ar_ds[i] for i in range(len(ar_ds))], [b"1", b"2", b"3", b"5", b"6", b"7"])
        with self.assertRaises(IndexError):
            _ = ar_ds[-1]
        with self.assertRaises(IndexError):
            _ = ar_ds[6]


if __name__ == "__main__":
    absltest.main()
