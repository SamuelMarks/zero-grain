"""Official Grain tests for ``DataLoader`` and ``PyGrainDatasetIterator``."""

from __future__ import annotations

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator, Sequence
import functools
import pathlib
import platform
import sys

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized as absl_parameterized
import numpy as np
import parameterized
from zero_grain import python as batch
from zero_grain import python as data_loader_lib
from zero_grain import python as options
from zero_grain import python as samplers
from zero_grain import python as sharding
from zero_grain import python as shared_memory_array
from zero_grain import python as transforms
from zero_grain._types import ElementTree, RNGGeneratorProtocol
from zero_grain.data_loader import TransformationSpec
from zero_grain.python import ArrayRecordDataSource
from zero_grain.python import BatchOperation
from zero_grain.python import FilterOperation
from zero_grain.python import MapOperation
from zero_grain.python import RangeDataSource
from zero_grain.python import SharedMemoryDataSource
from zero_grain.python import assert_equal_output_after_checkpoint

FLAGS = flags.FLAGS


def setup_module() -> None:
    """Set up the module for testing."""
    from zero_grain import python as grain  # pylint: disable=g-import-not-at-top

    srcdir = pathlib.Path(grain.__file__).parents[0] / "_src" / "python"
    FLAGS["test_srcdir"].parse(str(srcdir))


def map_function(data: ElementTree) -> ElementTree:
    """Increment ``data`` by 1.

    Args:
        data (ElementTree): Input numeric value.

    Returns:
        ElementTree: Incremented numeric value.
    """
    return int(data) + 1 if isinstance(data, int) else data


def condition_function(data: ElementTree) -> bool:
    """Return True if ``data`` is even.

    Args:
        data (ElementTree): Input integer value.

    Returns:
        bool: True if ``data`` is divisible by 2.
    """
    return isinstance(data, int) and data % 2 == 0


class FilterEven(transforms.Filter[ElementTree]):
    """Filter keeping only even integers."""

    def filter(self, x: ElementTree) -> bool:
        """Return True when ``x`` is even.

        Args:
            x (ElementTree): Input integer.

        Returns:
            bool: True if ``x % 2 == 0``.
        """
        return isinstance(x, int) and x % 2 == 0


class PlusOne(transforms.Map[ElementTree, ElementTree]):
    """Map transform adding 1 to each integer."""

    def map(self, x: ElementTree) -> ElementTree:
        """Add 1 to ``x``.

        Args:
            x (ElementTree): Input integer.

        Returns:
            ElementTree: ``x + 1``.
        """
        return int(x) + 1 if isinstance(x, int) else x


class PlusRandom(transforms.RandomMap[ElementTree, ElementTree]):
    """RandomMap transform adding a random integer offset."""

    def random_map(self, x: ElementTree, rng: RNGGeneratorProtocol) -> ElementTree:
        """Add a random integer from ``rng`` to ``x``.

        Args:
            x (ElementTree): Input integer.
            rng (RNGGeneratorProtocol): Per-record RNG.

        Returns:
            ElementTree: Randomly offset integer.
        """
        base = int(x) if isinstance(x, int) else 0
        return base + rng.integers(100_000)


class FailingMap(transforms.Map[ElementTree, ElementTree]):
    """Map transform that raises ``ZeroDivisionError``."""

    def map(self, x: ElementTree) -> ElementTree:
        """Raise ``ZeroDivisionError``.

        Args:
            x (ElementTree): Ignored input element.

        Returns:
            ElementTree: Never returns.
        """
        del x
        return int(1 / 0)


class NonPickableTransform(transforms.Map[ElementTree, ElementTree]):
    """Map transform that raises ``ValueError`` when pickled."""

    def __getstate__(self) -> dict[str, ElementTree]:
        """Raise ``ValueError`` to prevent pickling.

        Returns:
            dict[str, ElementTree]: Never returns.

        Raises:
            ValueError: Always raised.
        """
        raise ValueError("I shall not be pickled")

    def map(self, x: ElementTree) -> ElementTree:
        """Return ``x`` unchanged.

        Args:
            x (ElementTree): Input element.

        Returns:
            ElementTree: Unchanged input element.
        """
        return x


class RaisingTransform(transforms.Map[ElementTree, ElementTree]):
    """Map transform that raises ``AttributeError``."""

    def map(self, x: ElementTree) -> ElementTree:
        """Raise ``AttributeError``.

        Args:
            x (ElementTree): Ignored input element.

        Returns:
            ElementTree: Never returns.

        Raises:
            AttributeError: Always raised.
        """
        del x
        raise AttributeError("I shall raise")


class ExitingTransform(transforms.Map[ElementTree, ElementTree]):
    """Map transform that calls ``sys.exit(123)``."""

    def map(self, x: ElementTree) -> ElementTree:
        """Terminate process with exit code 123.

        Args:
            x (ElementTree): Ignored input element.

        Returns:
            ElementTree: Never returns.

        Raises:
            SystemExit: Always raised.
        """
        del x
        sys.exit(123)


class DuplicateElementFlatMap(transforms.FlatMap[ElementTree, ElementTree]):
    """FlatMap transform duplicating each element ``max_fan_out`` times."""

    max_fan_out: int = 7

    def flat_map(self, element: ElementTree) -> Iterator[ElementTree]:
        """Yield ``element`` ``max_fan_out`` times.

        Args:
            element (ElementTree): Input element.

        Yields:
            ElementTree: Duplicated element.
        """
        for _ in range(self.max_fan_out):
            yield element


class CopyNumPyArrayToSharedMemoryTest(absltest.TestCase):
    """Test class for copy num py array to shared memory."""

    def test_copy_numpy_array_to_shared_memory(self) -> None:
        """Test copy numpy array to shared memory."""
        element = np.array([1, 2, 3, 4, 5, 6, 7])
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIsInstance(result, shared_memory_array.SharedMemoryArrayMetadata)

    def test_copy_nested_numpy_array_to_shared_memory(self) -> None:
        """Test copy nested numpy array to shared memory."""
        element_1 = np.arange(5)
        element_2 = np.arange(5)
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map([element_1, element_2])
        assert isinstance(result, list)
        self.assertIsInstance(result[0], shared_memory_array.SharedMemoryArrayMetadata)
        self.assertIsInstance(result[1], shared_memory_array.SharedMemoryArrayMetadata)

    def test_copy_skipped_non_numpy_array(self) -> None:
        """Test copy skipped non numpy array."""
        element = "randomstring"
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIs(result, element)

    def test_copy_skipped_dtype_hasobject(self) -> None:
        """Test copy skipped dtype hasobject."""

        class DT:
            """Dummy Python class for object-dtype NumPy array."""

        element = np.array([127, 128, 129], dtype=np.dtype(DT))
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIs(result, element)

    def test_copy_skipped_flags_c_contiguous(self) -> None:
        """Test copy skipped flags c contiguous."""
        element = np.arange(9).reshape(3, 3)[:, (0, 1)]
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIs(result, element)


@absltest.skipIf(platform.system() == "Windows", "Skipped due to Windows paths.")
@parameterized.parameterized_class(
    [
        {"num_threads_per_worker": None},
        {"num_threads_per_worker": 0},
        {"num_threads_per_worker": 15},
    ]
)
class DataLoaderTest(absl_parameterized.TestCase):
    """Test class for data loader."""

    num_threads_per_worker: int | None

    def tearDown(self) -> None:
        """Tear down the test environment."""
        super().tearDown()

    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        self.testdata_dir = pathlib.Path(FLAGS.test_srcdir) / "testdata"
        self.read_options = (
            options.ReadOptions(num_threads=self.num_threads_per_worker)
            if (self.num_threads_per_worker is not None)
            else None
        )

    def _create_data_loader_for_short_sequence(
        self,
        transformations: Sequence[TransformationSpec[ElementTree, ElementTree]],
        *,
        worker_count: int = 0,
        seed: int | None = None,
    ) -> data_loader_lib.DataLoader[int]:
        """Create a ``DataLoader`` over ``RangeDataSource(0, 8, 1)``.

        Args:
            transformations (Sequence[TransformationSpec[ElementTree, ElementTree]]):
                Sequence of operations or transforms.
            worker_count (int): Number of worker processes.
            seed (int | None): Optional sampler seed.

        Returns:
            data_loader_lib.DataLoader[int]: Configured ``DataLoader`` instance.
        """
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source),
            shard_options=sharding.NoSharding(),
            seed=seed,
        )
        return data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=self.read_options,
        )

    def test_fails_to_pickle(self) -> None:
        """Test fails to pickle."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [NonPickableTransform()]
        data_loader = self._create_data_loader_for_short_sequence(transformations, worker_count=2)
        with self.assertRaisesRegex(ValueError, "I shall not be pickled"):
            list(data_loader)

    def test_propagates_transform_error_with_multiprocessing(self) -> None:
        """Test propagates transform error with multiprocessing."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [RaisingTransform()]
        data_loader = self._create_data_loader_for_short_sequence(transformations, worker_count=2)
        with self.assertRaisesRegex(Exception, "I shall raise"):
            list(data_loader)

    def test_reports_multiprocessing_worker_crash(self) -> None:
        """Test reports multiprocessing worker crash."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [ExitingTransform()]
        data_loader = self._create_data_loader_for_short_sequence(transformations, worker_count=2)
        with self.assertRaisesRegex(
            RuntimeError,
            "was terminated unexpectedly with exit code 123",
        ):
            list(data_loader)

    def test_data_loader_single_process(self) -> None:
        """Test data loader single process."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_single_process_random_map(self) -> None:
        """Test data loader single process random map."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusRandom(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations, seed=1)
        actual = list(data_loader)
        self.assertLen(actual, 4)
        for i in range(4):
            item = actual[i]
            assert isinstance(item, np.ndarray)
            self.assertEqual(item.shape, (2,))

    def test_data_loader_single_process_legacy_operations(self) -> None:
        """Test that old style operations that implement __call__() still work."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            MapOperation(map_function=map_function),
            FilterOperation(condition_function=condition_function),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_single_process_iterate_twice(self) -> None:
        """Test data loader single process iterate twice."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_in_memory_data_source(self) -> None:
        """Test data loader in memory data source."""
        data_source = SharedMemoryDataSource([0, 1, 2, 3, 4, 5, 6, 7])
        sampler = samplers.SequentialSampler(num_records=len(data_source), shard_options=sharding.NoSharding())
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        num_workers = 2
        data_loader = data_loader_lib.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_two_processes_no_shared_memory(self) -> None:
        """Test data loader two processes no shared memory."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        num_workers = 2
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_two_processes_with_shared_memory(self) -> None:
        """Test data loader two processes with shared memory."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        batch_op: BatchOperation[ElementTree] = BatchOperation(batch_size=2)
        batch_op._enable_shared_memory()
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            batch_op,
        ]
        num_workers = 2
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        actual = list(data_loader)
        self.assertLen(actual, 2)
        for item in actual:
            self.assertIsInstance(item, shared_memory_array.SharedMemoryArrayMetadata)
            assert isinstance(item, shared_memory_array.SharedMemoryArrayMetadata)
            self.assertEqual(item.shape, (2,))

    def test_data_loader_remote_exception(self) -> None:
        """Test data loader remote exception."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [FailingMap()]
        num_workers = 2
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        with self.assertRaises(ZeroDivisionError) as ctx:
            list(data_loader)
        self.assertIn("division by zero", str(ctx.exception))

    def test_data_loader_with_used_array_record_data_source(self) -> None:
        """Test data loader with used array record data source."""
        data_source = ArrayRecordDataSource(
            [
                str(self.testdata_dir / "digits.array_record-00000-of-00002"),
                str(self.testdata_dir / "digits.array_record-00001-of-00002"),
            ]
        )
        _ = data_source[0]
        sampler = samplers.SequentialSampler(num_records=len(data_source), shard_options=sharding.NoSharding())
        num_workers = 1
        data_loader = data_loader_lib.DataLoader(
            data_source=data_source,
            sampler=sampler,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        expected = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_with_invalid_number_of_workers(self) -> None:
        """Test a value error is raised when an invalid number of workers is used."""
        ar_data_source = ArrayRecordDataSource(
            [
                str(self.testdata_dir / "digits.array_record-00000-of-00002"),
                str(self.testdata_dir / "digits.array_record-00001-of-00002"),
            ]
        )
        sampler = samplers.SequentialSampler(num_records=len(ar_data_source), shard_options=sharding.NoSharding())
        num_workers = -1
        with self.assertRaises(ValueError):
            data_loader_lib.DataLoader(
                data_source=ar_data_source,
                sampler=sampler,
                worker_count=num_workers,
                read_options=self.read_options,
            )

    def create_checkpointing_dataloader(self, num_workers: int) -> data_loader_lib.DataLoader[int]:
        """Create a ``DataLoader`` instance for checkpointing tests.

        Args:
            num_workers (int): Number of worker processes.

        Returns:
            data_loader_lib.DataLoader[int]: Configured ``DataLoader``.
        """
        range_data_source = RangeDataSource(start=0, stop=16, step=1)
        sampler = samplers.IndexSampler(
            num_records=len(range_data_source),
            shard_options=sharding.NoSharding(),
            shuffle=False,
            num_epochs=1,
        )
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        return data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )

    @absl_parameterized.parameters(
        {
            "num_workers": 0,
            "steps_to_iterate": 0,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 0,
            "steps_to_iterate": 1,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 2,
            "steps_to_iterate": 1,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 2,
            "steps_to_iterate": 2,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 2,
            "expected": [
                np.array([4, 10]),
                np.array([2, 8]),
                np.array([6, 12]),
                np.array([16]),
                np.array([14]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 3,
            "expected": [
                np.array([4, 10]),
                np.array([2, 8]),
                np.array([6, 12]),
                np.array([16]),
                np.array([14]),
            ],
        },
    )
    def test_data_loader_checkpointing_object_reconstruction(
        self,
        num_workers: int,
        steps_to_iterate: int,
        expected: Sequence[np.ndarray],
    ) -> None:
        """Test ``DataLoader`` checkpointing when reconstructing a new iterator.

        Args:
            num_workers (int): Number of workers.
            steps_to_iterate (int): Steps to advance before checkpointing.
            expected (Sequence[np.ndarray]): Expected output sequence.
        """
        data_loader_iterator = iter(self.create_checkpointing_dataloader(num_workers))
        actual = [next(data_loader_iterator) for _ in range(steps_to_iterate)]
        state = data_loader_iterator.get_state()
        np.testing.assert_equal(next(data_loader_iterator), expected[steps_to_iterate])

        restored_data_loader = self.create_checkpointing_dataloader(num_workers)
        restored_data_loader_iterator = iter(restored_data_loader)
        restored_data_loader_iterator.set_state(state)

        for item in restored_data_loader_iterator:
            actual.append(item)

        np.testing.assert_equal(actual, expected)

    @absl_parameterized.parameters(
        {
            "num_workers": 0,
            "steps_to_iterate": 0,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 2,
            "steps_to_iterate": 0,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 2,
            "steps_to_iterate": 1,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 2,
            "steps_to_iterate": 2,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 0,
            "expected": [
                np.array([4, 10]),
                np.array([2, 8]),
                np.array([6, 12]),
                np.array([16]),
                np.array([14]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 2,
            "expected": [
                np.array([4, 10]),
                np.array([2, 8]),
                np.array([6, 12]),
                np.array([16]),
                np.array([14]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 3,
            "expected": [
                np.array([4, 10]),
                np.array([2, 8]),
                np.array([6, 12]),
                np.array([16]),
                np.array([14]),
            ],
        },
    )
    def test_data_loader_checkpointing_same_object(
        self,
        num_workers: int,
        steps_to_iterate: int,
        expected: Sequence[np.ndarray],
    ) -> None:
        """Test ``DataLoader`` checkpointing when restoring on the same iterator.

        Args:
            num_workers (int): Number of workers.
            steps_to_iterate (int): Steps to advance before checkpointing.
            expected (Sequence[np.ndarray]): Expected output sequence.
        """
        data_loader_iterator = iter(self.create_checkpointing_dataloader(num_workers))
        actual = [next(data_loader_iterator) for _ in range(steps_to_iterate)]
        state = data_loader_iterator.get_state()
        np.testing.assert_equal(next(data_loader_iterator), expected[steps_to_iterate])

        data_loader_iterator.set_state(state)
        for item in data_loader_iterator:
            actual.append(item)
        np.testing.assert_equal(actual, expected)

    @absl_parameterized.parameters({"num_workers": 0}, {"num_workers": 1}, {"num_workers": 2})
    def test_data_loader_sequential_checkpoint_restore_drift(self, num_workers: int) -> None:
        """Test sequential checkpoint get/set state calls do not cause drift.

        Args:
            num_workers (int): Number of workers.
        """
        data_loader_iterator = iter(self.create_checkpointing_dataloader(num_workers=num_workers))
        first_element = next(data_loader_iterator)
        np.testing.assert_equal(first_element, np.array([2, 4]))

        state1 = data_loader_iterator.get_state()
        data_loader_iterator.set_state(state1)

        state2 = data_loader_iterator.get_state()
        data_loader_iterator.set_state(state2)

        next_element = next(data_loader_iterator)
        np.testing.assert_equal(next_element, np.array([6, 8]))

    def test_batch_transform_mapped_to_batch_operation(self) -> None:
        """Test batch transform mapped to batch operation."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            transforms.Batch(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_with_batch_fn(self) -> None:
        """Test data loader with batch fn."""
        transformations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            BatchOperation(
                batch_size=3,
                batch_fn=functools.partial(batch.batch_and_pad, batch_size=3),
            ),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4, 6]), np.array([8, 0, 0])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_state_without_in_memory_data(self) -> None:
        """Test state without in memory data."""
        data_source = list(range(10000))
        loader = data_loader_lib.DataLoader(
            data_source=data_source,
            sampler=samplers.SequentialSampler(num_records=len(data_source)),
            read_options=self.read_options,
        )
        it = loader.__iter__()
        state = it.get_state()
        self.assertLess(len(state), 1000)

    def test_data_loader_with_flat_map(self) -> None:
        """Test data loader with flat map."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            DuplicateElementFlatMap(),
        ]
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            read_options=self.read_options,
        )
        np.testing.assert_equal(list(data_loader), [2] * 7 + [4] * 7 + [6] * 7 + [8] * 7)

    def test_data_loader_with_flat_map_checkpointing(self) -> None:
        """Test data loader with flat map checkpointing."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        operations: list[TransformationSpec[ElementTree, ElementTree]] = [
            PlusOne(),
            FilterEven(),
            DuplicateElementFlatMap(),
        ]
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            read_options=self.read_options,
        )
        assert_equal_output_after_checkpoint(data_loader)

    @absl_parameterized.product(worker_count=[0, 4], num_start_prefetch_calls=[1, 5])
    def test_start_prefetch(self, worker_count: int, num_start_prefetch_calls: int) -> None:
        """Test calling ``start_prefetch`` multiple times before iterating.

        Args:
            worker_count (int): Number of workers.
            num_start_prefetch_calls (int): Number of ``start_prefetch`` calls.
        """
        range_data_source = RangeDataSource(start=0, stop=16, step=1)
        sampler = samplers.SequentialSampler(num_records=len(range_data_source), shard_options=sharding.NoSharding())
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            read_options=self.read_options,
            worker_count=worker_count,
        )
        data_loader_iterator = data_loader.__iter__()
        for _ in range(num_start_prefetch_calls):
            data_loader_iterator.start_prefetch()
        self.assertEqual(list(data_loader_iterator), list(range(16)))


class PyGrainDatasetIteratorTest(absltest.TestCase):
    """Test class for py grain dataset iterator."""

    def test_str(self) -> None:
        """Test str."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source),
            shard_options=sharding.NoSharding(),
            seed=1,
        )
        loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            worker_count=3,
        )
        itr = iter(loader)

        expected_str = """PyGrainDatasetIterator(state={
    "version": 2,
    "last_seen_indices": {
        "0": -3,
        "1": -2,
        "2": -1
    },
    "last_worker_index": -1,
    "worker_count": 3,
    "sampler": "SequentialSampler(num_records=8, shard_options=NoSharding(shard_index=0, shard_count=1, drop_remainder=False))",
    "data_source": "RangeDataSource(start=0, stop=8, step=1)"
})"""

        self.assertEqual(expected_str, str(itr))


if __name__ == "__main__":
    absltest.main()
