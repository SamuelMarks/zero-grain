"""Module docstring."""

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
"""Tests for data loader."""

from collections.abc import Sequence
import functools
from multiprocessing import shared_memory
import pathlib
import platform
import sys
import threading
from typing import Any, Union
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized as absl_parameterized
from zero_grain import python as sharding
from zero_grain import python as transforms
import multiprocessing as mp
from zero_grain import python as data_loader_lib
from zero_grain import python as options
from zero_grain import python as samplers
from zero_grain.python import ArrayRecordDataSource
from zero_grain.python import RangeDataSource
from zero_grain.python import SharedMemoryDataSource
from zero_grain import python as batch
from zero_grain import python as process_prefetch
from zero_grain import python as source
from zero_grain import python as shared_memory_array
from zero_grain.python import BatchOperation
from zero_grain.python import FilterOperation
from zero_grain.python import MapOperation
from zero_grain.python import assert_equal_output_after_checkpoint
import numpy as np
import parameterized


FLAGS = flags.FLAGS


def setup_module():
    # Set the path to test data when run via pytest.
    # When run via bazel, FLAGS.test_srcdir is set from the
    # BUILD file, see args = ["--test_srcdir=grain/_src/python"]
    # in grain/_src/python/BUILD
    """Set up the module for testing."""
    from zero_grain import python as grain  # pylint: disable=g-import-not-at-top

    srcdir = pathlib.Path(grain.__file__).parents[0] / "_src" / "python"
    FLAGS["test_srcdir"].parse(str(srcdir))


def map_function(data):
    """Map a function over the data.

    Returns:
        The return value.

    """
    return data + 1


def condition_function(data):
    """Check a condition on the data.

    Returns:
        The return value.

    """
    return data % 2 == 0


class FilterEven(transforms.Filter):
    """Filter even."""

    def filter(self, x: int) -> bool:
        """Filter the data.

        Returns:
            The return value.

        """
        return x % 2 == 0


class PlusOne(transforms.Map):
    """Plus one."""

    def map(self, x: int) -> int:
        """Map the data.

        Returns:
            The return value.

        """
        return x + 1


class PlusRandom(transforms.RandomMap):
    """Plus random."""

    def random_map(self, x: int, rng: np.random.Generator) -> int:
        """Randomly map the data.

        Returns:
            The return value.

        """
        return x + rng.integers(100_000)


class FailingMap(transforms.Map):
    """Failing map."""

    def map(self, x):
        """Map the data."""
        del x
        1 / 0  # pylint: disable=pointless-statement


class NonPickableTransform(transforms.Map):
    """Non pickable transform."""

    def __getstate__(self):
        """Get the state for pickling.

        Raises:
            ValueError: An error occurred.

        """
        raise ValueError("I shall not be pickled")

    def map(self, x):
        """Map the data.

        Returns:
            The return value.

        """
        return x


class RaisingTransform(transforms.Map):
    """Raising transform."""

    def map(self, x):
        """Map the data.

        Raises:
            AttributeError: An error occurred.

        """
        raise AttributeError("I shall raise")


class ExitingTransform(transforms.Map):
    """Exiting transform."""

    def map(self, x):
        """Map the data.

        Raises:
            exit: An error occurred.

        """
        raise sys.exit(123)


class RandomTripletSource:
    """Random triplet source."""

    def __len__(self) -> int:
        """Return the length.

        Returns:
            The return value.

        """
        return 100_000

    def __getitem__(self, record_key: int):
        """Get an item by index.

        Returns:
            The return value.

        """
        return {
            "data": np.random.uniform(size=(3, 224, 224, 3)).astype(dtype=np.float32)
        }


class DuplicateElementFlatMap(transforms.FlatMap):
    """Duplicate element flat map."""

    max_fan_out: int = 7

    def flat_map(self, element: Any) -> Any:
        """Flat map the data.

        Yields:
            The yielded value.

        """
        for _ in range(self.max_fan_out):
            yield element


class CopyNumPyArrayToSharedMemoryTest(absltest.TestCase):
    """Test class for copy num py array to shared memory."""

    def test_copy_numpy_array_to_shared_memory(self):
        """Test copy numpy array to shared memory."""
        element = np.array([1, 2, 3, 4, 5, 6, 7])
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIsInstance(result, shared_memory_array.SharedMemoryArrayMetadata)

    def test_copy_nested_numpy_array_to_shared_memory(self):
        """Test copy nested numpy array to shared memory."""
        element_1 = np.arange(5)
        element_2 = np.arange(5)
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map([element_1, element_2])
        self.assertIsInstance(result[0], shared_memory_array.SharedMemoryArrayMetadata)
        self.assertIsInstance(result[1], shared_memory_array.SharedMemoryArrayMetadata)

    def test_copy_skipped_non_numpy_array(self):
        """Test copy skipped non numpy array."""
        element = "randomstring"
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        self.assertIs(result, element)

    def test_copy_skipped_dtype_hasobject(self):
        """Test copy skipped dtype hasobject."""

        class DT:
            """Dt."""

            pass

        element = np.array([127, 128, 129], dtype=np.dtype(DT))
        transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
        result = transform.map(element)
        print(result)
        self.assertIs(result, element)

    def test_copy_skipped_flags_c_contiguous(self):
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

    def tearDown(self):
        """Tear down the test environment."""
        super().tearDown()

    # Number of prefetch threads for each Grain worker
    num_threads_per_worker: int | None

    def setUp(self):
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
        transformations,
        *,
        worker_count: int = 0,
        seed: Union[int, None] = None,
    ) -> data_loader_lib.DataLoader:
        # Generates elements [0, 1, 2, 3, 4, 5, 6, 7].
        """Create a data loader for a short sequence.

        Returns:
            The return value.

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

    def test_fails_to_pickle(self):
        """Test fails to pickle."""
        transformations = [NonPickableTransform()]
        data_loader = self._create_data_loader_for_short_sequence(
            transformations, worker_count=2
        )
        with self.assertRaisesRegex(ValueError, "I shall not be pickled"):
            list(data_loader)

    def test_propagates_transform_error_with_multiprocessing(self):
        """Test propagates transform error with multiprocessing."""
        transformations = [RaisingTransform()]
        data_loader = self._create_data_loader_for_short_sequence(
            transformations, worker_count=2
        )
        with self.assertRaisesRegex(Exception, "I shall raise"):
            list(data_loader)

    def test_reports_multiprocessing_worker_crash(self):
        """Test reports multiprocessing worker crash."""
        transformations = [ExitingTransform()]
        data_loader = self._create_data_loader_for_short_sequence(
            transformations, worker_count=2
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "was terminated unexpectedly with exit code 123",
        ):
            list(data_loader)

    def test_data_loader_single_process(self):
        # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
        # Filter keeps only even elements [2, 4, 6, 8]
        # Batching batches each 2 consective elements, producing
        # [np.array([2, 4]), np.array([6, 8])]
        """Test data loader single process."""
        transformations = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_single_process_random_map(self):
        """Test data loader single process random map."""
        transformations = [
            PlusRandom(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(
            transformations, seed=1
        )
        actual = list(data_loader)
        # 4 batches of size 2.
        self.assertLen(actual, 4)
        for i in range(4):
            self.assertEqual(actual[i].shape, (2,))

    def test_data_loader_single_process_legacy_operations(self):
        """Test that old style operations that implement __call__() still work."""
        transformations = [
            MapOperation(map_function=map_function),
            FilterOperation(condition_function=condition_function),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_single_process_iterate_twice(self):
        """Test data loader single process iterate twice."""
        transformations = [
            PlusOne(),
            FilterEven(),
            BatchOperation(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        # First iteration.
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)
        # Second iteration.
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_in_memory_data_source(self):
        """Test data loader in memory data source."""
        data_source = SharedMemoryDataSource([0, 1, 2, 3, 4, 5, 6, 7])

        sampler = samplers.SequentialSampler(
            num_records=len(data_source), shard_options=sharding.NoSharding()
        )

        # Multiprocessing (with 2 processes), splits elements such that:
        # Process_0 gets [0, 2, 4, 6]
        # Process_1 gets [1, 3, 5, 7]
        # Afterwards, operations are executed on elements from each process.
        operations = [
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

    def test_data_loader_two_processes_no_shared_memory(self):
        # Generates elements [0, 1, 2, 3, 4, 5, 6, 7]
        """Test data loader two processes no shared memory."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)

        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )

        # Multiprocessing (with 2 processes), splits elements such that:
        # Process_0 gets [0, 2, 4, 6]
        # Process_1 gets [1, 3, 5, 7]
        # Afterwards, operations are executed on elements from each process.
        operations = [
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

    def test_data_loader_two_processes_with_shared_memory(self):
        # Generates elements [0, 1, 2, 3, 4, 5, 6, 7]
        """Test data loader two processes with shared memory."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)

        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )

        # Multiprocessing (with 2 processes), splits elements such that:
        # Process_0 gets [0, 2, 4, 6]
        # Process_1 gets [1, 3, 5, 7]
        # Afterwards, operations are executed on elements from each process.
        operations = [
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

    def test_data_loader_remote_exception(self):
        """Test data loader remote exception."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)

        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )

        operations = [FailingMap()]

        num_workers = 2
        data_loader = data_loader_lib.DataLoader(
            data_source=range_data_source,
            sampler=sampler,
            operations=operations,
            worker_count=num_workers,
            read_options=self.read_options,
        )
        with self.assertRaises(Exception) as e:
            list(data_loader)
            assert (
                "ZeroDivisionError: division by zero" in e.__cause__._traceback
            )  # pytype: disable=attribute-error

    def test_data_loader_with_used_array_record_data_source(
        self,
    ):
        """Test data loader with used array record data source."""
        data_source = ArrayRecordDataSource(
            [
                str(self.testdata_dir / "digits.array_record-00000-of-00002"),
                str(self.testdata_dir / "digits.array_record-00001-of-00002"),
            ]
        )

        data_source[0]  # pylint: disable=pointless-statement

        sampler = samplers.SequentialSampler(
            num_records=len(data_source), shard_options=sharding.NoSharding()
        )

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

    def test_data_loader_with_invalid_number_of_workers(self):
        """Test a value error is raised when an invlaid number of workers is used."""
        ar_data_source = ArrayRecordDataSource(
            [
                str(self.testdata_dir / "digits.array_record-00000-of-00002"),
                str(self.testdata_dir / "digits.array_record-00001-of-00002"),
            ]
        )

        sampler = samplers.SequentialSampler(
            num_records=len(ar_data_source), shard_options=sharding.NoSharding()
        )

        num_workers = -1
        with self.assertRaises(ValueError):
            data_loader_lib.DataLoader(
                data_source=ar_data_source,
                sampler=sampler,
                worker_count=num_workers,
                read_options=self.read_options,
            )

    def create_checkpointing_dataloader(
        self, num_workers: int
    ) -> data_loader_lib.DataLoader:
        """Create a DataLoader object for checkpointing tests.

        Returns:
            The return value.

        """
        range_data_source = RangeDataSource(start=0, stop=16, step=1)
        sampler = samplers.IndexSampler(
            num_records=len(range_data_source),
            shard_options=sharding.NoSharding(),
            shuffle=False,
            num_epochs=1,
        )
        operations = [
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
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 3,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
    )
    def test_data_loader_checkpointing_object_reconstruction(
        self,
        num_workers: int,
        steps_to_iterate: int,
        expected: Sequence[np.ndarray],
    ):
        """Test data loader checkpointing object reconstruction."""
        data_loader_iterator = iter(self.create_checkpointing_dataloader(num_workers))

        # actual contains elements obtained by iterating through dataloader before
        # getting state, as well as after state is restored. Should be identical
        # to elements obtained by iterating without checkpointing (expected.)
        actual = [next(data_loader_iterator) for i in range(steps_to_iterate)]

        state = data_loader_iterator.get_state()

        # Advance the iterator after getting the state. After restoring the iterator
        # to the state above, the element should appear again when iterating.
        np.testing.assert_equal(next(data_loader_iterator), expected[steps_to_iterate])

        # Create new objects (similar to after preemption) and attempt to restore
        # checkpointed state into them.
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
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
        {
            "num_workers": 3,
            "steps_to_iterate": 3,
            "expected": [
                np.array([2, 4]),
                np.array([6, 8]),
                np.array([10, 12]),
                np.array([14, 16]),
            ],
        },
    )
    def test_data_loader_checkpointing_same_object(
        self,
        num_workers: int,
        steps_to_iterate: int,
        expected: Sequence[np.ndarray],
    ):
        """Test data loader checkpointing same object."""
        data_loader_iterator = iter(self.create_checkpointing_dataloader(num_workers))

        # actual contains elements obtained by iterating through dataloader before
        # getting state, as well as after state is restored. Should be identical
        # to elements obtained by iterating without checkpointing (expected.)
        actual = [next(data_loader_iterator) for i in range(steps_to_iterate)]

        state = data_loader_iterator.get_state()

        # Advance the iterator after getting the state. After restoring the iterator
        # to the state above, the element should appear again when iterating.
        np.testing.assert_equal(next(data_loader_iterator), expected[steps_to_iterate])

        data_loader_iterator.set_state(state)
        for item in data_loader_iterator:
            actual.append(item)
        np.testing.assert_equal(actual, expected)

    @absl_parameterized.parameters(
        {"num_workers": 0}, {"num_workers": 1}, {"num_workers": 2}
    )
    def test_data_loader_sequential_checkpoint_restore_drift(self, num_workers: int):
        """Test data loader sequential checkpoint restore drift."""
        data_loader_iterator = iter(
            self.create_checkpointing_dataloader(num_workers=num_workers)
        )

        # Fetch the first element to have nontrivial initial state.
        first_element = next(data_loader_iterator)
        np.testing.assert_equal(first_element, np.array([2, 4]))

        # Perform sequential get/set state operations without advancing.
        state1 = data_loader_iterator.get_state()
        data_loader_iterator.set_state(state1)

        state2 = data_loader_iterator.get_state()
        data_loader_iterator.set_state(state2)

        # Assert that we can still advance the iterator and that the state is
        # still the same after all these operations.
        next_element = next(data_loader_iterator)
        np.testing.assert_equal(next_element, np.array([6, 8]))

    def test_batch_transform_mapped_to_batch_operation(self):
        # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
        # Filter keeps only even elements [2, 4, 6, 8]
        # Batching batches each 2 consective elements, producing
        # [np.array([2, 4]), np.array([6, 8])]
        """Test batch transform mapped to batch operation."""
        transformations = [
            PlusOne(),
            FilterEven(),
            transforms.Batch(batch_size=2),
        ]
        data_loader = self._create_data_loader_for_short_sequence(transformations)
        expected = [np.array([2, 4]), np.array([6, 8])]
        actual = list(data_loader)
        np.testing.assert_equal(actual, expected)

    def test_data_loader_with_batch_fn(self):
        # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
        # Filter keeps only even elements [2, 4, 6, 8]
        # Batching batches each 3 consective elements with batch_and_pad fn,
        # producing [np.array([2, 4, 6]), np.array([8, 0, 0])]
        """Test data loader with batch fn."""
        transformations = [
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

    def test_state_without_in_memory_data(self):
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

    def test_data_loader_with_flat_map(self):
        """Test data loader with flat map."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )
        operations = [
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
        np.testing.assert_equal(
            list(data_loader), [2] * 7 + [4] * 7 + [6] * 7 + [8] * 7
        )

    def test_data_loader_with_flat_map_checkpointing(self):
        """Test data loader with flat map checkpointing."""
        range_data_source = RangeDataSource(start=0, stop=8, step=1)
        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )
        operations = [
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
    def test_start_prefetch(self, worker_count: int, num_start_prefetch_calls: int):
        """Test start prefetch."""
        range_data_source = RangeDataSource(start=0, stop=16, step=1)
        sampler = samplers.SequentialSampler(
            num_records=len(range_data_source), shard_options=sharding.NoSharding()
        )
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

    def test_str(self):
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
