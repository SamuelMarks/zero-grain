"""Official Grain tests for pipeline operations."""

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

from collections.abc import Sequence
import dataclasses
import re
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from zero_grain import python as record
from zero_grain._types import ElementTree, RNGGeneratorProtocol, T_in
from zero_grain.python import BatchOperation
from zero_grain.python import FilterOperation
from zero_grain.python import MapOperation
from zero_grain.python import RandomMapOperation


class _PointTuple(NamedTuple):
    """Typed NamedTuple used in ``test_batch_named_tuples``."""

    x: int | np.ndarray
    y: str | np.ndarray


class OperationsTest(parameterized.TestCase):
    """Test class for operations."""

    def compare_output(
        self,
        actual: Sequence[record.Record[T_in]],
        expected: Sequence[record.Record[ElementTree]],
    ) -> None:
        """Compare actual and expected record lists ignoring RNG objects.

        Args:
            actual (Sequence[record.Record[T_in]]): Actual output record list.
            expected (Sequence[record.Record[ElementTree]]): Expected output record list.
        """
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            if a.metadata is not None:
                a.metadata.rng = None
            if b.metadata is not None:
                b.metadata.rng = None
            np.testing.assert_equal(dataclasses.astuple(a), dataclasses.astuple(b))

    def test_map_operation(self) -> None:
        """Test map operation."""

        def _inc(data: int) -> int:
            """Increment ``data`` by 1.

            Args:
                data (int): Input integer.

            Returns:
                int: Incremented integer.
            """
            return data + 1

        map_operation: MapOperation[int, int] = MapOperation(map_function=_inc)
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=3), 1),
                record.Record(record.RecordMetadata(index=1, record_key=2), 2),
                record.Record(record.RecordMetadata(index=2, record_key=1), 3),
                record.Record(record.RecordMetadata(index=3, record_key=0), 4),
            ]
        )

        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=0, record_key=None), 2),
            record.Record(record.RecordMetadata(index=1, record_key=None), 3),
            record.Record(record.RecordMetadata(index=2, record_key=None), 4),
            record.Record(record.RecordMetadata(index=3, record_key=None), 5),
        ]
        actual_output_data = list(map_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_random_map_operation(self) -> None:
        """Test random map operation."""
        delta = 0.1

        def _perturb(data: int, rng: RNGGeneratorProtocol) -> float:
            """Add uniform noise in ``[-delta, delta)`` to ``data``.

            Args:
                data (int): Base integer value.
                rng (RNGGeneratorProtocol): Per-record RNG.

            Returns:
                float: Perturbed float value.
            """
            return data + rng.uniform(-delta, delta)

        random_map_operation: RandomMapOperation[int, float] = RandomMapOperation(
            random_map_function=_perturb,
        )

        input_rng = np.random.Generator(np.random.Philox(key=0))
        input_record = record.Record(record.RecordMetadata(index=0, record_key=1, rng=input_rng), 1)
        input_data = iter([input_record])

        actual_output_data = list(random_map_operation(input_data))
        self.assertLen(actual_output_data, 1)
        actual_output_record = actual_output_data[0]
        self.assertIsInstance(actual_output_record, record.Record)
        assert actual_output_record.metadata is not None
        assert input_record.metadata is not None
        self.assertEqual(actual_output_record.metadata.index, input_record.metadata.index)
        self.assertIsNone(actual_output_record.metadata.record_key)
        self.assertEqual(actual_output_record.metadata.rng, input_rng)
        self.assertAlmostEqual(actual_output_record.data, input_record.data, delta=delta)

    def test_map_operation_with_callable_objects(self) -> None:
        """Test map operation with callable instances."""

        class CallableIncrement:
            """Callable incrementer class."""

            def __call__(self, x: int) -> int:
                """Increment ``x`` by 1.

                Args:
                    x (int): Input numeric value.

                Returns:
                    int: Incremented numeric value.
                """
                return x + 1

        map_operation: MapOperation[int, int] = MapOperation(map_function=CallableIncrement())
        input_data = iter([record.Record(record.RecordMetadata(index=0, record_key=3), 1)])

        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=0, record_key=None), 2)
        ]
        actual_output_data = list(map_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_filter_operation(self) -> None:
        """Test filter operation."""

        def _is_even(data: int) -> bool:
            """Return True if ``data`` is even.

            Args:
                data (int): Candidate integer.

            Returns:
                bool: True if even.
            """
            return data % 2 == 0

        filter_operation: FilterOperation[int] = FilterOperation(condition_function=_is_even)
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=3), 1),
                record.Record(record.RecordMetadata(index=1, record_key=2), 2),
                record.Record(record.RecordMetadata(index=2, record_key=1), 3),
                record.Record(record.RecordMetadata(index=3, record_key=0), 4),
            ]
        )

        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), 2),
            record.Record(record.RecordMetadata(index=3, record_key=None), 4),
        ]

        actual_output_data = list(filter_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_integer_scalars(self) -> None:
        """Test batch integer scalars."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=3), 1),
                record.Record(record.RecordMetadata(index=1, record_key=2), 2),
                record.Record(record.RecordMetadata(index=2, record_key=1), 3),
                record.Record(record.RecordMetadata(index=3, record_key=0), 4),
            ]
        )
        batch_operation: BatchOperation[int] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])),
            record.Record(record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])),
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_with_remainder(self) -> None:
        """Test batch with remainder."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), 1),
                record.Record(record.RecordMetadata(index=1, record_key=3), 2),
                record.Record(record.RecordMetadata(index=2, record_key=2), 3),
                record.Record(record.RecordMetadata(index=3, record_key=1), 4),
                record.Record(record.RecordMetadata(index=4, record_key=0), 5),
            ]
        )
        batch_operation: BatchOperation[int] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])),
            record.Record(record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])),
            record.Record(record.RecordMetadata(index=4, record_key=None), np.asarray([5])),
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_drop_remainder(self) -> None:
        """Test batch drop remainder."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), 1),
                record.Record(record.RecordMetadata(index=1, record_key=3), 2),
                record.Record(record.RecordMetadata(index=2, record_key=2), 3),
                record.Record(record.RecordMetadata(index=3, record_key=1), 4),
                record.Record(record.RecordMetadata(index=4, record_key=0), 5),
            ]
        )
        batch_operation: BatchOperation[int] = BatchOperation(batch_size=2, drop_remainder=True)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])),
            record.Record(record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])),
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_dict_of_numpy_arrays_no_shared_memory(self) -> None:
        """Test batch dict of numpy arrays no shared memory."""
        input_data = iter(
            [
                record.Record(
                    record.RecordMetadata(index=0, record_key=4),
                    {"a": np.array([1, 2], dtype=np.int32)},
                ),
                record.Record(
                    record.RecordMetadata(index=1, record_key=3),
                    {"a": np.array([3, 4], dtype=np.int64)},
                ),
            ]
        )
        batch_operation: BatchOperation[dict[str, np.ndarray]] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(
                record.RecordMetadata(index=1, record_key=None),
                {"a": np.array([[1, 2], [3, 4]])},
            )
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_tuples(self) -> None:
        """Test batch tuples."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), (1, 10.0, "text")),
                record.Record(
                    record.RecordMetadata(index=1, record_key=3),
                    (2, 20.0, "text_2"),
                ),
            ]
        )
        batch_operation: BatchOperation[tuple[int, float, str]] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(
                record.RecordMetadata(index=1, record_key=None),
                (
                    np.array([1, 2]),
                    np.array([10.0, 20.0]),
                    np.array(["text", "text_2"]),
                ),
            )
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_named_tuples(self) -> None:
        """Test batch named tuples."""
        input_data = iter(
            [
                record.Record(
                    record.RecordMetadata(index=0, record_key=4),
                    _PointTuple(x=1, y="str"),
                ),
                record.Record(
                    record.RecordMetadata(index=1, record_key=3),
                    _PointTuple(x=2, y="str_2"),
                ),
            ]
        )
        batch_operation: BatchOperation[_PointTuple] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(
                record.RecordMetadata(index=1, record_key=None),
                _PointTuple(x=np.array([1, 2]), y=np.array(["str", "str_2"])),
            )
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_lists(self) -> None:
        """Test batch lists."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), [1, 2]),
                record.Record(record.RecordMetadata(index=1, record_key=3), [3, 4]),
            ]
        )
        batch_operation: BatchOperation[list[int]] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(
                record.RecordMetadata(index=1, record_key=None),
                np.array([[1, 3], [2, 4]]),
            )
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_mixed_scalar_numbers(self) -> None:
        """Test batch mixed scalar numbers."""
        input_data: list[record.Record[int | float]] = [
            record.Record(record.RecordMetadata(index=0, record_key=4), 1),
            record.Record(record.RecordMetadata(index=1, record_key=3), 2.0),
            record.Record(record.RecordMetadata(index=2, record_key=2), int(np.int32(3))),
            record.Record(record.RecordMetadata(index=3, record_key=1), int(np.int64(4))),
            record.Record(record.RecordMetadata(index=4, record_key=0), float(5.0)),
        ]
        batch_operation: BatchOperation[int | float] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), np.array([1, 2])),
            record.Record(record.RecordMetadata(index=3, record_key=None), np.array([3, 4])),
            record.Record(record.RecordMetadata(index=4, record_key=None), np.array([5])),
        ]
        actual_output_data = list(batch_operation(iter(input_data)))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_lists_different_length(self) -> None:
        """Test batch lists different length."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), [1, 2]),
                record.Record(record.RecordMetadata(index=1, record_key=3), [3, 4, 5]),
            ]
        )
        batch_operation: BatchOperation[list[int]] = BatchOperation(batch_size=2)
        with np.testing.assert_raises_regex(
            TypeError,
            re.escape(
                "Record structures do not match. Record at position 0 has "
                "structure [1, 2], while records at positions [1] "
                "have structures [[3, 4, 5]]."
            ),
        ):
            list(batch_operation(input_data))

    def test_batch_dicts_different_keys(self) -> None:
        """Test batch dicts different keys."""
        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), {"a": 1}),
                record.Record(record.RecordMetadata(index=1, record_key=3), {"b": 2}),
            ]
        )
        batch_operation: BatchOperation[dict[str, int]] = BatchOperation(batch_size=2)
        with np.testing.assert_raises_regex(
            TypeError,
            re.escape(
                "Record structures do not match. Record at position 0 has "
                "structure {'a': 1}, while records at positions [1]"
                " have structures [{'b': 2}]."
            ),
        ):
            list(batch_operation(input_data))

    def test_batch_unsupported_type(self) -> None:
        """Test batch unsupported type."""

        @dataclasses.dataclass
        class UnSupportedType:
            """Un supported type."""

            some_value: int

        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=4), UnSupportedType(1)),
                record.Record(record.RecordMetadata(index=1, record_key=3), UnSupportedType(2)),
            ]
        )
        batch_operation: BatchOperation[UnSupportedType] = BatchOperation(batch_size=2)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(
                record.RecordMetadata(index=1, record_key=None),
                np.array([UnSupportedType(1), UnSupportedType(2)]),
            )
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)

    def test_batch_with_custom_batch_fn(self) -> None:
        """Test batch with custom batch fn."""

        def _custom_list_batch(xs: Sequence[int]) -> list[int]:
            """Convert sequence ``xs`` into a list.

            Args:
                xs (Sequence[int]): Input integer sequence.

            Returns:
                list[int]: List copy of ``xs``.
            """
            return list(xs)

        input_data = iter(
            [
                record.Record(record.RecordMetadata(index=0, record_key=3), 1),
                record.Record(record.RecordMetadata(index=1, record_key=2), 2),
                record.Record(record.RecordMetadata(index=2, record_key=1), 3),
                record.Record(record.RecordMetadata(index=3, record_key=0), 4),
            ]
        )
        batch_operation: BatchOperation[int] = BatchOperation(batch_size=2, batch_fn=_custom_list_batch)
        expected_output_data: list[record.Record[ElementTree]] = [
            record.Record(record.RecordMetadata(index=1, record_key=None), [1, 2]),
            record.Record(record.RecordMetadata(index=3, record_key=None), [3, 4]),
        ]
        actual_output_data = list(batch_operation(input_data))
        self.compare_output(actual_output_data, expected_output_data)


if __name__ == "__main__":
    absltest.main()
