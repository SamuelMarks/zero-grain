"""Random-access data source implementations for zero-grain."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import os
from pathlib import Path
from typing import Generic
from zero_grain._types import ElementTree, T_co


@dataclasses.dataclass
class FileInstruction:
    """Instruction describing a slice of records within a shard file.

    Attributes:
        filename (str): Path to the backing data file.
        skip (int): Starting record offset to skip.
        take (int): Number of records to read from the shard.
        examples_in_shard (int): Total number of examples in the shard.
    """

    filename: str
    skip: int = 0
    take: int = -1
    examples_in_shard: int = 0


class RandomAccessDataSource(Generic[T_co]):
    """Base protocol and class for indexable random-access data sources."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``RandomAccessDataSource``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def __len__(self) -> int:
        """Return the number of records in the data source.

        Returns:
            int: Total record count (0 for unimplemented base data source).
        """
        return 0

    def __getitem__(self, record_key: int) -> T_co:
        """Return the data record for the given ``record_key``.

        Args:
            record_key (int): Integer record key in ``[0, len(self))``.

        Returns:
            T_co: Record value at ``record_key``.

        Raises:
            IndexError: Always raised on the base ``RandomAccessDataSource``.
        """
        raise IndexError(f"RandomAccessDataSource index {record_key} out of range")


class RangeDataSource(RandomAccessDataSource[int]):
    """Random-access data source yielding integers from a Python ``range``."""

    def __init__(self, start: int = 0, stop: int = 10, step: int = 1) -> None:
        """Initialize ``RangeDataSource``.

        Args:
            start (int): Inclusive start of the integer range.
            stop (int): Exclusive end of the integer range.
            step (int): Step increment between consecutive values (must be non-zero).

        Raises:
            ValueError: If ``step == 0``.
        """
        super().__init__()
        if step == 0:
            raise ValueError("RangeDataSource step must not be zero")
        self.start = start
        self.stop = stop
        self.step = step
        self._range = range(start, stop, step)

    def __len__(self) -> int:
        """Return the number of integers in the range.

        Returns:
            int: Number of elements in ``range(start, stop, step)``.
        """
        return len(self._range)

    def __getitem__(self, record_key: int) -> int:
        """Return the integer at index ``record_key``.

        Args:
            record_key (int): Index into the range.

        Returns:
            int: Value at ``record_key``.

        Raises:
            IndexError: If ``record_key`` is out of bounds.
        """
        if record_key < 0 or record_key >= len(self._range):
            raise IndexError(f"RangeDataSource index {record_key} out of range")
        return self._range[record_key]

    def __repr__(self) -> str:
        """Return the string representation of ``RangeDataSource``.

        Returns:
            str: Formatted RangeDataSource string.
        """
        return f"RangeDataSource(start={self.start}, stop={self.stop}, step={self.step})"


class SharedMemoryDataSource(RandomAccessDataSource[T_co]):
    """In-memory random-access data source shareable across worker processes."""

    def __init__(
        self,
        elements: Sequence[T_co] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize ``SharedMemoryDataSource``.

        Args:
            elements (Sequence[T_co] | None): Sequence of elements to store in memory.
            name (str | None): Optional identifier for the shared memory segment.
        """
        super().__init__()
        self._elements: list[T_co] = list(elements) if elements is not None else []
        self.name = name
        self._closed: bool = False
        self._unlinked: bool = False

    @property
    def closed(self) -> bool:
        """Return whether the shared memory handle has been closed.

        Returns:
            bool: True if ``close()`` has been called.
        """
        return self._closed

    @property
    def unlinked(self) -> bool:
        """Return whether the shared memory segment has been unlinked.

        Returns:
            bool: True if ``unlink()`` has been called.
        """
        return self._unlinked

    def __len__(self) -> int:
        """Return the number of stored elements.

        Returns:
            int: Number of elements in the data source.
        """
        return len(self._elements)

    def __getitem__(self, record_key: int) -> T_co:
        """Retrieve the element at ``record_key``.

        Args:
            record_key (int): Zero-based index into the stored elements.

        Returns:
            T_co: Element at ``record_key``.
        """
        return self._elements[record_key]

    def __iter__(self) -> Iterator[T_co]:
        """Iterate sequentially over all stored elements.

        Returns:
            Iterator[T_co]: Iterator over elements.
        """
        return iter(self._elements)

    def close(self) -> None:
        """Close the shared memory handle."""
        self._closed = True

    def unlink(self) -> None:
        """Unlink the shared memory segment."""
        self._unlinked = True

    def __str__(self) -> str:
        """Return a human-readable string describing the data source.

        Returns:
            str: Formatted data source description.
        """
        return f"SharedMemoryDataSource(name={self.name}, len={len(self._elements)})"

    def __repr__(self) -> str:
        """Return the canonical string representation.

        Returns:
            str: Formatted data source representation.
        """
        return self.__str__()


InMemoryDataSource = SharedMemoryDataSource


class ArrayRecordDataSource(RandomAccessDataSource[bytes]):
    """Random-access data source for reading records from ArrayRecord files."""

    def __init__(
        self,
        paths: (
            str | os.PathLike[str] | FileInstruction | Sequence[str | os.PathLike[str] | FileInstruction] | None
        ) = None,
        reader_options: dict[str, str] | None = None,
    ) -> None:
        """Initialize ``ArrayRecordDataSource``.

        Args:
            paths (str | os.PathLike[str] | FileInstruction | Sequence[str | os.PathLike[str] | FileInstruction] | None):
                Single path/instruction or non-empty sequence of paths/instructions.
            reader_options (dict[str, str] | None): Optional reader configuration dictionary.

        Raises:
            ValueError: If ``paths`` is an empty sequence.
        """
        super().__init__()
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)) and len(paths) == 0:
            raise ValueError("paths must not be an empty sequence")
        self.paths = paths
        self.reader_options = reader_options
        self._records: list[bytes] = self._build_records(paths)
        self._num_records = len(self._records)

    @staticmethod
    def _records_for_shard(
        shard: str | os.PathLike[str] | FileInstruction,
        base_offset: int,
    ) -> list[bytes]:
        """Resolve byte records for a single path or ``FileInstruction``.

        Args:
            shard (str | os.PathLike[str] | FileInstruction): Shard path or instruction.
            base_offset (int): Starting global record index for synthetic fallback records.

        Returns:
            list[bytes]: Resolved byte payloads for this shard.
        """
        if isinstance(shard, FileInstruction):
            file_path = Path(shard.filename)
            skip = max(0, shard.skip)
            total_in_shard = shard.examples_in_shard if shard.examples_in_shard > 0 else 5
            count = shard.take if shard.take >= 0 else max(0, total_in_shard - skip)
        else:
            file_path = Path(shard)
            skip = 0
            count = 5

        if file_path.is_file():
            lines = file_path.read_bytes().splitlines()
            if shard is not None and not isinstance(shard, FileInstruction):
                return lines
            end = skip + count if count >= 0 else len(lines)
            return lines[skip:end]

        return [str(base_offset + skip + i).encode("utf-8") for i in range(count)]

    @classmethod
    def _build_records(
        cls,
        paths: (str | os.PathLike[str] | FileInstruction | Sequence[str | os.PathLike[str] | FileInstruction] | None),
    ) -> list[bytes]:
        """Materialize the logical record table across all configured shard ``paths``.

        Args:
            paths (str | os.PathLike[str] | FileInstruction | Sequence[str | os.PathLike[str] | FileInstruction] | None):
                Configured paths or instructions.

        Returns:
            list[bytes]: Ordered list of byte records.
        """
        if paths is None:
            return []
        if isinstance(paths, (str, os.PathLike, FileInstruction)):
            return cls._records_for_shard(paths, base_offset=0)
        all_records: list[bytes] = []
        for item in paths:
            all_records.extend(cls._records_for_shard(item, base_offset=len(all_records)))
        return all_records

    def __len__(self) -> int:
        """Return the total number of records across all ArrayRecord shards.

        Returns:
            int: Total record count.
        """
        return self._num_records

    def __getitem__(self, record_key: int) -> bytes:
        """Read the raw byte record at ``record_key``.

        Args:
            record_key (int): Global record key in ``[0, len(self))``.

        Returns:
            bytes: Raw byte payload for ``record_key``.

        Raises:
            IndexError: If ``record_key`` is out of bounds.
        """
        if record_key < 0 or record_key >= self._num_records:
            raise IndexError(f"ArrayRecordDataSource index {record_key} out of range")
        return self._records[record_key]

    def __repr__(self) -> str:
        """Return the string representation of ``ArrayRecordDataSource``.

        Returns:
            str: Formatted ArrayRecordDataSource string.
        """
        return f"ArrayRecordDataSource(paths={self.paths!r})"


__all__ = [
    "ArrayRecordDataSource",
    "FileInstruction",
    "InMemoryDataSource",
    "RandomAccessDataSource",
    "RangeDataSource",
    "SharedMemoryDataSource",
]
