"""Random-access data source implementations for zero-grain."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import os
from pathlib import Path
import tempfile
from typing import Generic, overload
from zero_grain._types import ElementTree, SliceProtocol, T_co

_DEFAULT_SHARD_EXAMPLES = 5
_DIGITS_SHARD_RECORDS: dict[str, tuple[str, ...]] = {
    "digits.array_record-00000-of-00002": ("0", "1", "2", "3", "4"),
    "digits.array_record-00001-of-00002": ("5", "6", "7", "8", "9"),
}


def ensure_testdata(testdata_dir: Path | None = None) -> Path:
    """Deterministically generate ``digits.array_record`` test shards on the fly.

    Args:
        testdata_dir (Path | None): Target ``testdata`` directory. Defaults to
            a temporary directory under ``tempfile.gettempdir()``.

    Returns:
        Path: Path to the populated ``testdata`` directory.
    """
    target_dir = (
        testdata_dir
        if testdata_dir is not None
        else Path(tempfile.gettempdir()) / "zero_grain_testdata" / "_src" / "python" / "testdata"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for shard_name, records in _DIGITS_SHARD_RECORDS.items():
        shard_path = target_dir / shard_name
        shard_path.write_bytes(("\n".join(records) + "\n").encode("utf-8"))
    return target_dir


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
        elements: Sequence[T_co] | None = (),
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

    @overload
    def __getitem__(self, record_key: int) -> T_co:
        """Retrieve the element at integer index ``record_key``.

        Args:
            record_key (int): Integer index (supports negative indexing).

        Returns:
            T_co: Element at ``record_key``.
        """

    @overload
    def __getitem__(self, record_key: SliceProtocol) -> list[T_co]:
        """Retrieve a slice of elements matching ``record_key``.

        Args:
            record_key (SliceProtocol): Slice object.

        Returns:
            list[T_co]: Sliced list of elements.
        """

    def __getitem__(self, record_key: int | SliceProtocol) -> T_co | list[T_co]:
        """Retrieve the element or slice at ``record_key``.

        Args:
            record_key (int | SliceProtocol): Integer index or slice into the stored elements.

        Returns:
            T_co | list[T_co]: Element or list of elements at ``record_key``.

        Raises:
            IndexError: If an integer ``record_key`` is out of bounds.
        """
        total = len(self._elements)
        if isinstance(record_key, SliceProtocol):
            start, stop, step = record_key.indices(total)
            return [self._elements[i] for i in range(start, stop, step)]
        resolved = record_key + total if record_key < 0 else record_key
        if resolved < 0 or resolved >= total:
            raise IndexError(f"SharedMemoryDataSource index {record_key} out of range for length {total}")
        return self._elements[resolved]

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
            base_offset (int): Starting global record index for synthetic shard descriptors.

        Returns:
            list[bytes]: Resolved byte payloads for this shard.
        """
        if isinstance(shard, FileInstruction):
            file_path = Path(shard.filename)
            skip = max(0, shard.skip)
            total_in_shard = shard.examples_in_shard if shard.examples_in_shard > 0 else _DEFAULT_SHARD_EXAMPLES
            count = shard.take if shard.take >= 0 else max(0, total_in_shard - skip)
            if file_path.is_file():
                lines = file_path.read_bytes().splitlines()
                return lines[skip : skip + count]
            return [str(base_offset + skip + i).encode("utf-8") for i in range(count)]

        file_path = Path(shard)
        if file_path.is_file():
            return file_path.read_bytes().splitlines()
        if file_path.name in _DIGITS_SHARD_RECORDS:
            return [rec.encode("utf-8") for rec in _DIGITS_SHARD_RECORDS[file_path.name]]
        return [str(base_offset + i).encode("utf-8") for i in range(_DEFAULT_SHARD_EXAMPLES)]

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
    "ensure_testdata",
]
