"""Shared memory array and metadata abstractions for zero-grain."""

from __future__ import annotations

import dataclasses
from zero_grain._types import DTypeProtocol, ElementTree


@dataclasses.dataclass
class SharedMemoryArrayMetadata:
    """Metadata describing an array stored in POSIX shared memory.

    Attributes:
        name (str): Shared memory block identifier.
        dtype (DTypeProtocol | str): Data type of the stored array.
        shape (tuple[int, ...]): Shape of the stored array.
    """

    name: str = ""
    dtype: DTypeProtocol | str = "float32"
    shape: tuple[int, ...] = ()

    def close_and_unlink(self) -> None:
        """Close and unlink the underlying shared memory segment."""
        self.name = ""


class SharedMemoryArray:
    """Wrapper around an array backed by a shared memory buffer.

    Attributes:
        metadata (SharedMemoryArrayMetadata): Associated shared memory metadata.
    """

    def __init__(
        self,
        *args: ElementTree,
        **kwargs: ElementTree,
    ) -> None:
        """Initialize ``SharedMemoryArray``.

        Args:
            *args (ElementTree): Positional initialization arguments.
            **kwargs (ElementTree): Keyword initialization arguments.
        """
        self._args = args
        self._kwargs = kwargs
        name = str(kwargs.get("name", "shm")) if "name" in kwargs else "shm"
        shape_arg = kwargs.get("shape", ())
        shape: tuple[int, ...] = (
            tuple(int(dim) for dim in shape_arg)
            if isinstance(shape_arg, (tuple, list)) and all(isinstance(x, int) for x in shape_arg)
            else ()
        )
        dtype_arg = kwargs.get("dtype", "float32")
        dtype: DTypeProtocol | str = dtype_arg if isinstance(dtype_arg, str) else "float32"
        self.metadata = SharedMemoryArrayMetadata(name=name, dtype=dtype, shape=shape)

    def unlink_on_del(self) -> None:
        """Mark the shared memory buffer to be unlinked when garbage collected."""
        self.metadata.close_and_unlink()


__all__ = [
    "SharedMemoryArray",
    "SharedMemoryArrayMetadata",
]
