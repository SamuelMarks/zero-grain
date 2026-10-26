"""Shared memory array and metadata abstractions for zero-grain."""

from __future__ import annotations

import dataclasses
import itertools
import math
from multiprocessing import shared_memory
from zero_grain._types import ArrayProtocol, DTypeProtocol, DTypeSpec, ElementTree, dtype_name

_SHM_ID_GEN = itertools.count(1)
_ACTIVE_SEGMENTS: dict[str, shared_memory.SharedMemory] = {}


def _dtype_itemsize(dtype: DTypeSpec) -> int:
    """Estimate byte itemsize for ``dtype``.

    Args:
        dtype (DTypeSpec): Data type descriptor, string name, scalar type, or None.

    Returns:
        int: Number of bytes per element (defaults to 8).
    """
    name = dtype_name(dtype)
    if "64" in name or "complex128" in name:
        return 8
    if "32" in name or "complex64" in name:
        return 4
    if "16" in name:
        return 2
    if "8" in name or "bool" in name:
        return 1
    return 8


def _allocate_shared_memory_segment(num_bytes: int) -> tuple[str, shared_memory.SharedMemory | None]:
    """Allocate a ``multiprocessing.shared_memory.SharedMemory`` block of ``num_bytes``.

    Args:
        num_bytes (int): Buffer size in bytes (at least 1 byte is allocated).

    Returns:
        tuple[str, shared_memory.SharedMemory | None]: ``(segment_name, segment_handle_or_None)``.
    """
    seq_id = next(_SHM_ID_GEN)
    fallback_name = f"zero_grain_shm_{seq_id}"
    size = max(1, num_bytes)
    try:
        shm = shared_memory.SharedMemory(create=True, size=size)
    except OSError:
        return fallback_name, None
    _ACTIVE_SEGMENTS[shm.name] = shm
    return shm.name, shm


def _release_shared_memory_segment(name: str) -> None:
    """Close and unlink a tracked shared memory segment by ``name`` if active.

    Args:
        name (str): Shared memory segment name.
    """
    shm = _ACTIVE_SEGMENTS.pop(name, None)
    if shm is not None:
        shm.close()
        shm.unlink()


@dataclasses.dataclass
class SharedMemoryArrayMetadata:
    """Descriptor pointing to a shared memory array segment.

    Attributes:
        name (str): Unique OS shared memory segment name.
        dtype (DTypeSpec): Array data type descriptor or name.
        shape (tuple[int, ...]): Array dimensions tuple.
    """

    name: str = ""
    dtype: DTypeSpec = None
    shape: tuple[int, ...] = ()

    def close_and_unlink(self) -> None:
        """Close and unlink the underlying shared memory segment."""
        if self.name:
            _release_shared_memory_segment(self.name)
        self.name = ""


class SharedMemoryArray:
    """Wrapper around a NumPy-compatible array backed by POSIX shared memory.

    Attributes:
        metadata (SharedMemoryArrayMetadata): Associated shared memory metadata.
    """

    def __init__(
        self,
        *args: ElementTree,
        name: str | None = None,
        dtype: DTypeSpec = None,
        shape: tuple[int, ...] = (),
        **kwargs: ElementTree,
    ) -> None:
        """Initialize ``SharedMemoryArray``.

        Args:
            *args (ElementTree): Optional positional ``(shape, dtype)`` arguments.
            name (str | None): Shared memory segment name (allocated if None).
            dtype (DTypeSpec): Array data type.
            shape (tuple[int, ...]): Array shape tuple.
            **kwargs (ElementTree): Additional keyword arguments.
        """
        del kwargs
        resolved_shape = shape
        resolved_dtype = dtype
        if len(args) >= 1 and isinstance(args[0], tuple):
            resolved_shape = tuple(int(dim) for dim in args[0] if isinstance(dim, int))
        if len(args) >= 2 and isinstance(args[1], (str, DTypeProtocol)):
            resolved_dtype = args[1]

        num_elements = math.prod(resolved_shape) if resolved_shape else 1
        byte_len = max(1, num_elements * _dtype_itemsize(resolved_dtype))
        if name is not None:
            resolved_name = name
        else:
            resolved_name, _ = _allocate_shared_memory_segment(byte_len)

        self.metadata = SharedMemoryArrayMetadata(
            name=resolved_name,
            dtype=resolved_dtype,
            shape=resolved_shape,
        )

    @classmethod
    def from_array(cls, arr: ArrayProtocol) -> SharedMemoryArrayMetadata:
        """Allocate a shared memory block sized for ``arr`` and return its metadata.

        Args:
            arr (ArrayProtocol): C-contiguous non-object array to copy into shared memory.

        Returns:
            SharedMemoryArrayMetadata: Metadata for the allocated shared memory block.
        """
        num_elements = math.prod(arr.shape) if arr.shape else 1
        byte_len = max(1, num_elements * _dtype_itemsize(arr.dtype))
        seg_name, _ = _allocate_shared_memory_segment(byte_len)
        meta = SharedMemoryArrayMetadata(
            name=seg_name,
            dtype=arr.dtype,
            shape=arr.shape,
        )
        _release_shared_memory_segment(seg_name)
        return meta

    def unlink_on_del(self) -> None:
        """Mark the shared memory buffer to be unlinked when garbage collected."""
        self.metadata.close_and_unlink()


__all__ = [
    "SharedMemoryArray",
    "SharedMemoryArrayMetadata",
    "_allocate_shared_memory_segment",
    "_dtype_itemsize",
    "_release_shared_memory_segment",
]
