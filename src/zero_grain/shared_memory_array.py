"""Shared memory array and metadata abstractions for zero-grain."""

from __future__ import annotations

import atexit
import dataclasses
import itertools
import math
from multiprocessing import shared_memory
from zero_grain._types import (
    ArrayProtocol,
    BytesExportArrayProtocol,
    DTypeProtocol,
    DTypeSpec,
    ElementTree,
    dtype_name,
    get_numpy_module,
)

_SHM_ID_GEN = itertools.count(1)
_ACTIVE_SEGMENTS: dict[str, shared_memory.SharedMemory] = {}


def _cleanup_all_active_segments() -> None:
    """Close and unlink all remaining active shared memory segments."""
    for name in list(_ACTIVE_SEGMENTS.keys()):
        _release_shared_memory_segment(name)


atexit.register(_cleanup_all_active_segments)


def _dtype_itemsize(dtype: DTypeSpec) -> int:
    """Estimate byte itemsize for ``dtype``.

    Args:
        dtype (DTypeSpec): Data type descriptor, string name, scalar type, or None.

    Returns:
        int: Number of bytes per element (defaults to 8).
    """
    name = dtype_name(dtype)
    if "complex128" in name:
        return 16
    if "64" in name:
        return 8
    if "32" in name:
        return 4
    if "16" in name or "bfloat16" in name:
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

    def get_bytes(self) -> bytes:
        """Read the raw byte payload stored in the active shared memory segment.

        Returns:
            bytes: Raw bytes from the shared memory buffer, or ``b""`` if inactive.
        """
        shm = _ACTIVE_SEGMENTS.get(self.name)
        if shm is None or shm.buf is None:
            return b""
        num_elements = math.prod(self.shape) if self.shape else 1
        byte_len = min(len(shm.buf), max(1, num_elements * _dtype_itemsize(self.dtype)))
        return bytes(shm.buf[:byte_len])

    def to_numpy(self) -> ArrayProtocol:
        """Reconstruct a zero-filled or buffer-backed ``ArrayProtocol`` from this metadata.

        Returns:
            ArrayProtocol: Reconstructed array matching ``shape`` and ``dtype``.
        """
        import struct

        dt_str = dtype_name(self.dtype)
        num_elements = math.prod(self.shape) if self.shape else 1
        raw = self.get_bytes()
        fmt_char = "d"
        if "int32" in dt_str:
            fmt_char = "i"
        elif "int64" in dt_str:
            fmt_char = "q"
        elif "float32" in dt_str:
            fmt_char = "f"
        elif "uint8" in dt_str or "bool" in dt_str:
            fmt_char = "B"
        expected_bytes = num_elements * struct.calcsize(fmt_char)
        if raw and len(raw) >= expected_bytes and num_elements > 0:
            unpacked = list(struct.unpack(f"<{num_elements}{fmt_char}", raw[:expected_bytes]))
            flat_vals: list[ElementTree] = [bool(v) if "bool" in dt_str else v for v in unpacked]
        else:
            zero_val: ElementTree = False if "bool" in dt_str else (0 if "int" in dt_str else 0.0)
            flat_vals = [zero_val for _ in range(num_elements)]

        def _reshape_flat(vals: list[ElementTree], dims: tuple[int, ...]) -> ElementTree:
            """Reshape a flat list ``vals`` into nested lists matching ``dims``.

            Args:
                vals (list[ElementTree]): Flat values.
                dims (tuple[int, ...]): Target shape dimensions.

            Returns:
                ElementTree: Nested list or scalar.
            """
            if not dims:
                return vals[0] if vals else 0
            if len(dims) == 1:
                return vals[: dims[0]]
            stride = math.prod(dims[1:])
            return [_reshape_flat(vals[i * stride : (i + 1) * stride], dims[1:]) for i in range(dims[0])]

        nested = _reshape_flat(flat_vals, self.shape)
        np_mod = get_numpy_module()
        if np_mod is not None:
            return np_mod.asarray(nested, dtype=self.dtype)
        from zero_grain._types import _ZERO_JAX_NUMPY_ADAPTER

        return _ZERO_JAX_NUMPY_ADAPTER.asarray(nested, dtype=self.dtype)

    def __array__(self) -> ArrayProtocol:
        """Convert this metadata descriptor into an array via ``to_numpy()``.

        Returns:
            ArrayProtocol: Reconstructed array.
        """
        return self.to_numpy()


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
        raw_bytes = arr.tobytes() if isinstance(arr, BytesExportArrayProtocol) else b""
        num_elements = math.prod(arr.shape) if arr.shape else 1
        byte_len = max(1, len(raw_bytes), num_elements * _dtype_itemsize(arr.dtype))
        seg_name, shm = _allocate_shared_memory_segment(byte_len)
        if shm is not None and raw_bytes and shm.buf is not None:
            shm.buf[: len(raw_bytes)] = raw_bytes
        return SharedMemoryArrayMetadata(
            name=seg_name,
            dtype=arr.dtype,
            shape=arr.shape,
        )

    @classmethod
    def from_metadata(cls, metadata: SharedMemoryArrayMetadata) -> SharedMemoryArray:
        """Construct a ``SharedMemoryArray`` view bound to an existing ``metadata`` descriptor.

        Args:
            metadata (SharedMemoryArrayMetadata): Existing shared memory metadata descriptor.

        Returns:
            SharedMemoryArray: Wrapper bound to ``metadata``.
        """
        return cls(name=metadata.name, dtype=metadata.dtype, shape=metadata.shape)

    def to_numpy(self) -> ArrayProtocol:
        """Reconstruct an ``ArrayProtocol`` from the underlying shared memory segment.

        Returns:
            ArrayProtocol: Reconstructed array matching ``self.metadata``.
        """
        return self.metadata.to_numpy()

    def __array__(self) -> ArrayProtocol:
        """Convert this ``SharedMemoryArray`` into an array via ``to_numpy()``.

        Returns:
            ArrayProtocol: Reconstructed array.
        """
        return self.metadata.to_numpy()

    def unlink_on_del(self) -> None:
        """Mark the shared memory buffer to be unlinked when garbage collected."""
        self.metadata.close_and_unlink()


__all__ = [
    "SharedMemoryArray",
    "SharedMemoryArrayMetadata",
    "_allocate_shared_memory_segment",
    "_cleanup_all_active_segments",
    "_dtype_itemsize",
    "_release_shared_memory_segment",
]
