"""Strongly typed structural protocols, type variables, and aliases for zero-grain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import dataclasses
import sys
from typing import Generic, Protocol, TypedDict, TypeVar, Union, runtime_checkable
from typing_extensions import TypeAlias, TypeGuard

ScalarElement: TypeAlias = Union[int, float, bool, str, bytes, complex, None]


@runtime_checkable
class SliceProtocol(Protocol):
    """Protocol representing a Python ``slice`` object."""

    def indices(self, length: int) -> tuple[int, int, int]:
        """Compute ``(start, stop, step)`` for a sequence of ``length``.

        Args:
            length (int): Target sequence length.

        Returns:
            tuple[int, int, int]: Concrete ``(start, stop, step)`` indices.
        """


@runtime_checkable
class DTypeProtocol(Protocol):
    """Protocol representing an array data type descriptor."""

    @property
    def name(self) -> str:
        """Return the string name of the data type.

        Returns:
            str: Data type name.
        """

    @property
    def hasobject(self) -> bool:
        """Return True if the data type contains Python objects.

        Returns:
            bool: Whether the dtype holds Python object references.
        """


@runtime_checkable
class ArrayFlagsProtocol(Protocol):
    """Protocol representing memory layout flags of an array."""

    @property
    def c_contiguous(self) -> bool:
        """Return True if the array is C-contiguous in memory.

        Returns:
            bool: Whether the array is C-contiguous.
        """


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol representing a multidimensional array (such as numpy.ndarray)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array dimensions.

        Returns:
            tuple[int, ...]: Shape tuple.
        """

    @property
    def dtype(self) -> DTypeProtocol:
        """Return the array data type descriptor.

        Returns:
            DTypeProtocol: Data type object.
        """

    @property
    def flags(self) -> ArrayFlagsProtocol:
        """Return the array memory flags.

        Returns:
            ArrayFlagsProtocol: Memory layout flags.
        """

    def __len__(self) -> int:
        """Return the size of the leading dimension.

        Returns:
            int: Length of the first dimension.
        """

    def __getitem__(self, index: int) -> int | float | ArrayProtocol:
        """Return the element or sub-array at ``index``.

        Args:
            index (int): Positional index.

        Returns:
            int | float | ArrayProtocol: Indexed element or slice.
        """

    def __iter__(self) -> Iterator[int]:
        """Iterate over the leading dimension.

        Returns:
            Iterator[int]: Iterator over elements.
        """

    def tolist(self) -> list[ScalarElement] | ScalarElement:
        """Convert the array into nested Python lists or a scalar.

        Returns:
            list[ScalarElement] | ScalarElement: Python representation.
        """


@runtime_checkable
class SharedMemoryMetadataProtocol(Protocol):
    """Protocol representing shared memory array metadata."""

    name: str
    shape: tuple[int, ...]

    def close_and_unlink(self) -> None:
        """Close and unlink the shared memory segment."""


@runtime_checkable
class ArrayConvertibleProtocol(Protocol):
    """Protocol for objects implementing the NumPy ``__array__`` interface."""

    def __array__(self) -> ArrayProtocol:
        """Convert the instance into an array.

        Returns:
            ArrayProtocol: Resulting array.
        """


@runtime_checkable
class NamedTupleProtocol(Protocol):
    """Protocol for ``collections.namedtuple`` and ``typing.NamedTuple`` instances."""

    _fields: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of tuple fields.

        Returns:
            int: Field count.
        """

    def __getitem__(self, index: int) -> ElementTree:
        """Return the field value at ``index``.

        Args:
            index (int): Positional field index.

        Returns:
            ElementTree: Value at ``index``.
        """


@runtime_checkable
class DataclassProtocol(Protocol):
    """Protocol for dataclass instances."""

    @property
    def __dataclass_fields__(self) -> Mapping[str, dataclasses.Field[ScalarElement]]:
        """Return the mapping of dataclass field definitions.

        Returns:
            Mapping[str, dataclasses.Field[ScalarElement]]: Dataclass fields mapping.
        """


@runtime_checkable
class RNGGeneratorProtocol(Protocol):
    """Protocol for deterministic random number generators."""

    def random(self) -> float:
        """Return a random floating-point number in ``[0.0, 1.0)``.

        Returns:
            float: Uniformly distributed float in ``[0.0, 1.0)``.
        """

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return a random floating-point number in ``[low, high)``.

        Args:
            low (float): Lower bound of the interval.
            high (float): Upper bound of the interval.

        Returns:
            float: Sampled float value.
        """

    def integers(self, low: int, high: int | None = None) -> int:
        """Return a random integer from ``[0, low)`` or ``[low, high)``.

        Args:
            low (int): Lowest integer (or exclusive upper bound if ``high`` is None).
            high (int | None): Exclusive upper bound when provided.

        Returns:
            int: Sampled integer.
        """

    def permutation(self, x: int | Sequence[int]) -> list[int] | ArrayProtocol:
        """Return a randomly permuted sequence of integers.

        Args:
            x (int | Sequence[int]): Upper bound integer ``n`` or sequence to permute.

        Returns:
            list[int] | ArrayProtocol: Permuted sequence.
        """


@runtime_checkable
class NumpyRandomModuleProtocol(Protocol):
    """Protocol representing ``numpy.random`` module functions."""

    def default_rng(self, seed: int | None = None) -> RNGGeneratorProtocol:
        """Construct a default NumPy Generator.

        Args:
            seed (int | None): Optional integer seed.

        Returns:
            RNGGeneratorProtocol: Initialized generator.
        """


@runtime_checkable
class NumpyModuleProtocol(Protocol):
    """Protocol representing the ``numpy`` module when loaded in ``sys.modules``."""

    ndarray: type[ArrayProtocol]
    random: NumpyRandomModuleProtocol

    def array(self, obj: Sequence[ElementTree] | ElementTree) -> ArrayProtocol:
        """Create a NumPy array from ``obj``.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            ArrayProtocol: Constructed NumPy array.
        """

    def asarray(self, obj: Sequence[ElementTree] | ElementTree) -> ArrayProtocol:
        """Convert ``obj`` to a NumPy array.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            ArrayProtocol: Constructed NumPy array.
        """

    def zeros_like(self, a: ArrayProtocol) -> ArrayProtocol:
        """Return an array of zeros with the same shape and type as ``a``.

        Args:
            a (ArrayProtocol): Reference array.

        Returns:
            ArrayProtocol: Zero-filled array.
        """


@runtime_checkable
class AbslLoggingModuleProtocol(Protocol):
    """Protocol representing ``absl.logging`` when loaded in ``sys.modules``."""

    def warning(self, msg: str, *args: ScalarElement) -> None:
        """Log a warning message via absl logging.

        Args:
            msg (str): Warning message format string.
            *args (ScalarElement): Format arguments.
        """


class CheckpointStateDict(TypedDict, total=False):
    """Typed dictionary representing serialized ``DataLoaderIterator`` state."""

    version: int
    last_seen_indices: dict[str, int]
    last_worker_index: int
    worker_count: int
    sampler: str
    data_source: str


ElementTree: TypeAlias = Union[
    ScalarElement,
    ArrayProtocol,
    ArrayConvertibleProtocol,
    SharedMemoryMetadataProtocol,
    DataclassProtocol,
    NamedTupleProtocol,
    Sequence["ElementTree"],
    Mapping[str, "ElementTree"],
]

T = TypeVar("T", bound=ElementTree)
T_co = TypeVar("T_co", bound=ElementTree, covariant=True)
T_in = TypeVar("T_in", bound=ElementTree)
T_out = TypeVar("T_out", bound=ElementTree)


def get_numpy_module() -> NumpyModuleProtocol | None:
    """Return the loaded ``numpy`` module if present in ``sys.modules``.

    Returns:
        NumpyModuleProtocol | None: Loaded NumPy module or None.
    """
    mod = sys.modules.get("numpy")
    if isinstance(mod, NumpyModuleProtocol):
        return mod
    return None


def is_namedtuple_instance(value: ElementTree) -> TypeGuard[NamedTupleProtocol]:
    """Check whether ``value`` is a ``NamedTuple`` instance.

    Args:
        value (ElementTree): Candidate value to inspect.

    Returns:
        TypeGuard[NamedTupleProtocol]: True if ``value`` is a tuple with ``_fields``.
    """
    return isinstance(value, tuple) and isinstance(value, NamedTupleProtocol)


def is_ndarray_instance(value: ElementTree) -> TypeGuard[ArrayProtocol]:
    """Check whether ``value`` is a full ``ArrayProtocol`` instance.

    Args:
        value (ElementTree): Candidate value to inspect.

    Returns:
        TypeGuard[ArrayProtocol]: True if ``value`` satisfies ``ArrayProtocol``.
    """
    return isinstance(value, ArrayProtocol)


__all__ = [
    "AbslLoggingModuleProtocol",
    "ArrayConvertibleProtocol",
    "ArrayFlagsProtocol",
    "ArrayProtocol",
    "CheckpointStateDict",
    "DTypeProtocol",
    "DataclassProtocol",
    "ElementTree",
    "NamedTupleProtocol",
    "NumpyModuleProtocol",
    "NumpyRandomModuleProtocol",
    "RNGGeneratorProtocol",
    "ScalarElement",
    "SharedMemoryMetadataProtocol",
    "T",
    "T_co",
    "T_in",
    "T_out",
    "get_numpy_module",
    "is_namedtuple_instance",
    "is_ndarray_instance",
    "Callable",
    "Generic",
    "Iterable",
    "Iterator",
    "Mapping",
    "Sequence",
]
