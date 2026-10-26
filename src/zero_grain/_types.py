"""Strongly typed structural protocols, type variables, and aliases for zero-grain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import sys
from types import ModuleType
from typing import Protocol, TypedDict, TypeVar, Union, runtime_checkable
from ml_switcheroo_compiler import ops as _compiler_ops
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from typing_extensions import TypeAlias, TypeGuard
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec
from zero_jax.numpy import ndarray as JaxArray

ScalarElement: TypeAlias = Union[int, float, bool, str, bytes, complex, None]
_INITIAL_NUMPY_MODULE: ModuleType | None = sys.modules.get("numpy")


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


@runtime_checkable
class ObjectDTypeProtocol(DTypeProtocol, Protocol):
    """Protocol representing a NumPy-like data type exposing ``hasobject``."""

    @property
    def hasobject(self) -> bool:
        """Return True if the data type contains Python objects.

        Returns:
            bool: Whether the dtype holds Python object references.
        """


DTypeSpec: TypeAlias = Union[
    DTypeProtocol,
    JaxArrayLikeSpec,
    str,
    type[int],
    type[float],
    type[complex],
    type[bool],
    None,
]


def dtype_name(dtype: DTypeSpec) -> str:
    """Resolve a ``DTypeSpec`` descriptor, string, or Python scalar type into a canonical dtype name.

    Args:
        dtype (DTypeSpec): Data type object, string, scalar type, or None.

    Returns:
        str: Canonical dtype name string.
    """
    if isinstance(dtype, DTypeProtocol):
        return dtype.name
    return str(dtype or "float64")


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
    """Protocol representing a multidimensional array across NumPy, zero-jax, ZeroArray, and CompilerTensor."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array dimensions.

        Returns:
            tuple[int, ...]: Shape tuple.
        """

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions.

        Returns:
            int: Number of dimensions.
        """

    @property
    def dtype(self) -> DTypeSpec:
        """Return the array data type descriptor.

        Returns:
            DTypeSpec: Data type object, string, or scalar type.
        """


@runtime_checkable
class FlaggedArrayProtocol(ArrayProtocol, Protocol):
    """Protocol representing a multidimensional array exposing memory layout ``flags``."""

    @property
    def flags(self) -> ArrayFlagsProtocol:
        """Return the array memory flags.

        Returns:
            ArrayFlagsProtocol: Memory layout flags.
        """


@runtime_checkable
class ToListArrayProtocol(Protocol):
    """Protocol representing an array exposing a ``tolist()`` conversion method."""

    def tolist(self) -> list[ScalarElement] | ScalarElement:
        """Convert the array into nested Python lists or a scalar.

        Returns:
            list[ScalarElement] | ScalarElement: Python representation.
        """


@runtime_checkable
class ZeroArrayProtocol(ArrayProtocol, Protocol):
    """Protocol representing a ``zero_chex.types.ZeroArray`` exposing ``to_list()``."""

    def to_list(self) -> list[int | float | complex | bool]:
        """Return a flat list of scalar elements in the array.

        Returns:
            list[int | float | complex | bool]: Flat scalar values.
        """


@runtime_checkable
class DataclassProtocol(Protocol):
    """Protocol for dataclass instances."""

    @property
    def __dataclass_fields__(self) -> dict[str, dataclasses.Field[ScalarElement]]:
        """Return the mapping of dataclass field definitions.

        Returns:
            dict[str, dataclasses.Field[ScalarElement]]: Dataclass fields mapping.
        """


@runtime_checkable
class MappableDataclassProtocol(Protocol):
    """Protocol for ``zero_chex.dataclass`` instances registered with mappable access."""

    def __init__(self, **kwargs: ElementTree) -> None:
        """Initialize the mappable dataclass from field keyword arguments.

        Args:
            **kwargs (ElementTree): Field names and values.
        """

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of field names in the dataclass.

        Returns:
            tuple[str, ...]: Ordered field names.
        """

    def __chex_field_values__(self) -> tuple[ElementTree, ...]:
        """Return the ordered tuple of field values in the dataclass.

        Returns:
            tuple[ElementTree, ...]: Ordered field values.
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
    """Protocol representing ``numpy`` or ``zero_jax.numpy`` array creation functions."""

    ndarray: type[ArrayProtocol]
    random: NumpyRandomModuleProtocol

    def array(self, obj: Sequence[ElementTree] | ElementTree) -> ArrayProtocol:
        """Create an array from ``obj``.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            ArrayProtocol: Constructed array.
        """

    def asarray(self, obj: Sequence[ElementTree] | ElementTree) -> ArrayProtocol:
        """Convert ``obj`` to an array.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            ArrayProtocol: Constructed array.
        """

    def zeros_like(self, a: ArrayProtocol) -> ArrayProtocol:
        """Return an array of zeros with the same shape and type as ``a``.

        Args:
            a (ArrayProtocol): Reference array.

        Returns:
            ArrayProtocol: Zero-filled array.
        """


class _ZeroJaxRandomAdapter:
    """Adapter providing ``default_rng`` for ``_ZeroJaxNumpyAdapter``."""

    def default_rng(self, seed: int | None = None) -> RNGGeneratorProtocol:
        """Construct a ``DeterministicGenerator`` seeded with ``seed``.

        Args:
            seed (int | None): Optional integer seed.

        Returns:
            RNGGeneratorProtocol: Deterministic random generator.
        """
        from zero_grain._rng import DeterministicGenerator

        return DeterministicGenerator(seed=seed)


class _ZeroJaxNumpyAdapter:
    """Adapter wrapping ``zero_jax.numpy`` to satisfy ``NumpyModuleProtocol`` with dtype preservation."""

    ndarray: type[ArrayProtocol] = JaxArray
    random: NumpyRandomModuleProtocol = _ZeroJaxRandomAdapter()

    def array(self, obj: Sequence[ElementTree] | ElementTree) -> JaxArray:
        """Create a ``zero_jax`` array from ``obj``.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            JaxArray: Constructed ``zero_jax`` array.
        """
        snapshot = dict(sys.modules)
        if sys.modules.get("numpy") is None and _INITIAL_NUMPY_MODULE is not None:
            sys.modules["numpy"] = _INITIAL_NUMPY_MODULE
        try:
            if isinstance(obj, Sequence) and len(obj) > 0 and isinstance(obj[0], JaxArray):
                tensors = [item._tensor for item in obj if isinstance(item, JaxArray)]
                stacked = _compiler_ops.stack(tensors)
                assert isinstance(stacked, CompilerTensor)
                return JaxArray(stacked)
            raw_list = (
                [int(x) if isinstance(x, int) else float(x) for x in obj if isinstance(x, (int, float, bool))]
                if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))
                else (obj if isinstance(obj, (int, float, bool)) else 0)
            )
            created = _compiler_ops.array(raw_list)
            assert isinstance(created, CompilerTensor)
            return JaxArray(created)
        finally:
            if sys.modules.get("numpy") is not snapshot.get("numpy"):
                sys.modules.clear()
                sys.modules.update(snapshot)

    def asarray(self, obj: Sequence[ElementTree] | ElementTree) -> JaxArray:
        """Convert ``obj`` into a ``zero_jax`` array.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.

        Returns:
            JaxArray: Constructed ``zero_jax`` array.
        """
        return self.array(obj)

    def zeros_like(self, a: ArrayProtocol) -> JaxArray:
        """Return a zero-filled ``zero_jax`` array preserving ``a.shape`` and ``a.dtype``.

        Args:
            a (ArrayProtocol): Reference array.

        Returns:
            JaxArray: Zero-filled ``zero_jax`` array.
        """
        snapshot = dict(sys.modules)
        if sys.modules.get("numpy") is None and _INITIAL_NUMPY_MODULE is not None:
            sys.modules["numpy"] = _INITIAL_NUMPY_MODULE
        try:
            target_dt = CompilerDType(dtype_name(a.dtype))
            zeros_t = _compiler_ops.zeros(a.shape, dtype=target_dt)
            assert isinstance(zeros_t, CompilerTensor)
            return JaxArray(zeros_t)
        finally:
            if sys.modules.get("numpy") is not snapshot.get("numpy"):
                sys.modules.clear()
                sys.modules.update(snapshot)


_ZERO_JAX_NUMPY_ADAPTER: NumpyModuleProtocol = _ZeroJaxNumpyAdapter()


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
    JaxArray,
    ZeroArrayProtocol,
    CompilerTensor,
    ArrayConvertibleProtocol,
    SharedMemoryMetadataProtocol,
    MappableDataclassProtocol,
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
    """Return the loaded ``numpy`` module if in ``sys.modules``, falling back to ``zero_jax.numpy``.

    Returns:
        NumpyModuleProtocol | None: Loaded NumPy module, ``zero_jax.numpy`` adapter, or None.
    """
    mod = sys.modules.get("numpy")
    if isinstance(mod, NumpyModuleProtocol):
        return mod
    jnp_mod = sys.modules.get("zero_jax.numpy")
    if jnp_mod is None and "zero_jax.numpy" in sys.modules:
        return None
    return _ZERO_JAX_NUMPY_ADAPTER


def array_to_list(arr: ArrayProtocol) -> list[ScalarElement] | ScalarElement:
    """Convert any ``ArrayProtocol`` instance into nested Python lists or a scalar.

    Args:
        arr (ArrayProtocol): Array instance to convert.

    Returns:
        list[ScalarElement] | ScalarElement: Nested list or scalar representation.
    """
    if isinstance(arr, ToListArrayProtocol):
        return arr.tolist()
    assert isinstance(arr, ZeroArrayProtocol)
    return [x for x in arr.to_list()]


def is_dataclass_instance(value: ElementTree) -> TypeGuard[DataclassProtocol]:
    """Check whether ``value`` is a dataclass instance (and not a dataclass type).

    Args:
        value (ElementTree): Candidate value to inspect.

    Returns:
        TypeGuard[DataclassProtocol]: True if ``value`` is an initialized dataclass instance.
    """
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def is_mappable_dataclass_instance(value: ElementTree) -> TypeGuard[MappableDataclassProtocol]:
    """Check whether ``value`` is a ``zero_chex`` ``MappableDataclassProtocol`` instance.

    Args:
        value (ElementTree): Candidate value to inspect.

    Returns:
        TypeGuard[MappableDataclassProtocol]: True if ``value`` implements ``MappableDataclassProtocol``.
    """
    return isinstance(value, MappableDataclassProtocol) and not isinstance(value, type)


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
    "CompilerDType",
    "CompilerTensor",
    "DTypeProtocol",
    "DTypeSpec",
    "DataclassProtocol",
    "ElementTree",
    "FlaggedArrayProtocol",
    "JaxArray",
    "JaxArrayLikeSpec",
    "MappableDataclassProtocol",
    "NamedTupleProtocol",
    "NumpyModuleProtocol",
    "NumpyRandomModuleProtocol",
    "ObjectDTypeProtocol",
    "RNGGeneratorProtocol",
    "ScalarElement",
    "SharedMemoryMetadataProtocol",
    "SliceProtocol",
    "T",
    "T_co",
    "T_in",
    "T_out",
    "ToListArrayProtocol",
    "ZeroArrayProtocol",
    "array_to_list",
    "dtype_name",
    "get_numpy_module",
    "is_dataclass_instance",
    "is_mappable_dataclass_instance",
    "is_namedtuple_instance",
    "is_ndarray_instance",
]
