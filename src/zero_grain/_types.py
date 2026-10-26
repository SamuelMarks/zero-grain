"""Strongly typed structural protocols, type variables, and aliases for zero-grain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import sys
from typing import Protocol, TypedDict, TypeVar, Union, runtime_checkable
from ml_switcheroo_compiler.core.device import Device as CompilerDevice
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor, TensorConfig as CompilerTensorConfig
from typing_extensions import TypeAlias, TypeGuard
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec
from zero_jax.numpy import ndarray as JaxArray

ScalarElement: TypeAlias = Union[int, float, bool, str, bytes, complex, None]
NestedNumericList: TypeAlias = Union[int, float, bool, complex, list["NestedNumericList"]]
_PYTHON_SCALAR_DTYPE_NAMES: dict[type[int | float | bool | complex], str] = {
    bool: "bool",
    int: "int64",
    float: "float64",
    complex: "complex128",
}


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
    if isinstance(dtype, type) and dtype in _PYTHON_SCALAR_DTYPE_NAMES:
        return _PYTHON_SCALAR_DTYPE_NAMES[dtype]
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

    def tolist(self) -> list[ScalarElement] | list[NestedNumericList] | ScalarElement:
        """Convert the array into nested Python lists or a scalar.

        Returns:
            list[ScalarElement] | list[NestedNumericList] | ScalarElement: Nested list or scalar representation.
        """


@runtime_checkable
class BytesExportArrayProtocol(ArrayProtocol, Protocol):
    """Protocol representing an array exposing a ``tobytes()`` byte serialization method."""

    def tobytes(self) -> bytes:
        """Serialize the array buffer to contiguous raw bytes.

        Returns:
            bytes: Raw buffer bytes.
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

    def array(
        self,
        obj: Sequence[ElementTree] | ElementTree,
        dtype: DTypeSpec = None,
    ) -> ArrayProtocol:
        """Create an array from ``obj``.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.
            dtype (DTypeSpec): Optional explicit data type override.

        Returns:
            ArrayProtocol: Constructed array.
        """

    def asarray(
        self,
        obj: Sequence[ElementTree] | ElementTree,
        dtype: DTypeSpec = None,
    ) -> ArrayProtocol:
        """Convert ``obj`` to an array.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.
            dtype (DTypeSpec): Optional explicit data type override.

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


def _infer_nested_list_and_shape(
    obj: Sequence[ElementTree] | ElementTree,
) -> tuple[NestedNumericList, tuple[int, ...], str]:
    """Convert ``obj`` into nested numeric lists alongside its inferred ``(shape, dtype_name)``.

    Args:
        obj (Sequence[ElementTree] | ElementTree): Input sequence or scalar value.

    Returns:
        tuple[NestedNumericList, tuple[int, ...], str]: ``(nested_payload, shape, dtype_str)``.
    """
    if isinstance(obj, JaxArray):
        raw_data = obj._tensor._data
        if isinstance(raw_data, (int, float, bool)):
            return raw_data, obj.shape, dtype_name(obj.dtype)
        if isinstance(raw_data, Sequence) and not isinstance(raw_data, (str, bytes)):
            seq_items: list[ElementTree] = [item for item in raw_data if isinstance(item, (int, float, bool, list))]
            seq_res, _, _ = _infer_nested_list_and_shape(seq_items)
            return seq_res, obj.shape, dtype_name(obj.dtype)
        listed_jax = obj.tolist()
        return (
            [x for x in listed_jax if isinstance(x, (int, float, bool, complex, list))]
            if isinstance(listed_jax, list)
            else (listed_jax if isinstance(listed_jax, (int, float, bool, complex)) else 0),
            obj.shape,
            dtype_name(obj.dtype),
        )
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) == 0:
            return [], (0,), "float64"
        sub_results = [_infer_nested_list_and_shape(item) for item in obj]
        sub_lists: list[NestedNumericList] = [r[0] for r in sub_results]
        first_sub_shape = sub_results[0][1]
        dtypes = {r[2] for r in sub_results}
        merged_dt = "float64" if ("float64" in dtypes or "float32" in dtypes) else sub_results[0][2]
        return sub_lists, (len(obj), *first_sub_shape), merged_dt
    if isinstance(obj, bool):
        return obj, (), "bool"
    if isinstance(obj, int):
        return obj, (), "int64"
    if isinstance(obj, float):
        return obj, (), "float64"
    return 0, (), "int64"


def _make_zero_nested_list(shape: tuple[int, ...], zero_scalar: int | float | bool) -> NestedNumericList:
    """Build a nested list of zeros matching ``shape``.

    Args:
        shape (tuple[int, ...]): Target tensor shape.
        zero_scalar (int | float | bool): Scalar zero value.

    Returns:
        NestedNumericList: Nested zero list or scalar if 0-D.
    """
    if len(shape) <= 1:
        return [zero_scalar for _ in range(shape[0])] if shape else zero_scalar
    sub = _make_zero_nested_list(shape[1:], zero_scalar)
    return [sub for _ in range(shape[0])]


class _ZeroJaxNumpyAdapter:
    """Adapter wrapping ``zero_jax.numpy`` to satisfy ``NumpyModuleProtocol`` with dtype preservation."""

    ndarray: type[ArrayProtocol] = JaxArray
    random: NumpyRandomModuleProtocol = _ZeroJaxRandomAdapter()

    def array(
        self,
        obj: Sequence[ElementTree] | ElementTree,
        dtype: DTypeSpec = None,
    ) -> JaxArray:
        """Create a ``zero_jax`` array from ``obj`` without mutating ``sys.modules``.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.
            dtype (DTypeSpec): Optional explicit data type override.

        Returns:
            JaxArray: Constructed ``zero_jax`` array.
        """
        nested_data, shape, inferred_dt = _infer_nested_list_and_shape(obj)
        resolved_dt_name = dtype_name(dtype) if dtype is not None else inferred_dt
        target_dt = CompilerDType(resolved_dt_name)
        np_mod = sys.modules.get("numpy")
        if isinstance(np_mod, NumpyModuleProtocol):
            raw_arr = np_mod.asarray(nested_data, dtype=resolved_dt_name)
            return JaxArray(CompilerTensor(raw_arr, CompilerTensorConfig(shape, target_dt, CompilerDevice("cpu"))))
        return JaxArray(CompilerTensor(nested_data, CompilerTensorConfig(shape, target_dt, CompilerDevice("cpu"))))

    def asarray(
        self,
        obj: Sequence[ElementTree] | ElementTree,
        dtype: DTypeSpec = None,
    ) -> JaxArray:
        """Convert ``obj`` into a ``zero_jax`` array.

        Args:
            obj (Sequence[ElementTree] | ElementTree): Input sequence or value.
            dtype (DTypeSpec): Optional explicit data type override.

        Returns:
            JaxArray: Constructed ``zero_jax`` array.
        """
        return self.array(obj, dtype=dtype)

    def zeros_like(self, a: ArrayProtocol) -> JaxArray:
        """Return a zero-filled ``zero_jax`` array preserving ``a.shape`` and ``a.dtype``.

        Args:
            a (ArrayProtocol): Reference array.

        Returns:
            JaxArray: Zero-filled ``zero_jax`` array.
        """
        dt_str = dtype_name(a.dtype)
        target_dt = CompilerDType(dt_str)
        np_mod = sys.modules.get("numpy")
        if isinstance(np_mod, NumpyModuleProtocol):
            return JaxArray(
                CompilerTensor(np_mod.zeros_like(a), CompilerTensorConfig(a.shape, target_dt, CompilerDevice("cpu")))
            )
        zero_scalar: int | float | bool = False if dt_str == "bool" else (0 if "int" in dt_str else 0.0)
        zeros_payload = _make_zero_nested_list(a.shape, zero_scalar)
        return JaxArray(CompilerTensor(zeros_payload, CompilerTensorConfig(a.shape, target_dt, CompilerDevice("cpu"))))


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


def array_to_list(arr: ArrayProtocol) -> list[ScalarElement] | list[NestedNumericList] | ScalarElement:
    """Convert any ``ArrayProtocol`` instance into nested Python lists or a scalar.

    Args:
        arr (ArrayProtocol): Array instance to convert.

    Returns:
        list[ScalarElement] | list[NestedNumericList] | ScalarElement: Nested list or scalar representation.

    Raises:
        TypeError: If ``arr`` does not implement ``tolist()`` or ``to_list()``.
    """
    if isinstance(arr, ToListArrayProtocol):
        return arr.tolist()
    if isinstance(arr, ZeroArrayProtocol):
        zero_list: list[ScalarElement] = [x for x in arr.to_list()]
        return zero_list
    raise TypeError(f"Unsupported ArrayProtocol type for list conversion: {type(arr)}")


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
    "BytesExportArrayProtocol",
    "CheckpointStateDict",
    "CompilerDevice",
    "CompilerDType",
    "CompilerTensor",
    "CompilerTensorConfig",
    "DTypeProtocol",
    "DTypeSpec",
    "DataclassProtocol",
    "ElementTree",
    "FlaggedArrayProtocol",
    "JaxArray",
    "JaxArrayLikeSpec",
    "MappableDataclassProtocol",
    "NamedTupleProtocol",
    "NestedNumericList",
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
    "_ZERO_JAX_NUMPY_ADAPTER",
    "_infer_nested_list_and_shape",
    "_make_zero_nested_list",
    "array_to_list",
    "dtype_name",
    "get_numpy_module",
    "is_dataclass_instance",
    "is_mappable_dataclass_instance",
    "is_namedtuple_instance",
    "is_ndarray_instance",
]
