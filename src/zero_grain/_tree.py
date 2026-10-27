"""PyTree traversal, structure validation, batching, and padding utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from zero_flax.core import FrozenEntry
from zero_flax.nnx.state import StateEntry
import zero_jax
from zero_grain._types import (
    _ZERO_JAX_NUMPY_ADAPTER,
    ArrayConvertibleProtocol,
    CompilerDevice,
    CompilerDType,
    CompilerTensor,
    CompilerTensorConfig,
    ElementTree,
    FlaxState,
    FlaxVariable,
    FlaxVariableBase,
    FrozenDict,
    JaxArray,
    MappableDataclassProtocol,
    ScalarElement,
    SharedMemoryMetadataProtocol,
    ZeroArrayProtocol,
    _infer_nested_list_and_shape,
    _make_zero_nested_list,
    dtype_name,
    get_numpy_module,
    is_dataclass_instance,
    is_mappable_dataclass_instance,
    is_namedtuple_instance,
    is_ndarray_instance,
)


def _build_flax_state(d: Mapping[str, ElementTree]) -> ElementTree:
    """Construct a ``FlaxState`` from a mapping of ``ElementTree`` entries.

    Args:
        d (Mapping[str, ElementTree]): Source mapping.

    Returns:
        ElementTree: Constructed ``FlaxState`` instance.
    """
    entries: dict[str, StateEntry] = {
        k: v
        for k, v in d.items()
        if isinstance(v, (FlaxVariableBase, JaxArray, CompilerTensor, int, float, bool, str, FlaxState))
    }
    return FlaxState(entries)


def _build_frozen_dict(d: Mapping[str, ElementTree]) -> ElementTree:
    """Construct a ``FrozenDict`` from a mapping of ``ElementTree`` entries.

    Args:
        d (Mapping[str, ElementTree]): Source mapping.

    Returns:
        ElementTree: Constructed ``FrozenDict`` instance.
    """
    entries: dict[str, FrozenEntry] = {
        k: v
        for k, v in d.items()
        if isinstance(v, (JaxArray, CompilerTensor, int, float, bool, str, bytes, complex, FrozenDict))
    }
    return FrozenDict(entries)


def _reconstruct_mappable_dataclass(
    instance: MappableDataclassProtocol,
    field_names: tuple[str, ...],
    field_values: tuple[ElementTree, ...],
) -> ElementTree:
    """Reconstruct a ``MappableDataclassProtocol`` instance with updated ``field_values``.

    Args:
        instance (MappableDataclassProtocol): Template dataclass instance.
        field_names (tuple[str, ...]): Ordered field names.
        field_values (tuple[ElementTree, ...]): Ordered new field values.

    Returns:
        ElementTree: Reconstructed ``MappableDataclassProtocol`` instance.
    """
    cls = type(instance)
    return cls(**dict(zip(field_names, field_values)))


def _assert_same_structure(batch: Sequence[ElementTree]) -> None:
    """Verify that all elements in ``batch`` share an identical PyTree structure via ``zero_chex``.

    Args:
        batch (Sequence[ElementTree]): Sequence of elements to validate.

    Raises:
        TypeError: If any element in ``batch`` has a mismatched PyTree structure.
    """
    mappable_items: list[dict[str, int]] = [
        {name: 0 for name in x.__chex_mappable_fields__()} for x in batch if is_mappable_dataclass_instance(x)
    ]
    if len(mappable_items) > 1:
        import zero_optax as _zero_optax  # noqa: F401
        import zero_chex

        zero_chex.assert_trees_all_equal_structs(*mappable_items, exception_type=TypeError)


def _convert_leaf_batch(
    batch: Sequence[ElementTree],
    *,
    as_jax_array: bool = False,
) -> ElementTree:
    """Stack a homogeneous sequence of leaf values into a batched array or list.

    Args:
        batch (Sequence[ElementTree]): Sequence of leaf elements to stack.
        as_jax_array (bool): Whether to stack leaves directly into a ``zero_jax.numpy.ndarray``.

    Returns:
        ElementTree: Stacked ``zero_jax``/NumPy/CompilerTensor/ZeroArray or list.
    """
    import zero_optax as _zero_optax  # noqa: F401
    import zero_chex

    first = batch[0]
    if isinstance(first, JaxArray) or as_jax_array:
        jax_items: list[JaxArray] = []
        for item in batch:
            if isinstance(item, JaxArray):
                jax_items.append(item)
            elif isinstance(item, (int, float, bool)) or is_ndarray_instance(item):
                jax_items.append(_ZERO_JAX_NUMPY_ADAPTER.asarray(item))
            else:
                return list(batch)
        rank = len(jax_items[0].shape)
        if rank > 0:
            if any(len(arr.shape) != rank or arr.shape != jax_items[0].shape for arr in jax_items):
                raise ValueError("inhomogeneous leaf shapes in batch")
            zero_chex.assert_equal_shape_prefix(jax_items, rank, exception_type=ValueError)
        stacked_jax: JaxArray = _ZERO_JAX_NUMPY_ADAPTER.stack(jax_items, axis=0)
        zero_chex.assert_axis_dimension(stacked_jax, 0, len(batch), exception_type=ValueError)
        return stacked_jax
    if isinstance(first, CompilerTensor):
        tensors = [item for item in batch if isinstance(item, CompilerTensor)]
        dt_str = dtype_name(first.dtype)
        target_dt = CompilerDType(dt_str)
        sub_shapes = tuple(int(d) for d in first.shape)
        stacked_shape = (len(tensors), *sub_shapes)
        jax_wrappers: list[ElementTree] = [JaxArray(t) for t in tensors]
        nested_payload, _, _ = _infer_nested_list_and_shape(jax_wrappers)
        return CompilerTensor(nested_payload, CompilerTensorConfig(stacked_shape, target_dt, CompilerDevice("cpu")))
    if isinstance(first, ZeroArrayProtocol):
        from zero_chex.types import ZeroArray

        flat_data: list[int | float | complex | bool] = [
            val for item in batch if isinstance(item, ZeroArrayProtocol) for val in item.to_list()
        ]
        res_zero: ZeroArrayProtocol = ZeroArray(
            flat_data,
            shape=(len(batch), *first.shape),
            dtype=dtype_name(first.dtype),
        )
        zero_chex.assert_axis_dimension(res_zero, 0, len(batch), exception_type=ValueError)
        return res_zero
    np_mod = get_numpy_module()
    if np_mod is not None:
        try:
            return np_mod.array(batch)
        except (ValueError, TypeError):
            return list(batch)
    return list(batch)


def _batch_elements(
    batch: Sequence[ElementTree],
    *,
    as_jax_array: bool = False,
) -> ElementTree:
    """Collate a sequence of structured elements into a single batched PyTree.

    Note:
        Following Google Grain's ``tree_lib`` PyTree semantics, Python ``list``
        and ``tuple`` instances are treated as PyTree containers whose positional
        children are recursively batched across records and (for ``list``) stacked
        into a NumPy or ``zero_jax`` array.

    Args:
        batch (Sequence[ElementTree]): Sequence of elements sharing identical tree structure.
        as_jax_array (bool): Whether to emit ``zero_jax.numpy.ndarray`` leaf arrays.

    Returns:
        ElementTree: Batched PyTree with leaves stacked along the leading dimension.

    Raises:
        TypeError: If elements in ``batch`` have mismatched dictionary keys or sequence lengths.
    """
    if not batch:
        return list(batch)
    first = batch[0]
    if isinstance(first, FlaxVariable):
        var_cls = type(first)
        mismatched_var_pos: list[int] = [
            idx for idx, item in enumerate(batch) if not isinstance(item, FlaxVariable) or type(item) is not var_cls
        ]
        if mismatched_var_pos:
            mismatched_var_vals: list[ElementTree] = [batch[idx] for idx in mismatched_var_pos]
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {mismatched_var_pos} "
                f"have structures {mismatched_var_vals}."
            )
        var_values: list[ElementTree] = [item.value for item in batch if isinstance(item, FlaxVariable)]
        batched_var_val = _batch_elements(var_values, as_jax_array=as_jax_array)
        if isinstance(batched_var_val, JaxArray):
            return first.replace(value=batched_var_val)
        return first.replace(value=_ZERO_JAX_NUMPY_ADAPTER.asarray(batched_var_val))

    if is_mappable_dataclass_instance(first):
        field_names = first.__chex_mappable_fields__()
        first_cls = type(first)
        mismatched_dc_pos: list[int] = []
        mismatched_dc_vals: list[ElementTree] = []
        for idx, item in enumerate(batch):
            if (
                not is_mappable_dataclass_instance(item)
                or type(item) is not first_cls
                or item.__chex_mappable_fields__() != field_names
            ):
                mismatched_dc_pos.append(idx)
                mismatched_dc_vals.append(item)
        if mismatched_dc_pos:
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {mismatched_dc_pos} "
                f"have structures {mismatched_dc_vals}."
            )
        _assert_same_structure(batch)
        batched_dc_values: list[ElementTree] = []
        for field_idx, _ in enumerate(field_names):
            col_vals: list[ElementTree] = [
                item.__chex_field_values__()[field_idx] for item in batch if is_mappable_dataclass_instance(item)
            ]
            batched_dc_values.append(_batch_elements(col_vals, as_jax_array=as_jax_array))
        return _reconstruct_mappable_dataclass(first, field_names, tuple(batched_dc_values))

    if is_dataclass_instance(first):
        dc_cls = type(first)
        mismatched_std_dc_pos: list[int] = [idx for idx, item in enumerate(batch) if type(item) is not dc_cls]
        if mismatched_std_dc_pos:
            mismatched_std_dc_vals: list[ElementTree] = [batch[idx] for idx in mismatched_std_dc_pos]
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {mismatched_std_dc_pos} "
                f"have structures {mismatched_std_dc_vals}."
            )
        return _convert_leaf_batch(batch, as_jax_array=as_jax_array)

    if isinstance(first, Mapping):
        first_keys = set(first.keys())
        mismatched_positions: list[int] = []
        mismatched_values: list[ElementTree] = []
        for idx, item in enumerate(batch):
            if not isinstance(item, Mapping) or set(item.keys()) != first_keys:
                mismatched_positions.append(idx)
                mismatched_values.append(item)
        if mismatched_positions:
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {mismatched_positions} "
                f"have structures {mismatched_values}."
            )
        result_dict: dict[str, ElementTree] = {}
        for key in first:
            sub_batch: list[ElementTree] = [item[key] for item in batch if isinstance(item, Mapping)]
            result_dict[key] = _batch_elements(sub_batch, as_jax_array=as_jax_array)
        if isinstance(first, FlaxState):
            return _build_flax_state(result_dict)
        if isinstance(first, FrozenDict):
            return _build_frozen_dict(result_dict)
        return result_dict

    if is_namedtuple_instance(first):
        tuple_cls = type(first)
        field_count = len(first)
        first_fields = first._fields
        mismatched_nt_pos: list[int] = []
        mismatched_nt_vals: list[ElementTree] = []
        for idx, item in enumerate(batch):
            if (
                not is_namedtuple_instance(item)
                or type(item) is not tuple_cls
                or item._fields != first_fields
                or len(item) != field_count
            ):
                mismatched_nt_pos.append(idx)
                mismatched_nt_vals.append(item)
        if mismatched_nt_pos:
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {mismatched_nt_pos} "
                f"have structures {mismatched_nt_vals}."
            )
        batched_fields: list[ElementTree] = []
        for field_idx in range(field_count):
            col: list[ElementTree] = [item[field_idx] for item in batch if isinstance(item, tuple)]
            batched_fields.append(_batch_elements(col, as_jax_array=as_jax_array))
        return tuple_cls(*batched_fields)

    if isinstance(first, (tuple, list)):
        first_len = len(first)
        bad_positions: list[int] = []
        bad_items: list[ElementTree] = []
        for idx, item in enumerate(batch):
            if not isinstance(item, (tuple, list)) or len(item) != first_len:
                bad_positions.append(idx)
                bad_items.append(item)
        if bad_positions:
            raise TypeError(
                f"Record structures do not match. Record at position 0 has "
                f"structure {first}, while records at positions {bad_positions} "
                f"have structures {bad_items}."
            )
        batched_tuple = tuple(
            _batch_elements([item[pos] for item in batch if isinstance(item, (tuple, list))], as_jax_array=as_jax_array)
            for pos in range(first_len)
        )
        if isinstance(first, list):
            if as_jax_array:
                return _ZERO_JAX_NUMPY_ADAPTER.asarray(list(batched_tuple))
            np_mod = get_numpy_module()
            if np_mod is not None:
                return np_mod.array(batched_tuple)
            return list(batched_tuple)
        return batched_tuple

    return _convert_leaf_batch(batch, as_jax_array=as_jax_array)


def _zero_like_element(element: ElementTree, pad_value: ScalarElement = 0) -> ElementTree:
    """Create a zero-valued padding element matching the structure of ``element``.

    Args:
        element (ElementTree): Reference element to mimic.
        pad_value (ScalarElement): Scalar fallback padding value.

    Returns:
        ElementTree: Zero-initialized element matching ``element``.
    """
    if isinstance(element, FlaxVariable):
        zero_inner = _zero_like_element(element.value, pad_value)
        if isinstance(zero_inner, JaxArray):
            return element.replace(value=zero_inner)
        return element.replace(value=_ZERO_JAX_NUMPY_ADAPTER.asarray(zero_inner))
    if is_mappable_dataclass_instance(element):
        field_names = element.__chex_mappable_fields__()
        zero_vals = tuple(_zero_like_element(v, pad_value) for v in element.__chex_field_values__())
        return _reconstruct_mappable_dataclass(element, field_names, zero_vals)
    if isinstance(element, Mapping):
        zero_dict = {k: _zero_like_element(v, pad_value) for k, v in element.items()}
        if isinstance(element, FlaxState):
            return _build_flax_state(zero_dict)
        if isinstance(element, FrozenDict):
            return _build_frozen_dict(zero_dict)
        return zero_dict
    if is_namedtuple_instance(element):
        tuple_cls = type(element)
        return tuple_cls(*(_zero_like_element(element[i], pad_value) for i in range(len(element))))
    if isinstance(element, tuple):
        return tuple(_zero_like_element(v, pad_value) for v in element)
    if isinstance(element, list):
        return [_zero_like_element(v, pad_value) for v in element]

    if isinstance(element, JaxArray):
        return _ZERO_JAX_NUMPY_ADAPTER.zeros_like(element)
    if isinstance(element, CompilerTensor):
        dt_str = dtype_name(element.dtype)
        target_dt = CompilerDType(dt_str)
        tensor_shape = tuple(int(d) for d in element.shape)
        zero_scalar: int | float | bool = False if dt_str == "bool" else (0 if "int" in dt_str else 0.0)
        zeros_payload = _make_zero_nested_list(tensor_shape, zero_scalar)
        return CompilerTensor(zeros_payload, CompilerTensorConfig(tensor_shape, target_dt, CompilerDevice("cpu")))
    if isinstance(element, ZeroArrayProtocol):
        from zero_chex.types import ZeroArray

        zero_arr: ZeroArrayProtocol = ZeroArray.zeros(
            element.shape,
            dtype=dtype_name(element.dtype),
        )
        return zero_arr

    np_mod = get_numpy_module()
    if np_mod is not None:
        if is_ndarray_instance(element):
            return np_mod.zeros_like(element)
        if isinstance(element, ArrayConvertibleProtocol):
            arr = element.__array__()
            return np_mod.zeros_like(arr)
    if isinstance(element, bool):
        return False
    if isinstance(element, int):
        return 0
    if isinstance(element, float):
        return 0.0
    return pad_value


def _pad_jax_leaf(leaf: ElementTree, pad_len: int, batch_size: int, pad_value: ScalarElement = 0) -> ElementTree:
    """Pad a ``JaxArray`` leaf along leading axis 0 using ``zero_jax.numpy.pad``.

    Args:
        leaf (ElementTree): Leaf value (padded if ``JaxArray``).
        pad_len (int): Number of trailing rows to pad along axis 0.
        batch_size (int): Expected leading dimension after padding.
        pad_value (ScalarElement): Constant value used for padding.

    Returns:
        ElementTree: Padded ``JaxArray`` leaf, or ``leaf`` unchanged if not a ``JaxArray``.
    """
    if isinstance(leaf, JaxArray):
        import zero_optax as _zero_optax  # noqa: F401
        import zero_chex

        pad_widths: tuple[tuple[int, int], ...] = ((0, pad_len), *((0, 0) for _ in range(leaf.ndim - 1)))
        const_val: int | float | bool = pad_value if isinstance(pad_value, (int, float, bool)) else 0
        padded: JaxArray = _ZERO_JAX_NUMPY_ADAPTER.pad(leaf, pad_widths, constant_values=const_val)
        zero_chex.assert_axis_dimension(padded, 0, batch_size, exception_type=ValueError)
        return padded
    return leaf


def _pad_batch_elements(
    elements: Sequence[ElementTree],
    batch_size: int,
    pad_value: ScalarElement = 0,
    *,
    as_jax_array: bool = False,
) -> ElementTree:
    """Batch ``elements`` and pad along the leading axis to ``batch_size``.

    Uses ``zero_jax.numpy.stack`` and ``zero_jax.numpy.pad`` directly when leaves
    are ``JaxArray`` instances or ``as_jax_array=True``.

    Args:
        elements (Sequence[ElementTree]): Input sequence of elements to batch.
        batch_size (int): Target batch size after padding.
        pad_value (ScalarElement): Scalar value used to pad incomplete batches.
        as_jax_array (bool): Whether to emit ``zero_jax.numpy.ndarray`` leaf arrays.

    Returns:
        ElementTree: Batched and padded PyTree.
    """
    import sys

    if not elements or batch_size <= 0:
        return []
    first = elements[0]
    prefix_elements: list[ElementTree] = list(elements[:batch_size])
    pad_len = batch_size - len(prefix_elements)
    if pad_len > 0 and sys.modules.get("numpy") is not None and (as_jax_array or isinstance(first, JaxArray)):
        stacked_prefix = _batch_elements(prefix_elements, as_jax_array=as_jax_array)
        return _map_tree(
            lambda leaf: _pad_jax_leaf(leaf, pad_len=pad_len, batch_size=batch_size, pad_value=pad_value),
            stacked_prefix,
        )
    if pad_len > 0:
        pad_elem = _zero_like_element(first, pad_value=pad_value)
        prefix_elements.extend([pad_elem for _ in range(pad_len)])
    return _batch_elements(prefix_elements, as_jax_array=as_jax_array)


def batch_and_pad(
    elements: Sequence[ElementTree],
    batch_size: int,
    pad_value: ScalarElement = 0,
    *,
    as_jax_array: bool = False,
) -> ElementTree:
    """Batch a sequence of elements and pad to ``batch_size`` if fewer elements are given.

    Args:
        elements (Sequence[ElementTree]): Input sequence of elements to batch.
        batch_size (int): Target batch size after padding.
        pad_value (ScalarElement): Scalar value used to pad incomplete batches.
        as_jax_array (bool): Whether to emit ``zero_jax.numpy.ndarray`` leaf arrays.

    Returns:
        ElementTree: Batched and padded PyTree.
    """
    return _pad_batch_elements(elements, batch_size=batch_size, pad_value=pad_value, as_jax_array=as_jax_array)


def _leaf_type_spec(leaf: ElementTree) -> str:
    """Return the structural type specification string for a single PyTree ``leaf``.

    Args:
        leaf (ElementTree): Leaf element to describe.

    Returns:
        str: Canonical type or dtype/shape string for ``leaf``.
    """
    if is_ndarray_instance(leaf):
        return f"{dtype_name(leaf.dtype)}{list(leaf.shape)}"
    if isinstance(leaf, SharedMemoryMetadataProtocol):
        return f"SharedMemoryArrayMetadata{list(leaf.shape)}"
    if isinstance(leaf, ArrayConvertibleProtocol):
        arr = leaf.__array__()
        return f"{dtype_name(arr.dtype)}{list(arr.shape)}"
    if leaf is None:
        return "NoneType"
    if isinstance(leaf, bool):
        return "bool"
    if isinstance(leaf, int):
        return "int"
    if isinstance(leaf, float):
        return "float"
    if isinstance(leaf, complex):
        return "complex"
    if isinstance(leaf, str):
        return "str"
    if isinstance(leaf, bytes):
        return "bytes"
    if is_dataclass_instance(leaf):
        return "dataclass"
    return "unknown"


def _map_tree(
    fn: Callable[[ElementTree], ElementTree],
    tree: ElementTree,
) -> ElementTree:
    """Map ``fn`` across leaves of ``tree`` while delegating leaf structure checks to ``zero_jax.tree_util``.

    Args:
        fn (Callable[[ElementTree], ElementTree]): Function applied to each leaf.
        tree (ElementTree): Input PyTree structure.

    Returns:
        ElementTree: Transformed PyTree with identical container hierarchy.
    """
    if isinstance(tree, FlaxVariable):
        mapped_inner = _map_tree(fn, tree.value)
        if isinstance(mapped_inner, JaxArray):
            return tree.replace(value=mapped_inner)
        return tree.replace(value=_ZERO_JAX_NUMPY_ADAPTER.asarray(mapped_inner))
    if is_mappable_dataclass_instance(tree):
        field_names = tree.__chex_mappable_fields__()
        mapped_vals = tuple(_map_tree(fn, v) for v in tree.__chex_field_values__())
        return _reconstruct_mappable_dataclass(tree, field_names, mapped_vals)
    if isinstance(tree, Mapping):
        mapped_dict = {k: _map_tree(fn, v) for k, v in tree.items()}
        if isinstance(tree, FlaxState):
            return _build_flax_state(mapped_dict)
        if isinstance(tree, FrozenDict):
            return _build_frozen_dict(mapped_dict)
        return mapped_dict
    if is_namedtuple_instance(tree):
        tuple_cls = type(tree)
        return tuple_cls(*(_map_tree(fn, tree[i]) for i in range(len(tree))))
    if isinstance(tree, tuple):
        return tuple(_map_tree(fn, v) for v in tree)
    if isinstance(tree, list):
        return [_map_tree(fn, v) for v in tree]
    zero_jax.tree_util.tree_map(lambda x: x, 0)
    return fn(tree)


def spec_like(tree: ElementTree) -> ElementTree:
    """Compute a structural type/shape specification matching ``tree``.

    Args:
        tree (ElementTree): Input PyTree structure.

    Returns:
        ElementTree: Specification tree with leaf types and shapes.
    """
    return _map_tree(_leaf_type_spec, tree)


def map_tree(
    fn: Callable[[ElementTree], ElementTree],
    tree: ElementTree,
) -> ElementTree:
    """Apply ``fn`` to every leaf in ``tree`` while preserving container structure.

    Args:
        fn (Callable[[ElementTree], ElementTree]): Function applied to each leaf.
        tree (ElementTree): Input PyTree structure.

    Returns:
        ElementTree: Transformed PyTree with identical container hierarchy.
    """
    return _map_tree(fn, tree)


__all__ = [
    "_assert_same_structure",
    "_batch_elements",
    "_convert_leaf_batch",
    "_leaf_type_spec",
    "_map_tree",
    "_pad_batch_elements",
    "batch_and_pad",
    "map_tree",
    "spec_like",
]
