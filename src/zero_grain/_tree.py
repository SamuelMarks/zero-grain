"""PyTree traversal, structure validation, batching, and padding utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from zero_grain._types import (
    _ZERO_JAX_NUMPY_ADAPTER,
    ArrayConvertibleProtocol,
    CompilerDevice,
    CompilerDType,
    CompilerTensor,
    CompilerTensorConfig,
    ElementTree,
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


def _convert_leaf_batch(batch: Sequence[ElementTree]) -> ElementTree:
    """Stack a homogeneous sequence of leaf values into a batched array or list.

    Args:
        batch (Sequence[ElementTree]): Sequence of leaf elements to stack.

    Returns:
        ElementTree: Stacked NumPy/zero_jax/CompilerTensor/ZeroArray or list.
    """
    first = batch[0]
    if isinstance(first, JaxArray):
        return _ZERO_JAX_NUMPY_ADAPTER.array(batch)
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
        return res_zero
    np_mod = get_numpy_module()
    if np_mod is not None:
        try:
            return np_mod.array(batch)
        except (ValueError, TypeError):
            return list(batch)
    return list(batch)


def _batch_elements(batch: Sequence[ElementTree]) -> ElementTree:
    """Collate a sequence of structured elements into a single batched PyTree.

    Note:
        Following Google Grain's ``tree_lib`` PyTree semantics, Python ``list``
        and ``tuple`` instances are treated as PyTree containers whose positional
        children are recursively batched across records and (for ``list``) stacked
        into a NumPy array when NumPy is available.

    Args:
        batch (Sequence[ElementTree]): Sequence of elements sharing identical tree structure.

    Returns:
        ElementTree: Batched PyTree with leaves stacked along the leading dimension.

    Raises:
        TypeError: If elements in ``batch`` have mismatched dictionary keys or sequence lengths.
    """
    if not batch:
        return list(batch)
    first = batch[0]
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
        batched_dc_values: list[ElementTree] = []
        for field_idx, _ in enumerate(field_names):
            col_vals: list[ElementTree] = [
                item.__chex_field_values__()[field_idx] for item in batch if is_mappable_dataclass_instance(item)
            ]
            batched_dc_values.append(_batch_elements(col_vals))
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
        return _convert_leaf_batch(batch)

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
            result_dict[key] = _batch_elements(sub_batch)
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
            batched_fields.append(_batch_elements(col))
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
            _batch_elements([item[pos] for item in batch if isinstance(item, (tuple, list))])
            for pos in range(first_len)
        )
        if isinstance(first, list):
            np_mod = get_numpy_module()
            if np_mod is not None:
                return np_mod.array(batched_tuple)
            return list(batched_tuple)
        return batched_tuple

    return _convert_leaf_batch(batch)


def _zero_like_element(element: ElementTree, pad_value: ScalarElement = 0) -> ElementTree:
    """Create a zero-valued padding element matching the structure of ``element``.

    Args:
        element (ElementTree): Reference element to mimic.
        pad_value (ScalarElement): Scalar fallback padding value.

    Returns:
        ElementTree: Zero-initialized element matching ``element``.
    """
    if is_mappable_dataclass_instance(element):
        field_names = element.__chex_mappable_fields__()
        zero_vals = tuple(_zero_like_element(v, pad_value) for v in element.__chex_field_values__())
        return _reconstruct_mappable_dataclass(element, field_names, zero_vals)
    if isinstance(element, Mapping):
        return {k: _zero_like_element(v, pad_value) for k, v in element.items()}
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


def batch_and_pad(
    elements: Sequence[ElementTree],
    batch_size: int,
    pad_value: ScalarElement = 0,
) -> ElementTree:
    """Batch a sequence of elements and pad to ``batch_size`` if fewer elements are given.

    Args:
        elements (Sequence[ElementTree]): Input sequence of elements to batch.
        batch_size (int): Target batch size after padding.
        pad_value (ScalarElement): Scalar value used to pad incomplete batches.

    Returns:
        ElementTree: Batched and padded PyTree.
    """
    if not elements or batch_size <= 0:
        return []
    first = elements[0]
    padded_elements: list[ElementTree] = list(elements[:batch_size])
    pad_len = batch_size - len(padded_elements)
    if pad_len > 0:
        pad_elem = _zero_like_element(first, pad_value=pad_value)
        padded_elements.extend([pad_elem for _ in range(pad_len)])
    return _batch_elements(padded_elements)


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


def spec_like(tree: ElementTree) -> ElementTree:
    """Compute a structural type/shape specification matching ``tree``.

    Args:
        tree (ElementTree): Input PyTree structure.

    Returns:
        ElementTree: Specification tree with leaf types and shapes.
    """
    if is_mappable_dataclass_instance(tree):
        field_names = tree.__chex_mappable_fields__()
        spec_vals = tuple(spec_like(v) for v in tree.__chex_field_values__())
        return _reconstruct_mappable_dataclass(tree, field_names, spec_vals)
    if isinstance(tree, Mapping):
        return {k: spec_like(v) for k, v in tree.items()}
    if is_namedtuple_instance(tree):
        tuple_cls = type(tree)
        return tuple_cls(*(spec_like(tree[i]) for i in range(len(tree))))
    if isinstance(tree, tuple):
        return tuple(spec_like(v) for v in tree)
    if isinstance(tree, list):
        return [spec_like(v) for v in tree]
    return _leaf_type_spec(tree)


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
    if is_mappable_dataclass_instance(tree):
        field_names = tree.__chex_mappable_fields__()
        mapped_vals = tuple(map_tree(fn, v) for v in tree.__chex_field_values__())
        return _reconstruct_mappable_dataclass(tree, field_names, mapped_vals)
    if isinstance(tree, Mapping):
        return {k: map_tree(fn, v) for k, v in tree.items()}
    if is_namedtuple_instance(tree):
        tuple_cls = type(tree)
        return tuple_cls(*(map_tree(fn, tree[i]) for i in range(len(tree))))
    if isinstance(tree, tuple):
        return tuple(map_tree(fn, v) for v in tree)
    if isinstance(tree, list):
        return [map_tree(fn, v) for v in tree]
    return fn(tree)


__all__ = [
    "_batch_elements",
    "_leaf_type_spec",
    "batch_and_pad",
    "map_tree",
    "spec_like",
]
