"""PyTree traversal, structure validation, batching, and padding utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from zero_grain._types import (
    ArrayConvertibleProtocol,
    ElementTree,
    ScalarElement,
    get_numpy_module,
    is_namedtuple_instance,
    is_ndarray_instance,
)


def _convert_leaf_batch(batch: Sequence[ElementTree]) -> ElementTree:
    """Stack a homogeneous sequence of leaf values into a batched array or list.

    Args:
        batch (Sequence[ElementTree]): Sequence of leaf elements to stack.

    Returns:
        ElementTree: Stacked NumPy array if NumPy is loaded and compatible, else a list.
    """
    np_mod = get_numpy_module()
    if np_mod is not None:
        try:
            return np_mod.array(batch)
        except Exception:
            return list(batch)
    return list(batch)


def _batch_elements(batch: Sequence[ElementTree]) -> ElementTree:
    """Collate a sequence of structured elements into a single batched PyTree.

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
    if isinstance(element, Mapping):
        return {k: _zero_like_element(v, pad_value) for k, v in element.items()}
    if is_namedtuple_instance(element):
        tuple_cls = type(element)
        return tuple_cls(*(_zero_like_element(element[i], pad_value) for i in range(len(element))))
    if isinstance(element, tuple):
        return tuple(_zero_like_element(v, pad_value) for v in element)
    if isinstance(element, list):
        return [_zero_like_element(v, pad_value) for v in element]

    np_mod = get_numpy_module()
    if np_mod is not None:
        if is_ndarray_instance(element):
            return np_mod.zeros_like(element)
        if isinstance(element, ArrayConvertibleProtocol):
            arr = element.__array__()
            return np_mod.zeros_like(arr)
    if isinstance(element, float):
        return 0.0
    if isinstance(element, int):
        return 0
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
    if not elements:
        return list(elements)
    first = elements[0]
    pad_len = batch_size - len(elements)
    padded_elements: list[ElementTree] = list(elements)
    if pad_len > 0:
        pad_elem = _zero_like_element(first, pad_value=pad_value)
        padded_elements.extend([pad_elem for _ in range(pad_len)])
    return _batch_elements(padded_elements)


def spec_like(tree: ElementTree) -> ElementTree:
    """Compute a structural type/shape specification matching ``tree``.

    Args:
        tree (ElementTree): Input PyTree structure.

    Returns:
        ElementTree: Specification tree with leaf types and shapes.
    """
    if isinstance(tree, Mapping):
        return {k: spec_like(v) for k, v in tree.items()}
    if is_namedtuple_instance(tree):
        tuple_cls = type(tree)
        return tuple_cls(*(spec_like(tree[i]) for i in range(len(tree))))
    if isinstance(tree, tuple):
        return tuple(spec_like(v) for v in tree)
    if isinstance(tree, list):
        return [spec_like(v) for v in tree]
    if is_ndarray_instance(tree):
        return f"{tree.dtype.name}{list(tree.shape)}"
    return type(tree).__name__


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
    "batch_and_pad",
    "map_tree",
    "spec_like",
]
