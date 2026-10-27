"""Testing utilities for verifying checkpoint determinism in zero-grain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union
from typing_extensions import TypeAlias
from zero_grain._tree import spec_like
from zero_grain._types import (
    _ZERO_JAX_NUMPY_ADAPTER,
    CheckpointStateDict,
    ElementTree,
    JaxArray,
    T_co,
    array_to_list,
    dtype_name,
    is_mappable_dataclass_instance,
    is_ndarray_instance,
)
from zero_grain.data_loader import DataLoader, DataLoaderIterator
from zero_grain.dataset import DatasetIterator, DatasetSelectionMap, IterDataset

ChexPyTree: TypeAlias = Union[
    JaxArray,
    tuple["ChexPyTree", ...],
    list["ChexPyTree"],
    dict[str, "ChexPyTree"],
]

_NUMERIC_DTYPES: frozenset[str] = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
    }
)


def _normalize_leaf_for_chex(leaf: ElementTree) -> JaxArray:
    """Convert any PyTree leaf into a ``zero_jax.numpy.ndarray`` for ``zero_chex`` shape, dtype, and value assertions.

    Args:
        leaf (ElementTree): Candidate PyTree leaf.

    Returns:
        JaxArray: ``zero_jax.numpy.ndarray`` encoding the leaf's shape, dtype, and value.
    """
    if isinstance(leaf, JaxArray):
        return leaf
    if is_ndarray_instance(leaf):
        dt = dtype_name(leaf.dtype)
        if dt in _NUMERIC_DTYPES:
            arr = _ZERO_JAX_NUMPY_ADAPTER.asarray(array_to_list(leaf), dtype=dt)
            return arr.reshape(leaf.shape) if arr.shape != leaf.shape else arr
        encoded = f"{dt}:{leaf.shape}:{array_to_list(leaf)!r}".encode("utf-8")
        return _ZERO_JAX_NUMPY_ADAPTER.asarray([int(b) for b in encoded], dtype="uint8")
    if isinstance(leaf, (bool, int, float)):
        return _ZERO_JAX_NUMPY_ADAPTER.asarray(leaf)
    encoded_str = f"{type(leaf)}:{leaf!r}".encode("utf-8")
    return _ZERO_JAX_NUMPY_ADAPTER.asarray([int(b) for b in encoded_str], dtype="uint8")


def _to_chex_pytree(tree: ElementTree) -> ChexPyTree:
    """Convert an ``ElementTree`` into a ``zero_chex``-compatible ``ChexPyTree`` of ``JaxArray`` leaves.

    Args:
        tree (ElementTree): Input PyTree structure.

    Returns:
        ChexPyTree: PyTree whose leaves are ``zero_jax.numpy.ndarray`` instances.
    """
    if is_mappable_dataclass_instance(tree):
        return {
            name: _to_chex_pytree(val)
            for name, val in zip(tree.__chex_mappable_fields__(), tree.__chex_field_values__())
        }
    if isinstance(tree, Mapping):
        return {str(k): _to_chex_pytree(v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_chex_pytree(v) for v in tree)
    if isinstance(tree, list):
        return [_to_chex_pytree(v) for v in tree]
    return _normalize_leaf_for_chex(tree)


def _assert_trees_equal_with_chex(left: ElementTree, right: ElementTree) -> None:
    """Assert that ``left`` and ``right`` have equal structures, shapes, dtypes, and values via ``zero_chex``.

    Args:
        left (ElementTree): First PyTree to compare.
        right (ElementTree): Second PyTree to compare.

    Raises:
        AssertionError: If structures, shapes, dtypes, or values differ.
    """
    import zero_optax as _zero_optax  # noqa: F401
    import zero_chex

    norm_left = _to_chex_pytree(left)
    norm_right = _to_chex_pytree(right)
    zero_chex.assert_trees_all_equal_shapes_and_dtypes(norm_left, norm_right)
    zero_chex.assert_trees_all_close(norm_left, norm_right)


def _trees_equal(left: ElementTree, right: ElementTree) -> bool:
    """Compare two PyTrees for structural, shape, dtype, and leaf value equality via ``zero_chex``.

    Args:
        left (ElementTree): First PyTree to compare.
        right (ElementTree): Second PyTree to compare.

    Returns:
        bool: True if both trees have identical structure, shapes, dtypes, and leaf values.
    """
    try:
        _assert_trees_equal_with_chex(left, right)
    except AssertionError:
        return False
    return True


def _state_to_tree(state: CheckpointStateDict) -> dict[str, ElementTree]:
    """Convert a ``CheckpointStateDict`` into an ``ElementTree``-compatible mapping.

    Args:
        state (CheckpointStateDict): Checkpoint state dictionary.

    Returns:
        dict[str, ElementTree]: Converted mapping with ``ElementTree`` values.
    """
    tree: dict[str, ElementTree] = {}
    if "version" in state:
        tree["version"] = state["version"]
    if "last_seen_indices" in state:
        tree["last_seen_indices"] = dict(state["last_seen_indices"])
    if "last_worker_index" in state:
        tree["last_worker_index"] = state["last_worker_index"]
    if "worker_count" in state:
        tree["worker_count"] = state["worker_count"]
    if "sampler" in state:
        tree["sampler"] = state["sampler"]
    if "data_source" in state:
        tree["data_source"] = state["data_source"]
    return tree


def _open_stateful_iterator(
    ds: DataLoader[T_co] | IterDataset[T_co],
) -> DataLoaderIterator[T_co] | DatasetIterator[T_co]:
    """Create a stateful iterator from ``ds``.

    Args:
        ds (DataLoader[T_co] | IterDataset[T_co]): Dataset or DataLoader to iterate.

    Returns:
        DataLoaderIterator[T_co] | DatasetIterator[T_co]: Fresh stateful iterator.
    """
    return ds.__iter__()


def assert_equal_output_after_checkpoint(
    ds: DataLoader[T_co] | IterDataset[T_co] | None = None,
) -> None:
    """Verify that restoring ``ds`` at every checkpoint step yields identical remaining elements.

    Uses ``zero_chex.assert_trees_all_equal_shapes_and_dtypes`` and
    ``zero_chex.assert_trees_all_close`` to validate state specs, restored states,
    and restored output elements.

    Args:
        ds (DataLoader[T_co] | IterDataset[T_co] | None):
            ``DataLoader`` or ``IterDataset`` to test (no-op if None).

    Raises:
        AssertionError: If restored states or values diverge at any checkpoint step.
    """
    if ds is None:
        return

    iterator = _open_stateful_iterator(ds)
    checkpoints: list[CheckpointStateDict] = []
    expected_values: list[ElementTree] = []
    state_spec: ElementTree = None
    current_state = iterator.get_state()

    for step, value in enumerate(iterator):
        current_spec = spec_like(_state_to_tree(current_state))
        if state_spec is None:
            state_spec = current_spec
        else:
            try:
                _assert_trees_equal_with_chex(state_spec, current_spec)
            except AssertionError as exc:
                raise AssertionError(f"State spec mismatch at step {step}: {state_spec} != {current_spec}") from exc
        checkpoints.append(current_state)
        expected_values.append(value)
        current_state = iterator.get_state()

    if not expected_values:
        raise AssertionError("Dataset did not produce any elements.")

    for step, state in enumerate(checkpoints):
        new_iterator = _open_stateful_iterator(ds)
        new_iterator.set_state(state)
        restored_state = new_iterator.get_state()
        try:
            _assert_trees_equal_with_chex(_state_to_tree(restored_state), _state_to_tree(state))
        except AssertionError as exc:
            raise AssertionError(f"Restored state mismatch at step {step}: {restored_state} != {state}") from exc
        new_values: list[ElementTree] = list(new_iterator)
        expected_slice: list[ElementTree] = expected_values[step:]
        try:
            _assert_trees_equal_with_chex(new_values, expected_slice)
        except AssertionError as exc:
            raise AssertionError(
                f"Restored values mismatch at step {step} for state {state}. "
                f"Expected: {expected_slice}, Actual: {new_values}"
            ) from exc


def fake_class(*args: ElementTree, **kwargs: ElementTree) -> type[DatasetSelectionMap]:
    """Return a lightweight placeholder class accepting arbitrary initialization arguments.

    Args:
        *args (ElementTree): Unused factory positional arguments.
        **kwargs (ElementTree): Unused factory keyword arguments.

    Returns:
        type[DatasetSelectionMap]: Generated placeholder class ``Fake``.
    """
    del args, kwargs

    class Fake(DatasetSelectionMap):
        """Placeholder class returned by ``fake_class``."""

        def __init__(self, *init_args: ElementTree, **init_kwargs: ElementTree) -> None:
            """Initialize ``Fake``.

            Args:
                *init_args (ElementTree): Positional arguments.
                **init_kwargs (ElementTree): Keyword arguments.
            """
            super().__init__(*init_args, **init_kwargs)
            self.args = init_args
            self.kwargs = init_kwargs

    return Fake


__all__ = [
    "_assert_trees_equal_with_chex",
    "_normalize_leaf_for_chex",
    "_open_stateful_iterator",
    "_state_to_tree",
    "_to_chex_pytree",
    "_trees_equal",
    "assert_equal_output_after_checkpoint",
    "fake_class",
]
