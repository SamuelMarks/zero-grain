"""Testing utilities for verifying checkpoint determinism in zero-grain."""

from __future__ import annotations

from zero_grain._tree import spec_like
from zero_grain._types import (
    CheckpointStateDict,
    ElementTree,
    T_co,
    array_to_list,
    is_ndarray_instance,
)
from zero_grain.data_loader import DataLoader, DataLoaderIterator
from zero_grain.dataset import DatasetIterator, DatasetSelectionMap, IterDataset


def _to_chex_pytree(values: list[ElementTree]) -> list[str]:
    """Convert a sequence of ``ElementTree`` outputs into a ``zero_chex``-compatible list.

    Args:
        values (list[ElementTree]): Sequence of dataset output elements.

    Returns:
        list[str]: Normalized list of element representations for ``zero_chex`` tree assertions.
    """
    return [repr(v) for v in values]


def _trees_equal(left: ElementTree, right: ElementTree) -> bool:
    """Compare two PyTrees for structural and leaf value equality.

    Args:
        left (ElementTree): First PyTree to compare.
        right (ElementTree): Second PyTree to compare.

    Returns:
        bool: True if both trees have identical structure and leaf values.
    """
    if is_ndarray_instance(left) and is_ndarray_instance(right):
        return left.shape == right.shape and array_to_list(left) == array_to_list(right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_trees_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        return all(_trees_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_trees_equal(left[k], right[k]) for k in left)
    return bool(left == right)


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
        elif not _trees_equal(state_spec, current_spec):
            raise AssertionError(f"State spec mismatch at step {step}: {state_spec} != {current_spec}")
        checkpoints.append(current_state)
        expected_values.append(value)
        current_state = iterator.get_state()

    if not expected_values:
        raise AssertionError("Dataset did not produce any elements.")

    for step, state in enumerate(checkpoints):
        new_iterator = _open_stateful_iterator(ds)
        new_iterator.set_state(state)
        restored_state = new_iterator.get_state()
        if not _trees_equal(_state_to_tree(restored_state), _state_to_tree(state)):
            raise AssertionError(f"Restored state mismatch at step {step}: {restored_state} != {state}")
        new_values: list[ElementTree] = list(new_iterator)
        expected_slice: list[ElementTree] = expected_values[step:]
        if not _trees_equal(new_values, expected_slice):
            raise AssertionError(
                f"Restored values mismatch at step {step} for state {state}. "
                f"Expected: {expected_slice}, Actual: {new_values}"
            )
        import zero_chex

        norm_new = _to_chex_pytree(new_values)
        norm_expected = _to_chex_pytree(expected_slice)
        zero_chex.assert_trees_all_equal_structs(norm_new, norm_expected)
        zero_chex.assert_trees_all_equal(norm_new, norm_expected)


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
    "_open_stateful_iterator",
    "_state_to_tree",
    "_trees_equal",
    "assert_equal_output_after_checkpoint",
    "fake_class",
]
