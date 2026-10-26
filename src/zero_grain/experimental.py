"""Experimental dataset classes for ``zero_grain.experimental`` and ``zero_grain.python.experimental``."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from zero_grain._rng import index_shuffle
from zero_grain._types import (
    ElementTree,
    T_co,
    array_to_list,
    get_numpy_module,
    is_ndarray_instance,
)
from zero_grain.dataset import IterDataset, MapDataset
from zero_grain.transforms import FlatMapTransform


class FlatMapIterDataset(IterDataset[ElementTree]):
    """Experimental ``IterDataset`` applying a flat-map transformation to a parent dataset."""

    def __init__(
        self,
        parent: IterDataset[ElementTree] | MapDataset[ElementTree],
        transform: (FlatMapTransform[ElementTree, ElementTree] | Callable[[ElementTree], Iterable[ElementTree]]),
    ) -> None:
        """Initialize ``FlatMapIterDataset``.

        Args:
            parent (IterDataset[ElementTree] | MapDataset[ElementTree]): Parent ``IterDataset`` or ``MapDataset``.
            transform (FlatMapTransform[ElementTree, ElementTree] | Callable[[ElementTree], Iterable[ElementTree]]):
                ``FlatMapTransform`` instance or callable expanding each element into an iterable.
        """
        self.parent = parent
        self.transform = transform
        expanded: list[ElementTree] = []
        max_fan_out: int | None = transform.max_fan_out if isinstance(transform, FlatMapTransform) else None
        for elem in parent:
            sub_iter = transform.flat_map(elem) if isinstance(transform, FlatMapTransform) else transform(elem)
            sub_items = list(sub_iter)
            if max_fan_out is not None and len(sub_items) > max_fan_out:
                raise ValueError(
                    f"FlatMapIterDataset element fan-out {len(sub_items)} exceeds max_fan_out {max_fan_out}"
                )
            expanded.extend(sub_items)
        super().__init__(MapDataset(expanded))


class ThreadPrefetchIterDataset(IterDataset[T_co]):
    """Experimental ``IterDataset`` buffering elements with configurable thread prefetching."""

    def __init__(
        self,
        parent: IterDataset[T_co] | MapDataset[T_co],
        prefetch_buffer_size: int,
    ) -> None:
        """Initialize ``ThreadPrefetchIterDataset``.

        Args:
            parent (IterDataset[T_co] | MapDataset[T_co]): Parent ``IterDataset`` or ``MapDataset``.
            prefetch_buffer_size (int): Prefetch buffer capacity (must be ``>= 0``).

        Raises:
            ValueError: If ``prefetch_buffer_size < 0``.
        """
        if prefetch_buffer_size < 0:
            raise ValueError(f"prefetch_buffer_size must be non-negative, got {prefetch_buffer_size}")
        self.parent = parent
        self.prefetch_buffer_size = prefetch_buffer_size
        base_map: MapDataset[T_co] = parent if isinstance(parent, MapDataset) else parent.dataset
        super().__init__(base_map)


class InterleaveIterDataset(IterDataset[ElementTree]):
    """Experimental ``IterDataset`` interleaving elements across a sequence of datasets."""

    def __init__(
        self,
        datasets: Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]],
        *,
        cycle_length: int,
        num_make_iter_threads: int = 1,
        iter_buffer_size: int = 1,
    ) -> None:
        """Initialize ``InterleaveIterDataset``.

        Args:
            datasets (Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]]):
                Sequence of child datasets to interleave.
            cycle_length (int): Number of datasets concurrently interleaved in the active window (must be ``>= 1``).
            num_make_iter_threads (int): Number of iterator construction threads (must be ``>= 1``).
            iter_buffer_size (int): Per-iterator prefetch buffer size (must be ``>= 1``).

        Raises:
            ValueError: If ``cycle_length < 1``, ``num_make_iter_threads < 1``, or ``iter_buffer_size < 1``.
        """
        if cycle_length < 1:
            raise ValueError(f"cycle_length must be >= 1, got {cycle_length}")
        if num_make_iter_threads < 1:
            raise ValueError(f"num_make_iter_threads must be >= 1, got {num_make_iter_threads}")
        if iter_buffer_size < 1:
            raise ValueError(f"iter_buffer_size must be >= 1, got {iter_buffer_size}")
        self.datasets = tuple(datasets)
        self.cycle_length = cycle_length
        self.num_make_iter_threads = num_make_iter_threads
        self.iter_buffer_size = iter_buffer_size
        interleaved = self._interleave_elements(self.datasets, cycle_length)
        super().__init__(MapDataset(interleaved))

    @staticmethod
    def _interleave_elements(
        datasets: Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]],
        cycle_length: int,
    ) -> list[ElementTree]:
        """Materialize the deterministic cycle-interleaved sequence across ``datasets``.

        Args:
            datasets (Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]]): Child datasets to interleave.
            cycle_length (int): Active cycle window length.

        Returns:
            list[ElementTree]: Interleaved output elements.
        """
        if not datasets:
            return []
        streams: list[list[ElementTree]] = [list(ds) for ds in datasets]
        positions: list[int] = [0 for _ in streams]
        window_size = min(cycle_length, len(streams))
        active_slots: list[int | None] = [i for i in range(window_size)]
        next_dataset_idx = window_size
        output: list[ElementTree] = []

        while any(slot is not None for slot in active_slots):
            for slot_pos in range(window_size):
                ds_idx = active_slots[slot_pos]
                if ds_idx is None:
                    continue
                while ds_idx is not None and positions[ds_idx] >= len(streams[ds_idx]):
                    if next_dataset_idx < len(streams):
                        ds_idx = next_dataset_idx
                        active_slots[slot_pos] = ds_idx
                        next_dataset_idx += 1
                    else:
                        active_slots[slot_pos] = None
                        ds_idx = None
                if ds_idx is not None:
                    elem = streams[ds_idx][positions[ds_idx]]
                    positions[ds_idx] += 1
                    output.append(elem)
                    if positions[ds_idx] >= len(streams[ds_idx]):
                        if next_dataset_idx < len(streams):
                            active_slots[slot_pos] = next_dataset_idx
                            next_dataset_idx += 1
                        else:
                            active_slots[slot_pos] = None
        return output


def _leaf_sequence_length(leaf: ElementTree) -> int:
    """Determine the sequence length of a single leaf value.

    Args:
        leaf (ElementTree): Leaf array, sequence, or scalar.

    Returns:
        int: Leading sequence dimension length (1 for scalars).
    """
    if is_ndarray_instance(leaf):
        return int(leaf.shape[0]) if len(leaf.shape) > 0 else 1
    if isinstance(leaf, (list, tuple)):
        return len(leaf)
    return 1


def _leaf_fits_capacity(elem_leaf: ElementTree, used_leaf: ElementTree, cap_leaf: ElementTree) -> bool:
    """Check whether ``elem_leaf`` fits within the remaining capacity ``cap_leaf - used_leaf``.

    Args:
        elem_leaf (ElementTree): Candidate element leaf or sub-tree.
        used_leaf (ElementTree): Currently used length tree.
        cap_leaf (ElementTree): Capacity length tree.

    Returns:
        bool: True if ``elem_leaf`` fits in all leaves.
    """
    if isinstance(cap_leaf, Mapping) and isinstance(elem_leaf, Mapping) and isinstance(used_leaf, Mapping):
        return all(
            _leaf_fits_capacity(elem_leaf[k], used_leaf[k], cap_leaf[k])
            for k in cap_leaf
            if k in elem_leaf and k in used_leaf
        )
    cap_int = int(cap_leaf) if isinstance(cap_leaf, (int, float)) else 0
    used_int = int(used_leaf) if isinstance(used_leaf, (int, float)) else 0
    return used_int + _leaf_sequence_length(elem_leaf) <= cap_int


def _zero_used_tree(length_struct: ElementTree) -> ElementTree:
    """Create a zero-initialized length tracking structure matching ``length_struct``.

    Args:
        length_struct (ElementTree): Capacity specification tree.

    Returns:
        ElementTree: Zero-initialized length structure.
    """
    if isinstance(length_struct, Mapping):
        return {k: _zero_used_tree(v) for k, v in length_struct.items()}
    return 0


def _add_used_tree(used_tree: ElementTree, elem_tree: ElementTree) -> ElementTree:
    """Add the sequence lengths of ``elem_tree`` into ``used_tree``.

    Args:
        used_tree (ElementTree): Current accumulated lengths.
        elem_tree (ElementTree): Newly packed element.

    Returns:
        ElementTree: Updated accumulated lengths.
    """
    if isinstance(used_tree, Mapping) and isinstance(elem_tree, Mapping):
        return {k: _add_used_tree(used_tree[k], elem_tree[k]) for k in used_tree if k in elem_tree}
    base_used = int(used_tree) if isinstance(used_tree, (int, float)) else 0
    return base_used + _leaf_sequence_length(elem_tree)


def _pack_leaf_sequences(
    leaves: Sequence[ElementTree],
    capacity: ElementTree,
    pad_val: ElementTree,
) -> ElementTree:
    """Concatenate and pad a list of packed ``leaves`` up to ``capacity``.

    Args:
        leaves (Sequence[ElementTree]): Sequence of leaf elements in a single packing bin.
        capacity (ElementTree): Target packed sequence length (or mapping of lengths).
        pad_val (ElementTree): Padding value (or mapping of padding values).

    Returns:
        ElementTree: Packed and padded sequence or array.
    """
    if isinstance(capacity, Mapping):
        return {
            k: _pack_leaf_sequences(
                [item[k] for item in leaves if isinstance(item, Mapping) and k in item],
                capacity[k],
                pad_val[k] if isinstance(pad_val, Mapping) and k in pad_val else 0,
            )
            for k in capacity
        }
    target_len = int(capacity) if isinstance(capacity, (int, float)) else 0
    scalar_pad: int | float = pad_val if isinstance(pad_val, (int, float)) and not isinstance(pad_val, bool) else 0
    flat_values: list[ElementTree] = []
    any_ndarray = False
    for leaf in leaves:
        if is_ndarray_instance(leaf):
            any_ndarray = True
            converted = array_to_list(leaf)
            if isinstance(converted, list):
                flat_values.extend(converted)
            else:
                flat_values.append(converted)
        elif isinstance(leaf, (list, tuple)):
            flat_values.extend(leaf)
        else:
            flat_values.append(leaf)
    while len(flat_values) < target_len:
        flat_values.append(scalar_pad)
    truncated = flat_values[:target_len]
    np_mod = get_numpy_module()
    if any_ndarray and np_mod is not None:
        return np_mod.asarray(truncated)
    return truncated


class FirstFitPackIterDataset(IterDataset[ElementTree]):
    """Experimental ``IterDataset`` packing variable-length sequences using the first-fit heuristic."""

    def __init__(
        self,
        parent: IterDataset[ElementTree] | MapDataset[ElementTree],
        *,
        length_struct: ElementTree,
        num_packing_bins: int,
        shuffle_bins: bool = True,
        padding_struct: ElementTree = None,
        max_sequences_per_bin: int | None = None,
    ) -> None:
        """Initialize ``FirstFitPackIterDataset``.

        Args:
            parent (IterDataset[ElementTree] | MapDataset[ElementTree]):
                Parent dataset yielding variable-length elements.
            length_struct (ElementTree): Maximum packed sequence length per feature leaf.
            num_packing_bins (int): Number of concurrent packing bins (must be ``>= 1``).
            shuffle_bins (bool): Whether to deterministically shuffle completed bins.
            padding_struct (ElementTree): Optional per-feature padding value structure.
            max_sequences_per_bin (int | None): Optional maximum number of sequences packed into a single bin.

        Raises:
            ValueError: If ``num_packing_bins < 1`` or ``max_sequences_per_bin < 1``.
        """
        if num_packing_bins < 1:
            raise ValueError(f"num_packing_bins must be >= 1, got {num_packing_bins}")
        if max_sequences_per_bin is not None and max_sequences_per_bin < 1:
            raise ValueError(f"max_sequences_per_bin must be >= 1, got {max_sequences_per_bin}")
        self.parent = parent
        self.length_struct = length_struct
        self.num_packing_bins = num_packing_bins
        self.shuffle_bins = shuffle_bins
        self.padding_struct = padding_struct
        self.max_sequences_per_bin = max_sequences_per_bin
        packed_bins = self._pack_elements(
            parent=parent,
            length_struct=length_struct,
            num_packing_bins=num_packing_bins,
            shuffle_bins=shuffle_bins,
            padding_struct=padding_struct,
            max_sequences_per_bin=max_sequences_per_bin,
        )
        super().__init__(MapDataset(packed_bins))

    @staticmethod
    def _pack_elements(
        parent: IterDataset[ElementTree] | MapDataset[ElementTree],
        length_struct: ElementTree,
        num_packing_bins: int,
        shuffle_bins: bool,
        padding_struct: ElementTree,
        max_sequences_per_bin: int | None,
    ) -> list[ElementTree]:
        """Pack elements from ``parent`` into bins using first-fit allocation.

        Args:
            parent (IterDataset[ElementTree] | MapDataset[ElementTree]): Parent dataset stream.
            length_struct (ElementTree): Target capacity per leaf.
            num_packing_bins (int): Number of concurrent bins.
            shuffle_bins (bool): Whether to shuffle emitted bins.
            padding_struct (ElementTree): Padding values per leaf.
            max_sequences_per_bin (int | None): Optional cap on sequences per bin.

        Returns:
            list[ElementTree]: Packed and padded bin elements.
        """
        bins: list[list[ElementTree]] = [[] for _ in range(num_packing_bins)]
        bin_used: list[ElementTree] = [_zero_used_tree(length_struct) for _ in range(num_packing_bins)]
        emitted: list[ElementTree] = []

        for elem in parent:
            placed = False
            for b_idx in range(num_packing_bins):
                if max_sequences_per_bin is not None and len(bins[b_idx]) >= max_sequences_per_bin:
                    continue
                if _leaf_fits_capacity(elem, bin_used[b_idx], length_struct):
                    bins[b_idx].append(elem)
                    bin_used[b_idx] = _add_used_tree(bin_used[b_idx], elem)
                    placed = True
                    break
            if not placed:
                if bins[0]:
                    emitted.append(_pack_leaf_sequences(bins[0], length_struct, padding_struct))
                bins.pop(0)
                bin_used.pop(0)
                bins.append([elem])
                bin_used.append(_add_used_tree(_zero_used_tree(length_struct), elem))

        for b_items in bins:
            if b_items:
                emitted.append(_pack_leaf_sequences(b_items, length_struct, padding_struct))

        if shuffle_bins and len(emitted) > 1:
            max_idx = len(emitted) - 1
            return [emitted[index_shuffle(i, max_index=max_idx, seed=0)] for i in range(len(emitted))]
        return emitted


__all__ = [
    "FirstFitPackIterDataset",
    "FlatMapIterDataset",
    "InterleaveIterDataset",
    "ThreadPrefetchIterDataset",
]
