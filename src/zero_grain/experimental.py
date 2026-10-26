"""Experimental dataset classes for ``zero_grain.experimental`` and ``zero_grain.python.experimental``."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from zero_grain._rng import index_shuffle
from zero_grain._tree import _reconstruct_mappable_dataclass
from zero_grain._types import (
    DTypeSpec,
    ElementTree,
    T_co,
    array_to_list,
    get_numpy_module,
    is_mappable_dataclass_instance,
    is_namedtuple_instance,
    is_ndarray_instance,
)
from zero_grain.data_sources import RandomAccessDataSource
from zero_grain.dataset import IterDataset, MapDataset
from zero_grain.options import ReadOptions
from zero_grain.transforms import FlatMapTransform


class _LazyStreamDataSource(RandomAccessDataSource[ElementTree]):
    """Demand-driven indexable data source materializing an iterator lazily as indices are requested."""

    def __init__(
        self,
        stream_factory: Callable[[], Iterator[ElementTree]],
        repr_label: str,
    ) -> None:
        """Initialize ``_LazyStreamDataSource``.

        Args:
            stream_factory (Callable[[], Iterator[ElementTree]]): Factory returning a fresh element iterator.
            repr_label (str): Canonical string representation for checkpointing.
        """
        super().__init__()
        self._stream_factory = stream_factory
        self._repr_label = repr_label
        self._cache: list[ElementTree] = []
        self._exhausted = False
        self._active_iter: Iterator[ElementTree] = stream_factory()

    def _pull_until(self, target_index: int | None = None) -> None:
        """Advance the underlying iterator until ``target_index`` is cached or the stream is exhausted.

        Args:
            target_index (int | None): Optional target index to reach (exhausts stream if None).
        """
        if self._exhausted:
            return
        while target_index is None or len(self._cache) <= target_index:
            try:
                self._cache.append(next(self._active_iter))
            except StopIteration:
                self._exhausted = True
                return

    def __len__(self) -> int:
        """Return total number of elements in the stream.

        Returns:
            int: Total stream length.
        """
        self._pull_until(None)
        return len(self._cache)

    def __getitem__(self, record_key: int) -> ElementTree:
        """Return the element at ``record_key``, advancing the stream only as far as needed.

        Args:
            record_key (int): Non-negative element index.

        Returns:
            ElementTree: Element at ``record_key``.

        Raises:
            IndexError: If ``record_key`` is out of bounds.
        """
        if record_key < 0:
            total = len(self)
            record_key += total
        else:
            self._pull_until(record_key)
        if record_key < 0 or record_key >= len(self._cache):
            raise IndexError(f"_LazyStreamDataSource index {record_key} out of range")
        return self._cache[record_key]

    def __repr__(self) -> str:
        """Return the string representation of this lazy stream source.

        Returns:
            str: Formatted string representation.
        """
        return self._repr_label


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
        self._validated_first_batch: list[ElementTree] | None = None
        self._validate_initial_fan_out()
        lazy_source = _LazyStreamDataSource(
            stream_factory=self._iter_flattened,
            repr_label=f"FlatMapIterDataset(parent={parent!r})",
        )
        super().__init__(MapDataset(lazy_source))

    def _expand_single(self, elem: ElementTree) -> list[ElementTree]:
        """Expand a single parent ``elem`` and validate ``max_fan_out``.

        Args:
            elem (ElementTree): Parent element.

        Returns:
            list[ElementTree]: Expanded sub-elements.

        Raises:
            ValueError: If fan-out exceeds ``transform.max_fan_out``.
        """
        max_fan_out: int | None = self.transform.max_fan_out if isinstance(self.transform, FlatMapTransform) else None
        sub_iter = (
            self.transform.flat_map(elem) if isinstance(self.transform, FlatMapTransform) else self.transform(elem)
        )
        sub_items = list(sub_iter)
        if max_fan_out is not None and len(sub_items) > max_fan_out:
            raise ValueError(f"FlatMapIterDataset element fan-out {len(sub_items)} exceeds max_fan_out {max_fan_out}")
        return sub_items

    def _validate_initial_fan_out(self) -> None:
        """Validate ``max_fan_out`` on the first parent element if ``transform`` defines ``max_fan_out``."""
        if not isinstance(self.transform, FlatMapTransform):
            return
        parent_it = iter(self.parent)
        try:
            first_elem = next(parent_it)
        except StopIteration:
            return
        self._validated_first_batch = self._expand_single(first_elem)

    def _iter_flattened(self) -> Iterator[ElementTree]:
        """Lazily yield flattened elements from ``self.parent``.

        Yields:
            ElementTree: Expanded sub-elements.
        """
        for idx, elem in enumerate(self.parent):
            if idx == 0 and self._validated_first_batch is not None:
                for sub_item in self._validated_first_batch:
                    yield sub_item
                continue
            for sub_item in self._expand_single(elem):
                yield sub_item


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
        parent_ro = parent.read_options if isinstance(parent, IterDataset) else None
        num_threads = parent_ro.num_threads if parent_ro is not None and prefetch_buffer_size > 0 else 0
        read_options = ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size)
        super().__init__(parent, read_options=read_options)


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
        lazy_source = _LazyStreamDataSource(
            stream_factory=lambda: self._iter_interleaved(self.datasets, self.cycle_length),
            repr_label=f"InterleaveIterDataset(cycle_length={cycle_length})",
        )
        super().__init__(MapDataset(lazy_source))

    @staticmethod
    def _iter_interleaved(
        datasets: Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]],
        cycle_length: int,
    ) -> Iterator[ElementTree]:
        """Lazily yield cycle-interleaved elements across ``datasets`` using active iterator slots.

        Args:
            datasets (Sequence[IterDataset[ElementTree] | MapDataset[ElementTree]]): Child datasets to interleave.
            cycle_length (int): Active cycle window length.

        Yields:
            ElementTree: Interleaved output elements.
        """
        if not datasets:
            return
        window_size = min(cycle_length, len(datasets))
        active_iters: list[Iterator[ElementTree] | None] = [iter(datasets[i]) for i in range(window_size)]
        next_dataset_idx = window_size

        while any(it is not None for it in active_iters):
            for slot_pos in range(window_size):
                cur_it = active_iters[slot_pos]
                while cur_it is not None:
                    try:
                        elem = next(cur_it)
                        yield elem
                        break
                    except StopIteration:
                        if next_dataset_idx < len(datasets):
                            cur_it = iter(datasets[next_dataset_idx])
                            active_iters[slot_pos] = cur_it
                            next_dataset_idx += 1
                        else:
                            active_iters[slot_pos] = None
                            cur_it = None


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
    if (
        is_mappable_dataclass_instance(cap_leaf)
        and is_mappable_dataclass_instance(elem_leaf)
        and is_mappable_dataclass_instance(used_leaf)
    ):
        return all(
            _leaf_fits_capacity(e_v, u_v, c_v)
            for e_v, u_v, c_v in zip(
                elem_leaf.__chex_field_values__(),
                used_leaf.__chex_field_values__(),
                cap_leaf.__chex_field_values__(),
            )
        )
    if isinstance(cap_leaf, Mapping) and isinstance(elem_leaf, Mapping) and isinstance(used_leaf, Mapping):
        return all(
            _leaf_fits_capacity(elem_leaf[k], used_leaf[k], cap_leaf[k])
            for k in cap_leaf
            if k in elem_leaf and k in used_leaf
        )
    if isinstance(cap_leaf, tuple) and isinstance(elem_leaf, tuple) and isinstance(used_leaf, tuple):
        return all(_leaf_fits_capacity(e_v, u_v, c_v) for e_v, u_v, c_v in zip(elem_leaf, used_leaf, cap_leaf))
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
    if is_mappable_dataclass_instance(length_struct):
        field_names = length_struct.__chex_mappable_fields__()
        zero_vals = tuple(_zero_used_tree(v) for v in length_struct.__chex_field_values__())
        return _reconstruct_mappable_dataclass(length_struct, field_names, zero_vals)
    if isinstance(length_struct, Mapping):
        return {k: _zero_used_tree(v) for k, v in length_struct.items()}
    if is_namedtuple_instance(length_struct):
        tuple_cls = type(length_struct)
        return tuple_cls(*(_zero_used_tree(length_struct[i]) for i in range(len(length_struct))))
    if isinstance(length_struct, tuple):
        return tuple(_zero_used_tree(v) for v in length_struct)
    return 0


def _add_used_tree(
    used_tree: ElementTree,
    elem_tree: ElementTree,
    cap_tree: ElementTree = None,
) -> ElementTree:
    """Add the sequence lengths of ``elem_tree`` into ``used_tree``, optionally clamped to ``cap_tree``.

    Args:
        used_tree (ElementTree): Current accumulated lengths.
        elem_tree (ElementTree): Newly packed element.
        cap_tree (ElementTree): Optional capacity tree to clamp accumulated lengths.

    Returns:
        ElementTree: Updated accumulated lengths.
    """
    if is_mappable_dataclass_instance(used_tree) and is_mappable_dataclass_instance(elem_tree):
        field_names = used_tree.__chex_mappable_fields__()
        cap_vals = (
            cap_tree.__chex_field_values__()
            if is_mappable_dataclass_instance(cap_tree)
            else tuple(None for _ in field_names)
        )
        added_vals = tuple(
            _add_used_tree(u_v, e_v, c_v)
            for u_v, e_v, c_v in zip(
                used_tree.__chex_field_values__(),
                elem_tree.__chex_field_values__(),
                cap_vals,
            )
        )
        return _reconstruct_mappable_dataclass(used_tree, field_names, added_vals)
    if isinstance(used_tree, Mapping) and isinstance(elem_tree, Mapping):
        return {
            k: _add_used_tree(
                used_tree[k],
                elem_tree[k],
                cap_tree[k] if isinstance(cap_tree, Mapping) and k in cap_tree else None,
            )
            for k in used_tree
            if k in elem_tree
        }
    if is_namedtuple_instance(used_tree) and isinstance(elem_tree, tuple):
        tuple_cls = type(used_tree)
        return tuple_cls(
            *(
                _add_used_tree(
                    used_tree[i],
                    elem_tree[i],
                    cap_tree[i] if isinstance(cap_tree, tuple) and i < len(cap_tree) else None,
                )
                for i in range(len(used_tree))
            )
        )
    if isinstance(used_tree, tuple) and isinstance(elem_tree, tuple):
        return tuple(
            _add_used_tree(
                u_v,
                e_v,
                cap_tree[i] if isinstance(cap_tree, tuple) and i < len(cap_tree) else None,
            )
            for i, (u_v, e_v) in enumerate(zip(used_tree, elem_tree))
        )
    base_used = int(used_tree) if isinstance(used_tree, (int, float)) else 0
    total = base_used + _leaf_sequence_length(elem_tree)
    return min(total, int(cap_tree)) if isinstance(cap_tree, (int, float)) and not isinstance(cap_tree, bool) else total


def _pack_leaf_sequences(
    leaves: Sequence[ElementTree],
    capacity: ElementTree,
    pad_val: ElementTree,
) -> ElementTree:
    """Concatenate and pad a list of packed ``leaves`` up to ``capacity``.

    Args:
        leaves (Sequence[ElementTree]): Sequence of leaf elements in a single packing bin.
        capacity (ElementTree): Target packed sequence length (or PyTree of lengths).
        pad_val (ElementTree): Padding value (or PyTree of padding values).

    Returns:
        ElementTree: Packed and padded sequence, array, or PyTree.
    """
    if is_mappable_dataclass_instance(capacity):
        field_names = capacity.__chex_mappable_fields__()
        cap_vals = capacity.__chex_field_values__()
        pad_vals = (
            pad_val.__chex_field_values__()
            if is_mappable_dataclass_instance(pad_val)
            else tuple(0 for _ in field_names)
        )
        packed_fields = tuple(
            _pack_leaf_sequences(
                [item.__chex_field_values__()[idx] for item in leaves if is_mappable_dataclass_instance(item)],
                cap_vals[idx],
                pad_vals[idx],
            )
            for idx in range(len(field_names))
        )
        return _reconstruct_mappable_dataclass(capacity, field_names, packed_fields)
    if isinstance(capacity, Mapping):
        return {
            k: _pack_leaf_sequences(
                [item[k] for item in leaves if isinstance(item, Mapping) and k in item],
                capacity[k],
                pad_val[k] if isinstance(pad_val, Mapping) and k in pad_val else 0,
            )
            for k in capacity
        }
    if is_namedtuple_instance(capacity):
        tuple_cls = type(capacity)
        return tuple_cls(
            *(
                _pack_leaf_sequences(
                    [item[idx] for item in leaves if isinstance(item, tuple) and idx < len(item)],
                    capacity[idx],
                    pad_val[idx] if isinstance(pad_val, tuple) and idx < len(pad_val) else 0,
                )
                for idx in range(len(capacity))
            )
        )
    if isinstance(capacity, tuple):
        return tuple(
            _pack_leaf_sequences(
                [item[idx] for item in leaves if isinstance(item, tuple) and idx < len(item)],
                capacity[idx],
                pad_val[idx] if isinstance(pad_val, tuple) and idx < len(pad_val) else 0,
            )
            for idx in range(len(capacity))
        )
    target_len = int(capacity) if isinstance(capacity, (int, float)) else 0
    scalar_pad: int | float = pad_val if isinstance(pad_val, (int, float)) and not isinstance(pad_val, bool) else 0
    flat_values: list[ElementTree] = []
    any_ndarray = False
    leaf_dtype: DTypeSpec = None
    for leaf in leaves:
        if is_ndarray_instance(leaf):
            if not any_ndarray:
                leaf_dtype = leaf.dtype
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
        return np_mod.asarray(truncated, dtype=leaf_dtype)
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
        parent_map = parent if isinstance(parent, MapDataset) else parent._materialize_as_map_dataset()
        bin_seed = parent_map._seed if parent_map._seed is not None else 0
        lazy_source = _LazyStreamDataSource(
            stream_factory=lambda: self._pack_elements(
                parent=self.parent,
                length_struct=self.length_struct,
                num_packing_bins=self.num_packing_bins,
                shuffle_bins=self.shuffle_bins,
                padding_struct=self.padding_struct,
                max_sequences_per_bin=self.max_sequences_per_bin,
                seed=bin_seed,
            ),
            repr_label=f"FirstFitPackIterDataset(num_packing_bins={num_packing_bins})",
        )
        super().__init__(MapDataset(lazy_source))

    @staticmethod
    def _pack_elements(
        parent: IterDataset[ElementTree] | MapDataset[ElementTree],
        length_struct: ElementTree,
        num_packing_bins: int,
        shuffle_bins: bool,
        padding_struct: ElementTree,
        max_sequences_per_bin: int | None,
        seed: int = 0,
    ) -> Iterator[ElementTree]:
        """Lazily pack elements from ``parent`` into bins using first-fit allocation.

        Args:
            parent (IterDataset[ElementTree] | MapDataset[ElementTree]): Parent dataset stream.
            length_struct (ElementTree): Target capacity per leaf.
            num_packing_bins (int): Number of concurrent bins.
            shuffle_bins (bool): Whether to shuffle emitted bins.
            padding_struct (ElementTree): Padding values per leaf.
            max_sequences_per_bin (int | None): Optional cap on sequences per bin.
            seed (int): Deterministic shuffle seed when ``shuffle_bins`` is True.

        Yields:
            ElementTree: Packed and padded bin elements.
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
                    bin_used[b_idx] = _add_used_tree(bin_used[b_idx], elem, length_struct)
                    placed = True
                    break
            if not placed:
                if bins[0]:
                    packed_bin = _pack_leaf_sequences(bins[0], length_struct, padding_struct)
                    if shuffle_bins:
                        emitted.append(packed_bin)
                    else:
                        yield packed_bin
                bins.pop(0)
                bin_used.pop(0)
                bins.append([elem])
                bin_used.append(_add_used_tree(_zero_used_tree(length_struct), elem, length_struct))

        for b_items in bins:
            if b_items:
                packed_bin = _pack_leaf_sequences(b_items, length_struct, padding_struct)
                if shuffle_bins:
                    emitted.append(packed_bin)
                else:
                    yield packed_bin

        if shuffle_bins:
            if len(emitted) > 1:
                max_idx = len(emitted) - 1
                for i in range(len(emitted)):
                    yield emitted[index_shuffle(i, max_index=max_idx, seed=seed)]
            else:
                for item in emitted:
                    yield item


__all__ = [
    "FirstFitPackIterDataset",
    "FlatMapIterDataset",
    "InterleaveIterDataset",
    "ThreadPrefetchIterDataset",
]
