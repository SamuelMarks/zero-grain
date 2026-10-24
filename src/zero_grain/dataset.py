"""MapDataset, IterDataset, and DatasetIterator abstractions for zero-grain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, overload
from zero_grain._rng import index_shuffle
from zero_grain._tree import spec_like
from zero_grain._types import (
    CheckpointStateDict,
    ElementTree,
    RNGGeneratorProtocol,
    SliceProtocol,
    T_co,
    T_out,
)
from zero_grain.data_loader import (
    DataLoader,
    DataLoaderIterator,
    TransformationSpec,
    normalize_operation,
)
from zero_grain.data_sources import (
    InMemoryDataSource,
    RandomAccessDataSource,
    RangeDataSource,
)
from zero_grain.operations import (
    BatchOperation,
    FilterOperation,
    MapOperation,
    MapWithIndexOperation,
    Operation,
    RandomMapOperation,
)
from zero_grain.options import ReadOptions
from zero_grain.samplers import IndexSampler, Sampler
from zero_grain.transforms import (
    FilterTransform,
    MapTransform,
    MapWithIndexTransform,
    RandomMapTransform,
)


def _apply_single_index_operations(
    raw_value: ElementTree,
    index: int,
    seed: int | None,
    operations: Sequence[Operation[ElementTree, ElementTree]],
) -> ElementTree:
    """Apply element-wise ``operations`` to ``raw_value`` at ``index``.

    Args:
        raw_value (ElementTree): Initial source element.
        index (int): Logical record index.
        seed (int | None): Base RNG seed for stochastic operations.
        operations (Sequence[Operation[ElementTree, ElementTree]]): Operations to evaluate.

    Returns:
        ElementTree: Transformed element (or None if filtered out by a ``FilterOperation``).
    """
    current: ElementTree = raw_value
    for op_idx, op in enumerate(operations):
        current = op.apply_element(current, index=index, seed=seed, op_seed=op_idx)
        if current is None and isinstance(op, FilterOperation):
            return None
    return current


class MapDataset(Generic[T_co]):
    """Random-access dataset supporting lazy functional transformations."""

    def __init__(
        self,
        parents: (MapDataset[T_co] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[T_co]]) = (),
        *,
        _source_override: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None = None,
        _operations_override: tuple[Operation[ElementTree, ElementTree], ...] | None = None,
        _seed_override: int | None = None,
        _shuffled_override: bool | None = None,
        _slice_range_override: range | None = None,
        _num_epochs_override: int | None = None,
    ) -> None:
        """Initialize ``MapDataset``.

        Args:
            parents (MapDataset[T_co] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[T_co]]):
                Parent dataset(s) or underlying data source.
            _source_override (RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None):
                Internal source override for derived datasets.
            _operations_override (tuple[Operation[ElementTree, ElementTree], ...] | None):
                Internal operations tuple override for derived datasets.
            _seed_override (int | None): Internal seed override for derived datasets.
            _shuffled_override (bool | None): Internal shuffle flag override for derived datasets.
            _slice_range_override (range | None): Internal slice range override for derived datasets.
            _num_epochs_override (int | None): Internal epoch count override for derived datasets.
        """
        self._seed: int | None = None
        self._shuffled: bool = False
        self._slice_range: range | None = None
        self._num_epochs: int | None = 1
        self.operations: tuple[Operation[ElementTree, ElementTree], ...] = ()
        if isinstance(parents, MapDataset):
            self.parents: tuple[MapDataset[ElementTree], ...] = (parents,)
            self.source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] = parents.source
            self.operations = parents.operations
            self._seed = parents._seed
            self._shuffled = parents._shuffled
            self._slice_range = parents._slice_range
            self._num_epochs = parents._num_epochs
        elif isinstance(parents, RandomAccessDataSource):
            self.parents = ()
            self.source = parents
        elif isinstance(parents, Sequence) and len(parents) > 0 and isinstance(parents[0], MapDataset):
            first_parent = parents[0]
            self.parents = tuple(p for p in parents if isinstance(p, MapDataset))
            self.source = first_parent.source
            self.operations = first_parent.operations
            self._seed = first_parent._seed
            self._shuffled = first_parent._shuffled
            self._slice_range = first_parent._slice_range
            self._num_epochs = first_parent._num_epochs
        else:
            self.parents = ()
            seq_items: list[T_co] = [
                item for item in (parents if isinstance(parents, Sequence) else ()) if not isinstance(item, MapDataset)
            ]
            self.source = InMemoryDataSource(seq_items)

        if _source_override is not None:
            self.source = _source_override
        if _operations_override is not None:
            self.operations = _operations_override
        if _seed_override is not None:
            self._seed = _seed_override
        if _shuffled_override is not None:
            self._shuffled = _shuffled_override
        if _slice_range_override is not None:
            self._slice_range = _slice_range_override
        if _num_epochs_override is not None:
            self._num_epochs = _num_epochs_override

    def _clone_with(
        self,
        *,
        operations: tuple[Operation[ElementTree, ElementTree], ...] | None = None,
        seed: int | None = None,
        shuffled: bool | None = None,
        slice_range: range | None = None,
        num_epochs: int | None = None,
    ) -> MapDataset[T_out]:
        """Construct a child ``MapDataset`` sharing this dataset's source and state.

        Args:
            operations (tuple[Operation[ElementTree, ElementTree], ...] | None): Updated operations tuple.
            seed (int | None): Updated RNG seed.
            shuffled (bool | None): Updated shuffle flag.
            slice_range (range | None): Updated index slice range.
            num_epochs (int | None): Updated epoch repetition count.

        Returns:
            MapDataset[T_out]: Derived child ``MapDataset``.
        """
        child: MapDataset[T_out] = MapDataset(
            _source_override=self.source,
            _operations_override=operations if operations is not None else self.operations,
            _seed_override=seed if seed is not None else self._seed,
            _shuffled_override=shuffled if shuffled is not None else self._shuffled,
            _slice_range_override=slice_range if slice_range is not None else self._slice_range,
            _num_epochs_override=num_epochs if num_epochs is not None else self._num_epochs,
        )
        child.parents = (self,)
        return child

    @classmethod
    def range(
        cls,
        start: int,
        stop: int | None = None,
        step: int = 1,
    ) -> MapDataset[int]:
        """Create a ``MapDataset`` yielding integers from ``range(start, stop, step)``.

        Args:
            start (int): Start value (or stop value if ``stop`` is None).
            stop (int | None): Exclusive stop value.
            step (int): Step increment.

        Returns:
            MapDataset[int]: Range-backed ``MapDataset``.
        """
        if stop is None:
            actual_start = 0
            actual_stop = start
        else:
            actual_start = start
            actual_stop = stop
        return MapDataset(RangeDataSource(actual_start, actual_stop, step))

    @classmethod
    def source_dataset(
        cls,
        source: RandomAccessDataSource[T_co] | Sequence[T_co],
    ) -> MapDataset[T_co]:
        """Create a ``MapDataset`` wrapping ``source``.

        Args:
            source (RandomAccessDataSource[T_co] | Sequence[T_co]): Backing data source.

        Returns:
            MapDataset[T_co]: Source-backed ``MapDataset``.
        """
        return cls(source)

    def _base_len(self) -> int:
        """Return the single-epoch length after applying any active slice.

        Returns:
            int: Single-epoch element count.
        """
        if self._slice_range is not None:
            return len(self._slice_range)
        return len(self.source)

    def __len__(self) -> int:
        """Return the number of elements in the dataset across configured epochs.

        Returns:
            int: Dataset length.
        """
        epochs = self._num_epochs if self._num_epochs is not None else 1
        return self._base_len() * epochs

    def _resolve_source_index(self, index: int) -> int:
        """Resolve a logical dataset ``index`` to the backing ``self.source`` index.

        Args:
            index (int): Non-negative logical dataset index in ``[0, len(self))``.

        Returns:
            int: Backing data source index.

        Raises:
            IndexError: If ``index`` is out of bounds.
        """
        total = MapDataset.__len__(self)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(f"MapDataset index {index} out of range for length {total}")
        base_len = self._base_len()
        epoch, pos_in_epoch = divmod(index, base_len)
        if self._shuffled and base_len > 1:
            base_seed = self._seed if self._seed is not None else 0
            per_epoch_seed = (base_seed + epoch) % (2**32)
            pos_in_epoch = index_shuffle(pos_in_epoch, max_index=base_len - 1, seed=per_epoch_seed)
        if self._slice_range is not None:
            return self._slice_range[pos_in_epoch]
        return pos_in_epoch

    @overload
    def __getitem__(self, index: int) -> ElementTree:
        """Return the transformed element at integer ``index``.

        Args:
            index (int): Zero-based element index.

        Returns:
            ElementTree: Transformed element at ``index``.
        """

    @overload
    def __getitem__(self, index: SliceProtocol) -> MapDataset[T_co]:
        """Return a sliced ``MapDataset`` for ``index``.

        Args:
            index (SliceProtocol): Python slice object.

        Returns:
            MapDataset[T_co]: Sliced ``MapDataset``.
        """

    def __getitem__(self, index: int | SliceProtocol) -> ElementTree | MapDataset[T_co]:
        """Return the transformed element at ``index`` or a sliced ``MapDataset``.

        Args:
            index (int | SliceProtocol): Zero-based element index or slice object.

        Returns:
            ElementTree | MapDataset[T_co]: Transformed element at ``index`` or sliced ``MapDataset``.
        """
        if isinstance(index, SliceProtocol):
            return self.slice(index)
        src_idx = self._resolve_source_index(index)
        raw_item = self.source[src_idx]
        if not self.operations:
            return raw_item
        return _apply_single_index_operations(
            raw_item,
            index=index,
            seed=self._seed,
            operations=self.operations,
        )

    def __iter__(self) -> DatasetIterator[T_co]:
        """Return an iterator over the dataset elements.

        Returns:
            DatasetIterator[T_co]: Iterator over this dataset.
        """
        return iter(self.to_iter_dataset())

    def seed(self, seed: int) -> MapDataset[T_co]:
        """Configure the default RNG ``seed`` for this dataset and child transformations.

        Args:
            seed (int): Integer seed value.

        Returns:
            MapDataset[T_co]: Dataset with updated seed.
        """
        self._seed = seed
        return self._clone_with(seed=seed)

    def shuffle(self, seed: int | None = None) -> MapDataset[T_co]:
        """Return a shuffled view of the dataset.

        Args:
            seed (int | None): Optional shuffle seed.

        Returns:
            MapDataset[T_co]: Shuffled dataset.
        """
        effective_seed = seed if seed is not None else self._seed
        return self._clone_with(shuffled=True, seed=effective_seed)

    def map(
        self,
        transform: (
            Callable[[ElementTree], ElementTree]
            | MapTransform[ElementTree, ElementTree]
            | Operation[ElementTree, ElementTree]
        ),
    ) -> MapDataset[T_out]:
        """Apply a map transformation to each element.

        Args:
            transform (Callable[[ElementTree], ElementTree] | MapTransform[ElementTree, ElementTree] | Operation[ElementTree, ElementTree]):
                Map transform or callable.

        Returns:
            MapDataset[T_out]: Transformed ``MapDataset``.
        """
        op: Operation[ElementTree, ElementTree] = (
            transform if isinstance(transform, Operation) else MapOperation(transform)
        )
        return self._clone_with(operations=(*self.operations, op))

    def map_with_index(
        self,
        transform: (
            Callable[[int, ElementTree], ElementTree]
            | MapWithIndexTransform[ElementTree, ElementTree]
            | MapWithIndexOperation[ElementTree, ElementTree]
        ),
    ) -> MapDataset[T_out]:
        """Apply an index-aware map transformation to each element.

        Args:
            transform (Callable[[int, ElementTree], ElementTree] | MapWithIndexTransform[ElementTree, ElementTree] | MapWithIndexOperation[ElementTree, ElementTree]):
                Index-aware transform or callable.

        Returns:
            MapDataset[T_out]: Transformed ``MapDataset``.
        """
        op: Operation[ElementTree, ElementTree] = (
            transform if isinstance(transform, Operation) else MapWithIndexOperation(transform)
        )
        return self._clone_with(operations=(*self.operations, op))

    def random_map(
        self,
        transform: (
            Callable[[ElementTree, RNGGeneratorProtocol], ElementTree]
            | RandomMapTransform[ElementTree, ElementTree]
            | RandomMapOperation[ElementTree, ElementTree]
        ),
        seed: int | None = None,
    ) -> MapDataset[T_out]:
        """Apply a stochastic transformation to each element.

        Args:
            transform (Callable[[ElementTree, RNGGeneratorProtocol], ElementTree] | RandomMapTransform[ElementTree, ElementTree] | RandomMapOperation[ElementTree, ElementTree]):
                Random map transform or callable.
            seed (int | None): Optional seed override.

        Returns:
            MapDataset[T_out]: Transformed ``MapDataset``.
        """
        op: Operation[ElementTree, ElementTree] = (
            transform if isinstance(transform, Operation) else RandomMapOperation(transform)
        )
        return self._clone_with(
            operations=(*self.operations, op),
            seed=seed if seed is not None else self._seed,
        )

    def filter(
        self,
        transform: (Callable[[ElementTree], bool] | FilterTransform[ElementTree] | FilterOperation[ElementTree]),
    ) -> FilterDataset[T_co]:
        """Filter elements matching ``transform``.

        Args:
            transform (Callable[[ElementTree], bool] | FilterTransform[ElementTree] | FilterOperation[ElementTree]):
                Predicate callable, ``FilterTransform``, or ``FilterOperation``.

        Returns:
            FilterDataset[T_co]: Filtered dataset.
        """
        op: Operation[ElementTree, ElementTree] = (
            transform if isinstance(transform, Operation) else FilterOperation(transform)
        )
        child: FilterDataset[T_co] = FilterDataset(
            self,
            _operations_override=(*self.operations, op),
        )
        return child

    def batch(
        self,
        batch_size: int,
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[ElementTree]], ElementTree] | None = None,
    ) -> BatchDataset[T_out]:
        """Batch consecutive elements into groups of ``batch_size``.

        Args:
            batch_size (int): Number of elements per batch.
            drop_remainder (bool): Whether to drop incomplete final batches.
            batch_fn (Callable[[Sequence[ElementTree]], ElementTree] | None): Custom collation function.

        Returns:
            BatchDataset[T_out]: Batched dataset.
        """
        op: BatchOperation[ElementTree] = BatchOperation(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            batch_fn=batch_fn,
        )
        child: BatchDataset[T_out] = BatchDataset(
            _source_override=self.source,
            _operations_override=(*self.operations, op),
            _seed_override=self._seed,
            _shuffled_override=self._shuffled,
            _slice_range_override=self._slice_range,
            _num_epochs_override=self._num_epochs,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            batch_op=op,
        )
        child.parents = (self,)
        return child

    def slice(self, sl: SliceProtocol) -> MapDataset[T_co]:
        """Return a lazily sliced view of the dataset.

        Args:
            sl (SliceProtocol): Python slice object.

        Returns:
            MapDataset[T_co]: Sliced ``MapDataset``.
        """
        current_range = self._slice_range if self._slice_range is not None else range(len(self.source))
        start, stop, step = sl.indices(len(current_range))
        new_range = current_range[start:stop:step]
        return self._clone_with(slice_range=new_range)

    def repeat(self, num_epochs: int | None = None) -> MapDataset[T_co]:
        """Repeat the dataset lazily for ``num_epochs`` epochs.

        Args:
            num_epochs (int | None): Number of epochs to repeat (defaults to 1 if None).

        Returns:
            MapDataset[T_co]: Repeated ``MapDataset``.
        """
        epochs = num_epochs if num_epochs is not None else 1
        current_epochs = self._num_epochs if self._num_epochs is not None else 1
        return self._clone_with(num_epochs=current_epochs * epochs)

    def to_iter_dataset(
        self,
        read_options: ReadOptions | None = None,
    ) -> IterDataset[T_co]:
        """Convert this ``MapDataset`` into an ``IterDataset``.

        Args:
            read_options (ReadOptions | None): Optional reading options.

        Returns:
            IterDataset[T_co]: Iterable dataset wrapper.
        """
        return IterDataset(self, read_options=read_options)


class FilterDataset(MapDataset[T_co]):
    """``MapDataset`` subclass representing a filtered dataset stage."""


class BatchDataset(MapDataset[T_co]):
    """``MapDataset`` subclass representing a batched dataset stage."""

    def __init__(
        self,
        parents: (MapDataset[T_co] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[T_co]]) = (),
        *,
        batch_size: int = 1,
        drop_remainder: bool = False,
        batch_op: BatchOperation[ElementTree] | None = None,
        _source_override: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None = None,
        _operations_override: tuple[Operation[ElementTree, ElementTree], ...] | None = None,
        _seed_override: int | None = None,
        _shuffled_override: bool | None = None,
        _slice_range_override: range | None = None,
        _num_epochs_override: int | None = None,
    ) -> None:
        """Initialize ``BatchDataset``.

        Args:
            parents (MapDataset[T_co] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[T_co]]):
                Parent dataset or data source.
            batch_size (int): Number of elements per batch.
            drop_remainder (bool): Whether to drop incomplete final batches.
            batch_op (BatchOperation[ElementTree] | None): Batch operation instance.
            _source_override (RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None):
                Internal source override.
            _operations_override (tuple[Operation[ElementTree, ElementTree], ...] | None):
                Internal operations override.
            _seed_override (int | None): Internal seed override.
            _shuffled_override (bool | None): Internal shuffle flag override.
            _slice_range_override (range | None): Internal slice range override.
            _num_epochs_override (int | None): Internal epoch count override.
        """
        super().__init__(
            parents,
            _source_override=_source_override,
            _operations_override=_operations_override,
            _seed_override=_seed_override,
            _shuffled_override=_shuffled_override,
            _slice_range_override=_slice_range_override,
            _num_epochs_override=_num_epochs_override,
        )
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self._batch_op: BatchOperation[ElementTree] = (
            batch_op if batch_op is not None else BatchOperation(batch_size, drop_remainder)
        )

    def __len__(self) -> int:
        """Return the number of batches in the dataset.

        Returns:
            int: Total batch count.
        """
        parent_len = super().__len__()
        if parent_len == 0:
            return 0
        if self.drop_remainder:
            return parent_len // self.batch_size
        return (parent_len + self.batch_size - 1) // self.batch_size

    @overload
    def __getitem__(self, index: int) -> ElementTree:
        """Return the collated batch at integer ``index``.

        Args:
            index (int): Batch index.

        Returns:
            ElementTree: Collated batch at ``index``.
        """

    @overload
    def __getitem__(self, index: SliceProtocol) -> MapDataset[T_co]:
        """Return a sliced ``MapDataset`` for ``index``.

        Args:
            index (SliceProtocol): Slice object.

        Returns:
            MapDataset[T_co]: Sliced dataset.
        """

    def __getitem__(self, index: int | SliceProtocol) -> ElementTree | MapDataset[T_co]:
        """Return the collated batch at ``index`` or a sliced ``MapDataset``.

        Args:
            index (int | SliceProtocol): Batch index or slice.

        Returns:
            ElementTree | MapDataset[T_co]: Collated batch at ``index`` or sliced dataset.

        Raises:
            IndexError: If ``index`` is out of bounds.
        """
        if isinstance(index, SliceProtocol):
            return self.slice(index)
        total_batches = len(self)
        if index < 0:
            index += total_batches
        if index < 0 or index >= total_batches:
            raise IndexError(f"BatchDataset index {index} out of range for {total_batches} batches")
        parent_len = super().__len__()
        start_elem = index * self.batch_size
        end_elem = min(start_elem + self.batch_size, parent_len)
        pre_batch_ops = tuple(op for op in self.operations if op is not self._batch_op)
        items: list[ElementTree] = []
        for elem_idx in range(start_elem, end_elem):
            src_idx = self._resolve_source_index(elem_idx)
            raw_val = self.source[src_idx]
            transformed = _apply_single_index_operations(
                raw_val,
                index=elem_idx,
                seed=self._seed,
                operations=pre_batch_ops,
            )
            if transformed is not None:
                items.append(transformed)
        return self._batch_op._collate_tree(items)


Dataset = MapDataset


class _SlicedDataSource(RandomAccessDataSource[ElementTree]):
    """Internal data source adapter applying a ``MapDataset`` slice range over ``source``."""

    def __init__(
        self,
        source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree],
        slice_range: range,
    ) -> None:
        """Initialize ``_SlicedDataSource``.

        Args:
            source (RandomAccessDataSource[ElementTree] | Sequence[ElementTree]): Underlying source.
            slice_range (range): Active slice index mapping.
        """
        super().__init__()
        self._source = source
        self._slice_range = slice_range

    def __len__(self) -> int:
        """Return the sliced length.

        Returns:
            int: Number of elements in the slice.
        """
        return len(self._slice_range)

    def __getitem__(self, record_key: int) -> ElementTree:
        """Return the element mapped by ``record_key`` through ``self._slice_range``.

        Args:
            record_key (int): Index within the slice.

        Returns:
            ElementTree: Underlying source element.
        """
        return self._source[self._slice_range[record_key]]

    def __repr__(self) -> str:
        """Return the string representation of the sliced source.

        Returns:
            str: Formatted string representation.
        """
        return repr(self._source)


class IterDataset(Generic[T_co]):
    """Iterable dataset supporting streaming transformations and checkpointing."""

    def __init__(
        self,
        parents: (MapDataset[T_co] | IterDataset[T_co] | Sequence[MapDataset[T_co] | IterDataset[T_co]]) = (),
        *,
        read_options: ReadOptions | None = None,
    ) -> None:
        """Initialize ``IterDataset``.

        Args:
            parents (MapDataset[T_co] | IterDataset[T_co] | Sequence[MapDataset[T_co] | IterDataset[T_co]]):
                Parent ``MapDataset`` or ``IterDataset`` instance(s).
            read_options (ReadOptions | None): Optional reading options.
        """
        self.read_options: ReadOptions | None = read_options
        if isinstance(parents, MapDataset):
            self.dataset: MapDataset[T_co] = parents
        elif isinstance(parents, IterDataset):
            self.dataset = parents.dataset
            if self.read_options is None:
                self.read_options = parents.read_options
        elif isinstance(parents, Sequence) and len(parents) > 0:
            first = parents[0]
            self.dataset = first if isinstance(first, MapDataset) else first.dataset
        else:
            self.dataset = MapDataset()

        base_len = self.dataset._base_len()
        effective_source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] = (
            _SlicedDataSource(self.dataset.source, self.dataset._slice_range)
            if self.dataset._slice_range is not None
            else self.dataset.source
        )
        sampler: Sampler = (
            IndexSampler(
                num_records=base_len,
                shuffle=self.dataset._shuffled,
                num_epochs=self.dataset._num_epochs,
                seed=self.dataset._seed,
            )
            if base_len > 0
            else Sampler()
        )
        self.dl: DataLoader[ElementTree] = DataLoader(
            data_source=effective_source,
            sampler=sampler,
            operations=self.dataset.operations,
            read_options=self.read_options,
        )
        self._iter: DataLoaderIterator[ElementTree] = iter(self.dl)

    def __iter__(self) -> DatasetIterator[T_co]:
        """Return a fresh ``DatasetIterator`` over this ``IterDataset``.

        Returns:
            DatasetIterator[T_co]: Fresh dataset iterator.
        """
        return DatasetIterator(self)

    def __next__(self) -> ElementTree:
        """Advance the default internal iterator and return the next element.

        Returns:
            ElementTree: Next element from the dataset.
        """
        return next(self._iter)

    def __str__(self) -> str:
        """Return a string representation of the ``IterDataset``.

        Returns:
            str: String representation of the underlying ``DataLoader``.
        """
        return str(self.dl)


class DatasetIterator(Generic[T_co]):
    """Stateful, checkpointable iterator over an ``IterDataset``."""

    def __init__(
        self,
        parents: (DatasetIterator[T_co] | IterDataset[T_co] | Sequence[DatasetIterator[T_co]]) = (),
    ) -> None:
        """Initialize ``DatasetIterator``.

        Args:
            parents (DatasetIterator[T_co] | IterDataset[T_co] | Sequence[DatasetIterator[T_co]]):
                Parent ``IterDataset`` or ``DatasetIterator`` instance(s).
        """
        if isinstance(parents, IterDataset):
            self._dl_iter: DataLoaderIterator[ElementTree] = iter(parents.dl)
        elif isinstance(parents, DatasetIterator):
            self._dl_iter = DataLoaderIterator(
                parents._dl_iter.data_loader,
                state=parents._dl_iter.get_state(),
                validate_state=False,
            )
        elif isinstance(parents, Sequence) and len(parents) > 0:
            first_iter = parents[0]
            self._dl_iter = DataLoaderIterator(
                first_iter._dl_iter.data_loader,
                state=first_iter._dl_iter.get_state(),
                validate_state=False,
            )
        else:
            self._dl_iter = DataLoaderIterator()

    def __iter__(self) -> DatasetIterator[T_co]:
        """Return ``self`` as an iterator.

        Returns:
            DatasetIterator[T_co]: Self reference.
        """
        return self

    def __next__(self) -> ElementTree:
        """Return the next element from the iterator.

        Returns:
            ElementTree: Next element.
        """
        return next(self._dl_iter)

    def get_state(self) -> CheckpointStateDict:
        """Capture the current iterator checkpoint state.

        Returns:
            CheckpointStateDict: Dictionary suitable for ``set_state`` restoration.
        """
        return self._dl_iter.get_state()

    def set_state(self, state: CheckpointStateDict) -> None:
        """Restore the iterator to a previously captured checkpoint ``state``.

        Args:
            state (CheckpointStateDict): Checkpoint dictionary from ``get_state()``.
        """
        self._dl_iter.set_state(state)

    def start_prefetch(self) -> None:
        """Start background prefetching on the underlying iterator."""
        self._dl_iter.start_prefetch()

    def __str__(self) -> str:
        """Return the formatted checkpoint state string of this iterator.

        Returns:
            str: Multi-line JSON-formatted iterator state string.
        """
        return str(self._dl_iter)

    def __repr__(self) -> str:
        """Return the canonical string representation of this iterator.

        Returns:
            str: Formatted iterator representation.
        """
        return str(self._dl_iter)


class DatasetSelectionMap:
    """Selection map mapping dataset indices across mixed or interleaved datasets."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``DatasetSelectionMap``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        """Return the total number of mapped elements.

        Returns:
            int: Length of the selection map.
        """
        if self.args and isinstance(self.args[0], int):
            return int(self.args[0])
        return 0

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Map global ``index`` to ``(dataset_index, dataset_element_index)``.

        Args:
            index (int): Global element index.

        Returns:
            tuple[int, int]: Pair of ``(dataset_index, element_index)``.
        """
        return (0, index)


def apply_transformations(
    ds: MapDataset[T_co],
    transformations: (
        TransformationSpec[ElementTree, ElementTree] | Sequence[TransformationSpec[ElementTree, ElementTree]]
    ),
) -> MapDataset[T_co]:
    """Apply one or more transformations to a ``MapDataset``.

    Args:
        ds (MapDataset[T_co]): Input ``MapDataset``.
        transformations (TransformationSpec[ElementTree, ElementTree] | Sequence[TransformationSpec[ElementTree, ElementTree]]):
            Single transformation or sequence of transformations.

    Returns:
        MapDataset[T_co]: Transformed ``MapDataset``.
    """
    specs: Sequence[TransformationSpec[ElementTree, ElementTree]] = (
        transformations if isinstance(transformations, Sequence) else (transformations,)
    )
    normalized = tuple(normalize_operation(t) for t in specs)
    return ds._clone_with(operations=(*ds.operations, *normalized))


def get_element_spec(ds: MapDataset[T_co] | IterDataset[T_co]) -> ElementTree:
    """Infer the structural element specification of ``ds``.

    Args:
        ds (MapDataset[T_co] | IterDataset[T_co]): Dataset to inspect.

    Returns:
        ElementTree: Inferred element specification or None if empty.
    """
    target = ds.dataset if isinstance(ds, IterDataset) else ds
    if len(target) == 0:
        return None
    first_elem = target[0]
    return spec_like(first_elem)


__all__ = [
    "BatchDataset",
    "Dataset",
    "DatasetIterator",
    "DatasetSelectionMap",
    "FilterDataset",
    "IterDataset",
    "MapDataset",
    "apply_transformations",
    "get_element_spec",
]
