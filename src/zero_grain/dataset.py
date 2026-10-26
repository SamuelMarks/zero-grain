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
from zero_grain.options import MultiprocessingOptions, ReadOptions
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


class DatasetSelectionMap:
    """Selection map mapping global dataset indices across mixed or interleaved datasets."""

    def __init__(
        self,
        *args: ElementTree,
        **kwargs: ElementTree,
    ) -> None:
        """Initialize ``DatasetSelectionMap``.

        Args:
            *args (ElementTree): Optional positional length or parent lengths.
            **kwargs (ElementTree): Optional keyword arguments (supports ``parent_lengths``).
        """
        self._args = args
        self._kwargs = kwargs
        self.args = args
        self.kwargs = kwargs
        kw_lengths = kwargs.get("parent_lengths")
        if isinstance(kw_lengths, (list, tuple)):
            self._parent_lengths: tuple[int, ...] = tuple(int(x) for x in kw_lengths if isinstance(x, int))
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            self._parent_lengths = tuple(int(x) for x in args[0] if isinstance(x, int))
        elif len(args) >= 1 and all(isinstance(x, int) for x in args):
            self._parent_lengths = tuple(int(x) for x in args if isinstance(x, int))
        else:
            self._parent_lengths = ()
        self._mapping: list[tuple[int, int]] = self._build_interleaved_mapping(self._parent_lengths)

    @staticmethod
    def _build_interleaved_mapping(lengths: Sequence[int]) -> list[tuple[int, int]]:
        """Build a round-robin ``(parent_index, element_index)`` table across ``lengths``.

        Args:
            lengths (Sequence[int]): Sequence of parent dataset lengths.

        Returns:
            list[tuple[int, int]]: Ordered index mapping table.
        """
        if not lengths:
            return []
        if len(lengths) == 1:
            return [(0, i) for i in range(max(0, lengths[0]))]
        max_len = max(lengths)
        table: list[tuple[int, int]] = []
        for elem_idx in range(max_len):
            for parent_idx, p_len in enumerate(lengths):
                if elem_idx < p_len:
                    table.append((parent_idx, elem_idx))
        return table

    def __len__(self) -> int:
        """Return the total number of mapped elements.

        Returns:
            int: Length of the selection map.
        """
        return len(self._mapping)

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Map global ``index`` to ``(dataset_index, dataset_element_index)``.

        Args:
            index (int): Global element index.

        Returns:
            tuple[int, int]: Pair of ``(dataset_index, element_index)``.

        Raises:
            IndexError: If ``index`` is out of bounds for a non-empty selection map.
        """
        if not self._mapping:
            if index < 0:
                raise IndexError(f"DatasetSelectionMap index {index} out of range")
            return (0, index)
        total = len(self._mapping)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(f"DatasetSelectionMap index {index} out of range for length {total}")
        return self._mapping[index]


class SelectionWithReplacementMap(DatasetSelectionMap):
    """Selection map sampling across multiple parent datasets with replacement according to weights."""

    def __init__(
        self,
        *args: ElementTree,
        **kwargs: ElementTree,
    ) -> None:
        """Initialize ``SelectionWithReplacementMap``.

        Args:
            *args (ElementTree): Optional positional parent lengths or weights.
            **kwargs (ElementTree): Optional keyword arguments (``parent_lengths``, ``weights``, ``total_size``).

        Raises:
            ValueError: If any weight is negative or all weights sum to zero.
        """
        super().__init__(*args, **kwargs)
        raw_weights = kwargs.get("weights")
        weights_list: list[float] = []
        if isinstance(raw_weights, (list, tuple)):
            for w in raw_weights:
                if isinstance(w, (int, float)):
                    weights_list.append(float(w))
        if not weights_list and self._parent_lengths:
            weights_list = [1.0 for _ in self._parent_lengths]
        if weights_list:
            if any(w < 0.0 for w in weights_list):
                raise ValueError("weights must be non-negative")
            weight_sum = sum(weights_list)
            if weight_sum <= 0.0:
                raise ValueError("sum of weights must be positive")
            self.weights: tuple[float, ...] = tuple(w / weight_sum for w in weights_list)
        else:
            self.weights = ()

        raw_total = kwargs.get("total_size")
        if isinstance(raw_total, int) and not isinstance(raw_total, bool):
            total_size = max(0, raw_total)
        else:
            total_size = sum(max(0, ln) for ln in self._parent_lengths)

        self._mapping = self._build_weighted_replacement_mapping(
            self._parent_lengths,
            self.weights,
            total_size,
        )

    @staticmethod
    def _build_weighted_replacement_mapping(
        lengths: Sequence[int],
        weights: Sequence[float],
        total_size: int,
    ) -> list[tuple[int, int]]:
        """Construct a deterministic Deficit-Round-Robin schedule with replacement across ``lengths``.

        Args:
            lengths (Sequence[int]): Parent dataset lengths.
            weights (Sequence[float]): Normalized parent weights summing to 1.0.
            total_size (int): Total number of elements to schedule.

        Returns:
            list[tuple[int, int]]: Sequence of ``(parent_index, element_index)`` pairs.
        """
        if not lengths or total_size <= 0:
            return []
        num_parents = len(lengths)
        norm_weights = list(weights) if len(weights) == num_parents else [1.0 / num_parents] * num_parents
        counts = [0 for _ in range(num_parents)]
        deficits = [0.0 for _ in range(num_parents)]
        schedule: list[tuple[int, int]] = []
        for _ in range(total_size):
            best_parent = 0
            best_score = -1e18
            for p_idx in range(num_parents):
                if lengths[p_idx] <= 0:
                    continue
                deficits[p_idx] += norm_weights[p_idx]
                if deficits[p_idx] > best_score:
                    best_score = deficits[p_idx]
                    best_parent = p_idx
            if lengths[best_parent] <= 0:
                break
            deficits[best_parent] -= 1.0
            elem_idx = counts[best_parent] % lengths[best_parent]
            counts[best_parent] += 1
            schedule.append((best_parent, elem_idx))
        return schedule


class _BoundMapDatasetSourceMethod:
    """Callable classmethod wrapper for ``MapDataset.source``."""

    def __init__(self, owner: type[MapDataset[ElementTree]]) -> None:
        """Initialize ``_BoundMapDatasetSourceMethod``.

        Args:
            owner (type[MapDataset[ElementTree]]): ``MapDataset`` class or subclass.
        """
        self._owner = owner

    def __call__(
        self,
        source: RandomAccessDataSource[T_co] | Sequence[T_co],
    ) -> MapDataset[T_co]:
        """Create a ``MapDataset`` wrapping ``source``.

        Args:
            source (RandomAccessDataSource[T_co] | Sequence[T_co]): Backing data source or sequence.

        Returns:
            MapDataset[T_co]: Source-backed ``MapDataset``.
        """
        return MapDataset(source)


class _MapDatasetSourceDescriptor:
    """Dual class/instance descriptor providing ``MapDataset.source(src)`` and ``instance.source``."""

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[MapDataset[ElementTree]],
    ) -> _BoundMapDatasetSourceMethod:
        """Return the bound classmethod when accessed on the ``MapDataset`` class.

        Args:
            instance (None): None when accessed on the class.
            owner (type[MapDataset[ElementTree]]): Owning ``MapDataset`` class.

        Returns:
            _BoundMapDatasetSourceMethod: Callable classmethod constructing a ``MapDataset``.
        """

    @overload
    def __get__(
        self,
        instance: MapDataset[ElementTree],
        owner: type[MapDataset[ElementTree]] | None = None,
    ) -> RandomAccessDataSource[ElementTree] | Sequence[ElementTree]:
        """Return the backing data source when accessed on a ``MapDataset`` instance.

        Args:
            instance (MapDataset[ElementTree]): ``MapDataset`` instance.
            owner (type[MapDataset[ElementTree]] | None): Owning class.

        Returns:
            RandomAccessDataSource[ElementTree] | Sequence[ElementTree]: Instance data source.
        """

    def __get__(
        self,
        instance: MapDataset[ElementTree] | None,
        owner: type[MapDataset[ElementTree]] | None = None,
    ) -> _BoundMapDatasetSourceMethod | RandomAccessDataSource[ElementTree] | Sequence[ElementTree]:
        """Dispatch between ``MapDataset.source(...)`` and ``instance.source``.

        Args:
            instance (MapDataset[ElementTree] | None): ``MapDataset`` instance or None.
            owner (type[MapDataset[ElementTree]] | None): Owning ``MapDataset`` class.

        Returns:
            _BoundMapDatasetSourceMethod | RandomAccessDataSource[ElementTree] | Sequence[ElementTree]:
                Bound classmethod or instance data source.
        """
        if instance is None:
            resolved_owner: type[MapDataset[ElementTree]] = owner if owner is not None else MapDataset
            return _BoundMapDatasetSourceMethod(resolved_owner)
        return instance._backing_source

    def __set__(
        self,
        instance: MapDataset[ElementTree],
        value: RandomAccessDataSource[ElementTree] | Sequence[ElementTree],
    ) -> None:
        """Store the backing data source on ``instance``.

        Args:
            instance (MapDataset[ElementTree]): Target ``MapDataset`` instance.
            value (RandomAccessDataSource[ElementTree] | Sequence[ElementTree]): Backing data source.
        """
        instance._backing_source = value


class MapDataset(Generic[T_co]):
    """Random-access dataset supporting lazy functional transformations and parent composition."""

    _backing_source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree]

    def __init__(
        self,
        parents: (
            MapDataset[ElementTree] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[ElementTree]]
        ) = (),
        selection_map: DatasetSelectionMap | None = None,
        *,
        _source_override: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None = None,
        _operations_override: tuple[Operation[ElementTree, ElementTree], ...] | None = None,
        _local_ops_override: tuple[Operation[ElementTree, ElementTree], ...] | None = None,
        _seed_override: int | None = None,
        _shuffled_override: bool | None = None,
        _slice_range_override: range | None = None,
        _num_epochs_override: int | None = None,
        _step_kind: str = "source",
        _step_slice: range | None = None,
        _step_epochs: int = 1,
        _step_shuffle_seed: int | None = None,
        _reassemble_every_epoch: bool = True,
    ) -> None:
        """Initialize ``MapDataset``.

        Args:
            parents (MapDataset[ElementTree] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[ElementTree]]):
                Parent dataset(s) or underlying data source.
            selection_map (DatasetSelectionMap | None): Optional multi-parent index selection map.
            _source_override (RandomAccessDataSource[ElementTree] | Sequence[ElementTree] | None):
                Internal source override for derived datasets.
            _operations_override (tuple[Operation[ElementTree, ElementTree], ...] | None):
                Internal cumulative operations tuple override for derived datasets.
            _local_ops_override (tuple[Operation[ElementTree, ElementTree], ...] | None):
                Internal stage-local operations tuple for derived datasets.
            _seed_override (int | None): Internal seed override for derived datasets.
            _shuffled_override (bool | None): Internal shuffle flag override for derived datasets.
            _slice_range_override (range | None): Internal slice range override for derived datasets.
            _num_epochs_override (int | None): Internal epoch count override for derived datasets.
            _step_kind (str): Stage evaluation kind (``source``, ``ops``, ``slice``, ``repeat``, ``shuffle``, ``mix``).
            _step_slice (range | None): Stage-local slice range when ``_step_kind == 'slice'``.
            _step_epochs (int): Stage-local repeat factor when ``_step_kind == 'repeat'``.
            _step_shuffle_seed (int | None): Stage-local shuffle seed when ``_step_kind == 'shuffle'``.
            _reassemble_every_epoch (bool): Whether ``repeat`` increments epoch offset across epochs.
        """
        self._seed: int | None = None
        self._shuffled: bool = False
        self._slice_range: range | None = None
        self._num_epochs: int | None = 1
        self.operations: tuple[Operation[ElementTree, ElementTree], ...] = ()
        self._local_ops: tuple[Operation[ElementTree, ElementTree], ...] = ()
        self._step_kind: str = _step_kind
        self._step_slice: range | None = _step_slice
        self._step_epochs: int = _step_epochs
        self._step_shuffle_seed: int | None = _step_shuffle_seed
        self._reassemble_every_epoch: bool = _reassemble_every_epoch
        self._selection_map: DatasetSelectionMap | None = selection_map

        if isinstance(parents, MapDataset):
            self.parents: tuple[MapDataset[ElementTree], ...] = (parents,)
            self._backing_source = parents._backing_source
            self.operations = parents.operations
            self._seed = parents._seed
            self._shuffled = parents._shuffled
            self._slice_range = parents._slice_range
            self._num_epochs = parents._num_epochs
            if _step_kind == "source":
                self._step_kind = "ops"
        elif isinstance(parents, RandomAccessDataSource):
            self.parents = ()
            self._backing_source = parents
        elif isinstance(parents, Sequence) and len(parents) > 0 and isinstance(parents[0], MapDataset):
            parent_tuple = tuple(p for p in parents if isinstance(p, MapDataset))
            self.parents = parent_tuple
            first_parent = parent_tuple[0]
            self._backing_source = first_parent._backing_source
            self.operations = first_parent.operations
            self._seed = first_parent._seed
            self._shuffled = first_parent._shuffled
            self._slice_range = first_parent._slice_range
            self._num_epochs = first_parent._num_epochs
            if len(parent_tuple) > 1 or selection_map is not None:
                self._step_kind = "mix"
                self._selection_map = (
                    selection_map
                    if selection_map is not None
                    else DatasetSelectionMap(parent_lengths=[len(p) for p in parent_tuple])
                )
            else:
                self._step_kind = "ops"
        else:
            self.parents = ()
            seq_items: list[T_co] = [
                item for item in (parents if isinstance(parents, Sequence) else ()) if not isinstance(item, MapDataset)
            ]
            self._backing_source = InMemoryDataSource(seq_items)

        if _source_override is not None:
            self._backing_source = _source_override
        if _operations_override is not None:
            self.operations = _operations_override
        if _local_ops_override is not None:
            self._local_ops = _local_ops_override
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
        local_ops: tuple[Operation[ElementTree, ElementTree], ...] = (),
        seed: int | None = None,
        shuffled: bool | None = None,
        slice_range: range | None = None,
        num_epochs: int | None = None,
        step_kind: str = "ops",
        step_slice: range | None = None,
        step_epochs: int = 1,
        step_shuffle_seed: int | None = None,
        reassemble_every_epoch: bool = True,
    ) -> MapDataset[T_out]:
        """Construct a child ``MapDataset`` delegating through ``self`` as its parent.

        Args:
            operations (tuple[Operation[ElementTree, ElementTree], ...] | None): Updated cumulative operations.
            local_ops (tuple[Operation[ElementTree, ElementTree], ...]): Stage-local operations for this node.
            seed (int | None): Updated RNG seed.
            shuffled (bool | None): Updated shuffle flag.
            slice_range (range | None): Updated index slice range.
            num_epochs (int | None): Updated epoch repetition count.
            step_kind (str): Stage evaluation kind for the new child node.
            step_slice (range | None): Stage-local slice range.
            step_epochs (int): Stage-local repeat factor.
            step_shuffle_seed (int | None): Stage-local shuffle seed.
            reassemble_every_epoch (bool): Whether ``repeat`` increments epoch offset across epochs.

        Returns:
            MapDataset[T_out]: Derived child ``MapDataset``.
        """
        child: MapDataset[T_out] = MapDataset(
            self,
            _source_override=self._backing_source,
            _operations_override=operations if operations is not None else self.operations,
            _local_ops_override=local_ops,
            _seed_override=seed if seed is not None else self._seed,
            _shuffled_override=shuffled if shuffled is not None else self._shuffled,
            _slice_range_override=slice_range if slice_range is not None else self._slice_range,
            _num_epochs_override=num_epochs if num_epochs is not None else self._num_epochs,
            _step_kind=step_kind,
            _step_slice=step_slice,
            _step_epochs=step_epochs,
            _step_shuffle_seed=step_shuffle_seed,
            _reassemble_every_epoch=reassemble_every_epoch,
        )
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

    source: _MapDatasetSourceDescriptor = _MapDatasetSourceDescriptor()

    @classmethod
    def mix(
        cls,
        datasets: Sequence[MapDataset[T_co]],
        weights: Sequence[float] | None = None,
    ) -> MapDataset[T_co]:
        """Mix multiple ``MapDataset`` instances deterministically according to ``weights``.

        Args:
            datasets (Sequence[MapDataset[T_co]]): Sequence of parent ``MapDataset`` objects.
            weights (Sequence[float] | None): Optional sampling weights per dataset.

        Returns:
            MapDataset[T_co]: Mixed ``MapDataset``.
        """
        if not datasets:
            return cls()
        lengths = [len(d) for d in datasets]
        if weights is None:
            sel_map: DatasetSelectionMap = DatasetSelectionMap(parent_lengths=lengths)
        else:
            sel_map = SelectionWithReplacementMap(
                parent_lengths=lengths,
                weights=list(weights),
                total_size=sum(lengths),
            )
        return cls.select_from_datasets(datasets=datasets, selection_map=sel_map)

    @classmethod
    def select_from_datasets(
        cls,
        datasets: Sequence[MapDataset[T_co]],
        selection_map: DatasetSelectionMap,
    ) -> MapDataset[T_co]:
        """Select elements across ``datasets`` according to ``selection_map``.

        Args:
            datasets (Sequence[MapDataset[T_co]]): Sequence of parent ``MapDataset`` objects.
            selection_map (DatasetSelectionMap): Index mapping from global index to ``(dataset_idx, elem_idx)``.

        Returns:
            MapDataset[T_co]: Combined ``MapDataset`` indexed via ``selection_map``.
        """
        if not datasets:
            return cls()
        return cls(datasets, selection_map=selection_map)

    def _base_len(self) -> int:
        """Return the single-epoch length after applying any active slice.

        Returns:
            int: Single-epoch element count.
        """
        if self._slice_range is not None:
            return len(self._slice_range)
        return len(self._backing_source)

    def __len__(self) -> int:
        """Return the number of elements in the dataset at this pipeline stage.

        Returns:
            int: Dataset length.
        """
        if self._step_kind == "source" or not self.parents:
            epochs = self._num_epochs if self._num_epochs is not None else 1
            return self._base_len() * epochs
        if self._step_kind == "mix" and self._selection_map is not None:
            return len(self._selection_map)
        parent_len = len(self.parents[0])
        if self._step_kind == "slice" and self._step_slice is not None:
            return len(self._step_slice)
        if self._step_kind == "repeat":
            return parent_len * self._step_epochs
        return parent_len

    def _get_item_with_epoch(self, index: int, epoch_offset: int = 0) -> ElementTree:
        """Evaluate the element at ``index`` propagating ``epoch_offset`` for repeated shuffles.

        Args:
            index (int): Non-negative logical index in ``[0, len(self))``.
            epoch_offset (int): Upstream repeat epoch index.

        Returns:
            ElementTree: Evaluated element (or ``None`` if filtered out).
        """
        if self._step_kind == "source" or not self.parents:
            return self._backing_source[index]
        if self._step_kind == "mix" and self._selection_map is not None:
            parent_idx, elem_idx = self._selection_map[index]
            return self.parents[parent_idx]._get_item_with_epoch(elem_idx, epoch_offset)

        parent = self.parents[0]
        if self._step_kind == "slice" and self._step_slice is not None:
            mapped_idx = self._step_slice[index]
            return parent._get_item_with_epoch(mapped_idx, epoch_offset)
        if self._step_kind == "repeat":
            p_len = len(parent)
            rep_epoch, pos_in_parent = divmod(index, p_len)
            next_epoch_offset = (epoch_offset + rep_epoch) if self._reassemble_every_epoch else epoch_offset
            return parent._get_item_with_epoch(pos_in_parent, next_epoch_offset)
        if self._step_kind == "shuffle":
            p_len = len(parent)
            if p_len > 1:
                base_seed = (
                    self._step_shuffle_seed
                    if self._step_shuffle_seed is not None
                    else (self._seed if self._seed is not None else 0)
                )
                per_epoch_seed = (base_seed + epoch_offset) % (2**32)
                shuffled_idx = index_shuffle(index, max_index=p_len - 1, seed=per_epoch_seed)
            else:
                shuffled_idx = index
            return parent._get_item_with_epoch(shuffled_idx, epoch_offset)

        raw_val = parent._get_item_with_epoch(index, epoch_offset)
        if raw_val is None:
            return None
        return _apply_single_index_operations(
            raw_val,
            index=index,
            seed=self._seed,
            operations=self._local_ops,
        )

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

        Raises:
            IndexError: If integer ``index`` is out of bounds.
        """
        if isinstance(index, SliceProtocol):
            return self.slice(index)
        total = len(self)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(f"MapDataset index {index} out of range for length {total}")
        return self._get_item_with_epoch(index, epoch_offset=0)

    def __iter__(self) -> DatasetIterator[T_co]:
        """Return an iterator over the dataset elements.

        Returns:
            DatasetIterator[T_co]: Iterator over this dataset.
        """
        return iter(self.to_iter_dataset())

    def seed(self, seed: int) -> MapDataset[T_co]:
        """Return a child ``MapDataset`` configured with default RNG ``seed``.

        Args:
            seed (int): Integer seed value.

        Returns:
            MapDataset[T_co]: Dataset with updated seed.
        """
        return self._clone_with(seed=seed, step_kind="ops")

    def shuffle(self, seed: int | None = None) -> MapDataset[T_co]:
        """Return a shuffled view of the dataset.

        Args:
            seed (int | None): Optional shuffle seed.

        Returns:
            MapDataset[T_co]: Shuffled dataset.
        """
        effective_seed = seed if seed is not None else self._seed
        return self._clone_with(
            shuffled=True,
            seed=effective_seed,
            step_kind="shuffle",
            step_shuffle_seed=effective_seed,
        )

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
        return self._clone_with(
            operations=(*self.operations, op),
            local_ops=(op,),
            step_kind="ops",
        )

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
        return self._clone_with(
            operations=(*self.operations, op),
            local_ops=(op,),
            step_kind="ops",
        )

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
            local_ops=(op,),
            seed=seed if seed is not None else self._seed,
            step_kind="ops",
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
            _local_ops_override=(op,),
            _step_kind="ops",
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
            self,
            _source_override=self._backing_source,
            _operations_override=(*self.operations, op),
            _seed_override=self._seed,
            _shuffled_override=self._shuffled,
            _slice_range_override=self._slice_range,
            _num_epochs_override=self._num_epochs,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            batch_op=op,
        )
        return child

    def slice(self, sl: SliceProtocol) -> MapDataset[T_co]:
        """Return a lazily sliced view of the dataset.

        Args:
            sl (SliceProtocol): Python slice object.

        Returns:
            MapDataset[T_co]: Sliced ``MapDataset``.
        """
        parent_len = len(self)
        start, stop, step = sl.indices(parent_len)
        step_range = range(start, stop, step)
        current_range = self._slice_range if self._slice_range is not None else range(len(self._backing_source))
        s0, s1, s2 = sl.indices(len(current_range))
        new_source_range = current_range[s0:s1:s2]
        return self._clone_with(
            slice_range=new_source_range,
            step_kind="slice",
            step_slice=step_range,
        )

    def repeat(
        self,
        num_epochs: int | None = None,
        *,
        reassemble_every_epoch: bool = True,
    ) -> MapDataset[T_co]:
        """Repeat the dataset lazily for ``num_epochs`` epochs.

        Args:
            num_epochs (int | None): Number of epochs to repeat (defaults to 1 if None).
            reassemble_every_epoch (bool): Whether to re-seed upstream shuffles each epoch.

        Returns:
            MapDataset[T_co]: Repeated ``MapDataset``.
        """
        epochs = num_epochs if num_epochs is not None else 1
        current_epochs = self._num_epochs if self._num_epochs is not None else 1
        return self._clone_with(
            num_epochs=current_epochs * epochs,
            step_kind="repeat",
            step_epochs=epochs,
            reassemble_every_epoch=reassemble_every_epoch,
        )

    def to_iter_dataset(
        self,
        read_options: ReadOptions | None = None,
        *,
        allow_nones: bool = False,
    ) -> IterDataset[T_co]:
        """Convert this ``MapDataset`` into an ``IterDataset``.

        Args:
            read_options (ReadOptions | None): Optional reading options.
            allow_nones (bool): Whether to yield ``None`` values produced by filter stages.

        Returns:
            IterDataset[T_co]: Iterable dataset wrapper.
        """
        return IterDataset(self, read_options=read_options, allow_nones=allow_nones)

    def mp_prefetch(
        self,
        options: MultiprocessingOptions | None = None,
        *,
        worker_init_fn: Callable[[int, int], None] | None = None,
        sequential_slice: bool = False,
    ) -> IterDataset[T_co]:
        """Convert this ``MapDataset`` to a multiprocessing-prefetched ``IterDataset``.

        Args:
            options (MultiprocessingOptions | None): Multiprocessing worker and buffer options.
            worker_init_fn (Callable[[int, int], None] | None): Optional worker initialization callback.
            sequential_slice (bool): Whether workers read sequential slices.

        Returns:
            IterDataset[T_co]: Prefetched ``IterDataset``.
        """
        resolved_opts = options if options is not None else MultiprocessingOptions()
        return IterDataset(
            self,
            multiprocessing_options=resolved_opts,
            worker_init_fn=worker_init_fn,
            sequential_slice=sequential_slice,
        )


class FilterDataset(MapDataset[T_co]):
    """``MapDataset`` subclass representing a filtered dataset stage."""


class BatchDataset(MapDataset[T_co]):
    """``MapDataset`` subclass representing a batched dataset stage."""

    def __init__(
        self,
        parents: (
            MapDataset[ElementTree] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[ElementTree]]
        ) = (),
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
            parents (MapDataset[ElementTree] | RandomAccessDataSource[T_co] | Sequence[T_co] | Sequence[MapDataset[ElementTree]]):
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
            _step_kind="batch",
        )
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self._batch_op: BatchOperation[ElementTree] = (
            batch_op if batch_op is not None else BatchOperation(batch_size, drop_remainder)
        )

    def _has_upstream_filter(self) -> bool:
        """Return True if any upstream operation is a ``FilterOperation``.

        Returns:
            bool: True when upstream contains a filter stage.
        """
        return any(isinstance(op, FilterOperation) for op in self.operations)

    def _compact_non_none_elements(self) -> list[ElementTree]:
        """Collect all non-None elements from the parent ``MapDataset``.

        Returns:
            list[ElementTree]: Filtered list of non-None parent elements.
        """
        parent = self.parents[0]
        parent_len = len(parent)
        items: list[ElementTree] = []
        for idx in range(parent_len):
            val = parent._get_item_with_epoch(idx, epoch_offset=0)
            if val is not None:
                items.append(val)
        return items

    def __len__(self) -> int:
        """Return the number of batches in the dataset.

        Returns:
            int: Total batch count.
        """
        if not self.parents:
            parent_len = self._base_len()
        elif self._has_upstream_filter():
            parent_len = len(self._compact_non_none_elements())
        else:
            parent_len = len(self.parents[0])
        if parent_len == 0:
            return 0
        if self.drop_remainder:
            return parent_len // self.batch_size
        return (parent_len + self.batch_size - 1) // self.batch_size

    def _get_item_with_epoch(self, index: int, epoch_offset: int = 0) -> ElementTree:
        """Evaluate the collated batch at ``index`` with ``epoch_offset``.

        Args:
            index (int): Non-negative batch index in ``[0, len(self))``.
            epoch_offset (int): Upstream repeat epoch offset.

        Returns:
            ElementTree: Collated batch at ``index``.
        """
        start_elem = index * self.batch_size
        if self._has_upstream_filter():
            compacted = self._compact_non_none_elements()
            end_elem = min(start_elem + self.batch_size, len(compacted))
            return self._batch_op._collate_tree(compacted[start_elem:end_elem])

        if not self.parents:
            return self._batch_op._collate_tree([])
        parent = self.parents[0]
        parent_len = len(parent)
        end_elem = min(start_elem + self.batch_size, parent_len)
        items: list[ElementTree] = [
            parent._get_item_with_epoch(elem_idx, epoch_offset) for elem_idx in range(start_elem, end_elem)
        ]
        return self._batch_op._collate_tree(items)

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
        return self._get_item_with_epoch(index, epoch_offset=0)


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


class _MapDatasetDataSource(RandomAccessDataSource[ElementTree]):
    """Internal data source adapter delegating ``__getitem__`` directly to a ``MapDataset``."""

    def __init__(self, dataset: MapDataset[ElementTree]) -> None:
        """Initialize ``_MapDatasetDataSource``.

        Args:
            dataset (MapDataset[ElementTree]): Backing ``MapDataset`` graph.
        """
        super().__init__()
        self._dataset = dataset

    def __getitem__(self, record_key: int) -> ElementTree:
        """Evaluate ``self._dataset[record_key]``.

        Args:
            record_key (int): Dataset index.

        Returns:
            ElementTree: Evaluated element (or ``None`` if filtered).
        """
        return self._dataset[record_key]

    def __repr__(self) -> str:
        """Return the underlying source representation.

        Returns:
            str: Formatted string representation.
        """
        return repr(self._dataset.source)


def _needs_graph_evaluation(ds: MapDataset[ElementTree]) -> bool:
    """Check whether ``ds`` requires parent-graph evaluation rather than flat ``DataLoader`` execution.

    Args:
        ds (MapDataset[ElementTree]): Dataset to inspect.

    Returns:
        bool: True if ``ds`` has multi-parent mixing or non-standard stage ordering.
    """
    seen_slice = False
    curr: MapDataset[ElementTree] | None = ds
    while curr is not None:
        if curr._step_kind == "mix":
            return True
        if curr._step_kind == "repeat" and not curr._reassemble_every_epoch:
            return True
        if curr._step_kind == "slice":
            seen_slice = True
        elif curr._step_kind in ("shuffle", "repeat") and seen_slice:
            return True
        curr = curr.parents[0] if curr.parents else None
    return False


class IterDataset(Generic[T_co]):
    """Iterable dataset supporting streaming transformations and checkpointing."""

    def __init__(
        self,
        parents: (MapDataset[T_co] | IterDataset[T_co] | Sequence[MapDataset[T_co] | IterDataset[T_co]]) = (),
        *,
        read_options: ReadOptions | None = None,
        allow_nones: bool = False,
        multiprocessing_options: MultiprocessingOptions | None = None,
        worker_init_fn: Callable[[int, int], None] | None = None,
        sequential_slice: bool = False,
    ) -> None:
        """Initialize ``IterDataset``.

        Args:
            parents (MapDataset[T_co] | IterDataset[T_co] | Sequence[MapDataset[T_co] | IterDataset[T_co]]):
                Parent ``MapDataset`` or ``IterDataset`` instance(s).
            read_options (ReadOptions | None): Optional reading options.
            allow_nones (bool): Whether to preserve ``None`` elements produced by filter stages.
            multiprocessing_options (MultiprocessingOptions | None): Optional multiprocessing worker configuration.
            worker_init_fn (Callable[[int, int], None] | None): Optional worker initialization callback.
            sequential_slice (bool): Whether workers read sequential slices.
        """
        self.read_options: ReadOptions | None = read_options
        self.allow_nones: bool = allow_nones
        self.multiprocessing_options: MultiprocessingOptions | None = multiprocessing_options
        self.worker_init_fn: Callable[[int, int], None] | None = worker_init_fn
        self.sequential_slice: bool = sequential_slice

        if isinstance(parents, MapDataset):
            self.dataset: MapDataset[T_co] = parents
        elif isinstance(parents, IterDataset):
            self.dataset = parents.dataset
            if self.read_options is None:
                self.read_options = parents.read_options
            self.allow_nones = self.allow_nones or parents.allow_nones
            self.multiprocessing_options = (
                self.multiprocessing_options
                if self.multiprocessing_options is not None
                else parents.multiprocessing_options
            )
            self.worker_init_fn = self.worker_init_fn if self.worker_init_fn is not None else parents.worker_init_fn
            self.sequential_slice = self.sequential_slice or parents.sequential_slice
        elif isinstance(parents, Sequence) and len(parents) > 0:
            first = parents[0]
            self.dataset = first if isinstance(first, MapDataset) else first.dataset
        else:
            self.dataset = MapDataset()

        worker_count = self.multiprocessing_options.num_workers if self.multiprocessing_options is not None else 0
        worker_buffer_size = (
            self.multiprocessing_options.per_worker_buffer_size if self.multiprocessing_options is not None else 1
        )
        enable_profiling = (
            self.multiprocessing_options.enable_profiling if self.multiprocessing_options is not None else False
        )
        if self.worker_init_fn is not None:
            effective_w = max(1, worker_count)
            for w_idx in range(effective_w):
                self.worker_init_fn(w_idx, effective_w)

        if _needs_graph_evaluation(self.dataset) or self.allow_nones:
            total_len = len(self.dataset)
            graph_source: RandomAccessDataSource[ElementTree] = _MapDatasetDataSource(self.dataset)
            graph_sampler: Sampler = IndexSampler(num_records=total_len) if total_len > 0 else Sampler()
            graph_ops: tuple[Operation[ElementTree, ElementTree], ...] = (
                () if self.allow_nones else (FilterOperation(lambda x: x is not None),)
            )
            self.dl: DataLoader[ElementTree] = DataLoader(
                data_source=graph_source,
                sampler=graph_sampler,
                operations=graph_ops,
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
                read_options=self.read_options,
                enable_profiling=enable_profiling,
            )
        else:
            base_len = self.dataset._base_len()
            effective_source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree] = (
                _SlicedDataSource(self.dataset._backing_source, self.dataset._slice_range)
                if self.dataset._slice_range is not None
                else self.dataset._backing_source
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
            self.dl = DataLoader(
                data_source=effective_source,
                sampler=sampler,
                operations=self.dataset.operations,
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
                read_options=self.read_options,
                enable_profiling=enable_profiling,
            )
        self._iter: DataLoaderIterator[ElementTree] = iter(self.dl)

    def _materialize_as_map_dataset(self) -> MapDataset[T_co]:
        """Return the backing ``MapDataset`` for this ``IterDataset``.

        Returns:
            MapDataset[T_co]: Backing ``MapDataset``.
        """
        return self.dataset

    def map(
        self,
        transform: (
            Callable[[ElementTree], ElementTree]
            | MapTransform[ElementTree, ElementTree]
            | Operation[ElementTree, ElementTree]
        ),
    ) -> IterDataset[T_out]:
        """Apply a streaming map transformation to each element of this ``IterDataset``.

        Args:
            transform (Callable[[ElementTree], ElementTree] | MapTransform[ElementTree, ElementTree] | Operation[ElementTree, ElementTree]):
                Map transform, operation, or callable.

        Returns:
            IterDataset[T_out]: Mapped ``IterDataset``.
        """
        base_ds = self._materialize_as_map_dataset()
        mapped_ds: MapDataset[T_out] = base_ds.map(transform)
        return IterDataset(
            mapped_ds,
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=self.multiprocessing_options,
            worker_init_fn=self.worker_init_fn,
            sequential_slice=self.sequential_slice,
        )

    def random_map(
        self,
        transform: (
            Callable[[ElementTree, RNGGeneratorProtocol], ElementTree]
            | RandomMapTransform[ElementTree, ElementTree]
            | RandomMapOperation[ElementTree, ElementTree]
        ),
        seed: int | None = None,
    ) -> IterDataset[T_out]:
        """Apply a streaming stochastic transformation to each element of this ``IterDataset``.

        Args:
            transform (Callable[[ElementTree, RNGGeneratorProtocol], ElementTree] | RandomMapTransform[ElementTree, ElementTree] | RandomMapOperation[ElementTree, ElementTree]):
                Random map transform, operation, or callable.
            seed (int | None): Optional RNG seed override.

        Returns:
            IterDataset[T_out]: Stochastically mapped ``IterDataset``.
        """
        base_ds = self._materialize_as_map_dataset()
        mapped_ds: MapDataset[T_out] = base_ds.random_map(transform, seed=seed)
        return IterDataset(
            mapped_ds,
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=self.multiprocessing_options,
            worker_init_fn=self.worker_init_fn,
            sequential_slice=self.sequential_slice,
        )

    def filter(
        self,
        transform: (Callable[[ElementTree], bool] | FilterTransform[ElementTree] | FilterOperation[ElementTree]),
    ) -> IterDataset[T_co]:
        """Filter elements of this ``IterDataset`` matching ``transform``.

        Args:
            transform (Callable[[ElementTree], bool] | FilterTransform[ElementTree] | FilterOperation[ElementTree]):
                Filter predicate callable, ``FilterTransform``, or ``FilterOperation``.

        Returns:
            IterDataset[T_co]: Filtered ``IterDataset``.
        """
        base_ds = self._materialize_as_map_dataset()
        filtered_ds = base_ds.filter(transform)
        return IterDataset(
            filtered_ds,
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=self.multiprocessing_options,
            worker_init_fn=self.worker_init_fn,
            sequential_slice=self.sequential_slice,
        )

    def batch(
        self,
        batch_size: int,
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[ElementTree]], ElementTree] | None = None,
    ) -> IterDataset[T_out]:
        """Batch consecutive elements of this ``IterDataset`` into groups of ``batch_size``.

        Args:
            batch_size (int): Number of elements per batch.
            drop_remainder (bool): Whether to drop incomplete final batches.
            batch_fn (Callable[[Sequence[ElementTree]], ElementTree] | None): Custom collation function.

        Returns:
            IterDataset[T_out]: Batched ``IterDataset``.
        """
        base_ds = self._materialize_as_map_dataset()
        batched_ds: BatchDataset[T_out] = base_ds.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            batch_fn=batch_fn,
        )
        return IterDataset(
            batched_ds,
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=self.multiprocessing_options,
            worker_init_fn=self.worker_init_fn,
            sequential_slice=self.sequential_slice,
        )

    @classmethod
    def mix(
        cls,
        datasets: Sequence[IterDataset[T_co] | MapDataset[T_co]],
        weights: Sequence[float] | None = None,
    ) -> IterDataset[T_co]:
        """Mix multiple ``IterDataset`` or ``MapDataset`` streams according to ``weights``.

        Args:
            datasets (Sequence[IterDataset[T_co] | MapDataset[T_co]]): Sequence of datasets to mix.
            weights (Sequence[float] | None): Optional sampling weights per dataset.

        Returns:
            IterDataset[T_co]: Mixed ``IterDataset``.
        """
        map_datasets: list[MapDataset[T_co]] = [
            d if isinstance(d, MapDataset) else d._materialize_as_map_dataset() for d in datasets
        ]
        mixed_map = MapDataset.mix(map_datasets, weights=weights)
        return cls(mixed_map)

    def prefetch(
        self,
        multiprocessing_options: MultiprocessingOptions,
        *,
        worker_init_fn: Callable[[int, int], None] | None = None,
    ) -> IterDataset[T_co]:
        """Configure multiprocessing prefetching on this ``IterDataset``.

        Args:
            multiprocessing_options (MultiprocessingOptions): Worker count and buffer configuration.
            worker_init_fn (Callable[[int, int], None] | None): Optional worker initialization callback.

        Returns:
            IterDataset[T_co]: Prefetched ``IterDataset``.
        """
        return IterDataset(
            self._materialize_as_map_dataset(),
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=multiprocessing_options,
            worker_init_fn=worker_init_fn,
            sequential_slice=self.sequential_slice,
        )

    def mp_prefetch(
        self,
        options: MultiprocessingOptions | None = None,
        *,
        worker_init_fn: Callable[[int, int], None] | None = None,
        sequential_slice: bool = False,
    ) -> IterDataset[T_co]:
        """Configure multiprocessing prefetching on this ``IterDataset``.

        Args:
            options (MultiprocessingOptions | None): Worker count and buffer configuration.
            worker_init_fn (Callable[[int, int], None] | None): Optional worker initialization callback.
            sequential_slice (bool): Whether workers read sequential slices.

        Returns:
            IterDataset[T_co]: Prefetched ``IterDataset``.
        """
        resolved_opts = options if options is not None else MultiprocessingOptions()
        return IterDataset(
            self._materialize_as_map_dataset(),
            read_options=self.read_options,
            allow_nones=self.allow_nones,
            multiprocessing_options=resolved_opts,
            worker_init_fn=worker_init_fn,
            sequential_slice=sequential_slice,
        )

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
    return ds._clone_with(
        operations=(*ds.operations, *normalized),
        local_ops=normalized,
        step_kind="ops",
    )


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
    "SelectionWithReplacementMap",
    "_MapDatasetDataSource",
    "_SlicedDataSource",
    "_needs_graph_evaluation",
    "apply_transformations",
    "get_element_spec",
]
