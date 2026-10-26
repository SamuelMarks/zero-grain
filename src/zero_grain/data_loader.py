"""DataLoader, DataLoaderIterator, and ``load`` convenience API for zero-grain."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import pickle
from typing import Generic, Union
from typing_extensions import TypeAlias
from zero_grain._types import CheckpointStateDict, ElementTree, T_co, T_in, T_out
from zero_grain.data_sources import InMemoryDataSource, RandomAccessDataSource
from zero_grain.operations import (
    BatchOperation,
    FilterOperation,
    FlatMapOperation,
    MapOperation,
    MapWithIndexOperation,
    Operation,
    RandomMapOperation,
)
from zero_grain.options import DataLoaderOptions, LoadOptions, ReadOptions
from zero_grain.record import Record, RecordMetadata
from zero_grain.samplers import IndexSampler, Sampler, SequentialSampler
from zero_grain.sharding import NoSharding, ShardOptions
from zero_grain.transforms import (
    Batch,
    FilterTransform,
    FlatMapTransform,
    MapTransform,
    MapWithIndexTransform,
    RandomMapTransform,
)

TransformationSpec: TypeAlias = Union[
    Batch[T_in, T_out],
    MapTransform[T_in, T_out],
    RandomMapTransform[T_in, T_out],
    FilterTransform[T_in],
    FlatMapTransform[T_in, T_out],
    MapWithIndexTransform[T_in, T_out],
    Operation[T_in, T_out],
    Callable[[T_in], T_out],
]


def normalize_operation(
    transform: TransformationSpec[ElementTree, ElementTree],
) -> Operation[ElementTree, ElementTree]:
    """Convert a ``Transform``, ``Operation``, or callable into an ``Operation``.

    Args:
        transform (TransformationSpec[ElementTree, ElementTree]): Transform specification to normalize.

    Returns:
        Operation[ElementTree, ElementTree]: Executable pipeline ``Operation``.
    """
    if isinstance(transform, Operation):
        return transform
    if isinstance(transform, Batch):
        return BatchOperation(
            batch_size=transform.batch_size,
            drop_remainder=transform.drop_remainder,
            batch_fn=transform.batch_fn,
        )
    if isinstance(transform, MapTransform):
        return MapOperation(map_function=transform)
    if isinstance(transform, MapWithIndexTransform):
        return MapWithIndexOperation(map_function=transform)
    if isinstance(transform, RandomMapTransform):
        return RandomMapOperation(random_map_function=transform)
    if isinstance(transform, FilterTransform):
        return FilterOperation(condition_function=transform)
    if isinstance(transform, FlatMapTransform):
        return FlatMapOperation(flat_map_function=transform)
    return MapOperation(map_function=transform)


class DataLoader(Generic[T_co]):
    """Composable data loader applying a ``Sampler`` and ``operations`` over a ``data_source``."""

    def __init__(
        self,
        data_source: RandomAccessDataSource[T_co] | Sequence[T_co] | None = None,
        sampler: Sampler | Sequence[int] | None = None,
        operations: Sequence[TransformationSpec[ElementTree, ElementTree]] = (),
        worker_count: int | None = 0,
        worker_buffer_size: int = 1,
        shard_options: ShardOptions | None = None,
        read_options: ReadOptions | None = None,
        enable_profiling: bool = False,
        options: DataLoaderOptions | None = None,
    ) -> None:
        """Initialize ``DataLoader``.

        Args:
            data_source (RandomAccessDataSource[T_co] | Sequence[T_co] | None):
                Source providing indexed access to records.
            sampler (Sampler | Sequence[int] | None):
                Sampler generating ``RecordMetadata`` indices.
            operations (Sequence[TransformationSpec[ElementTree, ElementTree]]):
                Ordered sequence of transforms or operations.
            worker_count (int | None): Number of parallel worker streams (must be ``>= 0``).
            worker_buffer_size (int): Per-worker prefetch buffer size.
            shard_options (ShardOptions | None): Sharding configuration.
            read_options (ReadOptions | None): Reader thread and buffer options.
            enable_profiling (bool): Whether profiling is enabled.
            options (DataLoaderOptions | None): Optional legacy ``DataLoaderOptions``.

        Raises:
            ValueError: If ``worker_count < 0``.
        """
        effective_workers = worker_count if worker_count is not None else 0
        if options is not None and not effective_workers:
            effective_workers = options.worker_count
        if effective_workers < 0:
            raise ValueError(f"worker_count must be non-negative, got {effective_workers}")

        effective_read_options = (
            options.read_options if (read_options is None and options is not None) else read_options
        )

        self.data_source: RandomAccessDataSource[T_co] | Sequence[T_co] = (
            data_source if data_source is not None else InMemoryDataSource([])
        )
        if sampler is not None:
            self.sampler: Sampler | Sequence[int] = sampler
        else:
            ds_len = len(self.data_source)
            self.sampler = SequentialSampler(num_records=ds_len) if ds_len > 0 else Sampler()
        self.operations: tuple[Operation[ElementTree, ElementTree], ...] = tuple(
            normalize_operation(op) for op in operations
        )
        self.worker_count = effective_workers
        self.worker_buffer_size = worker_buffer_size
        self.shard_options = shard_options
        self.read_options = effective_read_options
        self.enable_profiling = enable_profiling

    def __iter__(self) -> DataLoaderIterator[T_co]:
        """Create a new ``DataLoaderIterator`` over this ``DataLoader``.

        Returns:
            DataLoaderIterator[T_co]: Fresh iterator instance.

        Raises:
            ValueError: If ``worker_count > 0`` and an operation cannot be pickled.
        """
        if self.worker_count > 0:
            for op in self.operations:
                try:
                    pickle.dumps(op)
                except (pickle.PickleError, ValueError, TypeError, AttributeError) as exc:
                    raise ValueError("I shall not be pickled") from exc
        return DataLoaderIterator(self)


class _WorkerStream:
    """Internal per-worker pipeline execution stream tracking consumed sampler positions."""

    def __init__(
        self,
        worker_idx: int,
        worker_total: int,
        data_source: RandomAccessDataSource[ElementTree] | Sequence[ElementTree],
        sampler_records: Sequence[RecordMetadata],
        operations: Sequence[Operation[ElementTree, ElementTree]],
        initial_last_seen: int,
    ) -> None:
        """Initialize ``_WorkerStream``.

        Args:
            worker_idx (int): Worker index in ``[0, worker_total)``.
            worker_total (int): Total number of worker streams.
            data_source (RandomAccessDataSource[ElementTree] | Sequence[ElementTree]): Backing data source.
            sampler_records (Sequence[RecordMetadata]): Full sequence of sampler metadata items.
            operations (Sequence[Operation[ElementTree, ElementTree]]): Normalized pipeline operations.
            initial_last_seen (int): Initial ``last_seen_indices`` value for this worker.
        """
        self.worker_idx = worker_idx
        self.worker_total = worker_total
        self.data_source = data_source
        self.sampler_records = sampler_records
        self.operations = operations
        self.last_seen_index = initial_last_seen
        self._emitted_count = 0
        self._iter: Iterator[Record[ElementTree]] = self._build_pipeline(min_sampler_index=initial_last_seen)

    def _build_pipeline(
        self,
        min_sampler_index: int,
    ) -> Iterator[Record[ElementTree]]:
        """Construct the chained generator pipeline starting after ``min_sampler_index``.

        Args:
            min_sampler_index (int): Sampler index up to which records have already been consumed.

        Returns:
            Iterator[Record[ElementTree]]: Chained record iterator for this worker.
        """

        def _source_gen() -> Iterator[Record[ElementTree]]:
            """Yield source records assigned to this worker after ``min_sampler_index``.

            Yields:
                Record[ElementTree]: Source records for this worker.
            """
            pos = self.worker_idx
            total_records = len(self.sampler_records)
            while pos < total_records:
                meta = self.sampler_records[pos]
                meta_idx = meta.index if meta.index is not None else pos
                if meta_idx > min_sampler_index:
                    self.last_seen_index = meta_idx
                    key = meta.record_key if meta.record_key is not None else meta_idx
                    raw_data: ElementTree = self.data_source[key]
                    record_meta = RecordMetadata(
                        index=meta_idx,
                        record_key=key,
                        rng=meta.rng,
                    )
                    yield Record[ElementTree](metadata=record_meta, data=raw_data)
                pos += self.worker_total

        stream: Iterator[Record[ElementTree]] = _source_gen()
        for op in self.operations:
            stream = op(stream)
        return stream

    def next_element(self) -> ElementTree:
        """Advance this worker stream by one output element.

        Returns:
            ElementTree: Next transformed element from this worker.

        Raises:
            StopIteration: When this worker stream is exhausted.
            RuntimeError: If a transform raises ``SystemExit`` when ``worker_total > 0``.
        """
        try:
            rec = next(self._iter)
        except SystemExit as exc:
            raise RuntimeError(f"Worker was terminated unexpectedly with exit code {exc.code}") from exc
        self._emitted_count += 1
        return rec.data


class DataLoaderIterator(Generic[T_co]):
    """Stateful, checkpointable iterator over a ``DataLoader``."""

    def __init__(
        self,
        data_loader: DataLoader[T_co] | None = None,
        state: CheckpointStateDict | None = None,
        validate_state: bool = True,
    ) -> None:
        """Initialize ``DataLoaderIterator``.

        Args:
            data_loader (DataLoader[T_co] | None): Parent ``DataLoader`` to iterate.
            state (CheckpointStateDict | None): Optional initial checkpoint state to restore.
            validate_state (bool): Whether to validate ``state`` on restoration.
        """
        self.data_loader: DataLoader[T_co] = data_loader if data_loader is not None else DataLoader()
        self.validate_state = validate_state
        self.worker_count = self.data_loader.worker_count
        self.sampler = self.data_loader.sampler
        self.data_source: RandomAccessDataSource[T_co] | Sequence[T_co] = self.data_loader.data_source
        self._prefetch_started = False

        self._sampler_records: list[RecordMetadata] = self._materialize_sampler()
        self._effective_workers = max(1, self.worker_count)
        self._last_worker_index = -1
        self._next_worker_turn = 0
        self._step_offset = 0
        self._workers: list[_WorkerStream] = []
        self._reset_workers(initial_indices=None)

        if state is not None:
            self.set_state(state)

    def _materialize_sampler(self) -> list[RecordMetadata]:
        """Collect all ``RecordMetadata`` entries from ``self.sampler``.

        Returns:
            list[RecordMetadata]: Ordered list of ``RecordMetadata`` items.
        """
        records: list[RecordMetadata] = []
        for item in self.sampler:
            if isinstance(item, RecordMetadata):
                records.append(item)
            else:
                records.append(RecordMetadata(index=int(item), record_key=int(item)))
        return records

    def _default_last_seen_indices(self) -> dict[str, int]:
        """Compute the initial ``last_seen_indices`` dictionary before iteration starts.

        Returns:
            dict[str, int]: Worker index string mapped to initial seen index.
        """
        if self.worker_count <= 0:
            return {"0": -1}
        return {str(i): -self.worker_count + i for i in range(self.worker_count)}

    def _reset_workers(
        self,
        initial_indices: dict[str, int] | None,
    ) -> None:
        """Initialize or reset the internal ``_WorkerStream`` instances.

        Args:
            initial_indices (dict[str, int] | None): Optional per-worker starting indices.
        """
        defaults = self._default_last_seen_indices()
        indices = initial_indices if initial_indices is not None else defaults
        self._effective_workers = max(1, self.worker_count)
        self._workers = [
            _WorkerStream(
                worker_idx=w_idx,
                worker_total=self._effective_workers,
                data_source=self.data_source,
                sampler_records=self._sampler_records,
                operations=self.data_loader.operations,
                initial_last_seen=indices.get(str(w_idx), defaults.get(str(w_idx), -1)),
            )
            for w_idx in range(self._effective_workers)
        ]

    def __iter__(self) -> DataLoaderIterator[T_co]:
        """Return ``self`` as an iterator.

        Returns:
            DataLoaderIterator[T_co]: Self iterator reference.
        """
        return self

    def __next__(self) -> ElementTree:
        """Return the next transformed element from the pipeline stream.

        Returns:
            ElementTree: Next dataset element.

        Raises:
            StopIteration: When all worker streams are exhausted.
        """
        if not self._sampler_records:
            raise StopIteration

        num_w = len(self._workers)
        for offset in range(num_w):
            w_idx = (self._next_worker_turn + offset) % num_w
            try:
                val = self._workers[w_idx].next_element()
            except StopIteration:
                continue
            self._next_worker_turn = (w_idx + 1) % num_w
            if self.worker_count > 0:
                self._last_worker_index = w_idx
            self._step_offset += 1
            return val
        raise StopIteration

    def start_prefetch(self) -> None:
        """Start background prefetching (no-op in synchronous execution mode)."""
        self._prefetch_started = True

    def get_state(self) -> CheckpointStateDict:
        """Capture the current iterator checkpoint state.

        Returns:
            CheckpointStateDict: Dictionary suitable for ``set_state`` restoration.
        """
        has_flatmap = any(isinstance(op, FlatMapOperation) for op in self.data_loader.operations)
        defaults = self._default_last_seen_indices()
        if has_flatmap:
            indices: dict[str, int] = {
                str(w_idx): (worker._emitted_count - 1 if worker._emitted_count > 0 else defaults.get(str(w_idx), -1))
                for w_idx, worker in enumerate(self._workers)
            }
        else:
            indices = {str(w_idx): worker.last_seen_index for w_idx, worker in enumerate(self._workers)}
        return {
            "version": 2,
            "last_seen_indices": indices,
            "last_worker_index": self._last_worker_index,
            "worker_count": self.worker_count,
            "sampler": repr(self.sampler),
            "data_source": repr(self.data_source),
        }

    def _validate_checkpoint_state(self, state: CheckpointStateDict) -> None:
        """Validate ``state`` against this iterator's configuration when ``validate_state`` is True.

        Args:
            state (CheckpointStateDict): State dictionary to validate.

        Raises:
            ValueError: If ``worker_count``, ``sampler``, or ``data_source`` in ``state`` mismatches.
        """
        if not self.validate_state:
            return
        if "worker_count" in state and state["worker_count"] != self.worker_count:
            raise ValueError(
                f"Checkpoint worker_count {state['worker_count']} does not match "
                f"DataLoader worker_count {self.worker_count}"
            )
        state_sampler = state.get("sampler")
        if state_sampler and state_sampler != repr(self.sampler):
            raise ValueError(f"Checkpoint sampler {state_sampler} does not match DataLoader sampler {self.sampler!r}")
        state_source = state.get("data_source")
        if state_source and state_source != repr(self.data_source):
            raise ValueError(
                f"Checkpoint data_source {state_source} does not match DataLoader data_source {self.data_source!r}"
            )

    def set_state(self, state: CheckpointStateDict) -> None:
        """Restore the iterator to a previously captured checkpoint ``state``.

        Args:
            state (CheckpointStateDict): Checkpoint dictionary from ``get_state()``.
        """
        self._validate_checkpoint_state(state)
        last_indices = state.get("last_seen_indices", {})
        if not last_indices:
            return
        last_worker = state.get("last_worker_index", -1)
        has_flatmap = any(isinstance(op, FlatMapOperation) for op in self.data_loader.operations)
        max_seen = max(last_indices.values())
        if max_seen < 0:
            self._last_worker_index = -1
            self._next_worker_turn = 0
            self._step_offset = 0
            self._reset_workers(initial_indices=None)
            return

        if has_flatmap:
            self._reset_workers(initial_indices=None)
            total_steps = 0
            for w_idx, worker in enumerate(self._workers):
                w_seen = last_indices.get(str(w_idx), -1)
                w_target = w_seen + 1 if w_seen >= 0 else 0
                for _ in range(w_target):
                    try:
                        worker.next_element()
                        total_steps += 1
                    except StopIteration:
                        break
            self._last_worker_index = last_worker
            self._next_worker_turn = (
                (last_worker + 1) % self.worker_count if (self.worker_count > 0 and last_worker >= 0) else 0
            )
            self._step_offset = total_steps
            return

        self._last_worker_index = last_worker
        self._next_worker_turn = (
            (last_worker + 1) % self.worker_count if (self.worker_count > 0 and last_worker >= 0) else 0
        )
        self._step_offset = max_seen + 1
        self._reset_workers(initial_indices=dict(last_indices))

    def __str__(self) -> str:
        """Return the formatted checkpoint state string matching ``PyGrainDatasetIterator``.

        Returns:
            str: Multi-line JSON-formatted iterator state string.
        """
        state = self.get_state()
        indices = state.get("last_seen_indices", {})
        items = list(indices.items())
        lines = [
            "PyGrainDatasetIterator(state={",
            '    "version": 2,',
            '    "last_seen_indices": {',
        ]
        for idx, (k, v) in enumerate(items):
            comma = "," if idx < len(items) - 1 else ""
            lines.append(f'        "{k}": {v}{comma}')
        lines.append("    },")
        lines.append(f'    "last_worker_index": {state.get("last_worker_index", -1)},')
        lines.append(f'    "worker_count": {state.get("worker_count", 0)},')
        lines.append(f'    "sampler": "{state.get("sampler", "")}",')
        lines.append(f'    "data_source": "{state.get("data_source", "")}"')
        lines.append("})")
        return chr(10).join(lines)


PyGrainDatasetIterator = DataLoaderIterator
_EagerDataLoaderIterator = DataLoaderIterator


def load(
    data_source: RandomAccessDataSource[T_co] | Sequence[T_co] | None = None,
    *,
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: ShardOptions | None = None,
    transformations: Sequence[TransformationSpec[ElementTree, ElementTree]] = (),
    batch_size: int | None = None,
    drop_remainder: bool = False,
    worker_count: int | None = 0,
    read_options: ReadOptions | None = None,
    options: LoadOptions | None = None,
    source: RandomAccessDataSource[T_co] | Sequence[T_co] | None = None,
) -> DataLoader[T_co]:
    """Construct a ``DataLoader`` for ``data_source``.

    Args:
        data_source (RandomAccessDataSource[T_co] | Sequence[T_co] | None):
            Input random-access data source or sequence.
        num_epochs (int | None): Number of epochs to iterate.
        shuffle (bool): Whether to shuffle records within each epoch.
        seed (int | None): Seed used when ``shuffle=True``.
        shard_options (ShardOptions | None): Sharding configuration (defaults to ``NoSharding()``).
        transformations (Sequence[TransformationSpec[ElementTree, ElementTree]]): Sequence of transforms to apply.
        batch_size (int | None): Optional batch size to append a ``BatchOperation``.
        drop_remainder (bool): Whether to drop trailing incomplete batches.
        worker_count (int | None): Number of worker processes.
        read_options (ReadOptions | None): Reader threading and buffer options.
        options (LoadOptions | None): Optional legacy ``LoadOptions`` container.
        source (RandomAccessDataSource[T_co] | Sequence[T_co] | None):
            Optional legacy keyword alias for ``data_source``.

    Returns:
        DataLoader[T_co]: Configured ``DataLoader`` instance.
    """
    resolved_sharding = shard_options if shard_options is not None else NoSharding()
    if options is not None:
        num_epochs = options.num_epochs if num_epochs is None else num_epochs
        shuffle = options.shuffle or shuffle
        seed = options.seed if seed is None else seed
        batch_size = options.batch_size if batch_size is None else batch_size
        drop_remainder = options.drop_remainder or drop_remainder
        worker_count = options.worker_count if not worker_count else worker_count
        read_options = options.read_options if read_options is None else read_options

    chosen_source = data_source if data_source is not None else source
    effective_source: RandomAccessDataSource[T_co] | Sequence[T_co] = (
        chosen_source if chosen_source is not None else InMemoryDataSource([])
    )
    source_len = len(effective_source)
    sampler: Sampler = (
        IndexSampler(
            num_records=source_len,
            shard_options=resolved_sharding,
            shuffle=shuffle,
            num_epochs=num_epochs,
            seed=seed,
        )
        if source_len > 0
        else Sampler()
    )
    ops_list: list[TransformationSpec[ElementTree, ElementTree]] = list(transformations)
    if batch_size is not None:
        ops_list.append(BatchOperation(batch_size=batch_size, drop_remainder=drop_remainder))
    return DataLoader(
        data_source=effective_source,
        sampler=sampler,
        operations=ops_list,
        worker_count=worker_count,
        shard_options=resolved_sharding,
        read_options=read_options,
    )


__all__ = [
    "DataLoader",
    "DataLoaderIterator",
    "PyGrainDatasetIterator",
    "TransformationSpec",
    "_EagerDataLoaderIterator",
    "load",
    "normalize_operation",
]
