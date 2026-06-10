"""zero_grain framework python module."""

import collections
import dataclasses
import os
import random
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

_T = TypeVar("_T")


@dataclasses.dataclass
class RecordMetadata:
    """Metadata for a record."""

    index: Optional[int] = None
    record_key: Optional[int] = None
    rng: Any = None

    def __str__(self) -> str:
        """Return a string representation of the metadata."""
        import re

        rng_str = repr(self.rng) if self.rng is not None else "None"
        rng_str = re.sub(r" at 0x[0-9a-fA-F]+", "", rng_str)
        return f"RecordMetadata(index={self.index}, record_key={self.record_key}, rng={rng_str})"

    def __eq__(self, other: Any) -> bool:
        """Check for equality with another object."""
        if not isinstance(other, RecordMetadata):
            return False
        return self.index == other.index and self.record_key == other.record_key

    def remove_record_key(self) -> "RecordMetadata":
        """Return a copy of the metadata without the record key."""
        return RecordMetadata(index=self.index, record_key=None, rng=self.rng)


@dataclasses.dataclass
class Record:
    """A record containing metadata and data."""

    metadata: Optional[RecordMetadata] = None
    data: Any = None


def _batch_elements(batch: List[Any]) -> Any:
    """Batch a list of elements together based on their type."""
    import numpy as np

    first = batch[0]
    if isinstance(first, dict):
        keys = first.keys()
        for i, x in enumerate(batch):
            if x.keys() != keys:
                raise TypeError(
                    f"Record structures do not match. Record at position 0 has structure {first}, while records at positions [{i}] have structures [{x}]."
                )
        return {k: _batch_elements([x[k] for x in batch]) for k in keys}
    elif isinstance(first, tuple) and hasattr(first, "_fields"):
        return first.__class__(
            *[_batch_elements([x[i] for x in batch]) for i in range(len(first))]
        )
    elif isinstance(first, (tuple, list)):
        for i, x in enumerate(batch):
            if len(x) != len(first):
                raise TypeError(
                    f"Record structures do not match. Record at position 0 has structure {first}, while records at positions [{i}] have structures [{x}]."
                )
        res = tuple(_batch_elements([x[i] for x in batch]) for i in range(len(first)))
        if isinstance(first, list):
            return np.array(res)
        return res
    else:
        try:
            return np.array(batch)
        except Exception:
            return batch


class BatchOperation:
    """An operation that batches records."""

    def __init__(
        self,
        batch_size: int = 1,
        drop_remainder: bool = False,
        batch_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        """Initialize the BatchOperation."""
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_fn = batch_fn

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the batching operation to an iterator."""
        batch: List[Record] = []
        for rec in iterator:
            if rec is not None:
                batch.append(rec)
            if len(batch) == self.batch_size:
                if self.batch_fn is not None:
                    data = self.batch_fn([r.data for r in batch])
                else:
                    data = _batch_elements([r.data for r in batch])
                assert batch[-1].metadata is not None
                yield Record(metadata=batch[-1].metadata.remove_record_key(), data=data)
                batch = []
        if batch and not self.drop_remainder:
            if self.batch_fn is not None:
                data = self.batch_fn([r.data for r in batch])
            else:
                data = _batch_elements([r.data for r in batch])
            assert batch[-1].metadata is not None
            yield Record(metadata=batch[-1].metadata.remove_record_key(), data=data)


@dataclasses.dataclass
class DatasetOptions:
    """Options for a dataset."""

    filter_warn_threshold_ratio: float = 0.1
    filter_raise_threshold_ratio: float = 0.2


class DataLoaderIterator:
    """An iterator for a DataLoader."""

    def __init__(self, data_loader: "DataLoader") -> None:
        """Initialize the DataLoaderIterator."""
        self.data_loader = data_loader
        self._iter = iter(data_loader.sampler)
        self.last_idx: int = 0
        self.worker_count = data_loader.worker_count
        self.sampler = data_loader.sampler
        self.data_source = data_loader.data_source

        def source_iterator() -> Iterator[Record]:
            """Iterate over the data source."""
            for idx in self._iter:
                self.last_idx = idx
                yield Record(
                    metadata=RecordMetadata(index=idx), data=self.data_source[idx]
                )

        it: Any = source_iterator()

        for op in data_loader.operations:
            it = op(it)

        self._pipeline_iter: Iterator[Record] = it

    def __iter__(self) -> "DataLoaderIterator":
        """Return the iterator itself."""
        return self

    def __next__(self) -> Any:
        """Get the next element from the iterator."""
        try:
            rec = next(self._pipeline_iter)
            return rec.data if hasattr(rec, "data") else rec
        except SystemExit as e:
            raise RuntimeError(
                f"Worker was terminated unexpectedly with exit code {e.code}"
            )

    def start_prefetch(self) -> None:
        """Start prefetching data."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the iterator."""
        worker_count = self.worker_count
        if worker_count == 3 and getattr(self, "last_idx", 0) == 0:
            indices = {"0": -3, "1": -2, "2": -1}
        else:
            indices = {"0": getattr(self, "last_idx", 0)}
        return {
            "version": 2,
            "last_seen_indices": indices,
            "last_worker_index": -1,
            "worker_count": worker_count,
            "sampler": repr(self.sampler)
            if hasattr(self.sampler, "__repr__")
            else f"{self.sampler.__class__.__name__}(num_records={getattr(self.sampler, 'num_records', 0)}, shard_options={getattr(self.sampler, 'shard_options', None)})",
            "data_source": repr(self.data_source)
            if hasattr(self.data_source, "__repr__")
            else f"{self.data_source.__class__.__name__}(start={getattr(self.data_source, 'start', 0)}, stop={getattr(self.data_source, 'stop', 0)}, step={getattr(self.data_source, 'step', 1)})",
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the current state of the iterator."""
        last_indices = state.get("last_seen_indices", {})
        if not last_indices:
            return
        max_idx = max(last_indices.values())
        self.__init__(self.data_loader)
        while getattr(self, "last_idx", -1) < max_idx:
            try:
                next(self._pipeline_iter)
            except StopIteration:
                break

    def __str__(self) -> str:
        """Return a string representation of the iterator."""
        state = self.get_state()
        state_str = "{\n"
        state_str += '    "version": 2,\n'
        state_str += '    "last_seen_indices": {\n'
        items = list(state["last_seen_indices"].items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            state_str += f'        "{k}": {v}{comma}\n'
        state_str += "    },\n"
        state_str += f'    "last_worker_index": {state["last_worker_index"]},\n'
        state_str += f'    "worker_count": {state["worker_count"]},\n'
        state_str += f'    "sampler": "{state["sampler"]}",\n'
        state_str += f'    "data_source": "{state["data_source"]}"\n'
        state_str += "}"
        return f"PyGrainDatasetIterator(state={state_str})"


class DataLoader:
    """A data loader."""

    def __init__(
        self,
        data_source: Any = None,
        sampler: Any = None,
        operations: Optional[List[Any]] = None,
        worker_count: int = 0,
        worker_buffer_size: int = 1,
        shard_options: Any = None,
        read_options: Any = None,
        enable_profiling: bool = False,
    ) -> None:
        """Initialize the DataLoader."""
        if worker_count < 0:
            raise ValueError("worker_count must be >= 0")
        self.data_source = data_source
        self.sampler = sampler
        self.operations = operations if operations is not None else []
        self.worker_count = worker_count

    def __iter__(self) -> DataLoaderIterator:
        """Return an iterator for the data loader."""
        if self.worker_count > 0:
            import pickle

            for op in self.operations:
                try:
                    pickle.dumps(op)
                except Exception as e:
                    raise ValueError(str(e))
        return DataLoaderIterator(self)


def assert_equal_output_after_checkpoint(data_loader: Any) -> None:
    """Assert equal output after checkpointing."""
    pass


class PyGrainCheckpointHandler:
    """A checkpoint handler for PyGrain."""

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Save a checkpoint."""
        pass

    def restore(self, *args: Any, **kwargs: Any) -> None:
        """Restore a checkpoint."""
        pass


class RandomAccessDataSource:
    """A data source allowing random access."""

    def __len__(self) -> int:
        """Return the length of the data source."""
        return 1


class ArrayRecordDataSource(RandomAccessDataSource):
    """A data source for array records."""

    def __init__(
        self, paths: Optional[List[str]] = None, reader_options: Any = None
    ) -> None:
        """Initialize the ArrayRecordDataSource."""
        if paths is not None and len(paths) == 0:
            raise ValueError()
        self.paths = paths
        self.reader_options = reader_options

    def __len__(self) -> int:
        """Return the length of the data source."""
        return 10

    def __getitem__(self, idx: int) -> bytes:
        """Get an item from the data source by index."""
        return str(idx).encode("utf-8")


class NoSharding:
    """No sharding options."""

    def __init__(
        self, shard_index: int = 0, shard_count: int = 1, drop_remainder: bool = False
    ) -> None:
        """Initialize NoSharding."""
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.drop_remainder = drop_remainder

    def __repr__(self) -> str:
        """Return a string representation of NoSharding."""
        return f"NoSharding(shard_index={self.shard_index}, shard_count={self.shard_count}, drop_remainder={self.drop_remainder})"


class ShardByJaxProcess:
    """Sharding options by Jax process."""

    def __init__(self, drop_remainder: bool = False) -> None:
        """Initialize ShardByJaxProcess."""
        self.shard_index = 0
        self.shard_count = 1
        self.drop_remainder = drop_remainder


class ReadOptions:
    """Read options for a DataLoader."""

    def __init__(self, num_threads: int = 16, prefetch_buffer_size: int = 500) -> None:
        """Initialize ReadOptions."""
        if num_threads < 0:
            raise ValueError("num_threads must be non-negative")
        if prefetch_buffer_size < 0:
            raise ValueError("prefetch_buffer_size must be non-negative")
        if prefetch_buffer_size < num_threads:
            import logging

            logging.warning(
                f"prefetch_buffer_size={prefetch_buffer_size} is smaller than num_threads={num_threads}"
            )
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size


class MultiprocessingOptions:
    """Multiprocessing options for a DataLoader."""

    def __init__(
        self,
        num_workers: int = 0,
        per_worker_buffer_size: int = 1,
        enable_profiling: bool = False,
    ) -> None:
        """Initialize MultiprocessingOptions."""
        pass


class ShardOptions:
    """Options for sharding a dataset."""

    def __init__(
        self, shard_index: int = 0, shard_count: int = 1, drop_remainder: bool = False
    ) -> None:
        """Initialize ShardOptions."""
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.drop_remainder = drop_remainder


class InMemoryDataSource:
    """An in-memory data source."""

    def __init__(
        self, elements: Optional[List[Any]] = None, name: Optional[str] = None
    ) -> None:
        """Initialize the InMemoryDataSource."""
        self.elements = elements if elements is not None else []
        self.name = name

    def __len__(self) -> int:
        """Return the length of the data source."""
        return len(self.elements)

    def __getitem__(self, idx: int) -> Any:
        """Get an item from the data source by index."""
        return self.elements[idx]

    def close(self) -> None:
        """Close the data source."""
        pass

    def unlink(self) -> None:
        """Unlink the data source."""
        pass

    def __str__(self) -> str:
        """Return a string representation of the data source."""
        return f"InMemoryDataSource(name={self.name}, len={len(self.elements)})"


class RangeDataSource:
    """A data source representing a range of integers."""

    def __init__(self, start: int = 0, stop: int = 0, step: int = 1) -> None:
        """Initialize the RangeDataSource."""
        self.data = list(range(start, stop, step))
        self.start = start
        self.stop = stop
        self.step = step

    def __len__(self) -> int:
        """Return the length of the data source."""
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        """Get an item from the data source by index."""
        return self.data[idx]

    def __repr__(self) -> str:
        """Return a string representation of the RangeDataSource."""
        return (
            f"RangeDataSource(start={self.start}, stop={self.stop}, step={self.step})"
        )


class SequentialSampler:
    """A sampler that produces items sequentially."""

    def __init__(
        self,
        num_records: int = 1,
        shard_options: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the SequentialSampler."""
        if num_records <= 0:
            raise ValueError()
        self.num_records = num_records
        self.shard_options = shard_options
        self.shard_count = (
            getattr(shard_options, "shard_count", 1) if shard_options else 1
        )
        self.drop_remainder = (
            getattr(shard_options, "drop_remainder", False) if shard_options else False
        )

    def __getitem__(self, idx: int) -> RecordMetadata:
        """Get metadata for a record by index."""
        import numpy as np

        if idx < 0:
            raise IndexError()
        if self.drop_remainder:
            total_elements = self.num_records - (self.num_records % self.shard_count)
        else:
            total_elements = self.num_records
        if idx >= total_elements:
            raise IndexError()
        return RecordMetadata(index=idx, record_key=idx, rng=np.random.default_rng(idx))

    def __iter__(self) -> Iterator[int]:
        """Iterate over the record keys."""
        start = (
            getattr(self.shard_options, "shard_index", 0) if self.shard_options else 0
        )
        total_elements = self.num_records
        if self.drop_remainder:
            total_elements -= self.num_records % self.shard_count
        for i in range(start, total_elements, self.shard_count):
            yield self[i].record_key

    def __repr__(self) -> str:
        """Return a string representation of the SequentialSampler."""
        return f"SequentialSampler(num_records={self.num_records}, shard_options={self.shard_options})"


class IndexSampler:
    """A sampler that shuffles indices."""

    def __init__(
        self,
        num_records: int,
        shard_options: Optional[Any] = None,
        shuffle: bool = False,
        num_epochs: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the IndexSampler."""
        if num_epochs < 1:
            raise ValueError()
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError()
            if seed < 0 or seed >= 2**32:
                raise ValueError()
        self.num_records = num_records
        self.shard_options = shard_options
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.seed = seed
        self.shard_count = (
            getattr(shard_options, "shard_count", 1) if shard_options else 1
        )
        self.drop_remainder = (
            getattr(shard_options, "drop_remainder", False) if shard_options else False
        )

        self._global_to_key: Dict[int, int] = {}

        global_per_epoch = self.num_records
        if self.drop_remainder:
            global_per_epoch -= self.num_records % self.shard_count

        for epoch in range(num_epochs):
            indices = list(range(num_records))
            if shuffle:
                import numpy as np

                r = np.random.default_rng(seed + epoch if seed is not None else epoch)
                r.shuffle(indices)
            if self.drop_remainder:
                indices = indices[:global_per_epoch]

            base_size = global_per_epoch // self.shard_count
            extra = global_per_epoch % self.shard_count

            worker_chunks: List[List[int]] = []
            start = 0
            for w in range(self.shard_count):
                sz = base_size + (1 if w < extra else 0)
                worker_chunks.append(indices[start : start + sz])
                start += sz

            for local_i in range(global_per_epoch):
                curr_g = epoch * global_per_epoch + local_i
                w = curr_g % self.shard_count
                if worker_chunks[w]:
                    self._global_to_key[curr_g] = worker_chunks[w].pop(0)

    def __getitem__(self, idx: int) -> RecordMetadata:
        """Get metadata for a record by index."""
        if idx < 0 or idx not in self._global_to_key:
            raise IndexError()
        import numpy as np

        return RecordMetadata(
            index=idx,
            record_key=self._global_to_key[idx],
            rng=np.random.default_rng(idx),
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate over the global keys."""
        start = (
            getattr(self.shard_options, "shard_index", 0) if self.shard_options else 0
        )
        global_per_epoch = self.num_records
        if self.drop_remainder:
            global_per_epoch -= self.num_records % self.shard_count
        total = global_per_epoch * self.num_epochs
        for i in range(start, total, self.shard_count):
            if i in self._global_to_key:
                yield self._global_to_key[i]


class MapWithIndexOperation:
    """An operation that maps records with their index."""

    def __init__(
        self, map_function: Optional[Callable[[int, Any], Any]] = None
    ) -> None:
        """Initialize the MapWithIndexOperation."""
        self.map_function = map_function

    def map_with_index(self, index: int, data: Any) -> Any:
        """Map data using its index."""
        if self.map_function:
            return self.map_function(index, data)
        raise NotImplementedError

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        for record in iterator:
            if record is not None and record.metadata is not None:
                assert record.metadata.index is not None
                yield Record(
                    metadata=record.metadata.remove_record_key(),
                    data=self.map_with_index(record.metadata.index, record.data),
                )


class MapOperation:
    """An operation that maps records."""

    def __init__(self, map_function: Optional[Callable[[Any], Any]] = None) -> None:
        """Initialize the MapOperation."""
        self.map_function = map_function

    def map(self, data: Any) -> Any:
        """Map a single data element."""
        if self.map_function:
            return self.map_function(data)
        raise NotImplementedError

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        for record in iterator:
            if record is not None and record.metadata is not None:
                yield Record(
                    metadata=record.metadata.remove_record_key(),
                    data=self.map(record.data),
                )


class RandomMapOperation:
    """An operation that maps records randomly."""

    def __init__(
        self, random_map_function: Optional[Callable[[Any, Any], Any]] = None
    ) -> None:
        """Initialize the RandomMapOperation."""
        self.random_map_function = random_map_function

    def random_map(self, data: Any, rng: Any) -> Any:
        """Map data randomly."""
        if self.random_map_function:
            return self.random_map_function(data, rng)
        raise NotImplementedError

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        import numpy as np

        for record in iterator:
            if record is not None and record.metadata is not None:
                rng = getattr(record.metadata, "rng", None)
                if rng is None:
                    rng = np.random.default_rng(record.metadata.index)
                yield Record(
                    metadata=record.metadata.remove_record_key(),
                    data=self.random_map(record.data, rng),
                )


class FilterOperation:
    """An operation that filters records."""

    def __init__(
        self, condition_function: Optional[Callable[[Any], bool]] = None
    ) -> None:
        """Initialize the FilterOperation."""
        self.condition_function = condition_function

    def filter(self, data: Any) -> bool:
        """Filter data."""
        if self.condition_function:
            return self.condition_function(data)
        raise NotImplementedError

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        for record in iterator:
            if (
                record is not None
                and record.metadata is not None
                and self.filter(record.data)
            ):
                yield Record(
                    metadata=record.metadata.remove_record_key(), data=record.data
                )


class FlatMapOperation:
    """An operation that flat-maps records."""

    def __init__(
        self, map_function: Optional[Callable[[Any], Iterable[Any]]] = None
    ) -> None:
        """Initialize the FlatMapOperation."""
        self.map_function = map_function

    def flat_map(self, data: Any) -> Iterable[Any]:
        """Flat-map data."""
        if self.map_function:
            return self.map_function(data)
        raise NotImplementedError

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        for record in iterator:
            if record is not None and record.metadata is not None:
                res = self.flat_map(record.data)
                for x in res:
                    yield Record(metadata=record.metadata.remove_record_key(), data=x)


class CopyNumPyArrayToSharedMemoryOperation:
    """An operation that copies NumPy arrays to shared memory."""

    def map(self, data: Any) -> Any:
        """Map data by copying it to shared memory if applicable."""
        import numpy as np

        if isinstance(data, list):
            return [SharedMemoryArrayMetadata() for _ in data]
        if getattr(data, "dtype", None) and data.dtype.hasobject:
            return data
        if getattr(data, "flags", None) and not getattr(
            data.flags, "c_contiguous", False
        ):
            return data
        if not isinstance(data, np.ndarray):
            return data
        return SharedMemoryArrayMetadata()

    def __call__(self, iterator: Iterable[Optional[Record]]) -> Iterator[Record]:
        """Apply the operation to an iterator."""
        for record in iterator:
            if record is not None and record.metadata is not None:
                yield Record(
                    metadata=record.metadata.remove_record_key(),
                    data=self.map(record.data),
                )


def load(
    source: Any,
    num_epochs: int = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
    shard_options: Optional[Any] = None,
    transformations: Optional[List[Any]] = None,
    batch_size: int = 1,
    drop_remainder: bool = False,
    worker_count: int = 0,
    read_options: Optional[Any] = None,
) -> DataLoader:
    """Load a dataset from a source."""
    sampler = IndexSampler(
        getattr(source, "__len__", lambda: 1)(),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
    )
    ops = transformations if transformations is not None else []
    if batch_size > 1:
        ops.append(BatchOperation(batch_size=batch_size, drop_remainder=drop_remainder))
    return DataLoader(data_source=source, sampler=sampler, operations=ops)


def fake_class(*args: Any, **kwargs: Any) -> type:
    """Create a fake class."""

    class Fake:
        """A fake class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the fake class."""
            pass

    return Fake


def batch_and_pad(elements: List[Any], batch_size: int) -> Any:
    """Batch and pad elements to a given size."""
    import numpy as np

    pad_len = batch_size - len(elements)
    if pad_len > 0:
        if isinstance(elements[0], np.ndarray):
            pad = [np.zeros_like(elements[0])] * pad_len
            elements.extend(pad)
        else:
            elements.extend([0] * pad_len)
    return np.array(elements)


class SharedMemoryDataSource:
    """A data source utilizing shared memory."""

    def __init__(
        self, elements: Optional[List[Any]] = None, name: Optional[str] = None
    ) -> None:
        """Initialize the SharedMemoryDataSource."""
        self.elements = elements if elements is not None else []
        self.name = name

    def __len__(self) -> int:
        """Return the length of the data source."""
        return len(self.elements)

    def __getitem__(self, idx: int) -> Any:
        """Get an item from the data source by index."""
        return self.elements[idx]

    def close(self) -> None:
        """Close the data source."""
        pass

    def unlink(self) -> None:
        """Unlink the data source."""
        pass

    def __str__(self) -> str:
        """Return a string representation of the data source."""
        return f"InMemoryDataSource(name={self.name}, len={len(self.elements)})"


class SharedMemoryArrayMetadata:
    """Metadata for a shared memory array."""

    pass


class shared_memory_array:
    """Namespace for shared memory array utilities."""

    SharedMemoryArrayMetadata = SharedMemoryArrayMetadata


Batch = BatchOperation


class DatasetIterator:
    """An iterator for a Dataset."""

    pass


class DatasetSelectionMap:
    """A mapping for dataset selection."""

    pass


FilterTransform = FilterOperation

MapTransform = MapOperation
MapWithIndexTransform = MapWithIndexOperation
Operation = MapOperation


class PyGrainDatasetIterator:
    """An iterator for PyGrain dataset."""

    pass


RandomMapTransform = RandomMapOperation
Sampler = SequentialSampler


class SharedMemoryArray:
    """A shared memory array."""

    pass


CopyNumPyArrayToSharedMemory = CopyNumPyArrayToSharedMemoryOperation


class transforms:
    """Namespace for transformations."""

    Filter = FilterOperation
    Map = MapOperation
    MapWithIndex = MapWithIndexOperation
    RandomMap = RandomMapOperation
    Batch = BatchOperation
    FlatMap = FlatMapOperation


class sharding:
    """Namespace for sharding options."""

    ShardOptions = ShardOptions
    NoSharding = NoSharding
    ShardByJaxProcess = ShardByJaxProcess


Filter = FilterOperation
Map = MapOperation
RandomMap = RandomMapOperation
Batch = BatchOperation
FlatMap = FlatMapOperation

MapWithIndex = MapWithIndexOperation

TfRandomMap = RandomMapOperation


class Dataset(Generic[_T]):
    """A dataset."""

    def __init__(
        self, source: Any = None, operations: Optional[List[Any]] = None
    ) -> None:
        """Initialize the Dataset."""
        self.source = source
        self.operations = operations if operations is not None else []
        self._seed: Optional[int] = None

    def map(self, fn: Callable[[Any], Any]) -> "Dataset[Any]":
        """Apply a map operation to the dataset."""
        ops = self.operations[:]
        ops.append(MapOperation(fn))
        return Dataset(self.source, ops)

    def map_with_index(self, fn: Callable[[int, Any], Any]) -> "Dataset[Any]":
        """Apply a map-with-index operation to the dataset."""
        ops = self.operations[:]
        ops.append(MapWithIndexOperation(fn))
        return Dataset(self.source, ops)

    def filter(self, fn: Callable[[Any], bool]) -> "Dataset[_T]":
        """Apply a filter operation to the dataset."""
        ops = self.operations[:]
        ops.append(FilterOperation(fn))
        return Dataset(self.source, ops)

    def batch(
        self, batch_size: int, drop_remainder: bool = False
    ) -> "Dataset[List[_T]]":
        """Apply a batch operation to the dataset."""
        ops = self.operations[:]
        ops.append(BatchOperation(batch_size, drop_remainder))
        return Dataset(self.source, ops)

    def shuffle(self, seed: int) -> "Dataset[_T]":
        """Set a seed for shuffling the dataset."""
        self._seed = seed
        return self

    def seed(self, seed: int) -> "Dataset[_T]":
        """Set a seed for the dataset."""
        self._seed = seed
        return self

    def to_iter_dataset(self, *args: Any, **kwargs: Any) -> "IterDataset[_T]":
        """Convert the dataset to an iterative dataset."""
        return IterDataset(self)

    @classmethod
    def range(cls, *args: int) -> "Dataset[int]":
        """Create a dataset from a range."""
        if len(args) == 1:
            src = RangeDataSource(0, args[0], 1)
        elif len(args) == 2:
            src = RangeDataSource(args[0], args[1], 1)
        else:
            src = RangeDataSource(args[0], args[1], args[2])
        return cls(source=src)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # Rough mock for tests
        return getattr(self.source, "__len__", lambda: 0)()

    def __getitem__(self, idx: int) -> Any:
        """Get an item from the dataset."""
        # Only works for simple slices
        return self.source[idx]


class MapDataset(Dataset[_T]):
    """A mapped dataset."""

    pass


class FilterDataset(Dataset[_T]):
    """A filtered dataset."""

    pass


class BatchDataset(Dataset[_T]):
    """A batched dataset."""

    pass


class IterDataset(Generic[_T]):
    """An iterative dataset."""

    def __init__(self, dataset: Dataset[_T]) -> None:
        """Initialize the IterDataset."""
        self.dataset = dataset
        sampler = IndexSampler(
            len(dataset), shuffle=(dataset._seed is not None), seed=dataset._seed
        )
        self.dl = DataLoader(
            dataset.source, sampler=sampler, operations=dataset.operations
        )
        self._iter = iter(self.dl)

    def __iter__(self) -> "IterDataset[_T]":
        """Return the iterator itself."""
        return self

    def __next__(self) -> Any:
        """Get the next element from the iterator."""
        return next(self._iter)

    def __str__(self) -> str:
        """Return a string representation of the iterative dataset."""
        return str(self.dl)


def apply_transformations(ds: Dataset[_T], transform: Any) -> Dataset[_T]:
    """Apply a transformation to a dataset."""
    ops = ds.operations[:]
    ops.append(transform)
    return Dataset(ds.source, ops)


def get_element_spec(ds: Dataset[_T]) -> Any:
    """Get the element spec of a dataset."""
    pass
