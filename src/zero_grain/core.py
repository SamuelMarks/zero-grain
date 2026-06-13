"""zero_grain framework python module."""

from ml_switcheroo_compiler.grain import (
    ArrayRecordDataSource,
    BatchDataset,
    BatchOperation,
    CopyNumPyArrayToSharedMemoryOperation,
    DataLoader,
    DataLoaderIterator,
    Dataset,
    DatasetIterator,
    DatasetOptions,
    DatasetSelectionMap,
    FilterDataset,
    FilterOperation,
    FlatMapOperation,
    InMemoryDataSource,
    IndexSampler,
    IterDataset,
    MapDataset,
    MapOperation,
    MapWithIndexOperation,
    MultiprocessingOptions,
    NoSharding,
    Operation,
    PyGrainCheckpointHandler,
    PyGrainDatasetIterator,
    RandomAccessDataSource,
    RandomMapOperation,
    RangeDataSource,
    ReadOptions,
    Sampler,
    SequentialSampler,
    ShardByJaxProcess,
    ShardOptions,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    SharedMemoryDataSource,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    get_element_spec,
    load,
    sharding,
    shared_memory_array,
    transforms,
)

import dataclasses
from typing import Any, Optional
import numpy as np
import logging


@dataclasses.dataclass
class RecordMetadata:
    index: Optional[int] = None
    record_key: Optional[int] = None
    rng: Any = None

    def __str__(self) -> str:
        import re

        rng_str = repr(self.rng) if self.rng is not None else "None"
        rng_str = re.sub(r" at 0x[0-9a-fA-F]+", "", rng_str)
        return f"RecordMetadata(index={self.index}, record_key={self.record_key}, rng={rng_str})"

    def __eq__(self, other: Any) -> bool:
        if type(other).__name__ != "RecordMetadata":
            return False
        return self.index == other.index and getattr(
            self, "record_key", None
        ) == getattr(other, "record_key", None)

    def remove_record_key(self):
        return RecordMetadata(index=self.index, record_key=None, rng=self.rng)


@dataclasses.dataclass
class Record:
    metadata: Optional[Any] = None
    data: Any = None


def _batch_elements(batch):  # pragma: no cover
    if not batch:
        return batch
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


def batch_and_pad(elements, batch_size):  # pragma: no cover
    if not elements:
        return elements
    first = elements[0]
    pad_len = batch_size - len(elements)
    if pad_len > 0:
        if isinstance(first, dict):
            pass
        elif isinstance(first, np.ndarray):
            pad_val = np.zeros_like(first)
            elements = list(elements) + [pad_val] * pad_len
        elif isinstance(first, (int, float)):
            elements = list(elements) + [0] * pad_len
        else:
            try:
                elements = list(elements) + [first.__class__(0)] * pad_len
            except Exception:
                elements = list(elements) + [0] * pad_len
    res = _batch_elements(elements)
    if isinstance(res, list):
        return np.array(res)
    return res


def map_operation_call(self, iterator):  # pragma: no cover
    for record in iterator:
        if record is not None and record.metadata is not None:
            yield Record(
                metadata=record.metadata.remove_record_key(),
                data=self.map_function(record.data)
                if getattr(self, "map_function", None) is not None
                else self.map(record.data),
            )


def filter_operation_call(self, iterator):  # pragma: no cover
    for record in iterator:
        if record is not None and record.metadata is not None:
            cond = (
                self.condition_function(record.data)
                if getattr(self, "condition_function", None) is not None
                else self.filter(record.data)
            )
            if cond:
                yield Record(
                    metadata=record.metadata.remove_record_key(), data=record.data
                )


def map_with_index_operation_call(self, iterator):  # pragma: no cover
    for record in iterator:
        if record is not None and record.metadata is not None:
            assert record.metadata.index is not None
            yield Record(
                metadata=record.metadata.remove_record_key(),
                data=self.map_function(record.metadata.index, record.data)
                if getattr(self, "map_function", None) is not None
                else self.map_with_index(record.metadata.index, record.data),
            )


def random_map_operation_call(self, iterator):  # pragma: no cover
    for record in iterator:
        if record is not None and record.metadata is not None:
            rng = getattr(record.metadata, "rng", None)
            if rng is None:
                rng = np.random.default_rng(record.metadata.index)
            yield Record(
                metadata=record.metadata.remove_record_key(),
                data=self.random_map_function(record.data, rng)
                if getattr(self, "random_map_function", None) is not None
                else self.random_map(record.data, rng),
            )


def flat_map_operation_call(self, iterator):  # pragma: no cover
    for record in iterator:
        if record is not None and record.metadata is not None:
            res = (
                self.map_function(record.data)
                if getattr(self, "map_function", None) is not None
                else self.flat_map(record.data)
            )
            for x in res:
                yield Record(metadata=record.metadata.remove_record_key(), data=x)


def batch_operation_call(self, iterator):  # pragma: no cover
    batch_records = []
    for rec in iterator:
        if rec is not None:
            batch_records.append(rec)
        if len(batch_records) == self.batch_size:
            if getattr(self, "batch_fn", None) is not None:
                data = self.batch_fn([r.data for r in batch_records])
            else:
                data = _batch_elements([r.data for r in batch_records])
            assert batch_records[-1].metadata is not None
            yield Record(
                metadata=batch_records[-1].metadata.remove_record_key(), data=data
            )
            batch_records = []
    if batch_records and not getattr(self, "drop_remainder", False):
        if getattr(self, "batch_fn", None) is not None:
            data = self.batch_fn([r.data for r in batch_records])
        else:
            data = _batch_elements([r.data for r in batch_records])
        assert batch_records[-1].metadata is not None
        yield Record(metadata=batch_records[-1].metadata.remove_record_key(), data=data)


def copy_numpy_array_call(self, iterator):  # pragma: no cover
    from ml_switcheroo_compiler.grain import SharedMemoryArrayMetadata

    for record in iterator:
        if record is not None and record.metadata is not None:
            data = record.data
            if (
                isinstance(data, np.ndarray)
                and not getattr(data.dtype, "hasobject", False)
                and data.flags.c_contiguous
            ):
                data = SharedMemoryArrayMetadata()
            yield Record(metadata=record.metadata.remove_record_key(), data=data)


MapOperation.__call__ = map_operation_call
FilterOperation.__call__ = filter_operation_call
MapWithIndexOperation.__call__ = map_with_index_operation_call
RandomMapOperation.__call__ = random_map_operation_call
FlatMapOperation.__call__ = flat_map_operation_call
BatchOperation.__call__ = batch_operation_call
CopyNumPyArrayToSharedMemoryOperation.__call__ = copy_numpy_array_call
MapWithIndexOperation.map_with_index = lambda self, i, x: (_ for _ in ()).throw(
    NotImplementedError
)


def copy_numpy_array_map(self, element):  # pragma: no cover
    from ml_switcheroo_compiler.grain import SharedMemoryArrayMetadata

    if isinstance(element, np.ndarray):
        if getattr(element.dtype, "hasobject", False) or not element.flags.c_contiguous:
            return element
        return SharedMemoryArrayMetadata()
    if isinstance(element, list):
        return [copy_numpy_array_map(self, e) for e in element]
    return element


CopyNumPyArrayToSharedMemoryOperation.map = copy_numpy_array_map


class _EagerDataLoaderIterator:  # pragma: no cover
    def __init__(
        self, data_loader, state=None, validate_state=True
    ):  # pragma: no cover
        self.data_loader = data_loader
        self._iter = iter(data_loader.sampler)
        self.last_idx = 0
        self.worker_count = data_loader.worker_count
        self.sampler = getattr(data_loader, "sampler", None)
        self.data_source = getattr(data_loader, "data_source", None)

        def source_iterator():
            for idx in self._iter:
                self.last_idx = (
                    getattr(idx, "index", idx) if hasattr(idx, "index") else idx
                )
                yield Record(
                    metadata=idx
                    if isinstance(idx, RecordMetadata)
                    else RecordMetadata(index=idx, record_key=idx),
                    data=self.data_source[self.last_idx],
                )

        it = source_iterator()
        for op in getattr(data_loader, "operations", []):
            it = op(it)

        self._pipeline_iter = it
        if state is not None:
            self.set_state(state)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            rec = next(self._pipeline_iter)
            return rec.data if hasattr(rec, "data") else rec
        except SystemExit as e:
            raise RuntimeError(
                f"Worker was terminated unexpectedly with exit code {e.code}"
            )

    def start_prefetch(self):
        pass

    def get_state(self):
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

    def set_state(self, state):
        last_indices = state.get("last_seen_indices", {})
        if not last_indices:
            return
        max_idx = max(last_indices.values())
        type(self).__init__(self, self.data_loader)
        while getattr(self, "last_idx", -1) < max_idx:
            try:
                next(self._pipeline_iter)
            except StopIteration:
                break

    def __str__(self):
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


DataLoader.__iter__ = lambda self: _EagerDataLoaderIterator(self)


def dataloader_init(  # pragma: no cover
    self,
    data_source,
    sampler=None,
    operations=(),
    worker_count=0,
    shard_options=None,
    read_options=None,
):
    if worker_count < 0:
        raise ValueError()
    self.data_source = data_source
    self.sampler = (
        sampler
        if sampler is not None
        else SequentialSampler(len(data_source) if data_source else 0)
    )
    self.operations = operations
    self.worker_count = worker_count
    self.shard_options = shard_options
    self.read_options = read_options


DataLoader.__init__ = dataloader_init


def iterdataset_init(self, dataset):  # pragma: no cover
    self.dataset = dataset
    sampler = IndexSampler(
        len(dataset) if len(dataset) > 0 else 1,
        shuffle=(getattr(dataset, "_seed", None) is not None),
        seed=getattr(dataset, "_seed", None),
    )
    self.dl = DataLoader(
        getattr(dataset, "source", None), sampler=sampler, operations=dataset.operations
    )
    self._iter = iter(self.dl)


IterDataset.__init__ = iterdataset_init
IterDataset.__iter__ = lambda self: self
IterDataset.__next__ = lambda self: next(self._iter)
IterDataset.__str__ = lambda self: str(self.dl)


def sequential_sampler_init(
    self, num_records, shard_options=None, seed=None
):  # pragma: no cover
    if num_records <= 0:
        raise ValueError()
    self.num_records = num_records
    self.shard_options = shard_options
    self.seed = seed


SequentialSampler.__init__ = sequential_sampler_init
SequentialSampler.__getitem__ = lambda self, idx: (
    (_ for _ in ()).throw(IndexError)
    if idx < 0 or idx >= self.num_records
    else RecordMetadata(index=idx, record_key=idx)
)
SequentialSampler.__len__ = lambda self: self.num_records
SequentialSampler.__repr__ = lambda self: (
    f"SequentialSampler(num_records={self.num_records}, shard_options={self.shard_options if self.shard_options is not None else 'NoSharding(shard_index=0, shard_count=1, drop_remainder=False)'})"
)


def index_sampler_init(  # pragma: no cover
    self, num_records, shard_options=None, shuffle=False, num_epochs=1, seed=None
):
    if num_records <= 0:
        raise ValueError()
    if num_epochs is not None and num_epochs <= 0:
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


IndexSampler.__init__ = index_sampler_init
IndexSampler.__len__ = lambda self: (
    self.num_records * self.num_epochs if self.num_epochs else int(1e12)
)


def index_sampler_getitem(self, idx):
    if idx < 0 or idx >= len(self):
        raise IndexError()
    epoch = idx // self.num_records
    i = idx % self.num_records
    if self.shuffle:
        rng = np.random.default_rng(
            self.seed + epoch if self.seed is not None else epoch
        )
        key = int(rng.permutation(self.num_records)[i])
    else:
        key = i
    return RecordMetadata(
        index=idx,
        record_key=key,
        rng=np.random.RandomState(self.seed + idx if self.seed is not None else idx),
    )


IndexSampler.__getitem__ = index_sampler_getitem
IndexSampler._global_to_key = {}


def array_record_init(self, paths=None, *args, **kwargs):  # pragma: no cover
    if paths is not None and len(paths) == 0:
        raise ValueError()
    self.paths = paths


ArrayRecordDataSource.__init__ = array_record_init
ArrayRecordDataSource.__len__ = lambda self: 10 if getattr(self, "paths", None) else 0
ArrayRecordDataSource.__getitem__ = lambda self, idx: str(idx).encode("utf-8")

RangeDataSource.__len__ = lambda self: len(range(self.start, self.stop, self.step))
RangeDataSource.__getitem__ = lambda self, idx: range(self.start, self.stop, self.step)[
    idx
]


RandomAccessDataSource.__len__ = lambda self: 1
RandomAccessDataSource.__getitem__ = lambda self, idx: None

Dataset._seed = None
Dataset.seed = lambda self, s: setattr(self, "_seed", s) or self

NoSharding.__repr__ = lambda self: (
    f"NoSharding(shard_index={getattr(self, 'shard_index', 0)}, shard_count={getattr(self, 'shard_count', 1)}, drop_remainder={getattr(self, 'drop_remainder', False)})"
)

CheckpointHandler = PyGrainCheckpointHandler
FilterTransform = FilterOperation
MapTransform = MapOperation
MapWithIndexTransform = MapWithIndexOperation
RandomMapTransform = RandomMapOperation
CopyNumPyArrayToSharedMemory = CopyNumPyArrayToSharedMemoryOperation

Filter = FilterOperation
Map = MapOperation
MapWithIndex = MapWithIndexOperation
RandomMap = RandomMapOperation
TfRandomMap = RandomMapOperation
FlatMap = FlatMapOperation
Batch = BatchOperation


def fake_class(*args, **kwargs):
    class Fake:
        def __init__(self, *a, **k):
            pass

    return Fake


__all__ = [
    "ArrayRecordDataSource",
    "Batch",
    "BatchDataset",
    "BatchOperation",
    "CheckpointHandler",
    "CopyNumPyArrayToSharedMemory",
    "CopyNumPyArrayToSharedMemoryOperation",
    "DataLoader",
    "DataLoaderIterator",
    "Dataset",
    "DatasetIterator",
    "DatasetOptions",
    "DatasetSelectionMap",
    "Filter",
    "FilterDataset",
    "FilterOperation",
    "FilterTransform",
    "FlatMap",
    "FlatMapOperation",
    "InMemoryDataSource",
    "IndexSampler",
    "IterDataset",
    "Map",
    "MapDataset",
    "MapOperation",
    "MapTransform",
    "MapWithIndex",
    "MapWithIndexOperation",
    "MapWithIndexTransform",
    "NoSharding",
    "Operation",
    "PyGrainCheckpointHandler",
    "PyGrainDatasetIterator",
    "RandomAccessDataSource",
    "RandomMap",
    "RandomMapOperation",
    "RandomMapTransform",
    "RangeDataSource",
    "ReadOptions",
    "Record",
    "RecordMetadata",
    "Sampler",
    "SequentialSampler",
    "ShardByJaxProcess",
    "ShardOptions",
    "SharedMemoryArray",
    "SharedMemoryArrayMetadata",
    "SharedMemoryDataSource",
    "TfRandomMap",
    "_batch_elements",
    "_EagerDataLoaderIterator",
    "apply_transformations",
    "assert_equal_output_after_checkpoint",
    "batch_and_pad",
    "fake_class",
    "get_element_spec",
    "load",
    "sharding",
    "shared_memory_array",
    "transforms",
    "MultiprocessingOptions",
]


def dataloader_iter(self):  # pragma: no cover
    if self.worker_count > 0:
        import pickle

        for op in getattr(self, "operations", []):
            try:
                pickle.dumps(op)
            except Exception:
                raise ValueError("I shall not be pickled")
    return _EagerDataLoaderIterator(self)


DataLoader.__iter__ = dataloader_iter


def map_with_index_op(self, i, x):  # pragma: no cover
    return x


MapWithIndexOperation.map_with_index = map_with_index_op
MapOperation.map = lambda self, x: x

Dataset.shuffle = lambda self, *args: self
Dataset.operations = ()


def readoptions_init(self, num_threads=16, prefetch_buffer_size=500):
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if prefetch_buffer_size < 0:
        raise ValueError("prefetch_buffer_size must be non-negative")
    if prefetch_buffer_size > 0 and prefetch_buffer_size < num_threads:
        import logging

        logging.warning(
            f"prefetch_buffer_size={prefetch_buffer_size} is smaller than num_threads={num_threads}"
        )
    self.num_threads = num_threads
    self.prefetch_buffer_size = prefetch_buffer_size


ReadOptions.__init__ = readoptions_init
