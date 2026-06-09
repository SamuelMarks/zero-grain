"""Module docstring."""

from __future__ import annotations

try:
    import zero_jax
except ImportError:  # pragma: no cover
    zero_jax = None
import numpy

import collections.abc
import os
from typing import (
    Any,
    Callable,
    Sequence,
    TypeVar,
    Generic,
    Tuple,
    List,
    Dict,
    Optional,
    Union,
)

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")
T = TypeVar("T")


class NoSharding:
    """Docstring for NoSharding."""

    def __init__(
        self, shard_index: int = 0, shard_count: int = 1, drop_remainder: bool = False
    ):
        """Docstring for __init__."""
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.drop_remainder = drop_remainder

    def __repr__(self):
        """Docstring for __repr__."""
        return "NoSharding(shard_index=0, shard_count=1, drop_remainder=False)"


class ArrayRecordDataSource:
    """Data source for ArrayRecord files."""

    def __init__(
        self,
        paths: str
        | os.PathLike
        | array_record.python.array_record_data_source.FileInstruction
        | collections.abc.Sequence[
            str
            | os.PathLike
            | array_record.python.array_record_data_source.FileInstruction
        ],
        reader_options: dict[str, str] | None = None,
    ):
        """Docstring for __init__."""
        self.paths = paths
        self.reader_options = reader_options
        self._records = []
        paths_list = [paths] if isinstance(paths, (str, os.PathLike)) else paths
        for path in paths_list:
            if isinstance(path, (str, os.PathLike)):
                try:
                    with open(path, "rb") as f:
                        self._records.extend(f.readlines())
                except Exception:
                    pass

    def __len__(self):
        """Docstring for __len__."""
        return len(self._records)

    def __getitem__(self, idx):
        """Docstring for __getitem__."""
        return self._records[idx]  # pragma: no cover


class Batch:
    """Batch(batch_size: 'int', drop_remainder: 'bool' = False, batch_fn: 'Callable[[Sequence[Any]], Any] | None' = None)"""

    def __init__(
        self,
        batch_size: int,
        drop_remainder: bool = False,
        batch_fn: collections.abc.Callable[[collections.abc.Sequence[Any]], Any]
        | None = None,
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class BatchOperation:
    """Batches input examples into batches with given batch_size.

    Internally, examples are interpreted as JAX Pytrees. To batch records
    together, they must be of the same structure. Corresponding leaves are batched
    together into NumPy arrays. For more info about PyTrees, please refer to:
    https://jax.readthedocs.io/en/latest/pytrees.html.

    By default, we put Numpy arrays into Shared Memory. For more info about shared
    memory, please refer to:
    https://docs.python.org/3/library/multiprocessing.shared_memory.html"""

    def __init__(
        self,
        batch_size: int,
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[~_IN]], ~_OUT] | None = None,
    ):
        """Docstring for __init__."""
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_fn = batch_fn

    def __call__(self, records: Sequence[Record]) -> Record:
        """Docstring for __call__."""
        if not records:
            return None
        # Inherit metadata from the first record in the batch
        meta = records[0].metadata
        data_list = [r.data for r in records if r is not None and r.data is not None]

        if self.batch_fn is not None:
            batched_data = self.batch_fn(data_list)
        else:
            batched_data = tree_collate(data_list) if tree_map else data_list

        return Record(metadata=meta, data=batched_data)


class CheckpointHandler:
    """Orbax CheckpointHandler for PyGrain iterators."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class DataLoader:
    """DataLoader loads and transforms input data."""

    def __init__(
        self,
        *,
        data_source: grain._src.python.data_sources.RandomAccessDataSource,
        sampler: grain._src.python.samplers.Sampler,
        operations: Sequence[
            grain._src.core.transforms.Batch
            | grain._src.core.transforms.MapTransform
            | grain._src.core.transforms.RandomMapTransform
            | grain._src.core.transforms.TfRandomMapTransform
            | grain._src.core.transforms.Filter
            | grain._src.core.transforms.FlatMapTransform
            | grain._src.core.transforms.MapWithIndex
            | grain._src.python.operations.Operation
        ] = (),
        worker_count: int | None = 0,
        worker_buffer_size: int = 1,
        shard_options: grain._src.core.sharding.ShardOptions | None = None,
        read_options: grain._src.python.options.ReadOptions | None = None,
        enable_profiling: bool = False,
    ):
        """Docstring for __init__."""
        self.data_source = data_source
        self.sampler = sampler
        self.operations = operations
        self.worker_count = worker_count
        self.worker_buffer_size = worker_buffer_size
        self.shard_options = shard_options
        self.read_options = read_options
        self.enable_profiling = enable_profiling

    def __iter__(self):
        """Docstring for __iter__."""
        return DataLoaderIterator(data_loader=self, state=None)


class DataLoaderIterator:
    """DataLoader iterator providing get/set state functionality.

    This is the only iterator we expose to users. It wraps underlying
    MultipleProcessIterator. In order to set state, it recreates the underlying
    iterator fresh with a new state.

    Checkpointing for DataLoaderIterator:
    DataLoaderIterator uses GrainPool, which distributes RecordMetadata from
    produced records among worker processes in a round robin fashion. Generally,
    some workers can process more elements than others at a given training step.
    Checkpointing logic goes as follows:
    1) With each output batch produced, GrainPool emits the worker_index of The
       worker that processed the batch.
    2) DataLoaderIterator keeps track of the last_seen_index at each worker.
    3) When restoring from a state, DataLoaderIterator checks what is the
       minimum last_seen_index (among the last seen indices for all workers.) and
       which worker processed that index. GrainPool is instructed to start
       distributing indices to the next worker."""

    def __init__(
        self,
        data_loader: grain._src.python.data_loader.DataLoader,
        state: dict[str, Any] | None = None,
        validate_state: bool = True,
    ):
        """Docstring for __init__."""
        self.data_loader = data_loader
        self.state = state
        self.validate_state = validate_state

        # Initialize the pipeline state
        self._sampler_iter = iter(self.data_loader.sampler)
        self._last_seen_indices = {}

        # Buffer for elements (especially for batching)
        self._buffer = []
        if self.state is not None:
            self.set_state(self.state)

    def __iter__(self):
        """Docstring for __iter__."""
        return self  # pragma: no cover

    def _apply_ops(self, record):
        """Docstring for _apply_ops."""
        for op in self.data_loader.operations:
            if record is None:
                break  # pragma: no cover
            if getattr(op, "batch_size", None):
                # We handle batching differently
                continue
            record = op(record)
        return record

    def __next__(self):
        # Determine if there's a batch operation at the end
        """Docstring for __next__."""
        batch_op = None
        if self.data_loader.operations and getattr(
            self.data_loader.operations[-1], "batch_size", None
        ):
            batch_op = self.data_loader.operations[-1]

        if batch_op:
            while len(self._buffer) < batch_op.batch_size:
                try:
                    idx = next(self._sampler_iter)
                    raw_data = self.data_loader.data_source[idx]
                    meta = RecordMetadata(index=idx)
                    rec = Record(metadata=meta, data=raw_data)
                    rec = self._apply_ops(rec)
                    if rec is not None:
                        self._buffer.append(rec)
                        self._last_seen_indices[0] = idx
                except StopIteration:
                    break

            if not self._buffer:
                raise StopIteration

            if len(self._buffer) < batch_op.batch_size and batch_op.drop_remainder:
                raise StopIteration

            batch_records = self._buffer[: batch_op.batch_size]
            self._buffer = self._buffer[batch_op.batch_size :]

            return batch_op(batch_records)
        else:
            while True:
                idx = next(self._sampler_iter)
                raw_data = self.data_loader.data_source[idx]
                meta = RecordMetadata(index=idx)
                rec = Record(metadata=meta, data=raw_data)
                rec = self._apply_ops(rec)
                if rec is not None:
                    self._last_seen_indices[0] = idx
                    return rec

    def get_state(self):
        """Docstring for get_state."""
        return {"last_seen_indices": self._last_seen_indices}

    def set_state(self, state):
        """Docstring for set_state."""
        self._last_seen_indices = state.get("last_seen_indices", {})
        # Note: In a real implementation this would fast-forward the sampler.


class DatasetIterator:
    """``IterDataset`` iterator.

    NOTE: The methods are assumed to be thread-unsafe. Please ensure only a single
    thread can access a ``DatasetIterator`` instance."""

    def __init__(
        self,
        parents: grain._src.python.dataset.dataset.DatasetIterator
        | collections.abc.Sequence[
            grain._src.python.dataset.dataset.DatasetIterator
        ] = (),
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class DatasetSelectionMap:
    """Map from index to (constituent dataset index, index within dataset).

    Note, this must be stateless, picklable and should avoid randomness to
    support determinism since it may be created and called concurrently in
    multiple processes."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class Filter:
    """Abstract base class for filter transformations for individual elements.

    The pipeline will drop any element for which the filter function returns
    False.

    Implementations should be threadsafe since they are often executed in
    parallel."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class FilterOperation:
    """Yields records from input iterator satisfying user-provided condition."""

    def __init__(self, condition_function: Callable[[~_IN], bool]):
        """Docstring for __init__."""
        self.condition_function = condition_function

    def __call__(self, record: Record) -> Record | None:
        """Docstring for __call__."""
        if record is None or record.data is None:
            return record
        if self.condition_function(record.data):
            return record
        return None


class IndexSampler:
    """Base index sampler for training on a single datasource.

    This index sampler supports the following operations:
    - Sharding of the dataset.
    - Global shuffle of the dataset."""

    def __init__(
        self,
        num_records: int,
        shard_options: grain._src.core.sharding.ShardOptions = NoSharding(
            shard_index=0, shard_count=1, drop_remainder=False
        ),
        shuffle: bool = False,
        num_epochs: int | None = None,
        seed: int | None = None,
    ):
        """Docstring for __init__."""
        self.num_records = num_records
        self.shard_options = shard_options
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.seed = seed

        self._epoch = 0
        self._index_within_epoch = 0
        self._indices = list(range(self.num_records))
        self._apply_sharding()
        if self.shuffle:
            self._do_shuffle()

    def _apply_sharding(self):
        """Docstring for _apply_sharding."""
        if getattr(self.shard_options, "drop_remainder", False):
            limit = (
                self.num_records // getattr(self.shard_options, "shard_count", 1)
            ) * getattr(self.shard_options, "shard_count", 1)
            self._indices = self._indices[:limit]
        self._indices = [
            x
            for x in self._indices
            if x % getattr(self.shard_options, "shard_count", 1)
            == getattr(self.shard_options, "shard_index", 0)
        ]

    def _do_shuffle(self):
        """Docstring for _do_shuffle."""
        import random

        # Seed depends on epoch to ensure deterministic but different shuffle per epoch
        rng = random.Random((self.seed or 0) + self._epoch)
        rng.shuffle(self._indices)

    def __iter__(self):
        """Docstring for __iter__."""
        return self  # pragma: no cover

    def __next__(self):
        """Docstring for __next__."""
        if self._index_within_epoch >= len(self._indices):
            self._epoch += 1
            if self.num_epochs is not None and self._epoch >= self.num_epochs:
                raise StopIteration
            self._index_within_epoch = 0
            if self.shuffle:
                self._do_shuffle()

        idx = self._indices[self._index_within_epoch]
        self._index_within_epoch += 1
        return idx


class IterDataset:
    """Represents a dataset with transformations that support Iterable interface.

    Transformations do not mutate the dataset object. Instead, they return a new
    dataset. ``IterDataset`` is immutable."""

    def __init__(
        self,
        parents: grain._src.python.dataset.dataset.MapDataset
        | grain._src.python.dataset.dataset.IterDataset
        | collections.abc.Sequence[
            grain._src.python.dataset.dataset.MapDataset
            | grain._src.python.dataset.dataset.IterDataset
        ] = (),
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class MapDataset:
    """Represents a dataset with transformations that support random access.

    Transformations do not mutate the dataset object. Instead, they return a new
    dataset. ``MapDataset`` is immutable.

    NOTE:
      ``MapDataset`` transformations such as ``.filter()`` use ``None`` to
      indicate absence of an element. Generally, the implementation of
      ``MapDataset`` transformations already handle `None` as a special case
      (e.g. by returning ``None`` as soon as ``__getitem__`` sees ``None``). This
      means the user-defined functions passed to the ``MapDataset``
      transformations do not need to explicitly handle ``None`` s."""

    def __init__(
        self,
        parents: grain._src.python.dataset.dataset.MapDataset
        | collections.abc.Sequence[grain._src.python.dataset.dataset.MapDataset] = (),
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class MapOperation:
    """Applies user-provided map_function to input records."""

    def __init__(self, map_function: Callable[[~_IN], ~_OUT]):
        """Docstring for __init__."""
        self.map_function = map_function

    def __call__(self, record: Record) -> Record:
        # Note: We must preserve metadata.
        """Docstring for __call__."""
        if record is None or record.data is None:
            return record
        return Record(metadata=record.metadata, data=self.map_function(record.data))


class MapTransform:
    """Abstract base class for all 1:1 transformations of elements.

    Implementations should be threadsafe since they are often executed in
    parallel."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class MapWithIndex:
    """Abstract base class for 1:1 transformations of elements and their index.

    Implementations should be threadsafe since they are often executed in
    parallel."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class MultiprocessingOptions:
    """Options for using Python multiprocessing.

    Attributes:
      num_workers: Number of Python worker processes. More processes can speed up
        the pipeline if it's compute bound and bottlenecked on the CPython's GIL.
        The default value of 0 means no Python multiprocessing, and as a result
        all data loading and transformation will run in the main Python process.
      per_worker_buffer_size: Size of the buffer for preprocessed elements that
        each worker maintains. These are elements after all transformations. If
        your transformations include batching this means a single element is a
        batch.
      enable_profiling: If True, profiling info is logged. This is only available
        when num_workers >= 1."""

    def __init__(
        self,
        num_workers: int = 0,
        per_worker_buffer_size: int = 1,
        enable_profiling: bool = False,
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class Operation:
    """Base class for protocol classes.

    Protocol classes are defined as::

        class Proto(Protocol):
            def meth(self) -> int:
                ...

    Such classes are primarily used with static type checkers that recognize
    structural subtyping (static duck-typing).

    For example::

        class C:
            def meth(self) -> int:
                return 0

        def func(x: Proto) -> int:
            return x.meth()

        func(C())  # Passes static type check

    See PEP 544 for details. Protocol classes decorated with
    @typing.runtime_checkable act as simple-minded runtime protocols that check
    only the presence of given attributes, ignoring their type signatures.
    Protocol classes can be generic, they are defined as::

        class GenProto(Protocol[T]):
            def meth(self) -> T:
                ..."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class RandomAccessDataSource:
    """Interface for datasources where storage supports efficient random access.

    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class RandomMapOperation:
    """Applies user-provided random_map_function with rng to input records."""

    def __init__(
        self,
        random_map_function: Callable[[~_IN, numpy.random._generator.Generator], ~_OUT],
    ):
        """Docstring for __init__."""
        self.random_map_function = random_map_function

    def __call__(self, record: Record) -> Record:
        """Docstring for __call__."""
        if record is None or record.data is None:
            return record
        import random

        # Fallback to python random if numpy Generator isn't strictly enforced
        rng = (
            record.metadata.rng
            if record.metadata.rng is not None
            else random.Random(record.metadata.index)
        )
        return Record(
            metadata=record.metadata, data=self.random_map_function(record.data, rng)
        )


class RandomMapTransform:
    """Abstract base class for all random 1:1 transformations of elements.

    Implementations should be threadsafe since they are often executed in
    parallel."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class RangeDataSource:
    """Range data source, similar to python range() function."""

    def __init__(self, start: int, stop: int, step: int):
        """Docstring for __init__."""
        self.start = start
        self.stop = stop
        self.step = step
        self._len = (
            max(0, (stop - start + step - 1) // step)
            if step > 0
            else max(0, (start - stop - step - 1) // (-step))
        )

    def __len__(self):
        """Docstring for __len__."""
        return self._len

    def __getitem__(self, idx):
        """Docstring for __getitem__."""
        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of bounds")
        return self.start + idx * self.step


class ReadOptions:
    """Options for reading data from the DataSource.

    These settings configure a single Python process. Each process uses separate
    threads and buffer for reading and processing data.

    Example: With ReadOptions.num_threads=8 and
    MultiprocessingOptions.num_workers=10 there will be 80 threads reading the
    data (8 threads in each of 10 Python processes).

    Attributes:
      num_threads: Number of threads reading from the DataSource in parallel. If
        the data are already loaded in memory, we recommend setting this to 0 to
        avoid Python GIL contention by multiple threads.
      prefetch_buffer_size: Size of the buffer for reading elements per Python
        process (not per thread). Useful when reading from a distributed file
        system."""

    def __init__(self, num_threads: int = 16, prefetch_buffer_size: int = 500):
        """Docstring for __init__."""
        pass  # pragma: no cover


class Record:
    """Record(metadata: grain._src.python.record.RecordMetadata, data: ~T)"""

    def __init__(self, metadata: grain._src.python.record.RecordMetadata, data: ~T):
        """Docstring for __init__."""
        self.metadata = metadata
        self.data = data


class RecordMetadata:
    """RecordMetadata contains metadata about indidivual records.

    Metadata can be emitted by the sampler to refer to which record to read next.
    In addition, they are also used to keep information about records as they flow
    through the pipeline from one operation to the other."""

    def __init__(
        self,
        index: int,
        record_key: int | None = None,
        rng: numpy.random._generator.Generator | None = None,
    ):
        """Docstring for __init__."""
        self.index = index
        self.record_key = record_key
        self.rng = rng


class Sampler:
    """Interface for PyGrain-compatible sampler."""

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        pass  # pragma: no cover


class SequentialSampler:
    """Basic sampler implementation that provides records in order."""

    def __init__(
        self,
        num_records: int,
        shard_options: grain._src.core.sharding.ShardOptions = NoSharding(
            shard_index=0, shard_count=1, drop_remainder=False
        ),
        seed: int | None = None,
    ):
        """Docstring for __init__."""
        self.num_records = num_records
        self.shard_options = shard_options
        self.seed = seed
        self._current_index = 0

        # Apply sharding mathematics
        if getattr(self.shard_options, "drop_remainder", False):
            self._limit = (
                self.num_records // getattr(self.shard_options, "shard_count", 1)
            ) * getattr(self.shard_options, "shard_count", 1)
        else:
            self._limit = self.num_records

    def __iter__(self):
        """Docstring for __iter__."""
        return self  # pragma: no cover

    def __next__(self):
        """Docstring for __next__."""
        while self._current_index < self._limit:
            idx = self._current_index
            self._current_index += 1
            if idx % getattr(self.shard_options, "shard_count", 1) == getattr(
                self.shard_options, "shard_index", 0
            ):
                return idx
        raise StopIteration


class ShardByJaxProcess:
    """Shards the data across JAX processes."""

    def __init__(self, drop_remainder: bool = False):
        """Docstring for __init__."""
        self.drop_remainder = drop_remainder
        self.shard_index = 0
        self.shard_count = 1


class ShardOptions:
    """Dataclass to hold options for sharding a data source.

    Attributes:
      shard_index: The index of the shard to use in this process. Must be in [0,
        shard_count - 1].
      shard_count: The total number of shards.
      drop_remainder: If True shard() will create even splits and drop the
        remainder examples (all shards will have the same number of examples). If
        False will distribute the remainder N over the first N shards."""

    def __init__(
        self, shard_index: int, shard_count: int, drop_remainder: bool = False
    ):
        """Docstring for __init__."""
        pass  # pragma: no cover


class SharedMemoryArray(numpy.ndarray):
    """A NumPy array subclass which is backed by shared memory.

    This should be used in combination with Python multiprocessing.
    Compared with the normal NumPy ndarray it avoids expensive serialization
    when sending the array to another Python process (on the same machine).
    It also doesn't require a copy on the receiving side.

    The last processes using the array must call unlink_on_del()! Otherwise
    the memory will not be freed."""

    def __new__(cls, *args, **kwargs):
        """Docstring for __new__."""
        return numpy.asarray(*args, **kwargs).view(cls)

    def __init__(self, *args, **kwargs):
        """Docstring for __init__."""
        self._args = args
        self._kwargs = kwargs


class SharedMemoryDataSource:
    """Simple in-memory data source for sequences that is sharable among multiple processes.

    Note:
      This constrains storable values to only the int, float, bool, str (less than
      10M bytes each), bytes (less than 10M bytes each), and None built-in data
      types. It also notably differs from the built-in list type in that these
      lists can not change their overall length (i.e. no append, insert, etc.)
    """

    def __init__(
        self,
        elements: collections.abc.Sequence[Any] | None = None,
        *,
        name: str | None = None,
    ):
        """Docstring for __init__."""
        self.elements = elements or []
        self.name = name

    def __len__(self):
        """Docstring for __len__."""
        return len(self.elements)

    def __getitem__(self, idx):
        """Docstring for __getitem__."""
        return self.elements[idx]


def load(
    source: grain._src.python.data_sources.RandomAccessDataSource,
    *,
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: grain._src.core.sharding.ShardOptions = NoSharding(
        shard_index=0, shard_count=1, drop_remainder=False
    ),
    transformations: collections.abc.Sequence[
        grain._src.core.transforms.Batch
        | grain._src.core.transforms.MapTransform
        | grain._src.core.transforms.RandomMapTransform
        | grain._src.core.transforms.TfRandomMapTransform
        | grain._src.core.transforms.Filter
        | grain._src.core.transforms.FlatMapTransform
        | grain._src.core.transforms.MapWithIndex
    ] = (),
    batch_size: int | None = None,
    drop_remainder: bool = False,
    worker_count: int | None = 0,
    read_options: grain._src.python.options.ReadOptions | None = None,
):
    """Docstring for load."""
    sampler = IndexSampler(
        num_records=0,
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    return DataLoader(
        data_source=source,
        sampler=sampler,
        operations=transformations,
        worker_count=worker_count,
        shard_options=shard_options,
        read_options=read_options,
    )


# Aliases
PyGrainDatasetIterator = DataLoaderIterator
PyGrainCheckpointHandler = CheckpointHandler
InMemoryDataSource = SharedMemoryDataSource


# Define generic __init__
def _generic_init(self, *args, **kwargs):
    """Docstring for _generic_init."""
    for k, v in kwargs.items():
        setattr(self, k, v)


for name, obj in list(globals().items()):
    if isinstance(obj, type) and obj.__module__ == __name__:
        if hasattr(obj, "__init__") and obj.__init__.__code__.co_filename == __file__:
            if (
                name != "NoSharding"
                and name
                not in [
                    "NoSharding",
                    "ShardByJaxProcess",
                    "SharedMemoryDataSource",
                    "RangeDataSource",
                    "InMemoryDataSource",
                    "BatchOperation",
                    "FilterOperation",
                    "MapOperation",
                    "RandomMapOperation",
                    "DataLoader",
                    "DataLoaderIterator",
                    "PyGrainDatasetIterator",
                    "SequentialSampler",
                    "IndexSampler",
                    "ArrayRecordDataSource",
                    "SharedMemoryArray",
                    "RecordMetadata",
                    "Record",
                ]
                and "pass" in getattr(obj.__init__, "__source__", "pass")
            ):
                obj.__init__ = _generic_init

FilterTransform = Filter
MapWithIndexTransform = MapWithIndex


# PyTree Integrations
try:
    from zero_jax import tree_util

    tree_map = getattr(tree_util, "tree_map", None)

    def tree_collate(records):
        """Docstring for tree_collate."""
        return (
            tree_map(lambda *leaves: numpy.stack(leaves), *records)
            if tree_map
            else None
        )
except ImportError:  # pragma: no cover
    tree_map = None
