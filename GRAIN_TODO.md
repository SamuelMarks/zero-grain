Extracting target APIs from /Users/samuel/repos/zero-grain/src...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 100.0%

Breakdown by Module:
  - grain.python: 100.0% (33/33)

Mismatched APIs (0):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [x] | grain | grain.python | ArrayRecordDataSource | grain.python.ArrayRecordDataSource | `(paths: str | os.PathLike | array_record.python.array_record_data_source.FileInstruction | collections.abc.Sequence[str | os.PathLike | array_record.python.array_record_data_source.FileInstruction], reader_options: dict[str, str] | None=None)` | Data source for ArrayRecord files. |
| [x] | grain | grain.python | Batch | grain.python.Batch | `(batch_size: int, drop_remainder: bool=False, batch_fn: collections.abc.Callable[[collections.abc.Sequence[Any]], Any] | None=None)` | Batch(batch_size: 'int', drop_remainder: 'bool' = False, batch_fn: 'Callable[[Sequence[Any]], Any... |
| [x] | grain | grain.python | BatchOperation | grain.python.BatchOperation | `(batch_size: int, drop_remainder: bool=False, batch_fn: Callable[[Sequence[~_IN]], ~_OUT] | None=None)` | Batches input examples into batches with given batch_size. |
| [x] | grain | grain.python | DataLoader | grain.python.DataLoader | `(data_source: grain._src.python.data_sources.RandomAccessDataSource, sampler: grain._src.python.samplers.Sampler, operations: Sequence[grain._src.core.transforms.Batch | grain._src.core.transforms.MapTransform | grain._src.core.transforms.RandomMapTransform | grain._src.core.transforms.TfRandomMapTransform | grain._src.core.transforms.Filter | grain._src.core.transforms.FlatMapTransform | grain._src.core.transforms.MapWithIndex | grain._src.python.operations.Operation]=(), worker_count: int | None=0, worker_buffer_size: int=1, shard_options: grain._src.core.sharding.ShardOptions | None=None, read_options: grain._src.python.options.ReadOptions | None=None, enable_profiling: bool=False)` | DataLoader loads and transforms input data. |
| [x] | grain | grain.python | DatasetIterator | grain.python.DatasetIterator | `(parents: grain._src.python.dataset.dataset.DatasetIterator | collections.abc.Sequence[grain._src.python.dataset.dataset.DatasetIterator]=())` | ``IterDataset`` iterator. |
| [x] | grain | grain.python | DatasetSelectionMap | grain.python.DatasetSelectionMap | `(args, kwargs)` | Map from index to (constituent dataset index, index within dataset). |
| [x] | grain | grain.python | FilterOperation | grain.python.FilterOperation | `(condition_function: Callable[[~_IN], bool])` | Yields records from input iterator satisfying user-provided condition. |
| [x] | grain | grain.python | FilterTransform | grain.python.FilterTransform | `(args, kwargs)` | Abstract base class for filter transformations for individual elements. |
| [x] | grain | grain.python | InMemoryDataSource | grain.python.InMemoryDataSource | `(elements: collections.abc.Sequence[Any] | None=None, name: str | None=None)` | Simple in-memory data source for sequences that is sharable among multiple processes. |
| [x] | grain | grain.python | IndexSampler | grain.python.IndexSampler | `(num_records: int, shard_options: grain._src.core.sharding.ShardOptions=NoSharding(shard_index=0, shard_count=1, drop_remainder=False), shuffle: bool=False, num_epochs: int | None=None, seed: int | None=None)` | Base index sampler for training on a single datasource. |
| [x] | grain | grain.python | IterDataset | grain.python.IterDataset | `(parents: grain._src.python.dataset.dataset.MapDataset | grain._src.python.dataset.dataset.IterDataset | collections.abc.Sequence[grain._src.python.dataset.dataset.MapDataset | grain._src.python.dataset.dataset.IterDataset]=())` | Represents a dataset with transformations that support Iterable interface. |
| [x] | grain | grain.python | MapDataset | grain.python.MapDataset | `(parents: grain._src.python.dataset.dataset.MapDataset | collections.abc.Sequence[grain._src.python.dataset.dataset.MapDataset]=())` | Represents a dataset with transformations that support random access. |
| [x] | grain | grain.python | MapOperation | grain.python.MapOperation | `(map_function: Callable[[~_IN], ~_OUT])` | Applies user-provided map_function to input records. |
| [x] | grain | grain.python | MapTransform | grain.python.MapTransform | `(args, kwargs)` | Abstract base class for all 1:1 transformations of elements. |
| [x] | grain | grain.python | MapWithIndexTransform | grain.python.MapWithIndexTransform | `(args, kwargs)` | Abstract base class for 1:1 transformations of elements and their index. |
| [x] | grain | grain.python | MultiprocessingOptions | grain.python.MultiprocessingOptions | `(num_workers: int=0, per_worker_buffer_size: int=1, enable_profiling: bool=False)` | Options for using Python multiprocessing. |
| [x] | grain | grain.python | NoSharding | grain.python.NoSharding | `()` | Doesn't shard data. Each process will load all data. |
| [x] | grain | grain.python | Operation | grain.python.Operation | `(args, kwargs)` | Base class for protocol classes. |
| [x] | grain | grain.python | PyGrainCheckpointHandler | grain.python.PyGrainCheckpointHandler | `(args, kwargs)` | Orbax CheckpointHandler for PyGrain iterators. |
| [x] | grain | grain.python | PyGrainDatasetIterator | grain.python.PyGrainDatasetIterator | `(data_loader: grain._src.python.data_loader.DataLoader, state: dict[str, Any] | None, validate_state: bool=True)` | DataLoader iterator providing get/set state functionality. |
| [x] | grain | grain.python | RandomAccessDataSource | grain.python.RandomAccessDataSource | `(args, kwargs)` | Interface for datasources where storage supports efficient random access. |
| [x] | grain | grain.python | RandomMapOperation | grain.python.RandomMapOperation | `(random_map_function: Callable[[~_IN, numpy.random._generator.Generator], ~_OUT])` | Applies user-provided random_map_function with rng to input records. |
| [x] | grain | grain.python | RandomMapTransform | grain.python.RandomMapTransform | `(args, kwargs)` | Abstract base class for all random 1:1 transformations of elements. |
| [x] | grain | grain.python | RangeDataSource | grain.python.RangeDataSource | `(start: int, stop: int, step: int)` | Range data source, similar to python range() function. |
| [x] | grain | grain.python | ReadOptions | grain.python.ReadOptions | `(num_threads: int=16, prefetch_buffer_size: int=500)` | Options for reading data from the DataSource. |
| [x] | grain | grain.python | Record | grain.python.Record | `(metadata: grain._src.python.record.RecordMetadata, data: ~T)` | Record(metadata: grain._src.python.record.RecordMetadata, data: ~T) |
| [x] | grain | grain.python | RecordMetadata | grain.python.RecordMetadata | `(index: int, record_key: int | None=None, rng: numpy.random._generator.Generator | None=None)` | RecordMetadata contains metadata about indidivual records. |
| [x] | grain | grain.python | Sampler | grain.python.Sampler | `(args, kwargs)` | Interface for PyGrain-compatible sampler. |
| [x] | grain | grain.python | SequentialSampler | grain.python.SequentialSampler | `(num_records: int, shard_options: grain._src.core.sharding.ShardOptions=NoSharding(shard_index=0, shard_count=1, drop_remainder=False), seed: int | None=None)` | Basic sampler implementation that provides records in order. |
| [x] | grain | grain.python | ShardByJaxProcess | grain.python.ShardByJaxProcess | `(drop_remainder: bool=False)` | Shards the data across JAX processes. |
| [x] | grain | grain.python | ShardOptions | grain.python.ShardOptions | `(shard_index: int, shard_count: int, drop_remainder: bool=False)` | Dataclass to hold options for sharding a data source. |
| [x] | grain | grain.python | SharedMemoryArray | grain.python.SharedMemoryArray | `(args, kwargs)` | A NumPy array subclass which is backed by shared memory. |
| [x] | grain | grain.python | load | grain.python.load | `(source: grain._src.python.data_sources.RandomAccessDataSource, num_epochs: int | None=None, shuffle: bool=False, seed: int | None=None, shard_options: grain._src.core.sharding.ShardOptions=NoSharding(shard_index=0, shard_count=1, drop_remainder=False), transformations: collections.abc.Sequence[grain._src.core.transforms.Batch | grain._src.core.transforms.MapTransform | grain._src.core.transforms.RandomMapTransform | grain._src.core.transforms.TfRandomMapTransform | grain._src.core.transforms.Filter | grain._src.core.transforms.FlatMapTransform | grain._src.core.transforms.MapWithIndex]=(), batch_size: int | None=None, drop_remainder: bool=False, worker_count: int | None=0, read_options: grain._src.python.options.ReadOptions | None=None)` | Convenient method for simple pipelines on top of a data source. |
