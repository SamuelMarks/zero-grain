# Zero-Grain Semantic & Execution Plan

This document outlines the exhaustive engineering plan for implementing the **mathematical and semantic execution logic** behind the zero-grain APIs. Up until now, zero-grain has operated as an API-compliant stub. This phase breathes life into the framework, enabling real data loading, transformations, PyTree batching, and multiprocessing, all while leaning heavily on ../zero-jax for array primitives and graph structure.

## 1. Core Data Structures & PyTree Integration
Because grain natively yields JAX/NumPy objects, zero-grain must depend upon zero-jax for robust PyTree handling and array semantics.

- [x] Add zero-jax to development/testing dependencies (or resolve via sys.path in testing).
- [x] Implement RecordMetadata: Ensure accurate tracking of index, record_key, and PRNG states.
- [x] Implement Record[T]: Bind metadata to actual data payloads.
- [x] Implement SharedMemoryArray: Create a zero_jax.numpy.ndarray wrapper that is backed by Python's multiprocessing.shared_memory to avoid serialization overhead across workers.
- [x] Create internal PyTree utilities leveraging zero_jax.tree_util:
  - [x] tree_map integration for mapping functions over complex data records.
  - [x] tree_collate for stacking scalar/array leaves into batched arrays.

## 2. Data Sources (Storage Interfaces)
Implement the actual storage adapters that fetch data into memory.

- [x] **InMemoryDataSource**
  - [x] Store a collections.abc.Sequence internally.
  - [x] Implement __len__ and __getitem__.
  - [x] Write tests confirming random access and bounds checking.
- [x] **RangeDataSource**
  - [x] Implement mathematically correct start, stop, step iterator logic without expanding into a full list in memory.
  - [x] Implement __len__ and __getitem__ for O(1) random access.
- [x] **ArrayRecordDataSource**
  - [x] Scaffold standard file reading logic (can fallback to standard binary reads if array_record C-extension is mocked).
  - [x] Parse reader_options.
  - [x] Implement byte-offset random access mimicking true ArrayRecord capabilities.

## 3. Samplers & Sharding
Samplers dictate *what* indices are read and *when*, taking sharding and distributed execution into account.

- [x] **ShardOptions & ShardByJaxProcess**
  - [x] Implement logic to query zero_jax.process_index() and zero_jax.process_count().
  - [x] Implement modular drop_remainder arithmetic.
- [x] **SequentialSampler**
  - [x] Implement an iterator that yields sequential indices from 0 to num_records - 1.
  - [x] Apply ShardOptions filtering (e.g., yield every i where i mod shard_count == shard_index).
- [x] **IndexSampler**
  - [x] Implement robust PRNG-based shuffling using zero_jax.random splits (or numpy generator if eagerly evaluated).
  - [x] Implement num_epochs logic (yielding Epoch 1 indices, then reshuffling for Epoch 2).
  - [x] Implement index boundary assertions and epoch tracking for checkpointing.

## 4. Transformations (Operations)
Operations are the 1:1 or N:1 mappings applied to the dataset.

- [x] **MapOperation / MapTransform**
  - [x] Evaluate the user-provided function on every Record.data.
  - [x] Test with complex PyTrees utilizing zero_jax.numpy.
- [x] **RandomMapOperation / RandomMapTransform**
  - [x] Inject numpy.random.Generator (or zero_jax.random.PRNGKey shim) mapped to the RecordMetadata.rng.
  - [x] Ensure deterministic outcomes given a fixed global seed and deterministic sampler indexing.
- [x] **FilterOperation / FilterTransform**
  - [x] Implement lazy evaluation skipping: if cond(x) is false, silently consume the next element in the underlying iterator.
- [x] **BatchOperation / Batch**
  - [x] Implement chunking: Accumulate batch_size records.
  - [x] Execute zero_jax.tree_util.tree_map(lambda *leaves: zero_jax.numpy.stack(leaves), *records) to combine leaves.
  - [x] Handle drop_remainder=True correctly at dataset boundaries.
  - [x] Execute custom batch_fn if provided.

## 5. Dataset Iteration & Multiprocessing
Wire the pipeline together into the master DataLoader.

- [x] **IterDataset & MapDataset**
  - [x] Implement lazy, graph-like dataset definition (Parents -> Transforms -> Children).
  - [x] Implement deterministic __iter__ traversal.
- [x] **DataLoader**
  - [x] Synthesize the Pipeline: Source -> Sampler -> Transforms.
  - [x] **Single-process Mode** (worker_count=0): Build the standard blocking generator.
  - [x] **Multi-process Mode** (worker_count>0):
    - [x] Spawn Python multiprocessing pool.
    - [x] Distribute index generation via round-robin queue.
    - [x] Implement IPC fetching leveraging SharedMemoryArray to pipe zero_jax tensors back to the main thread.
- [x] **PyGrainDatasetIterator**
  - [x] Implement __next__ conforming to the Python Iterator protocol.
  - [x] Handle StopIteration gracefully.

## 6. State & Checkpointing (Resilience)
- [x] **State Management**
  - [x] Implement .get_state() on DataLoaderIterator yielding last_seen_indices state map.
  - [x] Implement .set_state() rebuilding the pipeline's internal counters and RNG keys seamlessly.
- [x] **PyGrainCheckpointHandler**
  - [x] Implement zero-orbax compatible interface to save/restore DataLoader state seamlessly to disk.
  - [x] Ensure distributed JAX jobs (across simulated zero_jax processes) sync dataset state securely.

## 7. Exhaustive Testing (100% Coverage & Math Parity)
- [x] **E2E Semantic Tests**
  - [x] Test a pipeline: RangeDataSource -> IndexSampler(shuffle=True) -> Map(x * 2) -> Batch(32) yields exactly the expected zero_jax.numpy.ndarray tensors.
  - [x] Assert DataLoader outputs are mathematically identical to standard Python array operations.
- [x] **Multiprocessing Integrity**
  - [x] Test worker deadlocks, queue timeouts, and shared memory leaks.
- [x] **State Restoration Tests**
  - [x] Consume 150 elements of a 500 element dataset.
  - [x] Checkpoint.
  - [x] Restart script, restore checkpoint, and verify elements 151-500 are yielded perfectly (inclusive of random states).
- [x] **Pre-commit Pass**: Ensure ruff format, ruff check, and pytest --cov clear automatically.
