"""Additional coverage tests for PyTree batching, operations, datasets, and samplers."""

from __future__ import annotations

from typing import NamedTuple
import numpy as np
import pytest
from zero_grain._types import ElementTree
from zero_grain.python import (
    BatchOperation,
    CopyNumPyArrayToSharedMemoryOperation,
    DataLoader,
    DataLoaderIterator,
    Dataset,
    DatasetSelectionMap,
    FilterOperation,
    FlatMapOperation,
    InMemoryDataSource,
    IndexSampler,
    IterDataset,
    MapOperation,
    MapWithIndexOperation,
    MultiprocessingOptions,
    NoSharding,
    PyGrainCheckpointHandler,
    PyGrainDatasetIterator,
    RandomAccessDataSource,
    RandomMapOperation,
    RangeDataSource,
    Record,
    RecordMetadata,
    SequentialSampler,
    ShardByJaxProcess,
    SharedMemoryDataSource,
    _batch_elements,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    batch_and_pad,
    create_rng,
    fake_class,
    get_element_spec,
    load,
)


class _PointTuple(NamedTuple):
    """Typed NamedTuple used in ``test_batch_elements_tuple_with_fields``."""

    x: int | np.ndarray
    y: int | np.ndarray


def test_record_metadata_eq() -> None:
    """Verify ``RecordMetadata`` equality comparisons."""
    meta = RecordMetadata(index=1, record_key=2)
    assert meta == RecordMetadata(index=1, record_key=2)
    assert meta != RecordMetadata(index=2, record_key=2)


def test_batch_elements_tuple_with_fields() -> None:
    """Verify ``_batch_elements`` preserves ``namedtuple`` types."""
    batch = [_PointTuple(1, 2), _PointTuple(3, 4)]
    res = _batch_elements(batch)
    assert isinstance(res, _PointTuple)
    assert isinstance(res.x, np.ndarray)
    assert isinstance(res.y, np.ndarray)
    assert res.x.tolist() == [1, 3]
    assert res.y.tolist() == [2, 4]


def test_batch_elements_mismatched_tuples() -> None:
    """Verify ``_batch_elements`` raises ``TypeError`` on mismatched tuple lengths."""
    batch: list[tuple[int, ...]] = [(1, 2), (1, 2, 3)]
    with pytest.raises(TypeError):
        _batch_elements(batch)


def test_batch_elements_lists() -> None:
    """Verify ``_batch_elements`` stacks lists into a NumPy array."""
    batch = [[1, 2], [3, 4]]
    res = _batch_elements(batch)
    assert isinstance(res, np.ndarray)
    assert res.tolist() == [[1, 3], [2, 4]]


def test_batch_elements_tuple_ndarray() -> None:
    """Verify ``_batch_elements`` handles tuples containing NumPy arrays."""
    batch = [(np.array([1]), 2), (np.array([3]), 4)]
    res = _batch_elements(batch)
    assert isinstance(res, tuple)
    assert len(res) == 2


def test_batch_elements_except() -> None:
    """Verify ``_batch_elements`` falls back to a list when ``__array__`` raises."""

    class Weird:
        """Class whose ``__array__`` method raises ``ValueError``."""

        def __array__(self) -> np.ndarray:
            """Raise ``ValueError`` on array conversion.

            Returns:
                np.ndarray: Never returns.

            Raises:
                ValueError: Always raised.
            """
            raise ValueError("cannot convert")

    batch = [Weird(), Weird()]
    res = _batch_elements(batch)
    assert res == batch


def test_data_loader_iterator_prefetch_state() -> None:
    """Verify ``DataLoaderIterator`` initial worker state and ``set_state`` edge cases."""
    dl = DataLoader(data_source=[1, 2, 3], sampler=[0, 1, 2], worker_count=3)
    it = iter(dl)
    it.start_prefetch()
    state = it.get_state()
    assert state["last_seen_indices"]["0"] == -3
    assert state["last_seen_indices"]["1"] == -2
    assert state["last_seen_indices"]["2"] == -1

    dl2 = DataLoader(data_source=[1, 2], sampler=[0, 1], worker_count=1)
    it2 = iter(dl2)
    state2 = it2.get_state()
    assert state2["last_seen_indices"]["0"] == -1
    it2.set_state({"last_seen_indices": {}})
    it2.set_state({"last_seen_indices": {"0": 10}})
    restored_it = DataLoaderIterator(dl2, state={"last_seen_indices": {"0": 10}})
    assert list(restored_it) == []


def test_checkpoint_handler() -> None:
    """Verify ``PyGrainCheckpointHandler`` no-op save and restore calls."""
    assert_equal_output_after_checkpoint(None)
    handler = PyGrainCheckpointHandler()
    handler.save()
    assert handler.restore() is None


def test_datasources_misc() -> None:
    """Verify miscellaneous data source methods and representations."""
    ds: RandomAccessDataSource[int] = RandomAccessDataSource()
    assert len(ds) == 0
    with pytest.raises(IndexError):
        _ = ds[0]

    in_mem = InMemoryDataSource([1, 2], name="test")
    assert len(in_mem) == 2
    assert in_mem[1] == 2
    assert (in_mem.closed, in_mem.unlinked) == (False, False)
    in_mem.close()
    in_mem.unlink()
    assert (in_mem.closed, in_mem.unlinked) == (True, True)
    assert str(in_mem) == "SharedMemoryDataSource(name=test, len=2)"

    sh_mem = SharedMemoryDataSource([1, 2], name="test")
    sh_mem.close()
    sh_mem.unlink()
    assert str(sh_mem) == "SharedMemoryDataSource(name=test, len=2)"

    rng_ds = RangeDataSource(0, 10, 2)
    assert repr(rng_ds) == "RangeDataSource(start=0, stop=10, step=2)"


def test_sharding_misc() -> None:
    """Verify ``ShardByJaxProcess`` properties."""
    shard = ShardByJaxProcess(drop_remainder=True)
    assert shard.shard_index == 0
    assert shard.shard_count == 1
    assert shard.drop_remainder is True


def test_options_misc() -> None:
    """Verify ``MultiprocessingOptions`` instantiation."""
    opts = MultiprocessingOptions(num_workers=1)
    assert opts.num_workers == 1


def test_index_sampler_seed_type() -> None:
    """Verify ``IndexSampler`` seed validation and sharded iteration."""
    with pytest.raises(TypeError):
        IndexSampler(10, seed=True)
    with pytest.raises(ValueError):
        IndexSampler(10, seed=-1)
    with pytest.raises(ValueError):
        IndexSampler(10, seed=2**32 + 1)

    idx_s = IndexSampler(10, shard_options=NoSharding(0, 2, True), num_epochs=1)
    assert len(idx_s) == 10
    assert len(list(iter(idx_s))) == 5
    idx_s_uneven = IndexSampler(9, shard_options=NoSharding(0, 2, True), num_epochs=1)
    assert len(idx_s_uneven) == 8
    assert len(list(iter(idx_s_uneven))) == 4


def test_operations_not_implemented() -> None:
    """Verify ``NotImplementedError`` on unconfigured operation instances."""
    op_map_index: MapWithIndexOperation[int, int] = MapWithIndexOperation()
    with pytest.raises(NotImplementedError):
        op_map_index.map_with_index(0, 1)

    op_map: MapOperation[int, int] = MapOperation()
    with pytest.raises(NotImplementedError):
        op_map.map(1)

    op_rmap: RandomMapOperation[int, int] = RandomMapOperation()
    with pytest.raises(NotImplementedError):
        op_rmap.random_map(1, create_rng(0))

    op_filter: FilterOperation[int] = FilterOperation()
    with pytest.raises(NotImplementedError):
        op_filter.filter(1)

    op_flat: FlatMapOperation[int, int] = FlatMapOperation()
    with pytest.raises(NotImplementedError):
        op_flat.flat_map(1)


def test_operations_implemented() -> None:
    """Verify configured operation methods execute as expected."""
    op_map_idx: MapWithIndexOperation[int, int] = MapWithIndexOperation(lambda i, x: x + i)
    assert op_map_idx.map_with_index(2, 1) == 3
    rec: Record[int] = Record(RecordMetadata(0, 0), 1)
    assert [r.data for r in op_map_idx([rec])] == [1]

    op_flat: FlatMapOperation[int, int] = FlatMapOperation(lambda x: [x, x + 1])
    assert list(op_flat.flat_map(1)) == [1, 2]

    op_copy = CopyNumPyArrayToSharedMemoryOperation()
    rec_tree: Record[ElementTree] = Record(RecordMetadata(0, 0), 1)
    assert [r.data for r in op_copy([rec_tree])] == [1]


def test_fake_class() -> None:
    """Verify ``fake_class`` returns an instantiable class."""
    cls = fake_class()
    inst = cls(3)
    assert len(inst) == 3
    assert inst[2] == (0, 2)


def test_batch_and_pad() -> None:
    """Verify ``batch_and_pad`` with arrays and scalars."""
    res = batch_and_pad([np.array([1]), np.array([2])], batch_size=3)
    assert isinstance(res, np.ndarray)
    assert res.tolist() == [[1], [2], [0]]

    res2 = batch_and_pad([1, 2], batch_size=3)
    assert isinstance(res2, np.ndarray)
    assert res2.tolist() == [1, 2, 0]

    res3 = batch_and_pad([_batch_elements([1]), _batch_elements([2])], batch_size=3)
    assert isinstance(res3, np.ndarray)
    assert res3.tolist() == [[1], [2], [0]]


def test_dataset_misc() -> None:
    """Verify ``Dataset`` methods and transformations."""
    ds = Dataset([1, 2, 3])
    ds2 = ds.seed(42)
    assert ds2._seed == 42
    assert ds._seed is None

    ds3 = ds.shuffle(12)
    assert ds3._seed == 12

    it_ds = ds.to_iter_dataset()
    assert isinstance(it_ds, IterDataset)
    assert str(it_ds) != ""
    it_ds_iter = iter(it_ds)
    assert next(it_ds_iter) == 1

    res = apply_transformations(ds, MapOperation(lambda x: int(x) * 2 if isinstance(x, int) else x))
    assert len(res.operations) == 1
    assert list(res) == [2, 4, 6]
    assert get_element_spec(ds) == "int"

    ds4: Dataset[int] = ds.map(lambda x: int(x) + 10 if isinstance(x, int) else x)
    assert len(ds4.operations) == 1
    assert list(ds4) == [11, 12, 13]

    ds5: Dataset[int] = ds.map_with_index(lambda i, x: i + (int(x) * 10 if isinstance(x, int) else 0))
    assert len(ds5.operations) == 1
    assert list(ds5) == [10, 21, 32]

    ds6 = ds.filter(lambda x: isinstance(x, int) and x >= 2)
    assert len(ds6.operations) == 1
    assert list(ds6) == [2, 3]

    ds7: Dataset[list[ElementTree]] = ds.batch(2, batch_fn=lambda xs: list(xs))
    assert len(ds7.operations) == 1
    assert list(ds7) == [[1, 2], [3]]

    rng = Dataset.range(10)
    assert len(rng) == 10
    assert len(Dataset.range(1, 10)) == 9
    assert len(Dataset.range(1, 10, 2)) == 5
    assert ds[0] == 1


def test_no_sharding_repr() -> None:
    """Verify ``NoSharding.__repr__`` formatting."""
    ns = NoSharding(1, 2, True)
    assert repr(ns) == "NoSharding(shard_index=1, shard_count=2, drop_remainder=True)"


def test_seq_sampler_repr() -> None:
    """Verify ``SequentialSampler.__repr__`` and indexing."""
    ss = SequentialSampler(10, NoSharding(0, 2, True))
    assert "SequentialSampler" in repr(ss)
    assert len(list(iter(ss))) == 5
    assert ss[0].record_key == 0


def test_misc_classes() -> None:
    """Verify ``DatasetSelectionMap``, ``PyGrainDatasetIterator``, and ``load``."""
    sel_map = DatasetSelectionMap()
    assert len(sel_map) == 0
    empty_it: PyGrainDatasetIterator[int] = PyGrainDatasetIterator()
    with pytest.raises(StopIteration):
        next(empty_it)
    dl = load([1, 2], batch_size=2)
    assert dl.worker_count == 0


def test_operations_with_none() -> None:
    """Verify operations skip ``None`` records cleanly."""
    op_batch: BatchOperation[int] = BatchOperation(batch_size=2)
    assert list(op_batch([None])) == []

    op_map_idx: MapWithIndexOperation[int, int] = MapWithIndexOperation(lambda i, x: i + x)
    assert list(op_map_idx([None])) == []

    op_map: MapOperation[int, int] = MapOperation(lambda x: x + 1)
    assert list(op_map([None])) == []

    op_rmap: RandomMapOperation[int, int] = RandomMapOperation(lambda x, rng: x + rng.integers(1, 10))
    assert list(op_rmap([None])) == []

    op_filter: FilterOperation[int] = FilterOperation(lambda x: x > 0)
    assert list(op_filter([None])) == []

    op_flat: FlatMapOperation[int, int] = FlatMapOperation(lambda x: [x, x + 1])
    assert list(op_flat([None])) == []

    op_copy = CopyNumPyArrayToSharedMemoryOperation()
    assert list(op_copy([None])) == []


def test_index_sampler_missing_branch() -> None:
    """Verify ``IndexSampler`` iteration."""
    sampler = IndexSampler(10)
    assert [m.record_key for m in sampler] == list(range(10))


def test_load_batch_size_1() -> None:
    """Verify ``load`` with ``batch_size=1`` appends a ``BatchOperation``."""
    dl = load([1, 2], batch_size=1)
    assert len(dl.operations) == 1
    batches = list(dl)
    assert len(batches) == 2
    assert isinstance(batches[0], np.ndarray) and batches[0].tolist() == [1]


def test_batch_and_pad_ndarray() -> None:
    """Verify ``batch_and_pad`` with a NumPy array element."""
    elements = [np.array([1, 2])]
    res = batch_and_pad(elements, 2)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 2)
    assert res[1][0] == 0


def test_batch_and_pad_dtype() -> None:
    """Verify ``batch_and_pad`` with a custom class implementing ``__array__``."""

    class MockTensor:
        """Tensor mock implementing ``__array__``."""

        dtype = np.float32

        def __array__(self) -> np.ndarray:
            """Convert ``MockTensor`` to a NumPy array.

            Returns:
                np.ndarray: 1D float32 array.
            """
            return np.array([1, 2], dtype=self.dtype)

    elements = [MockTensor()]
    res = batch_and_pad(elements, 2)
    assert isinstance(res, np.ndarray)
    assert len(res) == 2
