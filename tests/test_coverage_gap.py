import pytest
import numpy as np
from zero_grain.python import (
    RecordMetadata,
    Record,
    _batch_elements,
    DataLoaderIterator,
    DataLoader,
    assert_equal_output_after_checkpoint,
    PyGrainCheckpointHandler,
    RandomAccessDataSource,
    ShardByJaxProcess,
    MultiprocessingOptions,
    InMemoryDataSource,
    IndexSampler,
    MapWithIndexOperation,
    MapOperation,
    RandomMapOperation,
    FilterOperation,
    FlatMapOperation,
    CopyNumPyArrayToSharedMemoryOperation,
    fake_class,
    batch_and_pad,
    SharedMemoryDataSource,
    Dataset,
    apply_transformations,
    get_element_spec,
    ArrayRecordDataSource,
    NoSharding,
    RangeDataSource,
    SequentialSampler,
    IterDataset,
    DatasetSelectionMap,
    PyGrainDatasetIterator,
    load,
)


def test_record_metadata_eq():
    meta = RecordMetadata(index=1, record_key=2)
    assert meta != "not_a_meta"
    assert meta == RecordMetadata(index=1, record_key=2)
    assert meta != RecordMetadata(index=2, record_key=2)


def test_batch_elements_tuple_with_fields():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    batch = [Point(1, 2), Point(3, 4)]
    res = _batch_elements(batch)
    assert res.x.tolist() == [1, 3]
    assert res.y.tolist() == [2, 4]


def test_batch_elements_mismatched_tuples():
    batch = [(1, 2), (1, 2, 3)]
    with pytest.raises(TypeError):
        _batch_elements(batch)


def test_batch_elements_lists():
    batch = [[1, 2], [3, 4]]
    res = _batch_elements(batch)
    assert isinstance(res, np.ndarray)


def test_batch_elements_tuple_ndarray():
    batch = [(np.array([1]), 2), (np.array([3]), 4)]
    res = _batch_elements(batch)
    assert isinstance(res, tuple)


def test_batch_elements_except():
    class Weird:
        def __array__(self):
            raise ValueError()

    batch = [Weird(), Weird()]
    res = _batch_elements(batch)
    assert res == batch


def test_data_loader_iterator_prefetch_state():
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
    assert state2["last_seen_indices"]["0"] == 0
    it2.set_state({"last_seen_indices": {}})

    # Test StopIteration in set_state
    it2.set_state({"last_seen_indices": {"0": 10}})


def test_checkpoint_handler():
    assert_equal_output_after_checkpoint(None)
    handler = PyGrainCheckpointHandler()
    handler.save()
    handler.restore()


def test_datasources_misc():
    ds = RandomAccessDataSource()
    assert len(ds) == 1

    in_mem = InMemoryDataSource([1, 2], name="test")
    assert len(in_mem) == 2
    assert in_mem[1] == 2
    in_mem.close()
    in_mem.unlink()
    assert str(in_mem) == "InMemoryDataSource(name=test, len=2)"

    sh_mem = SharedMemoryDataSource([1, 2], name="test")
    sh_mem.close()
    sh_mem.unlink()
    assert str(sh_mem) == "InMemoryDataSource(name=test, len=2)"

    rng_ds = RangeDataSource(0, 10, 2)
    assert repr(rng_ds) == "RangeDataSource(start=0, stop=10, step=2)"


def test_sharding_misc():
    shard = ShardByJaxProcess(drop_remainder=True)
    assert shard.shard_index == 0
    assert shard.shard_count == 1
    assert shard.drop_remainder is True


def test_options_misc():
    MultiprocessingOptions(num_workers=1)


def test_index_sampler_seed_type():
    with pytest.raises(TypeError):
        IndexSampler(10, seed="not_an_int")
    with pytest.raises(ValueError):
        IndexSampler(10, seed=-1)
    with pytest.raises(ValueError):
        IndexSampler(10, seed=2**32 + 1)

    idx_s = IndexSampler(10, shard_options=NoSharding(0, 2, True))
    list(iter(idx_s))


def test_operations_not_implemented():
    op_map_index = MapWithIndexOperation()
    with pytest.raises(NotImplementedError):
        op_map_index.map_with_index(0, 1)

    op_map = MapOperation()
    with pytest.raises(NotImplementedError):
        op_map.map(1)

    op_rmap = RandomMapOperation()
    with pytest.raises(NotImplementedError):
        op_rmap.random_map(1, None)

    op_filter = FilterOperation()
    with pytest.raises(NotImplementedError):
        op_filter.filter(1)

    op_flat = FlatMapOperation()
    with pytest.raises(NotImplementedError):
        op_flat.flat_map(1)


def test_operations_implemented():
    op_map_idx = MapWithIndexOperation(lambda i, x: x)
    assert op_map_idx.map_with_index(0, 1) == 1
    rec = Record(RecordMetadata(0, 0), 1)
    list(op_map_idx([rec]))

    op_flat = FlatMapOperation(lambda x: [x, x])
    assert op_flat.flat_map(1) == [1, 1]

    op_copy = CopyNumPyArrayToSharedMemoryOperation()
    list(op_copy([rec]))


def test_fake_class():
    cls = fake_class()
    cls()


def test_batch_and_pad():
    res = batch_and_pad([np.array([1]), np.array([2])], batch_size=3)
    assert len(res) == 3

    res2 = batch_and_pad([1, 2], batch_size=3)
    assert len(res2) == 3


def test_dataset_misc():
    ds = Dataset([1, 2, 3])
    ds2 = ds.seed(42)
    assert ds2._seed == 42

    ds3 = ds.shuffle(12)
    assert ds3._seed == 12

    it_ds = ds.to_iter_dataset()
    assert isinstance(it_ds, IterDataset)
    str(it_ds)
    it_ds_iter = iter(it_ds)
    next(it_ds_iter)

    res = apply_transformations(ds, MapOperation(lambda x: x))
    assert len(res.operations) == 1

    get_element_spec(ds)

    ds4 = ds.map(lambda x: x)
    assert len(ds4.operations) == 1

    ds5 = ds.map_with_index(lambda i, x: x)
    assert len(ds5.operations) == 1

    ds6 = ds.filter(lambda x: x)
    assert len(ds6.operations) == 1

    ds7 = ds.batch(1)
    assert len(ds7.operations) == 1

    rng = Dataset.range(10)
    assert len(rng) == 10
    Dataset.range(1, 10)
    Dataset.range(1, 10, 2)

    assert ds[0] == 1


def test_no_sharding_repr():
    ns = NoSharding(1, 2, True)
    assert repr(ns) == "NoSharding(shard_index=1, shard_count=2, drop_remainder=True)"


def test_seq_sampler_repr():
    ss = SequentialSampler(10, NoSharding(0, 2, True))
    assert "SequentialSampler" in repr(ss)
    list(iter(ss))

    ss[0]


def test_misc_classes():
    DatasetSelectionMap()
    PyGrainDatasetIterator()
    dl = load([1, 2], batch_size=2)
    assert dl.worker_count == 0
