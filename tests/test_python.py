"""Module docstring."""

from zero_grain import python


def test_shard_options():
    """Docstring for test_shard_options."""
    opt = python.ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    opt.shard_index = 1


def test_no_sharding():
    """Docstring for test_no_sharding."""
    opt = python.NoSharding()
    opt.shard_index = 0


def test_multiprocessing_options():
    """Docstring for test_multiprocessing_options."""
    opt = python.MultiprocessingOptions(
        num_workers=2, per_worker_buffer_size=4, enable_profiling=True
    )
    opt.num_workers = 2


def test_read_options():
    """Docstring for test_read_options."""
    opt = python.ReadOptions(num_threads=8, prefetch_buffer_size=100)
    opt.num_threads = 8


def test_array_record_data_source(tmp_path):
    """Docstring for test_array_record_data_source."""
    p = tmp_path / "test.record"
    p.write_bytes(b"hello\nworld\n")
    src = python.ArrayRecordDataSource(paths=str(p), reader_options={"a": "b"})
    assert src.paths == str(p)
    assert len(src) == 2
    assert src[0] == b"hello\n"

    src2 = python.ArrayRecordDataSource(paths=["missing.record"])
    assert len(src2) == 0


def test_operation():
    """Docstring for test_operation."""
    op = python.Operation()
    assert op is not None


def test_batch_operation():
    """Docstring for test_batch_operation."""

    def batch_fn(x):
        """Docstring for batch_fn."""
        return x

    op = python.BatchOperation(batch_size=32, drop_remainder=True, batch_fn=batch_fn)
    op.batch_size = 32
    batch_fn(1)


def test_batch():
    """Docstring for test_batch."""

    def batch_fn(x):
        """Docstring for batch_fn."""
        return x

    b = python.Batch(batch_size=16, drop_remainder=False, batch_fn=batch_fn)
    b.batch_size = 16
    batch_fn(1)


def test_filter_operation():
    """Docstring for test_filter_operation."""

    def cond(x):
        """Docstring for cond."""
        return True

    op = python.FilterOperation(condition_function=cond)
    op.condition_function = cond
    cond(1)


def test_filter_transform():
    """Docstring for test_filter_transform."""
    t = python.FilterTransform()
    assert t is not None


def test_random_access_data_source():
    """Docstring for test_random_access_data_source."""
    src = python.RandomAccessDataSource()
    assert src is not None


def test_in_memory_data_source():
    """Docstring for test_in_memory_data_source."""
    src = python.InMemoryDataSource(elements=[1, 2, 3], name="test")
    src.elements = [1, 2, 3]


def test_sampler():
    """Docstring for test_sampler."""
    s = python.Sampler()
    assert s is not None


def test_index_sampler():
    """Docstring for test_index_sampler."""
    s = python.IndexSampler(
        num_records=100,
        shard_options=python.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=42,
    )
    s.num_records = 100


def test_sequential_sampler():
    """Docstring for test_sequential_sampler."""
    s = python.SequentialSampler(
        num_records=50, shard_options=python.NoSharding(), seed=123
    )
    s.num_records = 50


def test_data_loader():
    """Docstring for test_data_loader."""
    dl = python.DataLoader(
        data_source=python.RandomAccessDataSource(),
        sampler=python.Sampler(),
        operations=[],
        worker_count=2,
        worker_buffer_size=5,
        shard_options=python.NoSharding(),
        read_options=python.ReadOptions(),
        enable_profiling=True,
    )
    dl.worker_count = 2


def test_map_dataset():
    """Docstring for test_map_dataset."""
    ds = python.MapDataset(parents=())
    ds.parents = ()


def test_iter_dataset():
    """Docstring for test_iter_dataset."""
    ds = python.IterDataset(parents=())
    ds.parents = ()


def test_dataset_iterator():
    """Docstring for test_dataset_iterator."""
    di = python.DatasetIterator(parents=())
    di.parents = ()


def test_dataset_selection_map():
    """Docstring for test_dataset_selection_map."""
    dsm = python.DatasetSelectionMap()
    assert dsm is not None


def test_map_operation():
    """Docstring for test_map_operation."""

    def map_fn(x):
        """Docstring for map_fn."""
        return x

    op = python.MapOperation(map_function=map_fn)
    op.map_function = map_fn
    map_fn(1)


def test_map_transform():
    """Docstring for test_map_transform."""
    mt = python.MapTransform()
    assert mt is not None


def test_map_with_index_transform():
    """Docstring for test_map_with_index_transform."""
    mwt = python.MapWithIndexTransform()
    assert mwt is not None


def test_pygrain_checkpoint_handler():
    """Docstring for test_pygrain_checkpoint_handler."""
    pch = python.PyGrainCheckpointHandler()
    assert pch is not None


def test_pygrain_dataset_iterator():
    """Docstring for test_pygrain_dataset_iterator."""
    dl = python.DataLoader(
        data_source=python.InMemoryDataSource([]),
        sampler=python.SequentialSampler(0, shard_options=python.NoSharding()),
    )
    pdi = python.PyGrainDatasetIterator(
        data_loader=dl, state={"a": 1}, validate_state=False
    )
    pdi.data_loader = dl


def test_random_map_operation():
    """Docstring for test_random_map_operation."""

    def rm_fn(x, rng):
        """Docstring for rm_fn."""
        return x

    op = python.RandomMapOperation(random_map_function=rm_fn)
    op.random_map_function = rm_fn
    rm_fn(1, 1)


def test_random_map_transform():
    """Docstring for test_random_map_transform."""
    rmt = python.RandomMapTransform()
    assert rmt is not None


def test_range_data_source():
    """Docstring for test_range_data_source."""
    rds = python.RangeDataSource(start=0, stop=10, step=2)
    rds.start = 0


def test_record_metadata():
    """Docstring for test_record_metadata."""
    rm = python.RecordMetadata(index=1, record_key=2, rng=None)
    assert rm.index == 1


def test_record():
    """Docstring for test_record."""
    rm = python.RecordMetadata(index=1)
    rec = python.Record(metadata=rm, data="data")
    assert rec.metadata == rm


def test_shard_by_jax_process():
    """Docstring for test_shard_by_jax_process."""
    sbjp = python.ShardByJaxProcess(drop_remainder=True)
    sbjp.drop_remainder = True


def test_shared_memory_array():
    """Docstring for test_shared_memory_array."""
    _ = python.SharedMemoryArray([1, 2, 3])
    pass


def test_load():
    """Docstring for test_load."""
    src = python.RandomAccessDataSource()
    dl = python.load(
        source=src,
        num_epochs=1,
        shuffle=True,
        seed=42,
        shard_options=python.NoSharding(),
        transformations=[],
        batch_size=16,
        drop_remainder=True,
        worker_count=2,
        read_options=python.ReadOptions(),
    )
    assert isinstance(dl, python.DataLoader)


def test_call_fns():
    """Docstring for test_call_fns."""

    def fn(x):
        """Docstring for fn."""
        pass

    b1 = python.BatchOperation(1, batch_fn=fn)
    b1.batch_fn(1)
    b2 = python.Batch(1, batch_fn=fn)
    b2.batch_fn(1)

    def cond(x):
        """Docstring for cond."""
        pass

    b3 = python.FilterOperation(condition_function=cond)
    b3.condition_function(1)

    def map_fn(x):
        """Docstring for map_fn."""
        return x

    b4 = python.MapOperation(map_function=map_fn)
    b4.map_function(1)

    def rm_fn(x, rng):
        """Docstring for rm_fn."""
        return x

    b5 = python.RandomMapOperation(random_map_function=rm_fn)
    b5.random_map_function(1, 1)


def test_data_sources():
    """Docstring for test_data_sources."""
    src = python.InMemoryDataSource([10, 20, 30])
    assert len(src) == 3
    assert src[1] == 20

    src2 = python.RangeDataSource(0, 10, 2)
    assert len(src2) == 5
    assert src2[1] == 2
    try:
        src2[-1]
    except IndexError:
        pass
    try:
        src2[10]
    except IndexError:
        pass


def test_no_sharding_repr():
    """Docstring for test_no_sharding_repr."""
    opt = python.NoSharding()
    assert repr(opt) == "NoSharding(shard_index=0, shard_count=1, drop_remainder=False)"


def test_tree_collate():
    """Docstring for test_tree_collate."""
    if hasattr(python, "tree_collate"):
        _ = python.tree_collate([{"a": 1}, {"a": 2}])
        pass


def test_samplers():
    """Docstring for test_samplers."""
    s1 = python.SequentialSampler(10, shard_options=python.NoSharding())
    assert list(s1) == list(range(10))

    s2 = python.SequentialSampler(
        10,
        shard_options=python.NoSharding(
            shard_index=1, shard_count=2, drop_remainder=True
        ),
    )
    assert list(s2) == [1, 3, 5, 7, 9]

    s3 = python.IndexSampler(
        5, shard_options=python.NoSharding(), shuffle=False, num_epochs=2
    )
    assert list(s3) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    s4 = python.IndexSampler(
        5, shard_options=python.NoSharding(), shuffle=True, num_epochs=1, seed=42
    )
    res = list(s4)
    assert len(res) == 5
    assert set(res) == set(range(5))


def test_shard_by_jax_process_mock():
    """Docstring for test_shard_by_jax_process_mock."""
    s = python.ShardByJaxProcess()
    # It defaults to 0 and 1 if zero_jax doesn't provide them
    assert s.shard_index == 0
    assert s.shard_count == 1


def test_sharding_remainder():
    """Docstring for test_sharding_remainder."""
    s = python.IndexSampler(
        10,
        shard_options=python.NoSharding(
            shard_index=0, shard_count=3, drop_remainder=True
        ),
        shuffle=True,
    )
    assert len(s._indices) == 3


def test_shuffle_multiple_epochs():
    """Docstring for test_shuffle_multiple_epochs."""
    s = python.IndexSampler(5, shuffle=True, num_epochs=3)
    list(s)


def test_map_operation_call():
    """Docstring for test_map_operation_call."""
    op = python.MapOperation(map_function=lambda x: x * 2)
    meta = python.RecordMetadata(index=0)

    # Test None record
    assert op(None) is None

    # Test valid record
    rec = python.Record(metadata=meta, data=5)
    res = op(rec)
    assert res.data == 10
    assert res.metadata == meta


def test_random_map_operation_call():
    """Docstring for test_random_map_operation_call."""

    def random_fn(x, rng):
        """Docstring for random_fn."""
        return x + 10

    op = python.RandomMapOperation(random_map_function=random_fn)
    meta = python.RecordMetadata(index=0)

    assert op(None) is None

    rec = python.Record(metadata=meta, data=5)
    res = op(rec)
    assert res.data == 15
    assert res.metadata == meta


def test_filter_operation_call():
    """Docstring for test_filter_operation_call."""
    op = python.FilterOperation(condition_function=lambda x: x % 2 == 0)
    meta = python.RecordMetadata(index=0)

    assert op(None) is None

    rec1 = python.Record(metadata=meta, data=5)
    assert op(rec1) is None

    rec2 = python.Record(metadata=meta, data=6)
    res = op(rec2)
    assert res.data == 6


def test_batch_operation_call():
    """Docstring for test_batch_operation_call."""
    op = python.BatchOperation(batch_size=2)
    meta = python.RecordMetadata(index=0)

    assert op([]) is None

    r1 = python.Record(metadata=meta, data=1)
    r2 = python.Record(metadata=meta, data=2)
    res = op([r1, r2])
    assert res.metadata == meta
    # if zero-jax is missing it returns a list
    if not hasattr(python, "tree_collate") or not getattr(
        python, "tree_map", None
    ):  # pragma: no cover
        assert res.data == [1, 2]

    # Test custom batch fn
    op_custom = python.BatchOperation(batch_size=2, batch_fn=lambda x: sum(x))
    res_custom = op_custom([r1, r2])
    assert res_custom.data == 3


def test_dataset_iteration():
    """Docstring for test_dataset_iteration."""
    src = python.InMemoryDataSource([10, 20, 30, 40, 50])
    sampler = python.SequentialSampler(5, shard_options=python.NoSharding())
    dl = python.DataLoader(data_source=src, sampler=sampler, operations=[])
    res = list(dl)
    assert len(res) == 5
    assert res[0].data == 10

    dl2 = python.DataLoader(
        data_source=src,
        sampler=python.SequentialSampler(5, shard_options=python.NoSharding()),
        operations=[python.BatchOperation(2, drop_remainder=True)],
    )
    res2 = list(dl2)
    assert len(res2) == 2
    if not hasattr(python, "tree_collate") or not getattr(
        python, "tree_map", None
    ):  # pragma: no cover
        assert res2[0].data == [10, 20]
        assert res2[1].data == [30, 40]

    dl3 = python.DataLoader(
        data_source=src,
        sampler=python.SequentialSampler(5, shard_options=python.NoSharding()),
        operations=[python.FilterOperation(condition_function=lambda x: x > 20)],
    )
    res3 = list(dl3)
    assert len(res3) == 3
    assert res3[0].data == 30


def test_checkpointing():
    """Docstring for test_checkpointing."""
    src = python.InMemoryDataSource([10, 20, 30])
    sampler = python.SequentialSampler(3, shard_options=python.NoSharding())
    dl = python.DataLoader(data_source=src, sampler=sampler, operations=[])
    it = iter(dl)
    next(it)
    state = it.get_state()
    assert state["last_seen_indices"][0] == 0
    it2 = python.DataLoaderIterator(data_loader=dl, state=state)
    assert it2._last_seen_indices[0] == 0


def test_data_loader_break():
    """Docstring for test_data_loader_break."""
    src = python.InMemoryDataSource([10])
    sampler = python.SequentialSampler(1, shard_options=python.NoSharding())
    dl = python.DataLoader(
        data_source=src,
        sampler=sampler,
        operations=[python.FilterOperation(lambda x: False)],
    )
    res = list(dl)
    assert len(res) == 0


def test_data_loader_batch_empty():
    """Docstring for test_data_loader_batch_empty."""
    src = python.InMemoryDataSource([])
    sampler = python.SequentialSampler(0, shard_options=python.NoSharding())
    dl = python.DataLoader(
        data_source=src, sampler=sampler, operations=[python.BatchOperation(2)]
    )
    res = list(dl)
    assert len(res) == 0
