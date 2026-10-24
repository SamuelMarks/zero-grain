"""Comprehensive unit tests for the ``zero_grain`` package."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import numpy as np
import pytest

from zero_grain import (
    ArrayRecordDataSource,
    Batch,
    BatchDataset,
    BatchOperation,
    BatchTransform,
    CheckpointHandler,
    CopyNumPyArrayToSharedMemory,
    CopyNumPyArrayToSharedMemoryOperation,
    DataLoader,
    DataLoaderOptions,
    Dataset,
    DatasetIterator,
    DatasetOptions,
    DatasetSelectionMap,
    DeterministicGenerator,
    FileInstruction,
    Filter,
    FilterDataset,
    FilterOperation,
    FilterTransform,
    FlatMap,
    FlatMapOperation,
    FlatMapTransform,
    IndexSampler,
    InMemoryDataSource,
    IterDataset,
    LoadOptions,
    Map,
    MapDataset,
    MapOperation,
    MapTransform,
    MapWithIndex,
    MapWithIndexOperation,
    MapWithIndexTransform,
    MultiprocessingOptions,
    NoSharding,
    Operation,
    PyGrainCheckpointHandler,
    PyGrainDatasetIterator,
    RandomAccessDataSource,
    RandomMap,
    RandomMapOperation,
    RandomMapTransform,
    RangeDataSource,
    ReadOptions,
    Record,
    RecordMetadata,
    Sampler,
    SequentialSampler,
    ShardByJaxProcess,
    ShardOptions,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    SharedMemoryDataSource,
    TfRandomMap,
    TfRandomMapTransform,
    _EagerDataLoaderIterator,
    _batch_elements,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    batch_and_pad,
    create_rng,
    derive_rng,
    even_split,
    fake_class,
    get_element_spec,
    load,
    map_tree,
    python,
    sharding,
    shared_memory_array,
    spec_like,
    transforms,
)
from zero_grain._types import ElementTree, RNGGeneratorProtocol


class _SamplePair(NamedTuple):
    """Sample NamedTuple for testing PyTree batching."""

    left: int | str
    right: int | str


def test_record_metadata() -> None:
    """Verify ``RecordMetadata`` initialization, equality, string formatting, and key removal."""
    rng = DeterministicGenerator(seed=7, algorithm_name="Philox")
    rm1 = RecordMetadata(index=1, record_key=10, rng=rng)
    rm2 = RecordMetadata(index=1, record_key=10, rng=rng)
    rm3 = RecordMetadata(index=2)
    other_val: ElementTree = "not metadata"
    assert rm1 == rm2
    assert rm1 != rm3
    assert rm1 != other_val
    assert str(rm1) == "RecordMetadata(index=1, record_key=10, rng=Generator(Philox))"
    assert repr(rm3) == "RecordMetadata(index=2, record_key=None, rng=None)"
    stripped = rm1.remove_record_key()
    assert rm1.record_key == 10
    assert stripped.record_key is None
    assert stripped.index == 1


def test_record() -> None:
    """Verify ``Record`` container attributes and equality."""
    rm = RecordMetadata(index=1)
    rec = Record(metadata=rm, data={"a": 1})
    assert rec.metadata == rm
    assert rec.data == {"a": 1}


def test_rng_utilities() -> None:
    """Verify ``DeterministicGenerator``, ``create_rng``, and ``derive_rng`` behavior."""
    gen1 = DeterministicGenerator(seed=42)
    gen2 = DeterministicGenerator(seed=42)
    gen3 = DeterministicGenerator(seed=99)
    assert gen1 == gen2
    assert gen1 != gen3
    assert 0.0 <= gen1.random() < 1.0
    assert 2.0 <= gen1.uniform(2.0, 5.0) < 5.0
    assert 0 <= gen1.integers(10) < 10
    assert 5 <= gen1.integers(5, 8) < 8
    perm4 = gen1.permutation(4)
    perm3 = gen1.permutation([10, 20, 30])
    assert sorted(list(perm4)) == [0, 1, 2, 3]
    assert sorted(list(perm3)) == [10, 20, 30]
    assert repr(gen1) == "Generator(PCG64)"

    derived = derive_rng(None, 3, op_seed=1)
    assert repr(derived) == "Generator(Philox)"
    np_rng = create_rng(123)
    assert 0.0 <= np_rng.random() < 1.0


def test_batch_and_dataset_options() -> None:
    """Verify ``Batch``, ``DatasetOptions``, ``MultiprocessingOptions``, and ``ReadOptions``."""
    b: Batch[int, list[int]] = Batch(batch_size=4, drop_remainder=True)
    assert b.batch_size == 4
    assert b.drop_remainder is True
    with pytest.raises(ValueError, match="batch_size must be positive"):
        Batch(batch_size=0)

    opt = DatasetOptions(
        filter_warn_threshold_ratio=0.5,
        filter_raise_threshold_ratio=0.8,
    )
    assert opt.filter_warn_threshold_ratio == 0.5
    assert opt.filter_raise_threshold_ratio == 0.8

    mp1 = MultiprocessingOptions(num_workers=2, per_worker_buffer_size=4)
    mp2 = MultiprocessingOptions(num_workers=2, per_worker_buffer_size=4)
    assert mp1 == mp2
    assert mp1 != MultiprocessingOptions(num_workers=1)
    with pytest.raises(ValueError, match="num_workers"):
        MultiprocessingOptions(num_workers=-1)
    with pytest.raises(ValueError, match="per_worker_buffer_size"):
        MultiprocessingOptions(per_worker_buffer_size=0)

    ro1 = ReadOptions(num_threads=4, prefetch_buffer_size=8)
    ro2 = ReadOptions(num_threads=4, prefetch_buffer_size=8)
    assert ro1 == ro2
    assert ro1 != ReadOptions(num_threads=2, prefetch_buffer_size=8)
    assert repr(ro1) == "ReadOptions(num_threads=4, prefetch_buffer_size=8)"


def test_range_data_source() -> None:
    """Verify ``RangeDataSource`` length, indexing, bounds, and validation."""
    rds = RangeDataSource(0, 10, 2)
    assert len(rds) == 5
    assert rds[0] == 0
    assert rds[4] == 8
    with pytest.raises(IndexError):
        _ = rds[5]
    with pytest.raises(IndexError):
        _ = rds[-1]
    with pytest.raises(ValueError, match="step must not be zero"):
        RangeDataSource(0, 10, 0)
    assert repr(rds) == "RangeDataSource(start=0, stop=10, step=2)"

    base_ds: RandomAccessDataSource[int] = RandomAccessDataSource()
    assert len(base_ds) == 0
    with pytest.raises(IndexError):
        _ = base_ds[0]


def test_in_memory_and_shared_memory_data_sources() -> None:
    """Verify ``InMemoryDataSource`` and ``SharedMemoryDataSource`` operations."""
    ims = InMemoryDataSource([1, 2, 3], name="test")
    assert len(ims) == 3
    assert ims[1] == 2
    assert list(ims) == [1, 2, 3]
    ims.close()
    ims.unlink()
    assert "name=test, len=3" in str(ims)
    assert repr(ims) == str(ims)

    smds = SharedMemoryDataSource([4, 5], name="shared")
    assert len(smds) == 2
    assert smds[0] == 4
    smds.close()
    smds.unlink()
    assert str(smds) == "SharedMemoryDataSource(name=shared, len=2)"


def test_array_record_data_source(tmp_path: Path) -> None:
    """Verify ``ArrayRecordDataSource`` and ``FileInstruction`` behavior.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    fi = FileInstruction(filename="shard_0", skip=0, take=5, examples_in_shard=5)
    ards_single = ArrayRecordDataSource(fi)
    assert len(ards_single) == 5
    assert ards_single[0] == b"0"
    with pytest.raises(IndexError):
        _ = ards_single[10]

    ards_multi = ArrayRecordDataSource(["a", "b"], reader_options={"opt": "1"})
    assert len(ards_multi) == 10
    assert "ArrayRecordDataSource" in repr(ards_multi)

    ards_none = ArrayRecordDataSource(None)
    assert len(ards_none) == 0

    real_file = tmp_path / "records.txt"
    real_file.write_bytes(chr(10).encode("utf-8").join([b"rec0", b"rec1", b"rec2"]))
    ards_file = ArrayRecordDataSource(str(real_file))
    assert len(ards_file) == 3
    assert [ards_file[i] for i in range(3)] == [b"rec0", b"rec1", b"rec2"]

    fi_real = FileInstruction(filename=str(real_file), skip=1, take=2, examples_in_shard=3)
    ards_fi_real = ArrayRecordDataSource(fi_real)
    assert [ards_fi_real[i] for i in range(len(ards_fi_real))] == [b"rec1", b"rec2"]


def test_sharding() -> None:
    """Verify ``ShardOptions``, ``NoSharding``, ``ShardByJaxProcess``, and ``even_split``."""
    so = ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    assert so.shard_index == 1
    assert so == ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    assert so != ShardOptions(shard_index=0, shard_count=2, drop_remainder=True)
    with pytest.raises(ValueError, match="Invalid shard_index"):
        ShardOptions(shard_index=2, shard_count=2)

    ns = NoSharding()
    assert repr(ns) == "NoSharding(shard_index=0, shard_count=1, drop_remainder=False)"
    sb = ShardByJaxProcess(drop_remainder=True)
    assert sb.drop_remainder is True
    assert sharding.ShardOptions is ShardOptions
    assert even_split(10, ShardOptions(0, 3, drop_remainder=True)) == (0, 3)
    assert even_split(10, ShardOptions(0, 3, drop_remainder=False)) == (0, 4)


def test_samplers() -> None:
    """Verify ``Sampler``, ``SequentialSampler``, and ``IndexSampler``."""
    base_sampler = Sampler()
    assert len(base_sampler) == 0
    assert list(base_sampler) == []
    with pytest.raises(IndexError):
        _ = base_sampler[0]

    ss = SequentialSampler(num_records=2, seed=5)
    assert ss[0].record_key == 0
    assert ss[0].rng is not None
    with pytest.raises(IndexError):
        _ = ss[3]
    with pytest.raises(IndexError):
        _ = ss[-1]
    assert [m.record_key for m in ss] == [0, 1]
    assert "SequentialSampler(num_records=2" in repr(ss)

    iss = IndexSampler(num_records=3, num_epochs=2)
    assert iss[4].record_key == 1
    assert len(iss) == 6
    assert len(list(iss)) == 6
    assert "IndexSampler(num_records=3" in repr(iss)
    with pytest.raises(ValueError, match="num_records must be positive"):
        IndexSampler(num_records=0)


def test_transforms_and_operations() -> None:
    """Verify transform base classes, operation classes, and PyTree batching."""
    mt: MapTransform[int, int] = MapTransform()
    mwit: MapWithIndexTransform[int, int] = MapWithIndexTransform()
    rmt: RandomMapTransform[int, int] = RandomMapTransform()
    ft: FilterTransform[int] = FilterTransform()
    fmt: FlatMapTransform[int, int] = FlatMapTransform()
    with pytest.raises(NotImplementedError):
        mt.map(1)
    with pytest.raises(NotImplementedError):
        mwit.map_with_index(0, 1)
    with pytest.raises(NotImplementedError):
        rmt.random_map(1, create_rng(0))
    with pytest.raises(NotImplementedError):
        ft.filter(1)
    with pytest.raises(NotImplementedError):
        fmt.flat_map(1)

    mo: MapOperation[int, int] = MapOperation()
    mwio: MapWithIndexOperation[int, int] = MapWithIndexOperation()
    rmo: RandomMapOperation[int, int] = RandomMapOperation()
    fo: FilterOperation[int] = FilterOperation()
    fmo: FlatMapOperation[int, int] = FlatMapOperation()
    with pytest.raises(NotImplementedError):
        mo.map(1)
    with pytest.raises(NotImplementedError):
        mwio.map_with_index(0, 1)
    with pytest.raises(NotImplementedError):
        rmo.random_map(1, create_rng(0))
    with pytest.raises(NotImplementedError):
        fo.filter(1)
    with pytest.raises(NotImplementedError):
        fmo.flat_map(1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchOperation(batch_size=0)

    base_op: Operation[int, int] = Operation()
    assert base_op.apply_element(5) == 5
    in_rec = Record(metadata=RecordMetadata(index=0, record_key=0), data=5)
    with pytest.raises(NotImplementedError):
        list(base_op([in_rec]))

    mwi_op: MapWithIndexOperation[int, int] = MapWithIndexOperation(lambda i, x: i + x)
    out_mwi = list(mwi_op([in_rec]))
    assert out_mwi[0].data == 5

    rm_op: RandomMapOperation[int, int] = RandomMapOperation(lambda x, rng: x + rng.integers(0, 1))
    out_rm = list(rm_op([Record(metadata=RecordMetadata(index=0, rng=None), data=7)]))
    assert out_rm[0].data == 7

    fm_op: FlatMapOperation[int, int] = FlatMapOperation(lambda x: [x, x + 1])
    out_fm = list(fm_op([Record(metadata=RecordMetadata(index=0), data=3)]))
    assert [r.data for r in out_fm] == [3, 4]

    batch_op: BatchOperation[int] = BatchOperation(
        batch_size=2,
        drop_remainder=False,
        batch_fn=lambda xs: sum(xs),
    )
    out_b = list(
        batch_op(
            [
                Record(metadata=RecordMetadata(index=0), data=2),
                Record(metadata=RecordMetadata(index=1), data=3),
                Record(metadata=RecordMetadata(index=2), data=4),
            ]
        )
    )
    assert [r.data for r in out_b] == [5, 4]

    shm_op = CopyNumPyArrayToSharedMemoryOperation()
    shm_rec: Record[ElementTree] = Record(metadata=RecordMetadata(index=0), data=np.array([1, 2]))
    shm_out = list(shm_op([shm_rec]))
    assert isinstance(shm_out[0].data, SharedMemoryArrayMetadata)

    assert np.array_equal(_batch_elements([1, 2]), np.array([1, 2]))
    assert _batch_elements([]) == []
    assert batch_and_pad([], 2) == []
    padded_pair = batch_and_pad([_SamplePair(1, 2)], 2)
    assert isinstance(padded_pair, _SamplePair)
    padded_tuple = batch_and_pad([(1.5, "a")], 2)
    assert isinstance(padded_tuple, tuple)
    padded_list = batch_and_pad([[1, 2]], 2)
    assert isinstance(padded_list, (list, np.ndarray))
    assert len(padded_list) == 2

    mapped = map_tree(lambda x: x, _SamplePair(1, 2))
    assert mapped == _SamplePair(1, 2)
    assert spec_like(_SamplePair(1, 2)) == _SamplePair("int", "int")
    assert spec_like((1, [2])) == ("int", ["int"])


def test_data_loader_and_checkpoint_handler(tmp_path: Path) -> None:
    """Verify ``DataLoader``, ``DataLoaderIterator``, and ``PyGrainCheckpointHandler``.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    dl_opts = DataLoaderOptions(worker_count=2)
    dl = DataLoader(data_source=InMemoryDataSource([10, 20]), options=dl_opts)
    assert dl.worker_count == 2
    it = iter(dl)
    assert iter(it) is it
    it.start_prefetch()
    assert next(it) == 10

    ch = PyGrainCheckpointHandler()
    ch.save(None, None)
    assert ch.restore(None, None) is None
    assert ch.restore(tmp_path / "nonexistent", it) is it

    ckpt_dir = tmp_path / "ckpt"
    ch.save(ckpt_dir, it)
    raw_state: PyGrainDatasetIterator[int] | None = None
    restored = ch.restore(ckpt_dir, raw_state)
    assert isinstance(restored, dict)

    it2 = iter(dl)
    ch.restore(ckpt_dir, it2)
    assert next(it2) == 20
    with pytest.raises(StopIteration):
        next(it2)

    with pytest.raises(ValueError, match="worker_count"):
        it2.set_state({"worker_count": 99})

    assert_equal_output_after_checkpoint(None)
    assert_equal_output_after_checkpoint(Dataset.range(3).to_iter_dataset())


def test_datasets() -> None:
    """Verify ``MapDataset``, ``FilterDataset``, ``BatchDataset``, ``IterDataset``, and ``DatasetIterator``."""
    ds = Dataset.range(5)
    assert len(ds) == 5
    assert ds[1] == 1
    assert ds[-1] == 4
    with pytest.raises(IndexError):
        _ = ds[5]
    with pytest.raises(IndexError):
        _ = ds[-6]
    assert list(ds[1:4]) == [1, 2, 3]
    assert len(Dataset.range(1, 4)) == 3
    assert len(Dataset.source_dataset([10, 20])) == 2
    ds_from_parents: MapDataset[int] = MapDataset([ds])
    assert len(ds_from_parents) == 5
    ds_empty: MapDataset[int] = MapDataset(())
    assert len(ds_empty) == 0

    def _double(x: ElementTree) -> ElementTree:
        """Double integer ``x``.

        Args:
            x (ElementTree): Input element.

        Returns:
            ElementTree: ``x * 2``.
        """
        return int(x) * 2 if isinstance(x, int) else x

    def _add_idx(i: int, x: ElementTree) -> ElementTree:
        """Add ``i`` and ``x``.

        Args:
            i (int): Index.
            x (ElementTree): Input element.

        Returns:
            ElementTree: ``i + x``.
        """
        return i + int(x) if isinstance(x, int) else x

    def _keep_rng(x: ElementTree, rng: RNGGeneratorProtocol) -> ElementTree:
        """Return ``x`` with deterministic RNG consumed.

        Args:
            x (ElementTree): Input element.
            rng (RNGGeneratorProtocol): Per-record RNG.

        Returns:
            ElementTree: ``x``.
        """
        _ = rng.random()
        return x

    def _gt_one(x: ElementTree) -> bool:
        """Return True if ``x > 1``.

        Args:
            x (ElementTree): Input element.

        Returns:
            bool: True when greater than 1.
        """
        return isinstance(x, int) and x > 1

    md: MapDataset[int] = ds.map(_double)
    mwi: MapDataset[int] = ds.map_with_index(_add_idx)
    rmd: MapDataset[int] = ds.random_map(_keep_rng, seed=3)
    fd = ds.filter(_gt_one)
    bd: BatchDataset[list[int]] = ds.batch(2)
    bd_drop: BatchDataset[list[int]] = ds.batch(2, drop_remainder=True)

    assert list(md) == [0, 2, 4, 6, 8]
    assert md[2] == 4
    assert list(mwi) == [0, 2, 4, 6, 8]
    assert mwi[3] == 6
    assert list(rmd) == [0, 1, 2, 3, 4]
    assert rmd[4] == 4
    assert list(fd) == [2, 3, 4]
    assert fd[0] is None
    assert fd[2] == 2
    assert len(bd) == 3
    assert len(bd_drop) == 2
    assert len(BatchDataset()) == 0
    b0 = bd[0]
    b_last = bd[-1]
    assert isinstance(b0, np.ndarray) and b0.tolist() == [0, 1]
    assert isinstance(b_last, np.ndarray) and b_last.tolist() == [4]
    assert isinstance(bd[0:2], MapDataset)
    with pytest.raises(IndexError):
        _ = bd[5]
    assert [x.tolist() for x in bd if isinstance(x, np.ndarray)] == [[0, 1], [2, 3], [4]]

    shuffled_ds = ds.shuffle(seed=9)
    assert len(shuffled_ds) == 5
    assert set(shuffled_ds) == {0, 1, 2, 3, 4}
    assert list(shuffled_ds) != [0, 1, 2, 3, 4]
    assert [shuffled_ds[i] for i in range(5)] == list(shuffled_ds)
    seeded_ds = ds.seed(1)
    assert seeded_ds._seed == 1
    sliced_ds = ds.slice(slice(1, 4))
    assert list(sliced_ds) == [1, 2, 3]
    assert [sliced_ds[i] for i in range(len(sliced_ds))] == [1, 2, 3]
    repeated_ds = ds.repeat(2)
    assert list(repeated_ds) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert [repeated_ds[i] for i in range(len(repeated_ds))] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    id_ds = ds.to_iter_dataset()
    assert next(id_ds) == 0
    assert next(id_ds) == 1
    assert "DataLoader" in str(id_ds)

    id_from_iter = IterDataset(id_ds)
    assert list(id_from_iter) == [0, 1, 2, 3, 4]
    id_from_seq = IterDataset([ds])
    assert list(id_from_seq) == [0, 1, 2, 3, 4]
    id_from_iter_seq = IterDataset([id_ds])
    assert list(id_from_iter_seq) == [0, 1, 2, 3, 4]
    id_empty: IterDataset[int] = IterDataset(())
    assert list(id_empty) == []

    di1 = DatasetIterator(id_ds)
    di1.start_prefetch()
    assert "PyGrainDatasetIterator" in str(di1)
    assert "PyGrainDatasetIterator" in repr(di1)
    di2 = DatasetIterator(di1)
    di3 = DatasetIterator([di1])
    di_empty: DatasetIterator[int] = DatasetIterator(())
    assert iter(di2) is di2
    assert next(di3) == 0
    with pytest.raises(StopIteration):
        next(di_empty)


def test_utils_and_misc() -> None:
    """Verify ``apply_transformations``, ``get_element_spec``, ``load``, and namespace exports."""
    ds = Dataset.range(4)

    def _inc_elem(x: ElementTree) -> ElementTree:
        """Increment integer ``x``.

        Args:
            x (ElementTree): Input element.

        Returns:
            ElementTree: ``x + 1``.
        """
        return int(x) + 1 if isinstance(x, int) else x

    inc_op: MapOperation[ElementTree, ElementTree] = MapOperation(_inc_elem)
    transformed = apply_transformations(ds, inc_op)
    assert list(transformed) == [1, 2, 3, 4]
    assert get_element_spec(ds) == "int"
    assert get_element_spec(ds.to_iter_dataset()) == "int"
    empty_map_ds: MapDataset[int] = MapDataset(())
    assert get_element_spec(empty_map_ds) is None

    opts = LoadOptions(
        num_epochs=1,
        shuffle=False,
        batch_size=2,
        drop_remainder=False,
        worker_count=1,
    )
    dl = load(InMemoryDataSource([1, 2, 3, 4]), options=opts)
    assert dl.worker_count == 1
    batches = list(dl)
    assert len(batches) == 2

    sma = SharedMemoryArray(name="custom_shm", shape=(2, 3), dtype="int32")
    assert sma.metadata.shape == (2, 3)
    sma.unlink_on_del()
    sm_meta = SharedMemoryArrayMetadata(name="seg", shape=(2,))
    assert sm_meta == SharedMemoryArrayMetadata(name="seg", shape=(2,))
    assert sm_meta != SharedMemoryArrayMetadata(name="other", shape=(2,))
    assert "SharedMemoryArrayMetadata" in repr(sm_meta)
    sm_meta.close_and_unlink()
    assert sm_meta.name == ""

    FakeCls = fake_class()
    inst = FakeCls(1, k=2)
    assert inst.args == (1,)
    assert inst.kwargs == {"k": 2}

    assert shared_memory_array.SharedMemoryArrayMetadata is SharedMemoryArrayMetadata
    assert transforms.BatchTransform is BatchTransform
    assert transforms.Map is Map
    assert transforms.MapWithIndex is MapWithIndex
    assert transforms.RandomMap is RandomMap
    assert transforms.TfRandomMap is TfRandomMap
    assert transforms.TfRandomMapTransform is TfRandomMapTransform
    assert transforms.Filter is Filter
    assert transforms.FlatMap is FlatMap
    assert CheckpointHandler is PyGrainCheckpointHandler
    assert CopyNumPyArrayToSharedMemory is CopyNumPyArrayToSharedMemoryOperation
    assert _EagerDataLoaderIterator is PyGrainDatasetIterator
    assert python.DataLoader is DataLoader
    empty_bd: BatchDataset[int] = BatchDataset()
    empty_fd: FilterDataset[int] = FilterDataset()
    assert isinstance(DatasetSelectionMap(), DatasetSelectionMap)
    assert isinstance(empty_bd, BatchDataset)
    assert isinstance(empty_fd, FilterDataset)
