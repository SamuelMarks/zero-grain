"""Comprehensive unit tests for the ``zero_grain`` package."""

from __future__ import annotations

import dataclasses
from multiprocessing import shared_memory
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
    PyGrainCheckpointRestore,
    PyGrainCheckpointSave,
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
    SelectionWithReplacementMap,
    SequentialSampler,
    ShardByJaxProcess,
    ShardOptions,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    SharedMemoryDataSource,
    TfRandomMap,
    TfRandomMapTransform,
    Transformation,
    _EagerDataLoaderIterator,
    _batch_elements,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    batch_and_pad,
    create_rng,
    derive_rng,
    even_split,
    experimental,
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
from zero_grain._tree import _leaf_type_spec
from zero_grain._types import ElementTree, RNGGeneratorProtocol, is_dataclass_instance
from zero_grain.experimental import (
    FirstFitPackIterDataset,
    FlatMapIterDataset,
    InterleaveIterDataset,
    ThreadPrefetchIterDataset,
)
from zero_grain.shared_memory_array import (
    _allocate_shared_memory_segment,
    _dtype_itemsize,
    _release_shared_memory_segment,
)


class _SamplePair(NamedTuple):
    """Sample NamedTuple for testing PyTree batching."""

    left: int | str | list[int]
    right: int | str | list[int]


@dataclasses.dataclass
class _SampleBox:
    """Sample dataclass for testing ``is_dataclass_instance`` and ``spec_like``."""

    value: int


class _AddRandomOffset(TfRandomMapTransform[int, int]):
    """Concrete ``TfRandomMapTransform`` subclass for behavioral testing."""

    def random_map(self, element: int, rng: RNGGeneratorProtocol) -> int:
        """Add a deterministic random integer in ``[10, 100)`` to ``element``.

        Args:
            element (int): Input integer.
            rng (RNGGeneratorProtocol): Per-record RNG.

        Returns:
            int: Offset integer.
        """
        return element + rng.integers(10, 100)


class _AddRandomOffsetAlias(TfRandomMap[int, int]):
    """Concrete ``TfRandomMap`` subclass for behavioral testing."""

    def random_map(self, element: int, rng: RNGGeneratorProtocol) -> int:
        """Add a deterministic random integer in ``[10, 100)`` to ``element``.

        Args:
            element (int): Input integer.
            rng (RNGGeneratorProtocol): Per-record RNG.

        Returns:
            int: Offset integer.
        """
        return element + rng.integers(10, 100)


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
    assert isinstance(perm4, list) and sorted(perm4) == [0, 1, 2, 3]
    assert isinstance(perm3, list) and sorted(perm3) == [10, 20, 30]
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

    bt: BatchTransform[int, list[int]] = BatchTransform(batch_size=2, drop_remainder=False)
    assert bt.batch_size == 2
    assert bt.drop_remainder is False

    opt = DatasetOptions(
        filter_warn_threshold_ratio=0.5,
        filter_raise_threshold_ratio=0.8,
    )
    assert opt.filter_warn_threshold_ratio == 0.5
    assert opt.filter_raise_threshold_ratio == 0.8
    with pytest.raises(ValueError, match="filter_warn_threshold_ratio"):
        DatasetOptions(filter_warn_threshold_ratio=1.5)
    with pytest.raises(ValueError, match="filter_raise_threshold_ratio"):
        DatasetOptions(filter_raise_threshold_ratio=-0.1)

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
    assert repr(so) == "ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)"
    assert so == ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    assert so != ShardOptions(shard_index=0, shard_count=2, drop_remainder=True)
    with pytest.raises(ValueError, match="Invalid shard_index"):
        ShardOptions(shard_index=2, shard_count=2)

    ns = NoSharding()
    assert repr(ns) == "NoSharding(shard_index=0, shard_count=1, drop_remainder=False)"
    sb = ShardByJaxProcess(drop_remainder=True)
    assert sb.drop_remainder is True
    assert repr(sb) == "ShardByJaxProcess(shard_index=0, shard_count=1, drop_remainder=True)"
    assert sharding.ShardOptions(0, 1) == ShardOptions(0, 1)
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

    iss_drop = IndexSampler(num_records=10, shard_options=NoSharding(0, 2, True), num_epochs=1)
    assert len(iss_drop) == 10
    iss_drop_uneven = IndexSampler(num_records=9, shard_options=NoSharding(0, 2, True), num_epochs=1)
    assert len(iss_drop_uneven) == 8


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

    map_alias: Map[int, int] = Map()
    mwi_alias: MapWithIndex[int, int] = MapWithIndex()
    rm_alias: RandomMap[int, int] = RandomMap()
    filt_alias: Filter[int] = Filter()
    flat_alias: FlatMap[int, int] = FlatMap()
    with pytest.raises(NotImplementedError):
        map_alias.map(1)
    with pytest.raises(NotImplementedError):
        mwi_alias.map_with_index(0, 1)
    with pytest.raises(NotImplementedError):
        rm_alias.random_map(1, create_rng(0))
    with pytest.raises(NotImplementedError):
        filt_alias.filter(1)
    with pytest.raises(NotImplementedError):
        flat_alias.flat_map(1)

    tf_rm = _AddRandomOffset()
    tf_rm_alias = _AddRandomOffsetAlias()
    assert tf_rm.random_map(5, create_rng(7)) == tf_rm_alias.random_map(5, create_rng(7))
    assert tf_rm.random_map(5, create_rng(7)) >= 15

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

    rm_op: RandomMapOperation[int, int] = RandomMapOperation(lambda x, rng: x + rng.integers(10, 1000))
    out_rm_0 = list(rm_op([Record(metadata=RecordMetadata(index=0, rng=None), data=7)]))
    out_rm_0_repeat = list(rm_op([Record(metadata=RecordMetadata(index=0, rng=None), data=7)]))
    out_rm_1 = list(rm_op([Record(metadata=RecordMetadata(index=1, rng=None), data=7)]))
    assert out_rm_0[0].data == out_rm_0_repeat[0].data
    assert out_rm_0[0].data != 7
    assert out_rm_0[0].data != out_rm_1[0].data

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
    shm_alias = CopyNumPyArrayToSharedMemory()
    shm_input = np.array([1, 2], dtype=np.int32)
    shm_rec: Record[ElementTree] = Record(metadata=RecordMetadata(index=0), data=shm_input)
    shm_out = list(shm_op([shm_rec]))
    assert isinstance(shm_out[0].data, SharedMemoryArrayMetadata)
    assert shm_out[0].data.shape == (2,)
    assert shm_out[0].data.dtype == np.dtype("int32")
    shm_handle_0 = shared_memory.SharedMemory(name=shm_out[0].data.name)
    try:
        shm_view_0: np.ndarray = np.ndarray((2,), dtype=np.int32, buffer=shm_handle_0.buf)
        assert shm_view_0.tolist() == [1, 2]
    finally:
        shm_handle_0.close()
    shm_out[0].data.close_and_unlink()

    shm_alias_meta = shm_alias.map(np.array([3, 4], dtype=np.int64))
    assert isinstance(shm_alias_meta, SharedMemoryArrayMetadata)
    assert shm_alias_meta.shape == (2,)
    assert shm_alias_meta.dtype == np.dtype("int64")
    shm_handle_1 = shared_memory.SharedMemory(name=shm_alias_meta.name)
    try:
        shm_view_1: np.ndarray = np.ndarray((2,), dtype=np.int64, buffer=shm_handle_1.buf)
        assert shm_view_1.tolist() == [3, 4]
    finally:
        shm_handle_1.close()
    shm_alias_meta.close_and_unlink()

    batched_ints = _batch_elements([1, 2])
    assert isinstance(batched_ints, np.ndarray)
    assert batched_ints.dtype == np.dtype("int64")
    assert batched_ints.tolist() == [1, 2]
    assert _batch_elements([]) == []
    assert batch_and_pad([], 2) == []
    padded_pair = batch_and_pad([_SamplePair(1, 2)], 2)
    assert isinstance(padded_pair, _SamplePair)
    assert isinstance(padded_pair.left, np.ndarray) and padded_pair.left.dtype == np.dtype("int64")
    assert padded_pair.left.tolist() == [1, 0]
    assert isinstance(padded_pair.right, np.ndarray) and padded_pair.right.dtype == np.dtype("int64")
    assert padded_pair.right.tolist() == [2, 0]
    padded_tuple = batch_and_pad([(1.5, "a")], 2)
    assert isinstance(padded_tuple, tuple) and len(padded_tuple) == 2
    assert isinstance(padded_tuple[0], np.ndarray) and padded_tuple[0].dtype == np.dtype("float64")
    assert padded_tuple[0].tolist() == [1.5, 0.0]
    assert isinstance(padded_tuple[1], np.ndarray) and padded_tuple[1].tolist() == ["a", "0"]
    padded_bool = batch_and_pad([True], 2)
    assert isinstance(padded_bool, np.ndarray) and padded_bool.dtype == np.dtype("bool")
    assert padded_bool.tolist() == [True, False]
    padded_list = batch_and_pad([[1, 2]], 2)
    assert isinstance(padded_list, np.ndarray) and padded_list.dtype == np.dtype("int64")
    assert padded_list.tolist() == [[1, 0], [2, 0]]

    mapped = map_tree(lambda x: int(x) * 10 if isinstance(x, int) else x, _SamplePair(1, 2))
    assert mapped == _SamplePair(10, 20)
    assert spec_like(_SamplePair(1, 2)) == _SamplePair("int", "int")
    assert spec_like((1, [2])) == ("int", ["int"])


def test_leaf_type_spec_and_shared_memory_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify all branches of ``_leaf_type_spec`` and ``shared_memory_array`` helpers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    box = _SampleBox(value=42)
    assert is_dataclass_instance(box) is True
    assert _leaf_type_spec(None) == "NoneType"
    assert _leaf_type_spec(True) == "bool"
    assert _leaf_type_spec(3) == "int"
    assert _leaf_type_spec(1.5) == "float"
    assert _leaf_type_spec(1 + 2j) == "complex"
    assert _leaf_type_spec("abc") == "str"
    assert _leaf_type_spec(b"abc") == "bytes"
    assert _leaf_type_spec(box) == "dataclass"
    assert _leaf_type_spec(SharedMemoryArrayMetadata(name="s", shape=(2, 3))) == ("SharedMemoryArrayMetadata[2, 3]")

    class _ConvLeaf:
        """Convertible leaf object."""

        def __array__(self) -> np.ndarray:
            """Return 1D float32 array.

            Returns:
                np.ndarray: 1D float32 array.
            """
            return np.zeros((4,), dtype=np.float32)

    assert _leaf_type_spec(_ConvLeaf()) == "float32[4]"
    assert _leaf_type_spec([1]) == "unknown"

    assert _dtype_itemsize("float64") == 8
    assert _dtype_itemsize("float32") == 4
    assert _dtype_itemsize("float16") == 2
    assert _dtype_itemsize("uint8") == 1
    assert _dtype_itemsize("custom_unknown") == 8

    sma_pos = SharedMemoryArray((2, 2), "float32")
    assert sma_pos.metadata.shape == (2, 2)
    assert sma_pos.metadata.dtype == "float32"
    sma_pos.unlink_on_del()
    sma_pos.unlink_on_del()

    def _raise_os_error(create: bool = True, size: int = 1) -> shared_memory.SharedMemory:
        """Simulate ``SharedMemory`` allocation failure.

        Args:
            create (bool): Create flag.
            size (int): Segment byte size.

        Returns:
            shared_memory.SharedMemory: Never returns.

        Raises:
            OSError: Always raised.
        """
        del create, size
        raise OSError("shm allocation failed")

    monkeypatch.setattr(shared_memory, "SharedMemory", _raise_os_error)
    fallback_name, seg = _allocate_shared_memory_segment(16)
    assert fallback_name.startswith("zero_grain_shm_")
    assert seg is None
    fallback_meta = SharedMemoryArray.from_array(np.array([10, 20], dtype=np.int32))
    assert fallback_meta.name.startswith("zero_grain_shm_")
    assert fallback_meta.shape == (2,)
    _release_shared_memory_segment("nonexistent_segment")


def test_data_loader_and_checkpoint_handler(tmp_path: Path) -> None:
    """Verify ``DataLoader``, ``DataLoaderIterator``, and ``PyGrainCheckpointHandler``.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    dl_opts = DataLoaderOptions(worker_count=2, read_options=ReadOptions(num_threads=4))
    dl = DataLoader(data_source=InMemoryDataSource([10, 20]), worker_count=None, options=dl_opts)
    assert dl.worker_count == 2
    assert dl.read_options is not None and dl.read_options.num_threads == 4
    it = iter(dl)
    assert iter(it) is it
    it.start_prefetch()
    assert next(it) == 10

    ch = CheckpointHandler()
    ch_alias = PyGrainCheckpointHandler()
    ch.save(None, None)
    assert ch_alias.restore(None, None) is None
    assert ch.restore(tmp_path / "nonexistent", it) is it

    ckpt_dir = tmp_path / "ckpt"
    ch.save(ckpt_dir, it)
    raw_state: PyGrainDatasetIterator[int] | None = None
    restored = ch.restore(ckpt_dir, raw_state)
    assert isinstance(restored, dict)

    it2 = _EagerDataLoaderIterator(dl)
    ch.restore(ckpt_dir, it2)
    assert next(it2) == 20
    with pytest.raises(StopIteration):
        next(it2)

    with pytest.raises(ValueError, match="worker_count"):
        it2.set_state({"worker_count": 99})
    with pytest.raises(ValueError, match="sampler"):
        it2.set_state({"sampler": "MismatchedSampler()"})
    with pytest.raises(ValueError, match="data_source"):
        it2.set_state({"data_source": "MismatchedDataSource()"})

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

    def _add_rng(x: ElementTree, rng: RNGGeneratorProtocol) -> ElementTree:
        """Return ``x + rng.integers(100, 200)``.

        Args:
            x (ElementTree): Input element.
            rng (RNGGeneratorProtocol): Per-record RNG.

        Returns:
            ElementTree: Offset element.
        """
        return (int(x) if isinstance(x, int) else 0) + rng.integers(100, 200)

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
    rmd: MapDataset[int] = ds.random_map(_add_rng, seed=3)
    rmd_repeat: MapDataset[int] = ds.random_map(_add_rng, seed=3)
    fd: FilterDataset[int] = ds.filter(_gt_one)
    bd: BatchDataset[list[int]] = ds.batch(2)
    bd_drop: BatchDataset[list[int]] = ds.batch(2, drop_remainder=True)

    assert list(md) == [0, 2, 4, 6, 8]
    assert md[2] == 4
    assert list(mwi) == [0, 2, 4, 6, 8]
    assert mwi[3] == 6
    assert list(rmd) == [rmd[i] for i in range(5)]
    assert list(rmd) == list(rmd_repeat)
    assert list(rmd) != [0, 1, 2, 3, 4]
    assert all(100 <= val < 205 for val in rmd if isinstance(val, int))
    assert list(fd) == [2, 3, 4]
    assert fd[0] is None
    assert fd[2] == 2
    fd_mapped: MapDataset[int] = fd.map(_double)
    assert fd_mapped[0] is None
    assert fd_mapped[2] == 4
    assert len(bd) == 3
    assert len(bd_drop) == 2
    empty_bd: BatchDataset[int] = BatchDataset(batch_size=2)
    assert len(empty_bd) == 0
    assert empty_bd._get_item_with_epoch(0) == []
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

    # Verify chained shuffle -> slice and repeat -> slice preserve stage order
    shuffled_sliced = shuffled_ds.slice(slice(1, 4))
    assert len(shuffled_sliced) == 3
    assert [shuffled_sliced[i] for i in range(3)] == [shuffled_ds[1], shuffled_ds[2], shuffled_ds[3]]
    assert list(shuffled_sliced) == [shuffled_ds[1], shuffled_ds[2], shuffled_ds[3]]
    assert repr(shuffled_sliced.to_iter_dataset().dl.data_source) == repr(ds.source)

    # Verify single-element shuffle
    single_shuffled = Dataset.range(1).shuffle(seed=4)
    assert list(single_shuffled) == [0]
    assert single_shuffled[0] == 0

    # Verify ds.seed(1) does not mutate parent ds._seed
    assert ds._seed is None
    seeded_ds = ds.seed(1)
    assert ds._seed is None
    assert seeded_ds._seed == 1

    sliced_ds = ds.slice(slice(1, 4))
    assert list(sliced_ds) == [1, 2, 3]
    assert [sliced_ds[i] for i in range(len(sliced_ds))] == [1, 2, 3]
    repeated_ds = ds.repeat(2)
    assert list(repeated_ds) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert [repeated_ds[i] for i in range(len(repeated_ds))] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    # Verify multi-parent MapDataset mixing with DatasetSelectionMap
    ds_a = Dataset.source_dataset([10, 20])
    ds_b = Dataset.source_dataset([100, 200])
    mixed_ds: MapDataset[int] = MapDataset([ds_a, ds_b])
    assert len(mixed_ds) == 4
    assert [mixed_ds[i] for i in range(4)] == [10, 100, 20, 200]
    assert list(mixed_ds) == [10, 100, 20, 200]

    sel_map = DatasetSelectionMap([2, 1])
    assert len(sel_map) == 3
    assert sel_map[0] == (0, 0)
    assert sel_map[1] == (1, 0)
    assert sel_map[-1] == (0, 1)
    with pytest.raises(IndexError):
        _ = sel_map[5]
    empty_sel_map = DatasetSelectionMap()
    assert len(empty_sel_map) == 0
    assert empty_sel_map[2] == (0, 2)
    with pytest.raises(IndexError):
        _ = empty_sel_map[-1]
    single_sel_map = DatasetSelectionMap(3)
    assert len(single_sel_map) == 3
    assert single_sel_map[2] == (0, 2)

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
    assert isinstance(batches[0], np.ndarray) and batches[0].dtype == np.dtype("int64")
    assert batches[0].tolist() == [1, 2]
    assert isinstance(batches[1], np.ndarray) and batches[1].dtype == np.dtype("int64")
    assert batches[1].tolist() == [3, 4]

    sma = SharedMemoryArray(name="custom_shm", shape=(2, 3), dtype="int32")
    assert sma.metadata.name == "custom_shm"
    assert sma.metadata.shape == (2, 3)
    assert sma.metadata.dtype == "int32"
    sma.unlink_on_del()
    assert sma.metadata.name == ""
    sm_meta = shared_memory_array.SharedMemoryArrayMetadata(name="seg", shape=(2,), dtype="float32")
    assert sm_meta == SharedMemoryArrayMetadata(name="seg", shape=(2,), dtype="float32")
    assert sm_meta != SharedMemoryArrayMetadata(name="other", shape=(2,), dtype="float32")
    assert repr(sm_meta) == "SharedMemoryArrayMetadata(name='seg', dtype='float32', shape=(2,))"
    sm_meta.close_and_unlink()
    assert sm_meta.name == ""

    FakeCls = fake_class()
    inst = FakeCls(1, k=2)
    assert inst.args == (1,)
    assert inst.kwargs == {"k": 2}
    assert len(inst) == 1
    assert inst[0] == (0, 0)

    # Behavioral verification of module namespace exports
    py_loader = python.DataLoader(data_source=InMemoryDataSource([5, 6]))
    assert list(py_loader) == [5, 6]
    tr_batch: BatchTransform[int, list[int]] = transforms.BatchTransform(batch_size=2)
    assert tr_batch.batch_size == 2
    assert tr_batch.drop_remainder is False
    filt_nonzero_op: FilterOperation[ElementTree] = FilterOperation(lambda x: isinstance(x, int) and x != 0)
    manual_fd: FilterDataset[int] = FilterDataset(
        ds,
        _operations_override=(filt_nonzero_op,),
        _local_ops_override=(filt_nonzero_op,),
    )
    assert [manual_fd[i] for i in range(len(manual_fd))] == [None, 1, 2, 3]
    assert list(manual_fd) == [1, 2, 3]


def test_ecosystem_orbax_checkpointing(tmp_path: Path) -> None:
    """Verify ``PyGrainCheckpointHandler``, ``Checkpointer``, ``CheckpointManager``, and ``CompositeCheckpointHandler``.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    import zero_orbax.checkpoint as ocp

    dl: DataLoader[int] = DataLoader(
        data_source=RangeDataSource(0, 6, 1),
        sampler=SequentialSampler(6),
    )
    it1 = iter(dl)
    assert next(it1) == 0
    assert next(it1) == 1

    handler = PyGrainCheckpointHandler()
    assert handler.metadata(None) is None
    checkpointer = ocp.Checkpointer(handler)

    # Save and restore on local disk via Checkpointer and PyGrainCheckpointHandler
    local_dir = tmp_path / "orbax_local"
    checkpointer.save(local_dir, it1)
    handler.save(local_dir, PyGrainCheckpointSave(it1))
    struct_local = handler.structure(local_dir)
    assert isinstance(struct_local, dict) and struct_local.get("version") == 2
    assert handler.structure(None) == {}
    assert handler.structure(tmp_path / "nonexistent_dir") == {}
    meta_local = handler.metadata(local_dir)
    assert isinstance(meta_local, dict) and meta_local.get("version") == 2
    meta_none = handler.metadata(None)
    assert isinstance(meta_none, dict) and meta_none == meta_local
    meta_nonexistent = handler.metadata(tmp_path / "nonexistent_dir")
    assert isinstance(meta_nonexistent, dict) and meta_nonexistent.get("finalized") is True

    it1_restored = iter(dl)
    handler.restore(local_dir, PyGrainCheckpointRestore(it1_restored))
    assert list(it1_restored) == [2, 3, 4, 5]
    it1_restored_cp = iter(dl)
    checkpointer.restore(local_dir, it1_restored_cp)
    assert list(it1_restored_cp) == [2, 3, 4, 5]

    # Save and restore DatasetIterator via virtual memory:// URI and CheckpointManager
    ds_iter = iter(MapDataset.range(5).to_iter_dataset())
    assert next(ds_iter) == 0
    assert next(ds_iter) == 1

    mem_uri = "memory://grain_ckpt_manager"
    mgr = ocp.CheckpointManager(
        mem_uri,
        checkpointers=ocp.Checkpointer(PyGrainCheckpointHandler()),
    )
    assert mgr.save(1, items=ds_iter) is True
    ds_iter_restored = iter(MapDataset.range(5).to_iter_dataset())
    mgr.restore(1, items=ds_iter_restored)
    assert list(ds_iter_restored) == [2, 3, 4]
    mgr.close()

    # Joint composite checkpointing of DataLoaderIterator (PyGrainCheckpointSave) and PyTree state (PyTreeSave)
    comp_dir = tmp_path / "orbax_composite"
    it_comp = iter(dl)
    assert next(it_comp) == 0
    assert next(it_comp) == 1
    assert next(it_comp) == 2

    comp_mgr = ocp.CheckpointManager(comp_dir, item_names=("data", "state"))
    assert (
        comp_mgr.save(
            1,
            args=ocp.args.Composite(
                data=PyGrainCheckpointSave(it_comp),
                state=ocp.args.PyTreeSave({"step": 3, "lr": 0.01}),
            ),
        )
        is True
    )
    it_comp_restored = iter(dl)
    restored_bundle = comp_mgr.restore(
        1,
        args=ocp.args.Composite(
            data=PyGrainCheckpointRestore(it_comp_restored),
            state=ocp.args.PyTreeRestore({"step": 0, "lr": 0.0}),
        ),
    )
    assert isinstance(restored_bundle, dict)
    assert restored_bundle["state"] == {"step": 3, "lr": 0.01}
    assert list(it_comp_restored) == [3, 4, 5]
    comp_mgr.close()

    comp_handler = ocp.CompositeCheckpointHandler("data", "state")
    comp_dir_direct = tmp_path / "orbax_composite_direct"
    it_comp2 = iter(dl)
    assert next(it_comp2) == 0
    comp_handler.save(
        comp_dir_direct,
        {"data": it_comp2, "state": {"step": 1}},
    )
    it_comp2_restored = iter(dl)
    restored_direct = comp_handler.restore(
        comp_dir_direct,
        {"data": it_comp2_restored, "state": {"step": 0}},
    )
    assert isinstance(restored_direct, dict)
    assert list(it_comp2_restored) == [1, 2, 3, 4, 5]
    comp_handler.close()
    checkpointer.close()

    # Verify _unwrap_iterator_arg edge cases (keyword item/args, positional args, non-iterator args)
    kw_dir = tmp_path / "kw_ckpt"
    handler.save(kw_dir, None, it1_restored, process_index=0, process_count=1)
    it_kw = iter(dl)
    handler.restore(kw_dir, None, args=PyGrainCheckpointRestore(it_kw))
    handler.save(kw_dir, item=PyGrainCheckpointSave(it_kw))
    handler.restore(kw_dir, item=it_kw)
    handler.save(kw_dir, PyGrainCheckpointSave(None))
    assert handler.restore(None, item=PyGrainCheckpointRestore(None)) is None


def test_ecosystem_chex_jax_flax_optax_integration() -> None:
    """Verify ``zero_chex``, ``zero_jax``, ``zero_flax``, and ``CompilerTensor`` integration."""
    from ml_switcheroo_compiler import ops as compiler_ops
    from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
    from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
    import zero_chex as chex
    from zero_chex.types import ZeroArray
    from zero_flax import nnx
    import zero_jax
    import zero_jax.numpy as jnp
    from zero_jax.numpy import ndarray as JaxArray

    @chex.dataclass
    class TrainingSample:
        """Sample record with feature vector ``x`` and scalar target ``y``."""

        x: JaxArray
        y: JaxArray

    @chex.dataclass
    class AltSample:
        """Alternative dataclass with different structure for mismatch testing."""

        z: int

    def _make_jax_arr(vals: list[float] | list[list[float]] | float) -> JaxArray:
        """Construct a typed ``JaxArray`` from Python floats.

        Args:
            vals (list[float] | list[list[float]] | float): Input float values.

        Returns:
            JaxArray: Constructed ``zero_jax`` array.
        """
        res = jnp.array(vals, dtype="float32")
        assert isinstance(res, JaxArray)
        return res

    records: list[TrainingSample] = [
        TrainingSample(x=_make_jax_arr([1.0, 2.0]), y=_make_jax_arr(3.0)),
        TrainingSample(x=_make_jax_arr([2.0, 3.0]), y=_make_jax_arr(5.0)),
        TrainingSample(x=_make_jax_arr([3.0, 4.0]), y=_make_jax_arr(7.0)),
        TrainingSample(x=_make_jax_arr([4.0, 5.0]), y=_make_jax_arr(9.0)),
    ]
    batched_ds: MapDataset[ElementTree] = MapDataset.source_dataset(records).batch(batch_size=2)
    batch0 = batched_ds[0]
    assert isinstance(batch0, TrainingSample)
    chex.assert_shape(batch0.x, (2, 2))
    chex.assert_shape(batch0.y, (2,))
    expected_b0 = TrainingSample(
        x=_make_jax_arr([[1.0, 2.0], [2.0, 3.0]]),
        y=_make_jax_arr([3.0, 5.0]),
    )
    chex.assert_trees_all_close({"x": batch0.x, "y": batch0.y}, {"x": expected_b0.x, "y": expected_b0.y})

    # Verify batch_and_pad, spec_like, and map_tree on chex.dataclass, CompilerTensor, and ZeroArray
    padded_dc = batch_and_pad([records[0]], batch_size=2)
    assert isinstance(padded_dc, TrainingSample)
    chex.assert_shape(padded_dc.x, (2, 2))
    dc_spec = spec_like(records[0])
    assert isinstance(dc_spec, TrainingSample)
    assert dc_spec.x == "float32[2]"
    mapped_dc = map_tree(lambda leaf: leaf, records[0])
    assert isinstance(mapped_dc, TrainingSample)

    with pytest.raises(TypeError, match="Record structures do not match"):
        _batch_elements([records[0], AltSample(z=1)])

    # CompilerTensor batching and padding
    t1 = compiler_ops.array([1, 2], dtype=CompilerDType("int32"))
    t2 = compiler_ops.array([3, 4], dtype=CompilerDType("int32"))
    assert isinstance(t1, CompilerTensor) and isinstance(t2, CompilerTensor)
    t_batched = _batch_elements([t1, t2])
    assert isinstance(t_batched, CompilerTensor) and t_batched.shape == (2, 2)
    t_padded = batch_and_pad([t1], batch_size=2)
    assert isinstance(t_padded, CompilerTensor) and t_padded.shape == (2, 2)

    # ZeroArray batching and padding
    z1 = ZeroArray([1.0, 2.0], shape=(2,), dtype="float32")
    z2 = ZeroArray([3.0, 4.0], shape=(2,), dtype="float32")
    z_batched = _batch_elements([z1, z2])
    assert isinstance(z_batched, ZeroArray) and z_batched.shape == (2, 2)
    z_padded = batch_and_pad([z1], batch_size=2)
    assert isinstance(z_padded, ZeroArray) and z_padded.shape == (2, 2)
    z_key = ZeroArray([0, 42], shape=(2,), dtype="int64")
    assert create_rng(z_key).integers(0, 1000) == create_rng(42).integers(0, 1000)

    # Sharding with zero_chex.set_n_cpu_devices and PRNG interoperability
    chex.set_n_cpu_devices(4)
    try:
        sharded = ShardByJaxProcess(drop_remainder=True)
        assert sharded.shard_index == 0
        assert sharded.shard_count == 4
    finally:
        chex.set_n_cpu_devices(1)

    prng_key = zero_jax.random.PRNGKey(42)
    assert isinstance(prng_key, JaxArray)
    rng_from_key = create_rng(prng_key)
    rng_from_int = create_rng(42)
    assert rng_from_key.integers(0, 1000) == rng_from_int.integers(0, 1000)

    flax_rngs = nnx.Rngs(77)
    rng_from_flax = create_rng(flax_rngs)
    assert 0 <= rng_from_flax.integers(0, 100) < 100
    derived_from_key = derive_rng(prng_key, index=0, op_seed=1)
    derived_from_int = derive_rng(42, index=0, op_seed=1)
    assert derived_from_key.integers(0, 1000) == derived_from_int.integers(0, 1000)

    # Feed zero_grain.DataLoader batches directly into zero_flax.nnx.Linear parameter update step
    dl_train: DataLoader[TrainingSample] = DataLoader(
        data_source=InMemoryDataSource(records),
        sampler=SequentialSampler(len(records)),
        operations=[BatchOperation(batch_size=2)],
    )
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    assert model.bias is not None
    initial_bias_list = model.bias.value.tolist()

    for batch in dl_train:
        assert isinstance(batch, TrainingSample)
        preds = model(batch.x)
        assert isinstance(preds, JaxArray)
        chex.assert_shape(preds, (2, 1))
        grad_k = jnp.ones_like(model.kernel.value)
        grad_b = jnp.ones_like(model.bias.value)
        assert isinstance(grad_k, JaxArray) and isinstance(grad_b, JaxArray)
        new_k = jnp.asarray(model.kernel.value - 0.05 * grad_k)
        new_b = jnp.asarray(model.bias.value - 0.05 * grad_b)
        assert isinstance(new_k, JaxArray) and isinstance(new_b, JaxArray)
        model.kernel.value = new_k
        model.bias.value = new_b

    assert model.bias.value.tolist() != initial_bias_list


def test_v005_methods_and_experimental_datasets(tmp_path: Path) -> None:
    """Verify v0.0.5 API surface including ``Transformation``, ``SelectionWithReplacementMap``, ``MapDataset``, ``IterDataset``, and ``experimental``.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    tr_base = Transformation(1, key="v")
    assert tr_base._args == (1,)
    assert tr_base._kwargs == {"key": "v"}
    assert issubclass(MapTransform, Transformation)
    assert issubclass(Batch, Transformation)

    # SequentialSampler num_epochs validation and multi-epoch indexing
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        SequentialSampler(num_records=3, num_epochs=0)
    seq_multi = SequentialSampler(num_records=2, num_epochs=2)
    assert len(seq_multi) == 4
    assert [seq_multi[i].record_key for i in range(4)] == [0, 1, 0, 1]
    assert [seq_multi[i].index for i in range(4)] == [0, 1, 2, 3]

    # SelectionWithReplacementMap branches
    sel_empty = SelectionWithReplacementMap()
    assert len(sel_empty) == 0
    assert sel_empty.weights == ()
    sel_default_w = SelectionWithReplacementMap(parent_lengths=[2, 2])
    assert len(sel_default_w) == 4
    assert sel_default_w.weights == (0.5, 0.5)
    assert [sel_default_w[i] for i in range(4)] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    sel_custom = SelectionWithReplacementMap(
        parent_lengths=[2, 3],
        weights=[3.0, 1.0],
        total_size=6,
    )
    assert len(sel_custom) == 6
    assert sel_custom.weights == (0.75, 0.25)
    assert [sel_custom[i] for i in range(6)] == [(0, 0), (0, 1), (1, 0), (0, 0), (0, 1), (0, 0)]
    sel_zero_total = SelectionWithReplacementMap(parent_lengths=[2, 2], total_size=0)
    assert len(sel_zero_total) == 0
    sel_zero_parents = SelectionWithReplacementMap(parent_lengths=[0, 0], weights=[1.0, 1.0], total_size=3)
    assert len(sel_zero_parents) == 0
    assert SelectionWithReplacementMap._build_weighted_replacement_mapping([2, 2], [0.5], 2) == [(0, 0), (1, 0)]
    with pytest.raises(ValueError, match="parent_lengths must be non-negative"):
        DatasetSelectionMap(parent_lengths=[2, -1])
    with pytest.raises(ValueError, match="weights length 1 must match parent_lengths length 2"):
        SelectionWithReplacementMap(parent_lengths=[2, 2], weights=[1.0, "ignored"])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        SelectionWithReplacementMap(parent_lengths=[2, 2], weights=[-1.0, 2.0])
    with pytest.raises(ValueError, match="sum of weights must be positive"):
        SelectionWithReplacementMap(parent_lengths=[2, 2], weights=[0.0, 0.0])

    # MapDataset.source descriptor (class and instance access + assignment)
    ds_src = MapDataset.source([10, 20, 30])
    assert list(ds_src) == [10, 20, 30]
    assert [ds_src.source[i] for i in range(len(ds_src.source))] == [10, 20, 30]
    ds_src.source = InMemoryDataSource([40, 50])
    assert list(ds_src) == [40, 50]

    # MapDataset.mix and MapDataset.select_from_datasets
    assert list(MapDataset.mix([])) == []
    assert list(MapDataset.select_from_datasets([], sel_default_w)) == []
    m1 = MapDataset.source([1, 2])
    m2 = MapDataset.source([10, 20])
    mixed_uniform = MapDataset.mix([m1, m2])
    assert list(mixed_uniform) == [1, 10, 2, 20]
    mixed_weighted = MapDataset.mix([m1, m2], weights=[0.75, 0.25])
    assert list(mixed_weighted) == [1, 2, 10, 1]
    single_sel = MapDataset.select_from_datasets([m1], DatasetSelectionMap(parent_lengths=[2]))
    assert list(single_sel) == [1, 2]

    # MapDataset.repeat(reassemble_every_epoch=...) and to_iter_dataset(allow_nones=...)
    shuffled_m1 = MapDataset.range(4).shuffle(seed=9)
    rep_reassemble = shuffled_m1.repeat(2, reassemble_every_epoch=True)
    rep_fixed = shuffled_m1.repeat(2, reassemble_every_epoch=False)
    rep_reassemble_list = list(rep_reassemble)
    rep_fixed_list = list(rep_fixed)
    assert len(rep_reassemble_list) == 8
    assert sorted(int(x) for x in rep_reassemble_list[:4] if isinstance(x, int)) == [0, 1, 2, 3]
    assert sorted(int(x) for x in rep_reassemble_list[4:] if isinstance(x, int)) == [0, 1, 2, 3]
    assert rep_reassemble_list[:4] != rep_reassemble_list[4:]
    assert rep_fixed_list[:4] == rep_fixed_list[4:]
    assert rep_fixed_list[:4] == list(shuffled_m1)

    filtered_m = MapDataset.range(4).filter(lambda x: isinstance(x, int) and x % 2 == 0)
    with_nones = list(filtered_m.to_iter_dataset(allow_nones=True))
    assert with_nones == [0, None, 2, None]

    # MapDataset.mp_prefetch and IterDataset streaming methods
    init_calls: list[tuple[int, int]] = []
    prefetched_iter = m1.mp_prefetch(
        MultiprocessingOptions(num_workers=0, per_worker_buffer_size=2),
        worker_init_fn=lambda idx, cnt: init_calls.append((idx, cnt)),
        sequential_slice=True,
    )
    assert init_calls == [(0, 1)]
    assert list(prefetched_iter) == [1, 2]

    inherited_iter = IterDataset(prefetched_iter)
    assert inherited_iter.sequential_slice is True
    assert list(inherited_iter) == [1, 2]
    inherited_nones = IterDataset(filtered_m.to_iter_dataset(allow_nones=True))
    assert inherited_nones.allow_nones is True
    assert list(inherited_nones) == [0, None, 2, None]

    s1: IterDataset[int] = m1.to_iter_dataset().map(lambda x: x * 2 if isinstance(x, int) else 0)
    s2: IterDataset[int] = s1.random_map(
        lambda x, rng: (x + int(rng.integers(0, 1))) if isinstance(x, int) else 0,
        seed=3,
    )
    s3: IterDataset[int] = s2.filter(lambda x: isinstance(x, int) and x > 0)
    s4: IterDataset[list[int]] = s3.batch(batch_size=2)
    stream_ds: IterDataset[list[int]] = s4.prefetch(MultiprocessingOptions(num_workers=0)).mp_prefetch()
    stream_batches = list(stream_ds)
    assert len(stream_batches) == 1
    assert isinstance(stream_batches[0], np.ndarray) and stream_batches[0].dtype == np.dtype("int64")
    assert stream_batches[0].tolist() == [2, 4]

    mixed_iter = IterDataset.mix([m1.to_iter_dataset(), m2], weights=[0.5, 0.5])
    assert list(mixed_iter) == [1, 10, 2, 20]
    multi_parent_iter = IterDataset([m1.to_iter_dataset(), m2])
    assert list(multi_parent_iter) == [1, 10, 2, 20]

    # BatchDataset followed by .slice(), .repeat(), and .shuffle()
    batched_sliced: MapDataset[list[int]] = MapDataset.range(6).batch(2).slice(slice(1, 3))
    batched_sliced_items = list(batched_sliced)
    assert len(batched_sliced_items) == 2
    assert all(isinstance(b, np.ndarray) and b.dtype == np.dtype("int64") for b in batched_sliced_items)
    assert [b.tolist() for b in batched_sliced_items if isinstance(b, np.ndarray)] == [[2, 3], [4, 5]]

    batched_repeated: MapDataset[list[int]] = MapDataset.range(4).batch(2).repeat(2)
    batched_repeated_items = list(batched_repeated)
    assert len(batched_repeated_items) == 4
    assert all(isinstance(b, np.ndarray) and b.dtype == np.dtype("int64") for b in batched_repeated_items)
    assert [b.tolist() for b in batched_repeated_items if isinstance(b, np.ndarray)] == [[0, 1], [2, 3], [0, 1], [2, 3]]

    batched_shuffled: MapDataset[list[int]] = MapDataset.range(6).batch(2).shuffle(seed=7)
    batched_shuffled_items = [b.tolist() for b in batched_shuffled if isinstance(b, np.ndarray)]
    assert len(batched_shuffled_items) == 3
    assert sorted(batched_shuffled_items) == [[0, 1], [2, 3], [4, 5]]

    # Experimental datasets (both zero_grain.experimental and zero_grain.python.experimental)
    assert experimental.FlatMapIterDataset is python.experimental.FlatMapIterDataset

    class _DupFlatMap(FlatMapTransform[ElementTree, ElementTree]):
        """Duplicate each integer element twice."""

        max_fan_out = 2

        def flat_map(self, element: ElementTree) -> list[ElementTree]:
            """Return two copies of ``element``.

            Args:
                element (ElementTree): Input element.

            Returns:
                list[ElementTree]: Two-element list.
            """
            return [element, element]

    class _TooManyFlatMap(FlatMapTransform[ElementTree, ElementTree]):
        """Exceed ``max_fan_out`` for error testing."""

        max_fan_out = 1

        def flat_map(self, element: ElementTree) -> list[ElementTree]:
            """Return two copies exceeding ``max_fan_out=1``.

            Args:
                element (ElementTree): Input element.

            Returns:
                list[ElementTree]: Two-element list.
            """
            return [element, element]

    from zero_grain.experimental import _LazyStreamDataSource

    lazy_src = _LazyStreamDataSource(stream_factory=lambda: iter([10, 20, 30]), repr_label="lazy")
    assert lazy_src[0] == 10
    assert lazy_src[0] == 10
    assert lazy_src[1] == 20
    assert len(lazy_src) == 3

    fm_ds = FlatMapIterDataset(m1, _DupFlatMap())
    assert fm_ds.dataset.source[0] == 1
    assert fm_ds.dataset.source[1] == 1
    assert repr(fm_ds.dataset.source).startswith("FlatMapIterDataset(")
    with pytest.raises(IndexError):
        _ = fm_ds.dataset.source[10]
    with pytest.raises(IndexError):
        _ = fm_ds.dataset.source[-10]
    assert list(fm_ds) == [1, 1, 2, 2]
    assert fm_ds.dataset[-1] == 2
    assert list(fm_ds.map(lambda x: int(x) * 10 if isinstance(x, int) else x)) == [10, 10, 20, 20]
    assert_equal_output_after_checkpoint(fm_ds)
    fm_empty = FlatMapIterDataset(MapDataset([]), _DupFlatMap())
    assert list(fm_empty) == []
    fm_callable = FlatMapIterDataset(m1.to_iter_dataset(), lambda x: [x])
    assert list(fm_callable) == [1, 2]
    with pytest.raises(ValueError, match="exceeds max_fan_out"):
        FlatMapIterDataset(m1, _TooManyFlatMap())

    with pytest.raises(ValueError, match="prefetch_buffer_size must be non-negative"):
        ThreadPrefetchIterDataset(m1, prefetch_buffer_size=-1)
    tp_map = ThreadPrefetchIterDataset(m1, prefetch_buffer_size=2)
    assert list(tp_map) == [1, 2]
    tp_iter = ThreadPrefetchIterDataset(m1.to_iter_dataset(), prefetch_buffer_size=0)
    assert list(tp_iter) == [1, 2]

    with pytest.raises(ValueError, match="cycle_length must be >= 1"):
        InterleaveIterDataset([m1], cycle_length=0)
    with pytest.raises(ValueError, match="num_make_iter_threads must be >= 1"):
        InterleaveIterDataset([m1], cycle_length=1, num_make_iter_threads=0)
    with pytest.raises(ValueError, match="iter_buffer_size must be >= 1"):
        InterleaveIterDataset([m1], cycle_length=1, iter_buffer_size=0)
    assert list(InterleaveIterDataset([], cycle_length=2)) == []
    m_empty: MapDataset[int] = MapDataset([])
    m3 = MapDataset.source([100, 200, 300])
    inter_ds = InterleaveIterDataset([m_empty, m1, m2, m3, m_empty], cycle_length=2)
    assert list(inter_ds) == [10, 1, 20, 2, 100, 200, 300]
    assert list(inter_ds.filter(lambda x: isinstance(x, int) and x >= 10)) == [10, 20, 100, 200, 300]
    assert_equal_output_after_checkpoint(inter_ds)

    with pytest.raises(ValueError, match="num_packing_bins must be >= 1"):
        FirstFitPackIterDataset(m1, length_struct=4, num_packing_bins=0)
    with pytest.raises(ValueError, match="max_sequences_per_bin must be >= 1"):
        FirstFitPackIterDataset(m1, length_struct=4, num_packing_bins=1, max_sequences_per_bin=0)

    seq_source = MapDataset.source([[1, 2], [3], [4, 5, 6]])
    packed_unshuffled = FirstFitPackIterDataset(
        seq_source,
        length_struct=3,
        num_packing_bins=1,
        shuffle_bins=False,
        padding_struct=0,
        max_sequences_per_bin=2,
    )
    assert list(packed_unshuffled) == [[1, 2, 3], [4, 5, 6]]
    assert_equal_output_after_checkpoint(packed_unshuffled)

    packed_shuffled = FirstFitPackIterDataset(
        MapDataset.source([1, 2, 3]),
        length_struct=2,
        num_packing_bins=1,
        shuffle_bins=True,
        max_sequences_per_bin=1,
    )
    shuffled_bins = list(packed_shuffled)
    assert sorted(b for b in shuffled_bins if isinstance(b, list)) == [[1, 0], [2, 0], [3, 0]]

    # Verify FirstFitPackIterDataset with NamedTuple, plain tuple, and chex.dataclass PyTrees
    import zero_chex as chex

    @chex.dataclass
    class _PackSample:
        """Mappable dataclass for packing test."""

        tokens: list[int] | int
        mask: list[int] | int

    packed_nt = FirstFitPackIterDataset(
        MapDataset.source([_SamplePair([1, 2], [10]), _SamplePair([3], [20, 30])]),
        length_struct=_SamplePair(3, 3),
        num_packing_bins=1,
        shuffle_bins=False,
        padding_struct=_SamplePair(0, -1),
    )
    assert list(packed_nt) == [_SamplePair([1, 2, 3], [10, 20, 30])]

    packed_tup = FirstFitPackIterDataset(
        MapDataset.source([([1], [2]), ([3], [4])]),
        length_struct=(3, 3),
        num_packing_bins=1,
        shuffle_bins=False,
        padding_struct=(0, 9),
    )
    assert list(packed_tup) == [([1, 3, 0], [2, 4, 9])]

    packed_dc = FirstFitPackIterDataset(
        MapDataset.source([_PackSample(tokens=[1, 2], mask=[1, 1]), _PackSample(tokens=[3], mask=[1])]),
        length_struct=_PackSample(tokens=4, mask=4),
        num_packing_bins=1,
        shuffle_bins=False,
    )
    assert list(packed_dc) == [_PackSample(tokens=[1, 2, 3, 0], mask=[1, 1, 1, 0])]

    arr_dict_source = MapDataset.source(
        [
            {"a": np.array([1, 2], dtype=np.int32), "b": np.array(5, dtype=np.int32)},
            {"a": np.array([3], dtype=np.int32), "b": np.array(6, dtype=np.int32)},
        ]
    )
    packed_dict = FirstFitPackIterDataset(
        arr_dict_source,
        length_struct={"a": 4, "b": 2},
        num_packing_bins=2,
        shuffle_bins=False,
        padding_struct={"a": 0, "b": 0},
    )
    packed_dict_items = list(packed_dict)
    assert len(packed_dict_items) == 1
    first_bin = packed_dict_items[0]
    assert isinstance(first_bin, dict)
    assert isinstance(first_bin["a"], np.ndarray) and first_bin["a"].dtype == np.dtype("int32")
    assert first_bin["a"].tolist() == [1, 2, 3, 0]
    assert isinstance(first_bin["b"], np.ndarray) and first_bin["b"].dtype == np.dtype("int32")
    assert first_bin["b"].tolist() == [5, 6]

    empty_bin_pack = FirstFitPackIterDataset(
        MapDataset.source([[1, 2, 3, 4]]),
        length_struct=2,
        num_packing_bins=2,
        shuffle_bins=False,
    )
    assert list(empty_bin_pack) == [[1, 2]]

    # Verify PyGrainCheckpointHandler round-trips on FlatMapIterDataset, InterleaveIterDataset,
    # FirstFitPackIterDataset, and chained BatchDataset pipelines
    ckpt_handler = PyGrainCheckpointHandler()

    fm_it1 = iter(fm_ds)
    assert next(fm_it1) == 1
    assert next(fm_it1) == 1
    fm_ckpt_dir = tmp_path / "ckpt_flatmap"
    ckpt_handler.save(fm_ckpt_dir, fm_it1)
    fm_it2 = iter(fm_ds)
    ckpt_handler.restore(fm_ckpt_dir, fm_it2)
    assert list(fm_it2) == [2, 2]

    inter_it1 = iter(inter_ds)
    assert next(inter_it1) == 10
    assert next(inter_it1) == 1
    assert next(inter_it1) == 20
    inter_ckpt_dir = tmp_path / "ckpt_interleave"
    ckpt_handler.save(inter_ckpt_dir, inter_it1)
    inter_it2 = iter(inter_ds)
    ckpt_handler.restore(inter_ckpt_dir, inter_it2)
    assert list(inter_it2) == [2, 100, 200, 300]

    pack_it1 = iter(packed_unshuffled)
    assert next(pack_it1) == [1, 2, 3]
    pack_ckpt_dir = tmp_path / "ckpt_pack"
    ckpt_handler.save(pack_ckpt_dir, pack_it1)
    pack_it2 = iter(packed_unshuffled)
    ckpt_handler.restore(pack_ckpt_dir, pack_it2)
    assert list(pack_it2) == [[4, 5, 6]]

    chained_batch_iter = batched_sliced.to_iter_dataset()
    assert_equal_output_after_checkpoint(chained_batch_iter)
    cb_it1 = iter(chained_batch_iter)
    first_cb = next(cb_it1)
    assert isinstance(first_cb, np.ndarray) and first_cb.tolist() == [2, 3]
    cb_ckpt_dir = tmp_path / "ckpt_chained_batch"
    ckpt_handler.save(cb_ckpt_dir, cb_it1)
    cb_it2 = iter(chained_batch_iter)
    ckpt_handler.restore(cb_ckpt_dir, cb_it2)
    remaining_cb = list(cb_it2)
    assert len(remaining_cb) == 1
    assert isinstance(remaining_cb[0], np.ndarray) and remaining_cb[0].tolist() == [4, 5]

    from zero_grain.data_sources import ensure_testdata

    generated_dir = ensure_testdata()
    assert (generated_dir / "digits.array_record-00000-of-00002").is_file()
    assert (generated_dir / "digits.array_record-00001-of-00002").is_file()


def test_dataset_options_filter_thresholds() -> None:
    """Verify ``DatasetOptions`` warning and error threshold enforcement on ``FilterOperation``."""
    from unittest import mock

    records = [Record(RecordMetadata(index=i, record_key=i), i) for i in range(4)]

    # Below warning threshold -> no warning emitted
    opts_ok = DatasetOptions(filter_warn_threshold_ratio=0.6, filter_raise_threshold_ratio=0.9)
    op_ok: FilterOperation[int] = FilterOperation(lambda x: x >= 2, dataset_options=opts_ok)
    with mock.patch("logging.warning") as mock_warn_ok:
        kept_ok = list(op_ok(records))
        assert [r.data for r in kept_ok] == [2, 3]
        mock_warn_ok.assert_not_called()

    # Reaching warning threshold (3/4 = 0.75 >= 0.7) -> emits warning
    opts_warn = DatasetOptions(filter_warn_threshold_ratio=0.7, filter_raise_threshold_ratio=0.9)
    op_warn: FilterOperation[int] = FilterOperation(lambda x: x >= 3, dataset_options=opts_warn)
    with mock.patch("logging.warning") as mock_warn_hit:
        kept_warn = list(op_warn(records))
        assert [r.data for r in kept_warn] == [3]
        mock_warn_hit.assert_called_once()
        warn_arg = str(mock_warn_hit.call_args[0][0])
        assert "0.75 (3/4)" in warn_arg
        assert "filter_warn_threshold_ratio=0.7" in warn_arg

    # Reaching raise threshold (3/4 = 0.75 >= 0.75) -> raises ValueError
    opts_raise = DatasetOptions(filter_warn_threshold_ratio=0.5, filter_raise_threshold_ratio=0.75)
    op_raise: FilterOperation[int] = FilterOperation(lambda x: x >= 3, dataset_options=opts_raise)
    with pytest.raises(ValueError, match="filter_raise_threshold_ratio=0.75"):
        list(op_raise(records))

    # Verify DatasetOptions filter threshold enforcement under graph evaluation (e.g. filter -> shuffle)
    ds_graph_raise = (
        MapDataset.range(4).filter(lambda x: isinstance(x, int) and x >= 3, dataset_options=opts_raise).shuffle(seed=11)
    )
    with pytest.raises(ValueError, match="filter_raise_threshold_ratio=0.75"):
        list(ds_graph_raise)

    ds_graph_op_opts: MapDataset[int] = (
        MapDataset.range(4)
        .map(FilterOperation(lambda x: isinstance(x, int) and x >= 3, dataset_options=opts_raise))
        .shuffle(seed=11)
    )
    with pytest.raises(ValueError, match="filter_raise_threshold_ratio=0.75"):
        list(ds_graph_op_opts)


def test_multistage_graph_evaluation_and_sequential_slice_and_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-stage ``MapDataset`` graph parity, ``sequential_slice`` worker partitioning, and shared memory array reconstruction.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import sys
    from zero_grain._types import array_to_list
    from zero_grain.dataset import _DatasetStepKind, _normalize_step_kind
    from zero_grain.shared_memory_array import SharedMemoryArray, _dtype_itemsize

    assert _normalize_step_kind(_DatasetStepKind.OPS) is _DatasetStepKind.OPS
    assert _normalize_step_kind("shuffle") is _DatasetStepKind.SHUFFLE

    # Verify exact parity between [ds[i] ...] and list(ds) across multi-stage pipelines
    base = MapDataset.range(6)
    p1_mapped: MapDataset[int] = base.map_with_index(lambda idx, x: idx * 100 + (int(x) if isinstance(x, int) else 0))
    p1 = p1_mapped.shuffle(seed=7)
    assert list(p1) == [p1[i] for i in range(len(p1))]

    p2: MapDataset[int] = base.shuffle(seed=3).map_with_index(
        lambda idx, x: idx * 10 + (int(x) if isinstance(x, int) else 0)
    )
    assert list(p2) == [p2[i] for i in range(len(p2))]

    p3 = base.repeat(2).repeat(2)
    assert list(p3) == [p3[i] for i in range(len(p3))]

    p4_mapped: MapDataset[int] = base.map(lambda x: (int(x) if isinstance(x, int) else 0) + 1)
    p4 = p4_mapped.repeat(2)
    assert list(p4) == [p4[i] for i in range(len(p4))]

    # Verify sequential_slice=True vs False across 2 workers
    seq_ds = MapDataset.range(6)
    interleaved_workers = seq_ds.mp_prefetch(
        MultiprocessingOptions(num_workers=2),
        sequential_slice=False,
    )
    contiguous_workers = seq_ds.mp_prefetch(
        MultiprocessingOptions(num_workers=2),
        sequential_slice=True,
    )
    assert list(interleaved_workers) == [0, 1, 2, 3, 4, 5]
    # With 2 workers and sequential_slice=True: worker 0 gets [0, 1, 2], worker 1 gets [3, 4, 5],
    # and round-robin across workers yields [0, 3, 1, 4, 2, 5].
    assert list(contiguous_workers) == [0, 3, 1, 4, 2, 5]

    # SharedMemoryDataSource negative and slice indexing + bounds error
    shm_src = SharedMemoryDataSource([10, 20, 30, 40], name="slice_test")
    assert shm_src[-1] == 40
    assert shm_src[1:3] == [20, 30]
    with pytest.raises(IndexError, match="out of range"):
        _ = shm_src[10]
    with pytest.raises(IndexError, match="out of range"):
        _ = shm_src[-10]

    # SharedMemoryArray & SharedMemoryArrayMetadata .to_numpy() and __array__() across dtypes
    assert _dtype_itemsize("complex128") == 16
    assert _dtype_itemsize("bfloat16") == 2

    for dt_str, vals in (
        ("int32", [[1, 2], [3, 4]]),
        ("int64", [10, 20]),
        ("float32", [1.5, 2.5]),
        ("float64", 3.25),
        ("bool", [True, False]),
    ):
        arr = np.array(vals, dtype=np.dtype(dt_str))
        meta = SharedMemoryArray.from_array(arr)
        reconstructed = meta.to_numpy()
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == arr.shape
        assert reconstructed.tolist() == arr.tolist()
        sma = SharedMemoryArray.from_metadata(meta)
        assert array_to_list(sma.to_numpy()) == arr.tolist()
        assert array_to_list(sma.__array__()) == arr.tolist()
        assert array_to_list(meta.__array__()) == arr.tolist()
        meta.close_and_unlink()
        # After unlinking, to_numpy() returns a zero-filled array of matching shape
        zero_reconstructed = meta.to_numpy()
        assert zero_reconstructed.shape == arr.shape

    empty_sma = SharedMemoryArray(shape=(), dtype="int32")
    assert array_to_list(empty_sma.to_numpy()) == 0
    empty_sma.unlink_on_del()

    # Verify fallback to _ZERO_JAX_NUMPY_ADAPTER when numpy is absent
    meta_fb = SharedMemoryArray.from_array(np.array([5, 6], dtype=np.int32))
    meta_fb_str = SharedMemoryArrayMetadata(name=meta_fb.name, dtype="int32", shape=meta_fb.shape)
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "zero_jax.numpy", None)
    fb_arr = meta_fb_str.to_numpy()
    assert fb_arr.shape == (2,)
    assert fb_arr.ndim == 1
    meta_fb.close_and_unlink()
