"""Unit tests for repository quality, compliance, docstring, typing, and coverage edge cases."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import sys
import numpy as np
import pytest

from scripts import (
    calc_compliance,
    check_compliance,
    check_deps,
    check_docstrings,
    check_test_completeness,
    check_typing_discipline,
    update_badges,
)
from zero_grain import (
    BatchDataset,
    BatchOperation,
    CopyNumPyArrayToSharedMemoryOperation,
    DataLoader,
    DataLoaderIterator,
    Dataset,
    FilterOperation,
    FilterTransform,
    FlatMapOperation,
    FlatMapTransform,
    IndexSampler,
    InMemoryDataSource,
    IterDataset,
    LoadOptions,
    MapOperation,
    MapWithIndexOperation,
    MapWithIndexTransform,
    Operation,
    RandomMapOperation,
    ReadOptions,
    Record,
    RecordMetadata,
    SequentialSampler,
    ShardOptions,
    _batch_elements,
    apply_transformations,
    assert_equal_output_after_checkpoint,
    batch_and_pad,
    create_rng,
    get_element_spec,
    load,
    map_tree,
    spec_like,
)
from zero_grain._rng import index_shuffle
from zero_grain._types import CheckpointStateDict, ElementTree, get_numpy_module
from zero_grain.data_loader import normalize_operation
from zero_grain.testing import _trees_equal


class _DoubleWithIndex(MapWithIndexTransform[ElementTree, ElementTree]):
    """Sample ``MapWithIndexTransform`` subclass."""

    def map_with_index(self, index: int, element: ElementTree) -> ElementTree:
        """Return ``index + element * 2``.

        Args:
            index (int): Record index.
            element (ElementTree): Input integer.

        Returns:
            ElementTree: Computed integer.
        """
        return index + int(element) * 2 if isinstance(element, int) else element


class _DuplicateFlatMap(FlatMapTransform[ElementTree, ElementTree]):
    """Sample ``FlatMapTransform`` subclass."""

    def flat_map(self, element: ElementTree) -> list[ElementTree]:
        """Duplicate ``element`` into a two-element list.

        Args:
            element (ElementTree): Input integer.

        Returns:
            list[ElementTree]: Two-element list ``[element, element]``.
        """
        return [element, element]


class _PositiveFilter(FilterTransform[ElementTree]):
    """Sample ``FilterTransform`` subclass."""

    def filter(self, element: ElementTree) -> bool:
        """Return True if ``element > 0``.

        Args:
            element (ElementTree): Input integer.

        Returns:
            bool: True when positive.
        """
        return isinstance(element, int) and element > 0


class _ConvertibleMock:
    """Sample class implementing ``ArrayConvertibleProtocol``."""

    def __array__(self) -> np.ndarray:
        """Return a 1D NumPy array.

        Returns:
            np.ndarray: 1D array.
        """
        return np.array([1, 2], dtype=np.int32)


def test_pure_python_fallbacks_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RNG, PyTree batching, and padding with ``zero_jax.numpy`` and pure-Python fallbacks.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    from zero_grain._types import JaxArray, array_to_list, dtype_name

    assert dtype_name(int) == "int64"
    assert dtype_name(float) == "float64"
    assert dtype_name(bool) == "bool"
    assert dtype_name(complex) == "complex128"

    class _FakeArrayNoToList:
        """ArrayProtocol stub without tolist() or to_list()."""

        shape = (1,)
        ndim = 1
        dtype = "int32"

    with pytest.raises(TypeError, match="Unsupported ArrayProtocol"):
        array_to_list(_FakeArrayNoToList())

    jnp_with_np = get_numpy_module()
    assert jnp_with_np is not None
    from zero_grain._types import _ZERO_JAX_NUMPY_ADAPTER

    arr_with_np = _ZERO_JAX_NUMPY_ADAPTER.array([1, 2])
    assert isinstance(arr_with_np, JaxArray)
    zeros_with_np = _ZERO_JAX_NUMPY_ADAPTER.zeros_like(arr_with_np)
    assert isinstance(zeros_with_np, JaxArray)
    stacked_jax_with_np = _ZERO_JAX_NUMPY_ADAPTER.array([arr_with_np, arr_with_np])
    assert stacked_jax_with_np.shape == (2, 2)
    scalar_jax_with_np = _ZERO_JAX_NUMPY_ADAPTER.array(5)
    assert _ZERO_JAX_NUMPY_ADAPTER.array([scalar_jax_with_np]).shape == (1,)

    monkeypatch.setitem(sys.modules, "numpy", None)
    jnp_adapter = get_numpy_module()
    assert jnp_adapter is not None
    jnp_batched = _batch_elements([1, 2, 3])
    assert isinstance(jnp_batched, JaxArray)
    jnp_padded = batch_and_pad([jnp_batched], 2)
    assert isinstance(jnp_padded, JaxArray) and jnp_padded.shape == (2, 3)
    assert isinstance(jnp_adapter.asarray([1, 2]), JaxArray)
    assert jnp_adapter.array([]).shape == (0,)
    assert jnp_adapter.array([1.5, 2.5]).shape == (2,)
    assert jnp_adapter.array([True, False]).shape == (2,)
    assert jnp_adapter.array(["unsupported"]).shape == (1,)
    scalar_no_np = jnp_adapter.array(7)
    assert jnp_adapter.array([scalar_no_np]).shape == (1,)
    assert jnp_adapter.zeros_like(scalar_no_np).shape == ()
    jnp_2d = jnp_adapter.array([[1.0, 2.0], [3.0, 4.0]])
    assert jnp_adapter.zeros_like(jnp_2d).shape == (2, 2)
    jnp_rng = create_rng(7)
    assert 0.0 <= jnp_rng.random() < 1.0

    monkeypatch.setitem(sys.modules, "zero_jax.numpy", None)
    assert get_numpy_module() is None

    rng = create_rng(11)
    assert 0.0 <= rng.random() < 1.0

    assert _batch_elements([1, 2, 3]) == [1, 2, 3]
    assert _batch_elements([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert batch_and_pad([], 2) == []
    assert batch_and_pad([1], 0) == []
    assert batch_and_pad([1, 2, 3], 2) == [1, 2]
    assert batch_and_pad([1], 2) == [1, 0]
    assert batch_and_pad([1.5], 2) == [1.5, 0.0]
    assert batch_and_pad([True], 2) == [True, False]
    assert batch_and_pad(["a"], 2, pad_value="z") == ["a", "z"]
    assert batch_and_pad([{"k": 1}], 2) == {"k": [1, 0]}
    assert batch_and_pad([[1, 2]], 2) == [[1, 0], [2, 0]]
    assert batch_and_pad([(1, 2)], 2) == ([1, 0], [2, 0])


def test_numpy_array_creation_error_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ``_convert_leaf_batch`` falls back to a list if ``np.array`` raises an exception.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    class _BrokenNumpy:
        """Stub numpy module that fails on array()."""

        ndarray = np.ndarray
        random = np.random

        def array(self, obj: ElementTree) -> ElementTree:
            """Raise ValueError on array creation.

            Args:
                obj (ElementTree): Input element.

            Returns:
                ElementTree: Never returns.
            """
            del obj
            raise ValueError("broken array")

        def asarray(self, obj: ElementTree) -> ElementTree:
            """Raise ValueError.

            Args:
                obj (ElementTree): Input element.

            Returns:
                ElementTree: Never returns.
            """
            del obj
            raise ValueError("broken")

        def zeros_like(self, a: ElementTree) -> ElementTree:
            """Raise ValueError.

            Args:
                a (ElementTree): Input array.

            Returns:
                ElementTree: Never returns.
            """
            del a
            raise ValueError("broken")

    monkeypatch.setitem(sys.modules, "numpy", _BrokenNumpy())
    assert _batch_elements([1, 2]) == [1, 2]


def test_tree_and_operations_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover remaining branches in ``_tree.py``, ``operations.py``, ``options.py``, and ``samplers.py``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    from dataclasses import dataclass
    from typing import NamedTuple
    from unittest import mock

    assert map_tree(lambda x: int(x) + 10 if isinstance(x, int) else x, (1, 2)) == (11, 12)
    assert spec_like(np.zeros((2, 3), dtype=np.int32)) == "int32[2, 3]"

    class _NT1(NamedTuple):
        """First NamedTuple for structure mismatch testing."""

        a: int

    class _NT2(NamedTuple):
        """Second NamedTuple for structure mismatch testing."""

        b: int

    @dataclass
    class _StdDC1:
        """First standard dataclass for structure mismatch testing."""

        x: int

    @dataclass
    class _StdDC2:
        """Second standard dataclass for structure mismatch testing."""

        y: int

    with pytest.raises(TypeError, match="Record structures do not match"):
        _batch_elements([_NT1(1), _NT2(2)])
    with pytest.raises(TypeError, match="Record structures do not match"):
        _batch_elements([_StdDC1(1), _StdDC2(2)])

    monkeypatch.delitem(sys.modules, "absl.logging", raising=False)
    with mock.patch("logging.warning") as mock_warn:
        ro = ReadOptions(num_threads=8, prefetch_buffer_size=2)
        assert ro.num_threads == 8
        mock_warn.assert_called_once()

    ss = SequentialSampler(num_records=3, shard_options=None)
    assert len(ss) == 3

    op_mwi = normalize_operation(_DoubleWithIndex())
    assert isinstance(op_mwi, MapWithIndexOperation)
    rec: Record[ElementTree] = Record(metadata=RecordMetadata(index=2), data=5)
    assert list(op_mwi([rec, None]))[0].data == 12

    op_fm = normalize_operation(_DuplicateFlatMap())
    assert isinstance(op_fm, FlatMapOperation)
    assert [r.data for r in op_fm([rec, None])] == [5, 5]

    op_map: MapOperation[int, int] = MapOperation(lambda x: x + 1)
    rec_int: Record[int] = Record(metadata=RecordMetadata(index=2), data=5)
    assert [r.data for r in op_map([rec_int, None])] == [6]

    op_rm: RandomMapOperation[int, int] = RandomMapOperation(lambda x, r: x + r.integers(10, 20))
    rm_out = [r.data for r in op_rm([rec_int, None])]
    assert len(rm_out) == 1 and 15 <= rm_out[0] < 25

    op_filt: FilterOperation[ElementTree] = FilterOperation(_PositiveFilter())
    assert [r.data for r in op_filt([rec, None])] == [5]
    assert op_filt.apply_element(5) == 5
    assert op_filt.apply_element(-1) is None

    shm_elem_op = CopyNumPyArrayToSharedMemoryOperation()
    shm_applied = shm_elem_op.apply_element(np.array([10, 20], dtype=np.int32))
    from zero_grain.shared_memory_array import SharedMemoryArray, SharedMemoryArrayMetadata

    assert isinstance(shm_applied, SharedMemoryArrayMetadata)
    assert len(shm_applied.get_bytes()) == 8
    reconstructed_sma = SharedMemoryArray.from_metadata(shm_applied)
    assert reconstructed_sma.metadata == shm_applied
    shm_applied.close_and_unlink()
    assert shm_applied.get_bytes() == b""

    base_op: Operation[int, int] = Operation()
    with pytest.raises(NotImplementedError):
        list(base_op([rec_int, None]))

    batch_op: BatchOperation[int] = BatchOperation(
        batch_size=2,
        drop_remainder=False,
        batch_fn=lambda xs: list(xs),
    )
    no_meta_rec: Record[int] = Record(metadata=None, data=9)
    res_b = list(batch_op([None, no_meta_rec, no_meta_rec, no_meta_rec]))
    assert len(res_b) == 2

    shm_op = CopyNumPyArrayToSharedMemoryOperation()
    assert len(list(shm_op([rec, None]))) == 1

    conv_padded = batch_and_pad([_ConvertibleMock()], 2)
    assert isinstance(conv_padded, (list, np.ndarray))
    assert len(conv_padded) == 2
    assert map_tree(lambda x: int(x) + 10 if isinstance(x, int) else x, {"a": 1}) == {"a": 11}

    filt_norm = normalize_operation(_PositiveFilter())
    assert isinstance(filt_norm, FilterOperation)
    callable_norm = normalize_operation(lambda x: int(x) + 1 if isinstance(x, int) else x)
    assert isinstance(callable_norm, MapOperation)
    assert callable_norm.apply_element(4) == 5
    batch_op._enable_shared_memory()
    assert batch_op._use_shared_memory is True
    fm_with_transform: FlatMapOperation[ElementTree, ElementTree] = FlatMapOperation(_DuplicateFlatMap())
    assert fm_with_transform.flat_map(3) == [3, 3]

    dl_none_src: DataLoader[int] = load(None, batch_size=None)
    assert len(dl_none_src.data_source) == 0

    assert _trees_equal(np.array([1, 2]), np.array([1, 2])) is True
    assert _trees_equal(np.array([1, 2]), np.array([1, 3])) is False
    assert _trees_equal([1, 2], [1, 2]) is True
    assert _trees_equal((1, 2), (1, 2)) is True


def test_dataset_and_dataloader_branches() -> None:
    """Cover remaining branches in ``dataset.py``, ``data_loader.py``, ``samplers.py``, and ``_rng.py``."""
    with pytest.raises(ValueError, match="rounds"):
        index_shuffle(0, 5, 1, rounds=3)
    with pytest.raises(ValueError, match="index must be"):
        index_shuffle(-1, 5, 1)
    assert index_shuffle(0, 0, 1) == 0

    empty_shard_sampler = IndexSampler(
        num_records=1,
        shard_options=ShardOptions(shard_index=1, shard_count=2, drop_remainder=False),
    )
    with pytest.raises(IndexError, match="Empty shard"):
        _ = empty_shard_sampler[0]
    empty_shard_dl: DataLoader[int] = DataLoader(
        data_source=InMemoryDataSource([1]),
        sampler=empty_shard_sampler,
    )
    assert list(empty_shard_dl) == []

    from zero_grain.samplers import Sampler

    class _CustomMetaSampler(Sampler):
        """Custom ``Sampler`` subclass yielding a single ``RecordMetadata``."""

        def __iter__(self) -> Iterator[RecordMetadata]:
            """Yield one ``RecordMetadata``.

            Yields:
                RecordMetadata: Single metadata record.
            """
            yield RecordMetadata(index=0, record_key=0)

    dl_meta_seq: DataLoader[int] = DataLoader(
        data_source=InMemoryDataSource([42]),
        sampler=_CustomMetaSampler(),
    )
    assert list(dl_meta_seq) == [42]

    from zero_grain._types import JaxArray
    import zero_jax

    raw_key = zero_jax.random.PRNGKey(11)
    assert isinstance(raw_key, JaxArray)
    key_sampler = IndexSampler(num_records=3, shuffle=True, seed=raw_key)
    assert len(key_sampler) == 3

    from zero_grain.data_sources import RangeDataSource
    from zero_grain.dataset import FilterDataset

    fd_src = FilterDataset.source(RangeDataSource(0, 2, 1))
    assert isinstance(fd_src, FilterDataset)
    assert list(Dataset(fd_src)) == [0, 1]
    assert isinstance(BatchDataset.source([1, 2]), BatchDataset)
    single_parent_sel: Dataset[int] = Dataset([Dataset.range(2)], selection_map=None)
    assert list(single_parent_sel) == [0, 1]

    ds = Dataset.range(4)
    filtered_batched: BatchDataset[list[int]] = ds.filter(lambda x: isinstance(x, int) and x % 2 == 0).batch(2)
    assert len(filtered_batched) == 1
    b0 = filtered_batched[0]
    assert isinstance(b0, np.ndarray) and b0.tolist() == [0, 2]
    assert (filtered_batched[0] == b0).all()

    all_filtered_ds = ds.filter(lambda x: False)
    assert get_element_spec(all_filtered_ds) is None
    first_filtered_ds = ds.filter(lambda x: isinstance(x, int) and x > 0)
    assert get_element_spec(first_filtered_ds) == "int"

    repeat_then_slice = ds.repeat(2).slice(slice(1, 6))
    assert list(repeat_then_slice.to_iter_dataset()) == [1, 2, 3, 0, 1]

    from zero_grain.experimental import FirstFitPackIterDataset

    single_bin_shuffled = FirstFitPackIterDataset(
        Dataset.source([1]),
        length_struct=2,
        num_packing_bins=1,
        shuffle_bins=True,
    )
    assert list(single_bin_shuffled) == [[1, 0]]

    sliced_iter = ds.slice(slice(1, 3)).to_iter_dataset(read_options=ReadOptions(num_threads=2))
    assert len(sliced_iter.dl.data_source) == 2
    assert "RangeDataSource" in str(iter(sliced_iter))
    cloned_iter = IterDataset(sliced_iter, read_options=ReadOptions(num_threads=4))
    assert cloned_iter.read_options is not None and cloned_iter.read_options.num_threads == 4

    shuffled_unseeded = ds.shuffle()
    assert len(shuffled_unseeded) == 4
    assert sorted(int(v) for v in shuffled_unseeded if isinstance(v, int)) == [0, 1, 2, 3]

    rm_ds: Dataset[int] = ds.random_map(
        lambda x, rng: (int(x) if isinstance(x, int) else 0) + rng.integers(10, 20),
        seed=5,
    )
    assert len(rm_ds) == 4
    assert all(10 <= val < 24 for val in rm_ds if isinstance(val, int))

    map_op_ds: Dataset[int] = ds.map(MapOperation(lambda x: int(x) + 10 if isinstance(x, int) else x))
    assert list(map_op_ds) == [10, 11, 12, 13]
    assert [map_op_ds[i] for i in range(4)] == [10, 11, 12, 13]

    mwi_op_ds: Dataset[int] = ds.map_with_index(
        MapWithIndexOperation(lambda i, x: i * 10 + (int(x) if isinstance(x, int) else 0))
    )
    assert list(mwi_op_ds) == [0, 11, 22, 33]

    rm_op_ds: Dataset[int] = ds.random_map(
        RandomMapOperation(lambda x, r: (int(x) if isinstance(x, int) else 0) + r.integers(50, 60)),
        seed=9,
    )
    assert len(rm_op_ds) == 4
    assert all(50 <= val < 64 for val in rm_op_ds if isinstance(val, int))

    filt_op_ds = ds.filter(FilterOperation(lambda x: isinstance(x, int) and x >= 2))
    assert list(filt_op_ds) == [2, 3]
    assert filt_op_ds[0] is None
    assert filt_op_ds[2] == 2

    applied_ds = apply_transformations(
        ds,
        [MapOperation(lambda x: int(x) * 3 if isinstance(x, int) else x)],
    )
    assert list(applied_ds) == [0, 3, 6, 9]

    dl_seq_sampler = DataLoader(
        data_source=InMemoryDataSource([10, 20, 30]),
        sampler=[0, 1],
    )
    it = DataLoaderIterator(
        dl_seq_sampler,
        state={"version": 2, "last_seen_indices": {"0": 0}, "last_worker_index": 0},
    )
    assert next(it) == 20
    it.set_state({})

    fm_dl = DataLoader(
        data_source=InMemoryDataSource([1, 2]),
        sampler=SequentialSampler(2),
        operations=[FlatMapOperation(lambda x: [x])],
    )
    fm_it = iter(fm_dl)
    fm_it.set_state({"version": 2, "last_seen_indices": {"0": 10}, "last_worker_index": 0})
    with pytest.raises(StopIteration):
        next(fm_it)

    fm_dl_workers: DataLoader[int] = DataLoader(
        data_source=InMemoryDataSource([1, 2, 3, 4]),
        sampler=SequentialSampler(4),
        operations=[FlatMapOperation(_DuplicateFlatMap())],
        worker_count=2,
    )
    assert_equal_output_after_checkpoint(fm_dl_workers)

    dl_from_load = load(
        InMemoryDataSource([1, 2]),
        num_epochs=1,
        seed=5,
        batch_size=2,
        worker_count=2,
        read_options=ReadOptions(),
        options=load_opts_helper(),
    )
    assert dl_from_load.worker_count == 2


def load_opts_helper() -> LoadOptions:
    """Construct a ``LoadOptions`` instance with all fields populated.

    Returns:
        LoadOptions: Populated options instance.
    """
    return LoadOptions(
        num_epochs=2,
        shuffle=True,
        seed=7,
        batch_size=3,
        drop_remainder=True,
        worker_count=4,
        read_options=ReadOptions(),
    )


def test_testing_trees_equal_and_checkpoint_errors() -> None:
    """Verify ``_trees_equal`` and error branches in ``assert_equal_output_after_checkpoint``."""
    assert _trees_equal([1, 2], [1]) is False
    assert _trees_equal((1, 2), (1,)) is False
    assert _trees_equal({"a": 1}, {"b": 1}) is False
    assert _trees_equal({"a": 1}, {"a": 1}) is True

    empty_dl: DataLoader[int] = DataLoader(data_source=InMemoryDataSource([]))
    with pytest.raises(AssertionError, match="Dataset did not produce any elements"):
        assert_equal_output_after_checkpoint(empty_dl)

    class _SpecFlippingIterator(DataLoaderIterator[int]):
        """Iterator whose state structure changes on step 1."""

        def __init__(self, dl: DataLoader[int]) -> None:
            """Initialize ``_SpecFlippingIterator``.

            Args:
                dl (DataLoader[int]): Parent loader.
            """
            super().__init__(dl)
            self._calls = 0

        def get_state(self) -> CheckpointStateDict:
            """Return state with changing keys.

            Returns:
                CheckpointStateDict: State dict.
            """
            self._calls += 1
            if self._calls == 1:
                return {"version": 2}
            return {"last_worker_index": 2}

    class _SpecFlippingLoader(DataLoader[int]):
        """Loader returning ``_SpecFlippingIterator``."""

        def __iter__(self) -> DataLoaderIterator[int]:
            """Return flipping iterator.

            Returns:
                DataLoaderIterator[int]: Iterator.
            """
            return _SpecFlippingIterator(self)

    with pytest.raises(AssertionError, match="State spec mismatch"):
        assert_equal_output_after_checkpoint(_SpecFlippingLoader(data_source=InMemoryDataSource([1, 2])))

    class _BadRestoreIterator(DataLoaderIterator[int]):
        """Iterator that ignores ``set_state``."""

        def set_state(self, state: CheckpointStateDict) -> None:
            """Ignore state restoration.

            Args:
                state (CheckpointStateDict): Ignored state.
            """
            del state

    class _BadRestoreLoader(DataLoader[int]):
        """Loader returning ``_BadRestoreIterator``."""

        def __iter__(self) -> DataLoaderIterator[int]:
            """Return broken iterator.

            Returns:
                DataLoaderIterator[int]: Iterator.
            """
            return _BadRestoreIterator(self)

    with pytest.raises(AssertionError, match="Restored state mismatch"):
        assert_equal_output_after_checkpoint(_BadRestoreLoader(data_source=InMemoryDataSource([1, 2])))

    class _NondeterministicIterator(DataLoaderIterator[int]):
        """Iterator yielding incrementing values regardless of restored state."""

        _global_counter = 0

        def __next__(self) -> ElementTree:
            """Return incrementing counter for 2 steps.

            Returns:
                ElementTree: Counter value.

            Raises:
                StopIteration: After 2 elements per instance.
            """
            if self._step_offset >= 2:
                raise StopIteration
            self._step_offset += 1
            _NondeterministicIterator._global_counter += 1
            return _NondeterministicIterator._global_counter

    class _NondeterministicLoader(DataLoader[int]):
        """Loader returning ``_NondeterministicIterator``."""

        def __iter__(self) -> DataLoaderIterator[int]:
            """Return nondeterministic iterator.

            Returns:
                DataLoaderIterator[int]: Iterator.
            """
            return _NondeterministicIterator(self)

    with pytest.raises(AssertionError, match="Restored values mismatch"):
        assert_equal_output_after_checkpoint(_NondeterministicLoader(data_source=InMemoryDataSource([1, 2])))


def test_scripts_execution(tmp_path: Path) -> None:
    """Verify all scripts in ``scripts/`` achieve 100% line and branch coverage.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    assert calc_compliance.main([]) == 0
    assert calc_compliance._extract_module_exports(tmp_path / "nonexistent_init.py") == set()
    sample_init = tmp_path / "sample_init.py"
    sample_init.write_text(
        chr(10).join(
            [
                "from mod import A, B as C",
                "def helper_fn() -> None:",
                "    pass",
                "class HelperCls:",
                "    pass",
                "X = 1",
                '__all__ = ["A", "C", "X"]',
            ]
        ),
        encoding="utf-8",
    )
    assert {"A", "C", "helper_fn", "HelperCls", "X"} <= calc_compliance._extract_module_exports(sample_init)

    bad_snap = tmp_path / "bad_snap.json"
    bad_snap.write_text('{"categories": {"extras": [{"name": "MissingApi"}]}}', encoding="utf-8")
    assert calc_compliance.main([str(bad_snap)]) == 1
    empty_snap = tmp_path / "empty_snap.json"
    empty_snap.write_text('{"categories": {"extras": []}}', encoding="utf-8")
    assert calc_compliance.main([str(empty_snap)]) == 0

    assert check_compliance.main([str(tmp_path / "nonexistent_plan.md")]) == 0
    plan_file = tmp_path / "SEMANTIC_PLAN.md"
    plan_file.write_text(
        chr(10).join(
            [
                "# Plan",
                "## `zero_grain`",
                "- [ ] `DataLoader`",
                "- [ ] `MissingSymbol`",
                "## `zero_nonexistent_mod`",
                "- [ ] `Foo`",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compliance.main(["--fix", str(plan_file)]) == 1
    good_plan = tmp_path / "GOOD_PLAN.md"
    good_plan.write_text(
        chr(10).join(["## `zero_grain`", "- [x] `DataLoader`", ""]),
        encoding="utf-8",
    )
    assert check_compliance.main([str(good_plan)]) == 0
    assert check_compliance.main([]) == 0
    assert check_compliance.main(["--strict-v005"]) == 0

    assert check_compliance._extract_ast_signatures(tmp_path / "nonexistent.py") == ({}, {}, set())
    assert check_compliance.parse_todo_plan_unchecked(tmp_path / "nonexistent_todo.md") == set()
    assert check_compliance._collect_codebase_surface(tmp_path / "nonexistent_src") == (
        {},
        {},
        {
            "grain": set(),
            "grain.python": set(),
            "grain.experimental": set(),
            "grain.python.experimental": set(),
        },
    )
    assert check_compliance.verify_snapshot_v005(snapshot_path=tmp_path / "missing_snap.json")[0] == 0

    fake_src = tmp_path / "fake_src"
    (fake_src / "python").mkdir(parents=True)
    (fake_src / "__init__.py").write_text(
        chr(10).join(
            [
                "from .mod import MyCls, AliasCls, top_fn",
                "__all__ = ['MyCls', 'AliasCls', 'top_fn']",
            ]
        ),
        encoding="utf-8",
    )
    (fake_src / "python" / "__init__.py").write_text(
        "from ..mod import MyCls\n__all__ = ['MyCls']\n",
        encoding="utf-8",
    )
    (fake_src / "experimental.py").write_text(
        "class ExpCls:\n    '''Exp doc.'''\n    def __init__(self, a: int = 1) -> None:\n        '''Init.'''\n",
        encoding="utf-8",
    )
    (fake_src / "python" / "experimental.py").write_text(
        "from ..experimental import ExpCls\n__all__ = ['ExpCls']\n",
        encoding="utf-8",
    )
    (fake_src / "mod.py").write_text(
        chr(10).join(
            [
                "class MyCls:",
                "    '''Doc.'''",
                "    x = 1",
                "    y: int = 2",
                "    def __init__(self, pos_only: int, /, reg: int = 1, *args: int, kw: int = 2, **kwargs: int) -> None:",
                "        '''Init.'''",
                "    def documented_m(self, val: int) -> int:",
                "        '''Method doc.'''",
                "        return val",
                "    def nodoc_m(self, val) -> int:",
                "        return val",
                "AliasCls = MyCls",
                "UnknownAlias = ExternalThing",
                "NotAlias, OtherTarget = 1, 2",
                "def top_fn(a: int, unannotated_p=1) -> int:",
                "    return a",
            ]
        ),
        encoding="utf-8",
    )
    custom_snap = tmp_path / "custom_v005.json"
    custom_snap.write_text(
        chr(10).join(
            [
                "{",
                '  "categories": {',
                '    "array": [',
                '      "not_a_mapping",',
                '      {"name": "NoApiPath"},',
                '      {"name": "MyCls", "api_path": "grain.MyCls", "params": ["not_a_dict", {"name": "reg"}]},',
                '      {"name": "AliasCls", "api_path": "grain.AliasCls", "params": []},',
                '      {"name": "top_fn", "api_path": "grain.top_fn", "params": [{"name": "missing_p"}]},',
                '      {"name": "MissingTop", "api_path": "grain.MissingTop", "params": []},',
                '      {"name": "MyCls.documented_m", "api_path": "grain.MyCls.documented_m", "params": [{"name": "val"}]},',
                '      {"name": "nodoc_m", "api_path": "grain.MyCls.nodoc_m", "params": [{"name": "missing_m_p"}]},',
                '      {"name": "MyCls.missing_m", "api_path": "grain.MyCls.missing_m", "params": []},',
                '      {"name": "ExpCls", "api_path": "grain.experimental.ExpCls", "params": [{"name": "a"}]},',
                '      {"name": "UnexportedExp", "api_path": "grain.experimental.UnexportedExp", "params": []}',
                "    ]",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compliance.main(["--strict-v005", f"--snapshot={custom_snap}"]) == 1
    (fake_src / "unexp.py").write_text(
        "class UnexportedExp:\n    '''Doc.'''\n    pass\nclass UnexportedMain:\n    '''Doc.'''\n    pass\n",
        encoding="utf-8",
    )
    custom_snap_2 = tmp_path / "custom_v005_2.json"
    custom_snap_2.write_text(
        '{"categories": {"array": [{"name": "UnexportedMain", "api_path": "grain.UnexportedMain", "params": []}, {"name": "UnexportedExp", "api_path": "grain.experimental.UnexportedExp", "params": []}]}}',
        encoding="utf-8",
    )
    c1, t1, v1 = check_compliance.verify_snapshot_v005(
        snapshot_path=custom_snap,
        todo_plan_path=None,
        src_root=fake_src,
        strict_all=True,
    )
    assert t1 == 9 and c1 < t1 and len(v1) >= 5
    c2, t2, v2 = check_compliance.verify_snapshot_v005(
        snapshot_path=custom_snap_2,
        todo_plan_path=None,
        src_root=fake_src,
        strict_all=True,
    )
    assert t2 == 2 and c2 == 0 and len(v2) == 3

    todo_sample = tmp_path / "TODO_SAMPLE.md"
    todo_sample.write_text(
        "- [ ] Implement `zero_grain.UnexportedMain` and `grain.experimental.UnexportedExp`\n- [x] Done `grain.MyCls`\n",
        encoding="utf-8",
    )
    c3, t3, v3 = check_compliance.verify_snapshot_v005(
        snapshot_path=custom_snap_2,
        todo_plan_path=todo_sample,
        src_root=fake_src,
        strict_all=False,
    )
    assert c3 == 2 and t3 == 2 and v3 == []

    assert check_deps.main(["src"]) == 0
    assert check_deps.main([str(tmp_path / "missing_dir")]) == 1
    bad_dep_file = tmp_path / "bad_dep.py"
    bad_dep_file.write_text(
        chr(10).join(
            [
                "import third_party_pkg",
                "from another_pkg import thing",
                "from .local import ok",
                "from os import path",
            ]
        ),
        encoding="utf-8",
    )
    assert check_deps.main([str(bad_dep_file)]) == 1
    syntax_err_file = tmp_path / "syntax_err.py"
    syntax_err_file.write_text("def broken(:" + chr(10), encoding="utf-8")
    assert check_deps.main([str(syntax_err_file)]) == 1

    assert check_docstrings.main(["src", "scripts"]) == 0
    assert check_docstrings.main([str(tmp_path / "missing_doc_dir")]) == 1
    assert len(check_docstrings.check_file_docstrings(syntax_err_file)) == 1

    bad_doc_file = tmp_path / "bad_doc.py"
    bad_doc_file.write_text(
        chr(10).join(
            [
                "class NoDoc:",
                "    pass",
                "",
                "def no_fn_doc(x: int) -> int:",
                "    return x",
            ]
        ),
        encoding="utf-8",
    )
    assert check_docstrings.main([str(bad_doc_file)]) == 1

    placeholder_doc_file = tmp_path / "placeholder_doc.py"
    placeholder_doc_file.write_text(
        chr(10).join(
            [
                '"""Module docstring."""',
                "import typing",
                "",
                "class BadCls:",
                '    """Docstring for BadCls."""',
                "",
                "def bad_fn(a: int, *args: int, **kwargs: int) -> int:",
                '    """Docstring for bad_fn."""',
                "    return a",
                "",
                "def missing_param_fn(a: int, b: int) -> int:",
                '    """Compute sum.',
                "",
                "    Args:",
                "        a (int): First.",
                "        stale_arg (int): Extra.",
                "",
                "    Raises:",
                "        ValueError: Never.",
                '    """',
                "    def _inner() -> int:",
                "        return a",
                "    return _inner() + b",
                "",
                "def gen_fn():",
                '    """Generator missing yields section."""',
                "    def _nested() -> None:",
                "        pass",
                "    yield 1",
                "",
                "def none_return_fn() -> None:",
                '    """Return nothing."""',
                "    return None",
                "",
                "def noreturn_attr_fn() -> typing.NoReturn:",
                '    """Raise error."""',
                "    raise RuntimeError()",
                "",
                "def untyped_return_fn():",
                '    """Return 1."""',
                "    return 1",
                "",
                "def missing_returns_doc() -> int:",
                '    """Missing returns section."""',
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    assert check_docstrings.main([str(placeholder_doc_file)]) == 1

    bare_file = tmp_path / "bare.py"
    bare_file.write_text("# empty", encoding="utf-8")
    assert check_deps.check_file(bare_file) == []

    assert check_typing_discipline.main(["src", "scripts"]) == 0
    assert check_typing_discipline.main([str(tmp_path / "missing_typing_dir")]) == 1
    assert len(check_typing_discipline.check_file_typing(syntax_err_file)) == 1

    bad_typing_file = tmp_path / "bad_typing.py"
    bad_typing_file.write_text(
        chr(10).join(
            [
                "from typing import " + "Any, cast",
                "from typing_extensions import " + "Any as ExtAny",
                "import builtins",
                "import typing",
                "def untyped(a, *args, **kwargs):  #" + " type: ignore",
                "    a.__name__ = 'hack'",
                "    d = a.__" + "dict__",
                "    c = a.__" + "class__",
                "    x: " + "Any = getattr(a, 'b', None)",
                "    x2 = builtins." + "getattr(a, 'b', None)",
                "    fn_ref = builtins." + "setattr",
                "    t_any = typing." + "Any",
                "    y = cast(int, x)",
                "    z = typing.cast(int, y)",
                "    try:",
                "        v = vars(a)",
                "    except:",
                "        pass",
                "    try:",
                "        v2 = d",
                "    except Exception:",
                "        pass",
                "    return z",
            ]
        ),
        encoding="utf-8",
    )
    violations = check_typing_discipline.check_file_typing(bad_typing_file)
    assert len(violations) >= 12
    assert check_typing_discipline.main([str(bad_typing_file)]) == 1

    assert check_test_completeness.main([]) == 0
    fake_init = tmp_path / "fake_init.py"
    fake_init.write_text(
        chr(10).join(
            [
                "from foo import UntestedClassName",
                "from bar import _private_helper",
                '__all__ = ["UntestedClassName", "AnotherMissingSymbol"]',
            ]
        ),
        encoding="utf-8",
    )
    fake_tests_dir = tmp_path / "fake_tests"
    fake_tests_dir.mkdir()
    fake_test_file = fake_tests_dir / "test_sample.py"
    fake_test_file.write_text(
        chr(10).join(
            [
                "def helper_without_assert() -> None:",
                "    UntestedClassName()",
                "",
                "class SubCls(BaseSym, mod.AttrBase, GenBase[int], mod.GenAttr[int], make_base()):",
                "    def test_method(self) -> None:",
                "        res = obj.call_method(ArgSym, mod.ArgAttr, kw=KwSym)",
                "        self.assertEqual(res, 42)",
            ]
        ),
        encoding="utf-8",
    )
    tested_set = check_test_completeness.get_tested_names(fake_tests_dir)
    assert "UntestedClassName" not in tested_set
    assert {"BaseSym", "AttrBase", "GenBase", "GenAttr", "ArgSym", "ArgAttr", "KwSym"} <= tested_set
    assert check_test_completeness.main([str(fake_init), str(fake_tests_dir)]) == 1

    # Verify tautological assertion detection in check_test_completeness
    taut_tests_dir = tmp_path / "taut_tests"
    taut_tests_dir.mkdir()
    taut_file = taut_tests_dir / "test_taut.py"
    taut_file.write_text(
        chr(10).join(
            [
                "class TautSuite:",
                "    def test_eq_self(self) -> None:",
                "        OnlyInTautEq()",
                "        self.assertEqual(1, 1)",
                "    def test_true_const(self) -> None:",
                "        OnlyInTautTrue()",
                "        self.assertTrue(True)",
                "    def test_assert_self(self) -> None:",
                "        x = OnlyInAssertSelf()",
                "        assert x == x",
                "        assert True",
            ]
        ),
        encoding="utf-8",
    )
    taut_errs = check_test_completeness.find_tautological_assertions(taut_tests_dir)
    assert len(taut_errs) == 4
    assert check_test_completeness.get_tested_names(taut_tests_dir) == set()
    assert check_test_completeness.main([str(fake_init), str(taut_tests_dir)]) == 1

    assert update_badges.get_color(99.0) == "brightgreen"
    assert update_badges.get_color(85.0) == "green"
    assert update_badges.get_color(65.0) == "yellow"
    assert update_badges.get_color(40.0) == "red"
    assert update_badges.compute_doc_pct(()) == 100.0
    assert update_badges.compute_doc_pct((Path("src/zero_grain/record.py"),)) == 100.0
    assert update_badges.update_readme(tmp_path / "missing_readme.md") is False

    sample_readme_1 = tmp_path / "README1.md"
    sample_readme_1.write_text(
        chr(10).join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)",
                "[![API Compliance](https://img.shields.io/badge/api_compliance-50%25-red.svg)](#)",
            ]
        ),
        encoding="utf-8",
    )
    assert update_badges.main([str(sample_readme_1)]) == 0
    assert "100%25-brightgreen" in sample_readme_1.read_text(encoding="utf-8")

    sample_readme_2 = tmp_path / "README2.md"
    sample_readme_2.write_text(
        chr(10).join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)",
            ]
        ),
        encoding="utf-8",
    )
    assert update_badges.update_readme(sample_readme_2, doc_pct=100.0, api_pct=100.0) is True
    assert "api_compliance-100%25" in sample_readme_2.read_text(encoding="utf-8")

    sample_readme_3 = tmp_path / "README3.md"
    sample_readme_3.write_text("# Plain Readme", encoding="utf-8")
    assert update_badges.update_readme(sample_readme_3, doc_pct=100.0) is True
    assert sample_readme_3.read_text(encoding="utf-8") == "# Plain Readme"
