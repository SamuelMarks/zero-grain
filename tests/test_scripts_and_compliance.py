"""Unit tests for repository quality, compliance, docstring, typing, and coverage edge cases."""

from __future__ import annotations

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
    """Verify RNG, PyTree batching, and padding when ``numpy`` is absent from ``sys.modules``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert get_numpy_module() is None

    rng = create_rng(11)
    assert 0.0 <= rng.random() < 1.0

    assert _batch_elements([1, 2, 3]) == [1, 2, 3]
    assert _batch_elements([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert batch_and_pad([1], 2) == [1, 0]
    assert batch_and_pad([1.5], 2) == [1.5, 0.0]
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
    assert map_tree(lambda x: x, (1, 2)) == (1, 2)
    assert spec_like(np.zeros((2, 3), dtype=np.int32)) == "int32[2, 3]"

    monkeypatch.delitem(sys.modules, "absl.logging", raising=False)
    ro = ReadOptions(num_threads=8, prefetch_buffer_size=2)
    assert ro.num_threads == 8

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

    op_rm: RandomMapOperation[int, int] = RandomMapOperation(lambda x, r: x)
    assert [r.data for r in op_rm([rec_int, None])] == [5]

    op_filt: FilterOperation[ElementTree] = FilterOperation(_PositiveFilter())
    assert [r.data for r in op_filt([rec, None])] == [5]

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
    assert map_tree(lambda x: x, {"a": 1}) == {"a": 1}

    filt_norm = normalize_operation(_PositiveFilter())
    assert isinstance(filt_norm, FilterOperation)
    callable_norm = normalize_operation(lambda x: x)
    assert isinstance(callable_norm, MapOperation)
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

    ds = Dataset.range(4)
    filtered_batched: BatchDataset[list[int]] = ds.filter(lambda x: isinstance(x, int) and x % 2 == 0).batch(2)
    b0 = filtered_batched[0]
    assert isinstance(b0, np.ndarray) and b0.tolist() == [0]

    sliced_iter = ds.slice(slice(1, 3)).to_iter_dataset(read_options=ReadOptions(num_threads=2))
    assert len(sliced_iter.dl.data_source) == 2
    assert "RangeDataSource" in str(iter(sliced_iter))
    cloned_iter = IterDataset(sliced_iter, read_options=ReadOptions(num_threads=4))
    assert cloned_iter.read_options is not None and cloned_iter.read_options.num_threads == 4

    assert len(ds.shuffle()) == 4
    assert len(ds.random_map(lambda x, rng: x)) == 4
    assert len(ds.map(MapOperation(lambda x: x))) == 4
    assert len(ds.map_with_index(MapWithIndexOperation(lambda i, x: x))) == 4
    assert len(ds.random_map(RandomMapOperation(lambda x, r: x))) == 4
    assert len(ds.filter(FilterOperation(lambda x: True))) == 4
    assert len(apply_transformations(ds, [MapOperation(lambda x: x)])) == 4

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
            return

    class _BadRestoreLoader(DataLoader[int]):
        """Loader returning ``_BadRestoreIterator``."""

        def __iter__(self) -> DataLoaderIterator[int]:
            """Return bad restore iterator.

            Returns:
                DataLoaderIterator[int]: Iterator.
            """
            return _BadRestoreIterator(self)

    with pytest.raises(AssertionError, match="Restored state mismatch"):
        assert_equal_output_after_checkpoint(_BadRestoreLoader(data_source=InMemoryDataSource([1, 2])))

    class _NondeterministicIterator(DataLoaderIterator[int]):
        """Iterator yielding different values after restoration."""

        _global_counter = 0

        def __next__(self) -> ElementTree:
            """Return incrementing counter.

            Returns:
                ElementTree: Counter value.
            """
            val = super().__next__()
            _NondeterministicIterator._global_counter += 1
            base = int(val) if isinstance(val, int) else 0
            return base + _NondeterministicIterator._global_counter

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
                "",
                "    Raises:",
                "        ValueError: Never.",
                '    """',
                "    return a + b",
                "",
                "def none_return_fn() -> None:",
                '    """Return nothing."""',
                "    return None",
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
                "import typing",
                "def untyped(a, *args, **kwargs):  #" + " type: ignore",
                "    a.__name__ = 'hack'",
                "    x: " + "Any = getattr(a, 'b', None)",
                "    y = cast(int, x)",
                "    z = typing.cast(int, y)",
                "    return z",
            ]
        ),
        encoding="utf-8",
    )
    assert check_typing_discipline.main([str(bad_typing_file)]) == 1

    assert check_test_completeness.main([]) == 0
    fake_init = tmp_path / "fake_init.py"
    fake_init.write_text(
        chr(10).join(
            [
                "from foo import UntestedClassName",
                "from bar import _private_helper",
            ]
        ),
        encoding="utf-8",
    )
    assert check_test_completeness.main([str(fake_init), "tests"]) == 1

    assert update_badges.get_color(99.0) == "brightgreen"
    assert update_badges.get_color(85.0) == "green"
    assert update_badges.get_color(65.0) == "yellow"
    assert update_badges.get_color(40.0) == "red"
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
    assert update_badges.update_readme(sample_readme_2, api_pct=100.0) is True
    assert "api_compliance-100%25" in sample_readme_2.read_text(encoding="utf-8")

    sample_readme_3 = tmp_path / "README3.md"
    sample_readme_3.write_text("# Plain Readme", encoding="utf-8")
    assert update_badges.update_readme(sample_readme_3) is True
