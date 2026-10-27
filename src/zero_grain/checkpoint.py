"""Orbax-compatible checkpoint handler for ``DataLoaderIterator`` and ``DatasetIterator``."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from typing import TypeVar
from zero_grain._types import CheckpointStateDict, ElementTree
from zero_orbax.checkpoint import (
    AsyncCheckpointHandler,
    CheckpointHandlerRegistry,
    DefaultCheckpointHandlerRegistry,
    _CheckpointArgsBase,
    args as orbax_args,
)
from zero_orbax.core import AtomicWrite, FileSystemIO
from zero_orbax.types import GrainIteratorProtocol, JsonValue, PathLike, PyTree

_V_co = TypeVar("_V_co", covariant=True)


def _state_to_json_dict(state: CheckpointStateDict) -> dict[str, JsonValue]:
    """Convert a ``CheckpointStateDict`` into a shallow ``dict[str, JsonValue]`` copy.

    Args:
        state (CheckpointStateDict): Source checkpoint state dictionary.

    Returns:
        dict[str, JsonValue]: Copy with ``JsonValue``-typed entries.
    """
    return {
        "version": state.get("version", 2),
        "last_seen_indices": {k: v for k, v in state.get("last_seen_indices", {}).items()},
        "last_worker_index": state.get("last_worker_index", -1),
        "worker_count": state.get("worker_count", 0),
        "sampler": state.get("sampler", ""),
        "data_source": state.get("data_source", ""),
    }


def _extract_process_shard(
    pos_args: tuple[PyTree | _CheckpointArgsBase | ElementTree, ...],
    kwargs: dict[str, PyTree | _CheckpointArgsBase | ElementTree],
) -> tuple[int, int]:
    """Extract ``(process_index, process_count)`` from positional or keyword arguments.

    Args:
        pos_args (tuple[PyTree | _CheckpointArgsBase | ElementTree, ...]): Positional arguments.
        kwargs (dict[str, PyTree | _CheckpointArgsBase | ElementTree]): Keyword arguments.

    Returns:
        tuple[int, int]: ``(process_index, process_count)``.
    """
    kw_idx = kwargs.get("process_index")
    kw_cnt = kwargs.get("process_count")
    proc_idx = (
        int(pos_args[0])
        if len(pos_args) >= 1 and isinstance(pos_args[0], int)
        else (int(kw_idx) if isinstance(kw_idx, int) else 0)
    )
    proc_cnt = (
        int(pos_args[1])
        if len(pos_args) >= 2 and isinstance(pos_args[1], int)
        else (int(kw_cnt) if isinstance(kw_cnt, int) else 1)
    )
    return proc_idx, proc_cnt


def _coerce_state_dict(
    raw_map: CheckpointStateDict | Mapping[str, _V_co],
) -> CheckpointStateDict | None:
    """Coerce a mapping into a ``CheckpointStateDict`` if it represents iterator state.

    Args:
        raw_map (CheckpointStateDict | Mapping[str, _V_co]):
            Candidate state mapping.

    Returns:
        CheckpointStateDict | None: Coerced ``CheckpointStateDict`` or None.
    """
    raw_indices = raw_map.get("last_seen_indices")
    if not isinstance(raw_indices, Mapping):
        return None
    ver = raw_map.get("version", 2)
    last_w = raw_map.get("last_worker_index", -1)
    w_cnt = raw_map.get("worker_count", 0)
    sampler_str = raw_map.get("sampler", "")
    source_str = raw_map.get("data_source", "")
    return {
        "version": int(ver) if isinstance(ver, int) else 2,
        "last_seen_indices": {str(k): int(v) for k, v in raw_indices.items() if isinstance(v, int)},
        "last_worker_index": int(last_w) if isinstance(last_w, int) else -1,
        "worker_count": int(w_cnt) if isinstance(w_cnt, int) else 0,
        "sampler": str(sampler_str) if isinstance(sampler_str, str) else "",
        "data_source": str(source_str) if isinstance(source_str, str) else "",
    }


def _unwrap_iterator_arg(
    item: GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree,
    pos_args: tuple[GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree, ...],
    kwargs: dict[str, GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree],
) -> tuple[GrainIteratorProtocol | None, CheckpointStateDict | None]:
    """Extract a ``GrainIteratorProtocol`` or ``CheckpointStateDict`` from positional or keyword ``CheckpointArgs``.

    Args:
        item (GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree): Primary item or args wrapper.
        pos_args (tuple[GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree, ...]): Extra positional args.
        kwargs (dict[str, GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree]): Keyword args.

    Returns:
        tuple[GrainIteratorProtocol | None, CheckpointStateDict | None]:
            Pair of ``(iterator_or_none, explicit_state_dict_or_none)``.
    """
    candidate: GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree = item
    if candidate is None:
        if "args" in kwargs:
            candidate = kwargs["args"]
        elif len(pos_args) >= 1:
            candidate = pos_args[0]
    if isinstance(candidate, _CheckpointArgsBase):
        candidate = candidate.item
    if isinstance(candidate, GrainIteratorProtocol):
        return candidate, None
    if isinstance(candidate, Mapping):
        return None, _coerce_state_dict(candidate)
    return None, None


def validate_restored_state(
    state: CheckpointStateDict,
    iterator: GrainIteratorProtocol | None = None,
) -> None:
    """Validate a restored ``CheckpointStateDict`` using ``zero_chex`` assertions before mutating ``iterator``.

    Args:
        state (CheckpointStateDict): Restored checkpoint state dictionary.
        iterator (GrainIteratorProtocol | None): Target iterator to validate against when provided.

    Raises:
        ValueError: If ``version``, ``last_seen_indices``, ``sampler``, or ``data_source`` is invalid.
    """
    import zero_optax as _zero_optax  # noqa: F401
    import zero_chex

    version_val = state.get("version", 2)
    zero_chex.assert_scalar_non_negative(
        version_val,
        custom_message=f"Checkpoint version must be non-negative, got {version_val}",
        exception_type=ValueError,
    )
    indices = state.get("last_seen_indices")
    zero_chex.assert_equal(
        isinstance(indices, dict),
        True,
        custom_message="Checkpoint last_seen_indices must be a dict",
        exception_type=ValueError,
    )
    if iterator is not None:
        expected = iterator.get_state()
        zero_chex.assert_equal(
            state.get("version", 2),
            expected.get("version", 2),
            custom_message=(
                f"Checkpoint version {state.get('version', 2)} does not match expected {expected.get('version', 2)}"
            ),
            exception_type=ValueError,
        )
        state_sampler = state.get("sampler", "")
        expected_sampler = expected.get("sampler", "")
        if state_sampler and expected_sampler:
            zero_chex.assert_equal(
                state_sampler,
                expected_sampler,
                custom_message=f"Checkpoint sampler {state_sampler} does not match DataLoader sampler {expected_sampler}",
                exception_type=ValueError,
            )
        state_source = state.get("data_source", "")
        expected_source = expected.get("data_source", "")
        if state_source and expected_source:
            zero_chex.assert_equal(
                state_source,
                expected_source,
                custom_message=(
                    f"Checkpoint data_source {state_source} does not match DataLoader data_source {expected_source}"
                ),
                exception_type=ValueError,
            )


class CheckpointHandler(AsyncCheckpointHandler):
    """CheckpointHandler saving and restoring iterator states as JSON via ``zero_orbax``."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``CheckpointHandler``.

        Args:
            *args (ElementTree): Optional positional initialization arguments.
            **kwargs (ElementTree): Optional keyword initialization arguments.
        """
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._last_state: CheckpointStateDict | None = None

    def _resolve_checkpoint_filename(
        self,
        process_index: int = 0,
        process_count: int = 1,
    ) -> str:
        """Construct the checkpoint filename for a given process shard.

        Args:
            process_index (int): Zero-based process index.
            process_count (int): Total number of processes.

        Returns:
            str: Checkpoint JSON filename.
        """
        return f"process_{process_index}-of-{process_count}.json"

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree | _CheckpointArgsBase | ElementTree = None,
        *pos_args: PyTree | _CheckpointArgsBase | ElementTree,
        **kwargs: PyTree | _CheckpointArgsBase | ElementTree,
    ) -> None:
        """Save the state of ``item`` to ``directory`` using ``FileSystemIO`` and ``AtomicWrite``.

        Args:
            directory (PathLike | None): Target checkpoint directory or virtual URI.
            item (PyTree | _CheckpointArgsBase | ElementTree): Iterator, state dict, or ``CheckpointArgs`` to save.
            *pos_args (PyTree | _CheckpointArgsBase | ElementTree): Additional positional arguments.
            **kwargs (PyTree | _CheckpointArgsBase | ElementTree): Additional keyword arguments.
        """
        target_item, explicit_state = _unwrap_iterator_arg(item, pos_args, kwargs)
        if directory is None or (target_item is None and explicit_state is None):
            return
        proc_idx, proc_cnt = _extract_process_shard(pos_args, kwargs)
        dir_str = os.fspath(directory) if isinstance(directory, os.PathLike) else str(directory)
        FileSystemIO().makedirs(dir_str, exist_ok=True)
        state: CheckpointStateDict = target_item.get_state() if target_item is not None else (explicit_state or {})
        self._last_state = state
        filename = self._resolve_checkpoint_filename(
            process_index=proc_idx,
            process_count=proc_cnt,
        )
        checkpoint_file = f"{dir_str.rstrip('/')}/{filename}"
        AtomicWrite().atomic_write_bytes(
            checkpoint_file,
            json.dumps(state, indent=2).encode("utf-8"),
        )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree | _CheckpointArgsBase | ElementTree = None,
        *pos_args: PyTree | _CheckpointArgsBase | ElementTree,
        **kwargs: PyTree | _CheckpointArgsBase | ElementTree,
    ) -> PyTree:
        """Restore the state of ``item`` from ``directory`` using ``FileSystemIO``.

        Args:
            directory (PathLike | None): Source checkpoint directory or virtual URI.
            item (PyTree | _CheckpointArgsBase | ElementTree): Optional iterator or ``CheckpointArgs``.
            *pos_args (PyTree | _CheckpointArgsBase | ElementTree): Additional positional arguments.
            **kwargs (PyTree | _CheckpointArgsBase | ElementTree): Additional keyword arguments.

        Returns:
            PyTree: Restored iterator if ``item`` was given, raw state dict if only ``directory`` was given, or None.
        """
        target_item, _ = _unwrap_iterator_arg(item, pos_args, kwargs)
        if directory is None:
            return target_item
        proc_idx, proc_cnt = _extract_process_shard(pos_args, kwargs)
        dir_str = os.fspath(directory) if isinstance(directory, os.PathLike) else str(directory)
        filename = self._resolve_checkpoint_filename(
            process_index=proc_idx,
            process_count=proc_cnt,
        )
        checkpoint_file = f"{dir_str.rstrip('/')}/{filename}"
        io = FileSystemIO()
        if not io.exists(checkpoint_file):
            return target_item
        raw_bytes = io.read(checkpoint_file)
        state: CheckpointStateDict = json.loads(raw_bytes.decode("utf-8"))
        self._last_state = state
        if target_item is not None:
            validate_restored_state(state, target_item)
            target_item.set_state(state)
            return target_item
        validate_restored_state(state, None)
        state_pytree: dict[str, PyTree] = json.loads(raw_bytes.decode("utf-8"))
        return state_pytree

    def structure(
        self,
        directory: PathLike | None = None,
    ) -> PyTree:
        """Return the structural checkpoint state dictionary stored in ``directory``.

        Args:
            directory (PathLike | None): Checkpoint directory or virtual URI.

        Returns:
            PyTree: Restored checkpoint state dictionary, or empty dict if not found.
        """
        if directory is None:
            return {}
        restored = self.restore(directory)
        if isinstance(restored, dict):
            return restored
        return {}

    def metadata(
        self,
        directory: PathLike | None = None,
    ) -> dict[str, JsonValue] | None:
        """Return checkpoint state metadata associated with ``directory``.

        Args:
            directory (PathLike | None): Checkpoint directory or virtual URI.

        Returns:
            dict[str, JsonValue] | None: Checkpoint state dictionary if available, otherwise None.
        """
        if directory is not None:
            dir_str = os.fspath(directory) if isinstance(directory, os.PathLike) else str(directory)
            filename = self._resolve_checkpoint_filename()
            checkpoint_file = f"{dir_str.rstrip('/')}/{filename}"
            io = FileSystemIO()
            if io.exists(checkpoint_file):
                parsed: dict[str, JsonValue] = json.loads(io.read(checkpoint_file).decode("utf-8"))
                return parsed
            return super().metadata(dir_str)
        if self._last_state is not None:
            return _state_to_json_dict(self._last_state)
        return super().metadata(None)

    def close(self) -> None:
        """Release any resources held by the handler."""
        super().close()


PyGrainCheckpointHandler = CheckpointHandler


@orbax_args.register_with_handler(PyGrainCheckpointHandler, for_save=True)
class PyGrainCheckpointSave(orbax_args.CheckpointArgs):
    """``zero_orbax`` ``CheckpointArgs`` wrapper for saving a Grain iterator or state dictionary."""

    def __init__(
        self,
        item: GrainIteratorProtocol | CheckpointStateDict | PyTree = None,
    ) -> None:
        """Initialize ``PyGrainCheckpointSave``.

        Args:
            item (GrainIteratorProtocol | CheckpointStateDict | PyTree): Grain iterator or state dict to checkpoint.
        """
        coerced = _coerce_state_dict(item) if isinstance(item, Mapping) else None
        resolved_item: PyTree = (
            _state_to_json_dict(coerced) if coerced is not None else (item if not isinstance(item, Mapping) else None)
        )
        super().__init__(item=resolved_item)


@orbax_args.register_with_handler(PyGrainCheckpointHandler, for_restore=True)
class PyGrainCheckpointRestore(orbax_args.CheckpointArgs):
    """``zero_orbax`` ``CheckpointArgs`` wrapper for restoring a Grain iterator."""

    def __init__(
        self,
        item: GrainIteratorProtocol | PyTree = None,
    ) -> None:
        """Initialize ``PyGrainCheckpointRestore``.

        Args:
            item (GrainIteratorProtocol | PyTree): Grain iterator to restore in-place.
        """
        super().__init__(item=item)


def register_with_orbax(
    registry: CheckpointHandlerRegistry | None = None,
    item_name: str | None = None,
) -> CheckpointHandlerRegistry:
    """Register ``PyGrainCheckpointHandler``, ``PyGrainCheckpointSave``, and ``PyGrainCheckpointRestore`` in ``registry``.

    Args:
        registry (CheckpointHandlerRegistry | None): Target ``CheckpointHandlerRegistry`` (or new default registry).
        item_name (str | None): Optional named checkpoint item (e.g. ``"iterator"`` or ``"dataset"``).

    Returns:
        CheckpointHandlerRegistry: Registry populated with ``PyGrainCheckpointHandler`` bindings.
    """
    target_registry: CheckpointHandlerRegistry = (
        registry if registry is not None else DefaultCheckpointHandlerRegistry()
    )
    target_registry.add(item_name, PyGrainCheckpointSave, PyGrainCheckpointHandler, override=True)
    target_registry.add(item_name, PyGrainCheckpointRestore, PyGrainCheckpointHandler, override=True)
    if item_name is not None:
        target_registry.add(item_name, None, PyGrainCheckpointHandler, override=True)
    return target_registry


_DEFAULT_GRAIN_HANDLER_REGISTRY: CheckpointHandlerRegistry = register_with_orbax()


__all__ = [
    "CheckpointHandler",
    "PyGrainCheckpointHandler",
    "PyGrainCheckpointRestore",
    "PyGrainCheckpointSave",
    "register_with_orbax",
    "validate_restored_state",
]
