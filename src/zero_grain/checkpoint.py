"""Orbax-compatible checkpoint handler for ``DataLoaderIterator`` and ``DatasetIterator``."""

from __future__ import annotations

import json
import os
from zero_grain import _types, shared_memory_array
from zero_grain._types import CheckpointStateDict, ElementTree
from zero_orbax.checkpoint import (
    AsyncCheckpointHandler,
    _CheckpointArgsBase,
    args as orbax_args,
)
from zero_orbax.core import AtomicWrite, FileSystemIO
from zero_orbax.types import GrainIteratorProtocol, JsonValue, PathLike, PyTree

_ = (_types, shared_memory_array)


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


def _unwrap_iterator_arg(
    item: GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree,
    pos_args: tuple[GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree, ...],
    kwargs: dict[str, GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree],
) -> GrainIteratorProtocol | None:
    """Extract a ``GrainIteratorProtocol`` from positional or keyword ``CheckpointArgs``.

    Args:
        item (GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree): Primary item or args wrapper.
        pos_args (tuple[GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree, ...]): Extra positional args.
        kwargs (dict[str, GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree]): Keyword args.

    Returns:
        GrainIteratorProtocol | None: Unwrapped iterator instance or None.
    """
    candidate: GrainIteratorProtocol | _CheckpointArgsBase | ElementTree | PyTree = item
    if candidate is None:
        if "args" in kwargs:
            candidate = kwargs["args"]
        elif len(pos_args) >= 1:
            candidate = pos_args[0]
    if isinstance(candidate, _CheckpointArgsBase):
        inner = candidate.item
        return inner if isinstance(inner, GrainIteratorProtocol) else None
    return candidate if isinstance(candidate, GrainIteratorProtocol) else None


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
            item (PyTree | _CheckpointArgsBase | ElementTree): Iterator or ``CheckpointArgs`` to save.
            *pos_args (PyTree | _CheckpointArgsBase | ElementTree): Additional positional arguments.
            **kwargs (PyTree | _CheckpointArgsBase | ElementTree): Additional keyword arguments.
        """
        target_item = _unwrap_iterator_arg(item, pos_args, kwargs)
        if directory is None or target_item is None:
            return
        proc_idx, proc_cnt = _extract_process_shard(pos_args, kwargs)
        dir_str = os.fspath(directory) if isinstance(directory, os.PathLike) else str(directory)
        FileSystemIO().makedirs(dir_str, exist_ok=True)
        state = target_item.get_state()
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
        target_item = _unwrap_iterator_arg(item, pos_args, kwargs)
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
            target_item.set_state(state)
            return target_item
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
            last_dict: dict[str, JsonValue] = json.loads(json.dumps(self._last_state))
            return last_dict
        return super().metadata(None)

    def close(self) -> None:
        """Release any resources held by the handler."""
        super().close()


PyGrainCheckpointHandler = CheckpointHandler


@orbax_args.register_with_handler(PyGrainCheckpointHandler, for_save=True)
class PyGrainCheckpointSave(orbax_args.CheckpointArgs):
    """``zero_orbax`` ``CheckpointArgs`` wrapper for saving a Grain iterator."""

    def __init__(
        self,
        item: GrainIteratorProtocol | PyTree = None,
    ) -> None:
        """Initialize ``PyGrainCheckpointSave``.

        Args:
            item (GrainIteratorProtocol | PyTree): Grain iterator to checkpoint.
        """
        super().__init__(item=item)


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


__all__ = [
    "CheckpointHandler",
    "PyGrainCheckpointHandler",
    "PyGrainCheckpointRestore",
    "PyGrainCheckpointSave",
]
