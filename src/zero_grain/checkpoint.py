"""Orbax-compatible checkpoint handler for ``DataLoaderIterator`` and ``DatasetIterator``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable
from zero_grain._types import CheckpointStateDict, ElementTree


@runtime_checkable
class _StatefulIteratorProtocol(Protocol):
    """Protocol for iterators supporting checkpoint state extraction and restoration."""

    def get_state(self) -> CheckpointStateDict:
        """Capture the current iterator checkpoint state.

        Returns:
            CheckpointStateDict: Dictionary suitable for ``set_state`` restoration.
        """

    def set_state(self, state: CheckpointStateDict) -> None:
        """Restore the iterator to a previously captured checkpoint ``state``.

        Args:
            state (CheckpointStateDict): Checkpoint dictionary from ``get_state()``.
        """


class CheckpointHandler:
    """CheckpointHandler saving and restoring iterator states as JSON."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``CheckpointHandler``.

        Args:
            *args (ElementTree): Optional positional initialization arguments.
            **kwargs (ElementTree): Optional keyword initialization arguments.
        """
        self._args = args
        self._kwargs = kwargs

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
        directory: str | os.PathLike[str] | None = None,
        item: _StatefulIteratorProtocol | None = None,
        process_index: int = 0,
        process_count: int = 1,
    ) -> None:
        """Save the state of ``item`` to ``directory``.

        Args:
            directory (str | os.PathLike[str] | None): Target checkpoint directory.
            item (_StatefulIteratorProtocol | None): Iterator whose state should be saved.
            process_index (int): Zero-based process index for multi-host checkpointing.
            process_count (int): Total number of processes for multi-host checkpointing.
        """
        if directory is None or item is None:
            return
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        state = item.get_state()
        checkpoint_file = dir_path / self._resolve_checkpoint_filename(
            process_index=process_index,
            process_count=process_count,
        )
        checkpoint_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def restore(
        self,
        directory: str | os.PathLike[str] | None = None,
        item: _StatefulIteratorProtocol | None = None,
        process_index: int = 0,
        process_count: int = 1,
    ) -> _StatefulIteratorProtocol | CheckpointStateDict | None:
        """Restore the state of ``item`` from ``directory``.

        Args:
            directory (str | os.PathLike[str] | None): Source checkpoint directory.
            item (_StatefulIteratorProtocol | None): Optional iterator to restore in-place.
            process_index (int): Zero-based process index for multi-host checkpointing.
            process_count (int): Total number of processes for multi-host checkpointing.

        Returns:
            _StatefulIteratorProtocol | CheckpointStateDict | None:
                Restored iterator if ``item`` was given, raw state dict if only ``directory`` was given, or None.
        """
        if directory is None:
            return item
        checkpoint_file = Path(directory) / self._resolve_checkpoint_filename(
            process_index=process_index,
            process_count=process_count,
        )
        if not checkpoint_file.is_file():
            return item
        state: CheckpointStateDict = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if item is not None:
            item.set_state(state)
            return item
        return state


PyGrainCheckpointHandler = CheckpointHandler


__all__ = [
    "CheckpointHandler",
    "PyGrainCheckpointHandler",
]
