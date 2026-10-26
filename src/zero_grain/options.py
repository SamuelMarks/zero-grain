"""Execution, reading, and multiprocessing option classes for zero-grain."""

from __future__ import annotations

import dataclasses
import logging
import sys
from zero_grain._types import AbslLoggingModuleProtocol


def _emit_warning(msg: str) -> None:
    """Emit a warning message to both ``logging`` and ``absl.logging`` if loaded.

    Args:
        msg (str): Warning message text.
    """
    logging.warning(msg)
    absl_logging = sys.modules.get("absl.logging")
    if isinstance(absl_logging, AbslLoggingModuleProtocol):
        absl_logging.warning(msg)


@dataclasses.dataclass
class ReadOptions:
    """Options controlling multithreaded reading and prefetching from a data source.

    Attributes:
        num_threads (int): Number of reader threads per worker process.
        prefetch_buffer_size (int): Size of the per-worker read prefetch buffer.
    """

    num_threads: int = 16
    prefetch_buffer_size: int = 500

    def __post_init__(self) -> None:
        """Validate ``num_threads`` and ``prefetch_buffer_size`` after initialization.

        Raises:
            ValueError: If ``num_threads < 0`` or ``prefetch_buffer_size < 0``.
        """
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")
        if self.prefetch_buffer_size < 0:
            raise ValueError("prefetch_buffer_size must be non-negative")
        if 0 < self.prefetch_buffer_size < self.num_threads:
            msg = f"prefetch_buffer_size={self.prefetch_buffer_size} is smaller than num_threads={self.num_threads}"
            _emit_warning(msg)


@dataclasses.dataclass
class MultiprocessingOptions:
    """Options configuring worker processes for parallel pipeline execution.

    Attributes:
        num_workers (int): Number of child worker processes.
        per_worker_buffer_size (int): Output buffer size per worker.
        enable_profiling (bool): Whether worker profiling is enabled.
    """

    num_workers: int = 0
    per_worker_buffer_size: int = 1
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        """Validate ``num_workers`` and ``per_worker_buffer_size`` after initialization.

        Raises:
            ValueError: If ``num_workers < 0`` or ``per_worker_buffer_size < 1``.
        """
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.per_worker_buffer_size < 1:
            raise ValueError("per_worker_buffer_size must be at least 1")


@dataclasses.dataclass
class DatasetOptions:
    """Execution options for ``MapDataset`` and ``IterDataset`` pipelines.

    Attributes:
        filter_warn_threshold_ratio (float | None): Ratio of filtered elements triggering a warning in ``[0.0, 1.0]``.
        filter_raise_threshold_ratio (float | None): Ratio of filtered elements triggering an error in ``[0.0, 1.0]``.
    """

    filter_warn_threshold_ratio: float | None = 0.9
    filter_raise_threshold_ratio: float | None = 1.0

    def __post_init__(self) -> None:
        """Validate filter warning and error threshold ratios.

        Raises:
            ValueError: If a specified ratio is outside ``[0.0, 1.0]``.
        """
        if self.filter_warn_threshold_ratio is not None and not (0.0 <= self.filter_warn_threshold_ratio <= 1.0):
            raise ValueError(
                f"filter_warn_threshold_ratio must be in [0.0, 1.0], got {self.filter_warn_threshold_ratio}"
            )
        if self.filter_raise_threshold_ratio is not None and not (0.0 <= self.filter_raise_threshold_ratio <= 1.0):
            raise ValueError(
                f"filter_raise_threshold_ratio must be in [0.0, 1.0], got {self.filter_raise_threshold_ratio}"
            )

    def check_filter_ratio(self, total_seen: int, total_filtered: int) -> None:
        """Validate observed filter drop ratio against configured warn and raise thresholds.

        Args:
            total_seen (int): Total number of candidate records evaluated.
            total_filtered (int): Number of candidate records dropped by the filter.

        Raises:
            ValueError: If the filtered ratio reaches or exceeds ``filter_raise_threshold_ratio``.
        """
        if total_seen <= 0:
            return
        ratio = total_filtered / total_seen
        if self.filter_raise_threshold_ratio is not None and ratio >= self.filter_raise_threshold_ratio:
            raise ValueError(
                f"Filtered ratio {ratio:.2f} ({total_filtered}/{total_seen}) reached or exceeded "
                f"filter_raise_threshold_ratio={self.filter_raise_threshold_ratio}"
            )
        if self.filter_warn_threshold_ratio is not None and ratio >= self.filter_warn_threshold_ratio:
            _emit_warning(
                f"Filtered ratio {ratio:.2f} ({total_filtered}/{total_seen}) reached or exceeded "
                f"filter_warn_threshold_ratio={self.filter_warn_threshold_ratio}"
            )


@dataclasses.dataclass
class DataLoaderOptions:
    """Configuration container for ``DataLoader`` initialization.

    Attributes:
        worker_count (int): Number of worker processes.
        read_options (ReadOptions | None): Reader threading options.
    """

    worker_count: int = 0
    read_options: ReadOptions | None = None


@dataclasses.dataclass
class LoadOptions:
    """Configuration options for the ``load`` convenience function.

    Attributes:
        num_epochs (int | None): Number of epochs to iterate.
        shuffle (bool): Whether to shuffle records.
        seed (int | None): Random seed for shuffling.
        batch_size (int | None): Optional batch size.
        drop_remainder (bool): Whether to drop incomplete final batches.
        worker_count (int | None): Number of worker processes.
        read_options (ReadOptions | None): Reader threading options.
    """

    num_epochs: int | None = None
    shuffle: bool = False
    seed: int | None = None
    batch_size: int | None = None
    drop_remainder: bool = False
    worker_count: int | None = 0
    read_options: ReadOptions | None = None


__all__ = [
    "DataLoaderOptions",
    "DatasetOptions",
    "LoadOptions",
    "MultiprocessingOptions",
    "ReadOptions",
]
