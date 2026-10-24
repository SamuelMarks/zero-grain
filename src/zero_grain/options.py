"""Execution, reading, and multiprocessing option classes for zero-grain."""

from __future__ import annotations

import dataclasses
import logging
import sys
from zero_grain._types import AbslLoggingModuleProtocol, ElementTree


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
            logging.warning(msg)
            absl_logging = sys.modules.get("absl.logging")
            if isinstance(absl_logging, AbslLoggingModuleProtocol):
                absl_logging.warning(msg)


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


class DatasetOptions:
    """Execution options for ``MapDataset`` and ``IterDataset`` pipelines.

    Attributes:
        filter_warn_threshold_ratio (float): Ratio of filtered elements triggering a warning.
        filter_raise_threshold_ratio (float): Ratio of filtered elements triggering an error.
    """

    def __init__(
        self,
        filter_warn_threshold_ratio: float = 0.9,
        filter_raise_threshold_ratio: float = 1.0,
    ) -> None:
        """Initialize ``DatasetOptions``.

        Args:
            filter_warn_threshold_ratio (float): Warning threshold ratio for filtered records.
            filter_raise_threshold_ratio (float): Exception threshold ratio for filtered records.
        """
        self.filter_warn_threshold_ratio = filter_warn_threshold_ratio
        self.filter_raise_threshold_ratio = filter_raise_threshold_ratio


class DataLoaderOptions:
    """Legacy configuration container for ``DataLoader`` initialization.

    Attributes:
        worker_count (int): Number of worker processes.
        read_options (ReadOptions): Reader threading options.
    """

    def __init__(
        self,
        worker_count: int = 0,
        read_options: ReadOptions | None = None,
    ) -> None:
        """Initialize ``DataLoaderOptions``.

        Args:
            worker_count (int): Number of worker processes.
            read_options (ReadOptions | None): Optional ``ReadOptions`` instance.
        """
        self.worker_count = worker_count
        self.read_options = read_options if read_options is not None else ReadOptions()


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

    def __init__(
        self,
        num_epochs: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        batch_size: int | None = None,
        drop_remainder: bool = False,
        worker_count: int | None = 0,
        read_options: ReadOptions | None = None,
    ) -> None:
        """Initialize ``LoadOptions``.

        Args:
            num_epochs (int | None): Number of epochs to repeat.
            shuffle (bool): Whether to shuffle records each epoch.
            seed (int | None): Optional integer seed.
            batch_size (int | None): Optional batch size.
            drop_remainder (bool): Whether to drop partial trailing batches.
            worker_count (int | None): Number of workers.
            read_options (ReadOptions | None): Optional ``ReadOptions``.
        """
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.worker_count = worker_count
        self.read_options = read_options


__all__ = [
    "DataLoaderOptions",
    "DatasetOptions",
    "LoadOptions",
    "MultiprocessingOptions",
    "ReadOptions",
    "ElementTree",
]
