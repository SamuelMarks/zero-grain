"""Sharding configuration classes and interval splitting utilities for zero-grain."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ShardOptions:
    """Configuration specifying data sharding across workers or hosts.

    Attributes:
        shard_index (int): Zero-based index of the current shard.
        shard_count (int): Total number of shards across all processes.
        drop_remainder (bool): Whether to drop trailing records not evenly divisible.
    """

    shard_index: int = 0
    shard_count: int = 1
    drop_remainder: bool = False

    def __post_init__(self) -> None:
        """Validate ``shard_index`` and ``shard_count`` after initialization.

        Raises:
            ValueError: If ``shard_index < 0``, ``shard_count <= 0``, or ``shard_index >= shard_count``.
        """
        if self.shard_count <= 0 or self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise ValueError(
                f"Invalid shard_index={self.shard_index} and shard_count={self.shard_count}. "
                f"Must satisfy 0 <= shard_index < shard_count."
            )

    def __repr__(self) -> str:
        """Return the canonical string representation of the shard options.

        Returns:
            str: Formatted ShardOptions string.
        """
        return (
            f"{self.__class__.__name__}(shard_index={self.shard_index}, "
            f"shard_count={self.shard_count}, drop_remainder={self.drop_remainder})"
        )


class NoSharding(ShardOptions):
    """Default ``ShardOptions`` subclass representing a single unsharded process."""

    def __init__(
        self,
        shard_index: int = 0,
        shard_count: int = 1,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize ``NoSharding`` with single-process defaults.

        Args:
            shard_index (int): Zero-based shard index (defaults to 0).
            shard_count (int): Total shard count (defaults to 1).
            drop_remainder (bool): Drop remainder flag (defaults to False).
        """
        super().__init__(
            shard_index=shard_index,
            shard_count=shard_count,
            drop_remainder=drop_remainder,
        )


class ShardByJaxProcess(ShardOptions):
    """``ShardOptions`` subclass that shards records across JAX processes."""

    def __init__(self, drop_remainder: bool = False) -> None:
        """Initialize ``ShardByJaxProcess`` for the local process.

        Args:
            drop_remainder (bool): Whether to drop trailing records not divisible by process count.
        """
        super().__init__(
            shard_index=0,
            shard_count=1,
            drop_remainder=drop_remainder,
        )


def even_split(num_records: int, options: ShardOptions) -> tuple[int, int]:
    """Compute the ``(start, end)`` record key interval for ``options.shard_index``.

    Args:
        num_records (int): Total number of records in a single epoch.
        options (ShardOptions): Sharding configuration.

    Returns:
        tuple[int, int]: Inclusive start key and exclusive end key for the shard.
    """
    if options.drop_remainder:
        records_per_shard = num_records // options.shard_count
        start = options.shard_index * records_per_shard
        end = start + records_per_shard
        return start, end

    quotient = num_records // options.shard_count
    remainder = num_records % options.shard_count
    start = options.shard_index * quotient + min(options.shard_index, remainder)
    end = start + quotient + (1 if options.shard_index < remainder else 0)
    return start, end


__all__ = [
    "NoSharding",
    "ShardByJaxProcess",
    "ShardOptions",
    "even_split",
]
