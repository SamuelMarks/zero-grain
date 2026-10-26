"""Sharding configuration classes and interval splitting utilities for zero-grain."""

from __future__ import annotations

import dataclasses
import zero_jax


@dataclasses.dataclass
class ShardOptions:
    """Configuration specifying how a dataset is partitioned across shards.

    Attributes:
        shard_index (int): Zero-based index of the current shard in ``[0, shard_count)``.
        shard_count (int): Total number of shards (must be ``>= 1``).
        drop_remainder (bool): Whether to drop trailing records that do not divide evenly.
    """

    shard_index: int = 0
    shard_count: int = 1
    drop_remainder: bool = False

    def __post_init__(self) -> None:
        """Validate ``shard_index`` and ``shard_count`` bounds.

        Raises:
            ValueError: If ``shard_index < 0``, ``shard_count <= 0``, or ``shard_index >= shard_count``.
        """
        if self.shard_index < 0 or self.shard_count <= 0 or self.shard_index >= self.shard_count:
            raise ValueError(
                f"Invalid shard_index={self.shard_index} or shard_count={self.shard_count}. "
                "Require 0 <= shard_index < shard_count."
            )

    def _repr_name(self) -> str:
        """Return the class display name for ``__repr__``.

        Returns:
            str: Class name string.
        """
        return "ShardOptions"

    def __repr__(self) -> str:
        """Return the canonical string representation of this sharding option.

        Returns:
            str: Formatted representation matching Grain output.
        """
        return (
            f"{self._repr_name()}("
            f"shard_index={self.shard_index}, "
            f"shard_count={self.shard_count}, "
            f"drop_remainder={self.drop_remainder})"
        )


class NoSharding(ShardOptions):
    """Sharding configuration representing a single unpartitioned shard."""

    def __init__(
        self,
        shard_index: int = 0,
        shard_count: int = 1,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize ``NoSharding`` options.

        Args:
            shard_index (int): Shard index (defaults to 0).
            shard_count (int): Total shard count (defaults to 1).
            drop_remainder (bool): Whether to drop uneven remainder records.
        """
        super().__init__(
            shard_index=shard_index,
            shard_count=shard_count,
            drop_remainder=drop_remainder,
        )

    def _repr_name(self) -> str:
        """Return the class display name for ``__repr__``.

        Returns:
            str: Class name string.
        """
        return "NoSharding"


class ShardByJaxProcess(ShardOptions):
    """Sharding configuration partitioning records across active ``zero_jax`` / ``zero_chex`` processes."""

    def __init__(self, drop_remainder: bool = False) -> None:
        """Initialize ``ShardByJaxProcess`` from active ``zero_jax`` process or ``zero_chex`` device counts.

        Args:
            drop_remainder (bool): Whether to drop trailing records not divisible by process count.
        """
        from zero_chex.mocks import get_num_devices_available

        proc_idx = zero_jax.process_index()
        proc_cnt = max(zero_jax.process_count(), get_num_devices_available("cpu"))
        super().__init__(
            shard_index=proc_idx,
            shard_count=proc_cnt,
            drop_remainder=drop_remainder,
        )

    def _repr_name(self) -> str:
        """Return the class display name for ``__repr__``.

        Returns:
            str: Class name string.
        """
        return "ShardByJaxProcess"


def even_split(num_examples: int, options: ShardOptions) -> tuple[int, int]:
    """Compute the ``(start, end)`` half-open interval for ``options.shard_index``.

    Args:
        num_examples (int): Total number of examples in the unsharded dataset.
        options (ShardOptions): Sharding configuration.

    Returns:
        tuple[int, int]: ``(start_index, end_index)`` interval for the current shard.
    """
    quotient, remainder = divmod(num_examples, options.shard_count)
    if options.drop_remainder:
        start = options.shard_index * quotient
        end = start + quotient
        return start, end
    start = options.shard_index * quotient + min(options.shard_index, remainder)
    end = start + quotient + (1 if options.shard_index < remainder else 0)
    return start, end


__all__ = [
    "NoSharding",
    "ShardByJaxProcess",
    "ShardOptions",
    "even_split",
]
