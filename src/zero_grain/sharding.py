"""Sharding configuration classes and interval splitting utilities for zero-grain."""

from __future__ import annotations

import dataclasses
from zero_grain._tree import map_tree
from zero_grain._types import (
    _ZERO_JAX_NUMPY_ADAPTER,
    CompilerTensor,
    ElementTree,
    JaxArray,
    is_ndarray_instance,
)
import zero_jax
from zero_jax.sharding import Sharding as JaxSharding
from zero_orbax.core import GlobalArraySharding


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
        """Initialize ``ShardByJaxProcess`` from active ``zero_jax`` process and local device counts.

        Args:
            drop_remainder (bool): Whether to drop trailing records not divisible by process count.
        """
        from zero_chex.mocks import get_num_devices_available
        import zero_optax as _zero_optax  # noqa: F401
        import zero_chex

        proc_idx = zero_jax.process_index()
        proc_cnt = max(zero_jax.process_count(), get_num_devices_available("cpu"))
        local_devs = zero_jax.local_device_count()
        zero_chex.assert_devices_available(max(1, local_devs), "cpu", not_less_than=True)
        self.local_device_count: int = local_devs
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


def _place_leaf_on_sharding(
    leaf: ElementTree,
    sharding: JaxSharding | GlobalArraySharding | None,
    host_id: int,
    num_hosts: int,
) -> ElementTree:
    """Convert and place a single PyTree ``leaf`` onto ``sharding`` via ``GlobalArraySharding`` and ``zero_jax.device_put``.

    Args:
        leaf (ElementTree): Leaf element from a batched PyTree.
        sharding (JaxSharding | GlobalArraySharding | None): Target JAX or Orbax sharding.
        host_id (int): Active host index in ``[0, num_hosts)``.
        num_hosts (int): Total number of hosts.

    Returns:
        ElementTree: Sharded ``JaxArray`` leaf, or non-numeric leaf unchanged.
    """
    if isinstance(leaf, JaxArray):
        arr: JaxArray = leaf
    elif is_ndarray_instance(leaf) or isinstance(leaf, (int, float, bool)):
        arr = _ZERO_JAX_NUMPY_ADAPTER.asarray(leaf)
    else:
        return leaf

    orbax_sharding = sharding if isinstance(sharding, GlobalArraySharding) else GlobalArraySharding()
    if isinstance(sharding, GlobalArraySharding) or num_hosts > 1:
        local_slice = orbax_sharding.get_local_shard(arr._tensor, host_id=host_id, num_hosts=num_hosts)
        arr = JaxArray(local_slice) if isinstance(local_slice, CompilerTensor) else arr
    target_device = sharding if isinstance(sharding, JaxSharding) else None
    placed: JaxArray = zero_jax.device_put(arr, target_device)
    return placed


def put_batch_on_sharding(
    batch: ElementTree,
    sharding: JaxSharding | GlobalArraySharding | None = None,
    *,
    host_id: int | None = None,
    num_hosts: int | None = None,
) -> ElementTree:
    """Place all array leaves of ``batch`` onto ``sharding`` using ``zero_jax.device_put`` and ``GlobalArraySharding``.

    Args:
        batch (ElementTree): Batched PyTree structure to shard across devices/hosts.
        sharding (JaxSharding | GlobalArraySharding | None): Optional JAX ``Sharding`` or ``GlobalArraySharding``.
        host_id (int | None): Optional host index override (defaults to ``zero_jax.process_index()``).
        num_hosts (int | None): Optional host count override (defaults to ``zero_jax.process_count()``).

    Returns:
        ElementTree: Batched PyTree with leaves placed on ``sharding``.
    """
    resolved_host_id = zero_jax.process_index() if host_id is None else host_id
    resolved_num_hosts = max(1, zero_jax.process_count()) if num_hosts is None else num_hosts
    return map_tree(
        lambda leaf: _place_leaf_on_sharding(
            leaf,
            sharding=sharding,
            host_id=resolved_host_id,
            num_hosts=resolved_num_hosts,
        ),
        batch,
    )


__all__ = [
    "NoSharding",
    "ShardByJaxProcess",
    "ShardOptions",
    "even_split",
    "put_batch_on_sharding",
]
