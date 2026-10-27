"""Sampler classes generating deterministic ``RecordMetadata`` streams."""

from __future__ import annotations

from collections.abc import Iterator
from zero_flax.nnx.rnglib import RngStream as FlaxRngStream, Rngs as FlaxRngs
from zero_grain._rng import _resolve_seed_int, derive_rng, index_shuffle
from zero_grain._types import ArrayProtocol, ElementTree, SeedSpec
from zero_grain.record import RecordMetadata
from zero_grain.sharding import NoSharding, ShardOptions, even_split


class Sampler:
    """Base protocol and class for Grain record samplers."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``Sampler``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def __len__(self) -> int:
        """Return the number of records in the sampler.

        Returns:
            int: Total record count.
        """
        return 0

    def __getitem__(self, index: int) -> RecordMetadata:
        """Return the ``RecordMetadata`` for global ``index``.

        Args:
            index (int): Global step index.

        Returns:
            RecordMetadata: Metadata for ``index``.

        Raises:
            IndexError: Always raised on the base empty ``Sampler``.
        """
        raise IndexError(f"Sampler index {index} out of bounds")

    def __iter__(self) -> Iterator[RecordMetadata]:
        """Iterate over all ``RecordMetadata`` items in this sampler's shard.

        Returns:
            Iterator[RecordMetadata]: Empty iterator for the base ``Sampler``.
        """
        return iter(())


class SequentialSampler(Sampler):
    """Sampler emitting records in sequential order without shuffling."""

    def __init__(
        self,
        num_records: int = 1,
        shard_options: ShardOptions | None = None,
        num_epochs: int | None = None,
        seed: SeedSpec | None = None,
    ) -> None:
        """Initialize ``SequentialSampler``.

        Args:
            num_records (int): Total number of records in the dataset (must be ``> 0``).
            shard_options (ShardOptions | None): Sharding configuration across processes.
            num_epochs (int | None): Optional number of epochs to iterate (defaults to 1 if None).
            seed (SeedSpec | None): Optional seed, ``PRNGKey``, ``Rngs``, or ``RngStream`` for RNG attachment.

        Raises:
            ValueError: If ``num_records <= 0`` or ``num_epochs <= 0``.
        """
        super().__init__()
        if num_records <= 0:
            raise ValueError(f"num_records must be positive, got {num_records}")
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        self.num_records = num_records
        self.shard_options: ShardOptions = shard_options if shard_options is not None else NoSharding()
        self.num_epochs = num_epochs
        self.seed = _resolve_seed_int(seed)
        effective_epochs = num_epochs if num_epochs is not None else 1
        total_records = num_records * effective_epochs
        if self.shard_options.drop_remainder:
            self._max_index = (total_records // self.shard_options.shard_count) * self.shard_options.shard_count
        else:
            self._max_index = total_records

    def __len__(self) -> int:
        """Return the total number of accessible global indices.

        Returns:
            int: Total record count after applying ``drop_remainder``.
        """
        return self._max_index

    def __getitem__(self, index: int) -> RecordMetadata:
        """Return the ``RecordMetadata`` at global ``index``.

        Args:
            index (int): Non-negative global record index.

        Returns:
            RecordMetadata: Metadata containing ``index``, ``record_key``, and optional ``rng``.

        Raises:
            IndexError: If ``index < 0`` or ``index >= len(self)``.
        """
        if index < 0 or index >= self._max_index:
            raise IndexError(f"RecordMetadata index {index} is out of bounds for size {self._max_index}")
        rng = derive_rng(self.seed, index) if self.seed is not None else None
        return RecordMetadata(index=index, record_key=index % self.num_records, rng=rng)

    def __iter__(self) -> Iterator[RecordMetadata]:
        """Iterate over ``RecordMetadata`` entries belonging to the active shard.

        Yields:
            RecordMetadata: RecordMetadata for each index in ``shard_options``.
        """
        idx = self.shard_options.shard_index
        step = self.shard_options.shard_count
        while idx < self._max_index:
            yield self[idx]
            idx += step

    def __repr__(self) -> str:
        """Return the canonical string representation of ``SequentialSampler``.

        Returns:
            str: Formatted SequentialSampler string.
        """
        return f"SequentialSampler(num_records={self.num_records}, shard_options={self.shard_options!r})"


class IndexSampler(Sampler):
    """Sampler supporting multi-epoch iteration, sharding, and seeded shuffling."""

    def __init__(
        self,
        num_records: int = 1,
        shard_options: ShardOptions | None = None,
        shuffle: bool = False,
        num_epochs: int | None = None,
        seed: SeedSpec | ElementTree | None = None,
    ) -> None:
        """Initialize ``IndexSampler``.

        Args:
            num_records (int): Number of records in a single epoch (must be ``> 0``).
            shard_options (ShardOptions | None): Sharding configuration (defaults to ``NoSharding()``).
            shuffle (bool): Whether to shuffle record keys within each epoch.
            num_epochs (int | None): Optional number of epochs (None for 1 epoch).
            seed (SeedSpec | ElementTree | None): Integer seed, ``PRNGKey``, ``Rngs``, or ``RngStream``.

        Raises:
            ValueError: If ``num_records <= 0``, ``num_epochs <= 0``, or ``seed`` is outside ``[0, 2**32)``.
            TypeError: If ``seed`` is not an integer or supported PRNG key object.
        """
        super().__init__()
        if num_records <= 0:
            raise ValueError(f"num_records must be positive, got {num_records}")
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        validated_seed: int | None = None
        if seed is not None:
            if isinstance(seed, (ArrayProtocol, FlaxRngs, FlaxRngStream)):
                validated_seed = _resolve_seed_int(seed)
            elif not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError("seed must be an integer")
            else:
                if seed < 0 or seed >= 2**32:
                    raise ValueError("seed must be a 32-bit unsigned integer in [0, 2**32)")
                validated_seed = seed
        self.num_records = num_records
        self.shard_options: ShardOptions = shard_options if shard_options is not None else NoSharding()
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.seed: int | None = validated_seed

        self._shard_start, self._shard_end = even_split(num_records, self.shard_options)
        self._shard_len = max(0, self._shard_end - self._shard_start)

        effective_epochs = num_epochs if num_epochs is not None else 1
        max_index = effective_epochs * num_records
        if self.shard_options.drop_remainder:
            max_index = min(
                max_index,
                self._shard_len * self.shard_options.shard_count * effective_epochs,
            )
        self._max_index = max_index

    def __len__(self) -> int:
        """Return the total number of records across configured epochs.

        Returns:
            int: Total record count.
        """
        return self._max_index

    def __getitem__(self, index: int) -> RecordMetadata:
        """Return the ``RecordMetadata`` for global ``index``.

        Args:
            index (int): Non-negative global step index.

        Returns:
            RecordMetadata: Metadata containing ``index``, ``record_key``, and unique ``rng``.

        Raises:
            IndexError: If ``index < 0`` or ``index >= len(self)``.
        """
        if index < 0 or index >= self._max_index:
            raise IndexError(f"IndexSampler index {index} out of bounds for max_index {self._max_index}")

        shard_step = index // self.shard_options.shard_count
        if self._shard_len == 0:
            raise IndexError(f"Empty shard for index {index}")

        if self.shuffle:
            epoch, index_in_epoch = divmod(shard_step, self._shard_len)
            base_seed = self.seed if self.seed is not None else 0
            per_epoch_seed = (base_seed + epoch) % (2**32)
            shuffled_in_epoch = index_shuffle(
                index_in_epoch,
                max_index=self._shard_len - 1,
                seed=per_epoch_seed,
            )
            sharded_index = shuffled_in_epoch + epoch * self._shard_len
        else:
            sharded_index = shard_step

        shard_epoch, shard_pos = divmod(sharded_index, self._shard_len)
        parent_index = shard_epoch * self.num_records + shard_pos + self._shard_start
        record_key = parent_index % self.num_records
        record_rng = derive_rng(self.seed, index)
        return RecordMetadata(index=index, record_key=record_key, rng=record_rng)

    def __iter__(self) -> Iterator[RecordMetadata]:
        """Iterate over ``RecordMetadata`` entries for this sampler's shard.

        Yields:
            RecordMetadata: Sequential metadata entries for the active shard.
        """
        idx = self.shard_options.shard_index
        step = self.shard_options.shard_count
        while idx < self._max_index:
            yield self[idx]
            idx += step

    def __repr__(self) -> str:
        """Return the canonical string representation of ``IndexSampler``.

        Returns:
            str: Formatted IndexSampler string.
        """
        return (
            f"IndexSampler(num_records={self.num_records}, "
            f"shard_options={self.shard_options!r}, "
            f"shuffle={self.shuffle}, "
            f"num_epochs={self.num_epochs}, "
            f"seed={self.seed})"
        )


__all__ = [
    "IndexSampler",
    "Sampler",
    "SequentialSampler",
]
