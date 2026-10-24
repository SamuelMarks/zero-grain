"""Record and RecordMetadata data structures for zero-grain."""

from __future__ import annotations

import dataclasses
import re
from typing import Generic
from zero_grain._types import RNGGeneratorProtocol, T


@dataclasses.dataclass
class RecordMetadata:
    """Metadata tracking a record's position, source key, and deterministic RNG.

    Attributes:
        index (int | None): Global sequential index in the sampler stream.
        record_key (int | None): Key used to fetch the record from the data source.
        rng (RNGGeneratorProtocol | None): Per-record random number generator.
    """

    index: int | None = None
    record_key: int | None = None
    rng: RNGGeneratorProtocol | None = dataclasses.field(default=None, compare=False)

    def __str__(self) -> str:
        """Return a human-readable string representation with memory addresses stripped.

        Returns:
            str: Formatted RecordMetadata string.
        """
        rng_str = repr(self.rng) if self.rng is not None else "None"
        rng_str = re.sub(r" at 0x[0-9a-fA-F]+", "", rng_str)
        return f"RecordMetadata(index={self.index}, record_key={self.record_key}, rng={rng_str})"

    def __repr__(self) -> str:
        """Return the canonical string representation of the metadata.

        Returns:
            str: Canonical string representation.
        """
        return self.__str__()

    def remove_record_key(self) -> RecordMetadata:
        """Return a copy of this ``RecordMetadata`` with ``record_key`` set to ``None``.

        Returns:
            RecordMetadata: New metadata instance with ``record_key=None``.
        """
        return RecordMetadata(index=self.index, record_key=None, rng=self.rng)


@dataclasses.dataclass
class Record(Generic[T]):
    """Container pairing a ``RecordMetadata`` instance with a data payload.

    Attributes:
        metadata (RecordMetadata | None): Associated record metadata.
        data (T): Record payload element.
    """

    metadata: RecordMetadata | None
    data: T


__all__ = [
    "Record",
    "RecordMetadata",
]
