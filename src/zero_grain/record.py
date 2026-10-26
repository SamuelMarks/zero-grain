"""Record and RecordMetadata data structures for zero-grain."""

from __future__ import annotations

import dataclasses
import re
from typing import Generic
from zero_grain._types import RNGGeneratorProtocol, T

_MEM_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


@dataclasses.dataclass
class RecordMetadata:
    """Metadata attached to a single dataset ``Record``.

    Attributes:
        index (int | None): Global dataset step index.
        record_key (int | None): Backing data source index key.
        rng (RNGGeneratorProtocol | None): Per-record deterministic random number generator.
    """

    index: int | None = None
    record_key: int | None = None
    rng: RNGGeneratorProtocol | None = None

    def remove_record_key(self) -> RecordMetadata:
        """Return a new ``RecordMetadata`` instance with ``record_key`` set to ``None``.

        Returns:
            RecordMetadata: Copy of this metadata without ``record_key``.
        """
        return RecordMetadata(index=self.index, record_key=None, rng=self.rng)

    def __str__(self) -> str:
        """Return the string representation with memory addresses stripped from ``rng``.

        Returns:
            str: Formatted metadata string.
        """
        rng_str = "None" if self.rng is None else _MEM_ADDR_RE.sub("", repr(self.rng))
        return f"RecordMetadata(index={self.index}, record_key={self.record_key}, rng={rng_str})"

    def __repr__(self) -> str:
        """Return the canonical string representation of this metadata.

        Returns:
            str: Formatted metadata string.
        """
        return self.__str__()


@dataclasses.dataclass
class Record(Generic[T]):
    """Container pairing ``RecordMetadata`` with a data element ``data``.

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
