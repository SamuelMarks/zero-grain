"""Record and RecordMetadata data structures for zero-grain."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import re
from typing import Generic
from zero_grain._types import JaxArray, RNGGeneratorProtocol, T
import zero_jax

_MEM_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_RECORD_PYTREES_REGISTERED: list[bool] = [False]


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

    def __post_init__(self) -> None:
        """Ensure ``RecordMetadata`` and ``Record`` are registered with ``zero_chex`` on first instantiation."""
        if not _RECORD_PYTREES_REGISTERED[0]:
            register_record_pytrees()

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

    def __post_init__(self) -> None:
        """Ensure ``RecordMetadata`` and ``Record`` are registered with ``zero_chex`` on first instantiation."""
        if not _RECORD_PYTREES_REGISTERED[0]:
            register_record_pytrees()


def _chex_flatten_record_metadata(
    meta: RecordMetadata | None,
) -> tuple[tuple[int | None, ...], tuple[str, ...]]:
    """Flatten ``RecordMetadata`` for ``zero_chex.register_custom_pytree_node``.

    Args:
        meta (RecordMetadata | None): ``RecordMetadata`` instance to flatten.

    Returns:
        tuple[tuple[int | None, ...], tuple[str, ...]]: Field values and field names.
    """
    if isinstance(meta, RecordMetadata):
        return (meta.index, meta.record_key), ("index", "record_key")
    return (), ()


def _chex_unflatten_record_metadata(
    field_names: Sequence[str],
    field_values: Sequence[int | None],
) -> RecordMetadata:
    """Reconstruct ``RecordMetadata`` for ``zero_chex.register_custom_pytree_node``.

    Args:
        field_names (Sequence[str]): Ordered field names.
        field_values (Sequence[int | None]): Ordered child values.

    Returns:
        RecordMetadata: Reconstructed ``RecordMetadata`` instance.
    """
    mapped = dict(zip(field_names, field_values))
    raw_idx = mapped.get("index")
    raw_key = mapped.get("record_key")
    return RecordMetadata(
        index=int(raw_idx) if isinstance(raw_idx, int) else None,
        record_key=int(raw_key) if isinstance(raw_key, int) else None,
    )


def _chex_flatten_record(
    rec: Record[T] | None,
) -> tuple[tuple[JaxArray | int | float | bool | str | None, ...], tuple[str, ...]]:
    """Flatten ``Record`` for ``zero_chex.register_custom_pytree_node``.

    Args:
        rec (Record[T] | None): ``Record`` instance to flatten.

    Returns:
        tuple[tuple[JaxArray | int | float | bool | str | None, ...], tuple[str, ...]]:
            Tuple containing ``(rec.data,)`` and ``("data",)``.
    """
    if isinstance(rec, Record) and isinstance(rec.data, (JaxArray, int, float, bool, str, type(None))):
        return (rec.data,), ("data",)
    return (), ()


def _chex_unflatten_record(
    field_names: Sequence[str],
    field_values: Sequence[JaxArray | int | float | bool | str | None],
) -> Record[JaxArray | int | float | bool | str | None]:
    """Reconstruct ``Record`` for ``zero_chex.register_custom_pytree_node``.

    Args:
        field_names (Sequence[str]): Ordered field names.
        field_values (Sequence[JaxArray | int | float | bool | str | None]): Reconstructed data payload.

    Returns:
        Record[JaxArray | int | float | bool | str | None]: Reconstructed ``Record`` instance.
    """
    del field_names
    first_val: JaxArray | int | float | bool | str | None = field_values[0] if field_values else None
    return Record(metadata=None, data=first_val)


def register_record_pytrees() -> None:
    """Register ``RecordMetadata`` and ``Record`` with ``zero_chex`` and ``zero_jax.tree_util``."""
    import zero_optax as _zero_optax  # noqa: F401
    from zero_chex._src.tree_util import register_custom_pytree_node

    _RECORD_PYTREES_REGISTERED[0] = True
    register_custom_pytree_node(
        RecordMetadata,
        lambda node: _chex_flatten_record_metadata(node if isinstance(node, RecordMetadata) else None),
        lambda names, vals: _chex_unflatten_record_metadata(
            names,
            [int(v) if isinstance(v, int) else None for v in vals],
        ),
    )
    register_custom_pytree_node(
        Record,
        lambda node: _chex_flatten_record(node if isinstance(node, Record) else None),
        lambda names, vals: _chex_unflatten_record(
            names,
            [v if isinstance(v, (int, float, bool, str, JaxArray)) else None for v in vals],
        ),
    )
    zero_jax.tree_util.register_dataclass(
        RecordMetadata,
        data_fields=("index", "record_key"),
        meta_fields=("rng",),
    )
    zero_jax.tree_util.register_dataclass(
        Record,
        data_fields=("data",),
        meta_fields=("metadata",),
    )


zero_jax.tree_util.register_dataclass(
    RecordMetadata,
    data_fields=("index", "record_key"),
    meta_fields=("rng",),
)
zero_jax.tree_util.register_dataclass(
    Record,
    data_fields=("data",),
    meta_fields=("metadata",),
)


__all__ = [
    "Record",
    "RecordMetadata",
    "_chex_flatten_record",
    "_chex_flatten_record_metadata",
    "_chex_unflatten_record",
    "_chex_unflatten_record_metadata",
    "register_record_pytrees",
]
