"""Pipeline operation classes transforming streams of ``Record`` objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Generic
from zero_grain._rng import create_rng, derive_rng
from zero_grain._tree import _batch_elements, map_tree
from zero_grain._types import (
    ElementTree,
    RNGGeneratorProtocol,
    T_in,
    T_out,
    is_ndarray_instance,
)
from zero_grain.record import Record, RecordMetadata
from zero_grain.shared_memory_array import SharedMemoryArrayMetadata
from zero_grain.transforms import (
    FilterTransform,
    FlatMapTransform,
    MapTransform,
    MapWithIndexTransform,
    RandomMapTransform,
)

_SHM_COUNTER = 0


class Operation(Generic[T_in, T_out]):
    """Base protocol and class for iterator-to-iterator record operations."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``Operation``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def apply_element(
        self,
        element: T_in,
        index: int = 0,
        seed: int | None = None,
        op_seed: int = 0,
    ) -> ElementTree:
        """Apply this operation to a single indexed ``element``.

        Args:
            element (T_in): Input element.
            index (int): Logical record index.
            seed (int | None): Base RNG seed.
            op_seed (int): Operation-level seed offset.

        Returns:
            ElementTree: Transformed element.
        """
        del index, seed, op_seed
        return element

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_out]]:
        """Transform an input stream of ``Record`` instances.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Returns:
            Iterator[Record[T_out]]: Transformed downstream record iterator.

        Raises:
            NotImplementedError: If the subclass does not implement ``__call__``.
        """
        raise NotImplementedError("Subclasses of Operation must implement __call__()")


class MapOperation(Operation[T_in, T_out]):
    """Operation applying a deterministic function or ``MapTransform`` to each record."""

    def __init__(
        self,
        map_function: Callable[[T_in], T_out] | MapTransform[T_in, T_out] | None = None,
    ) -> None:
        """Initialize ``MapOperation``.

        Args:
            map_function (Callable[[T_in], T_out] | MapTransform[T_in, T_out] | None):
                Callable or ``MapTransform`` applied to each record's data.
        """
        super().__init__()
        self.map_function = map_function

    def map(self, element: T_in) -> T_out:
        """Apply the configured map function to a single ``element``.

        Args:
            element (T_in): Input element.

        Returns:
            T_out: Transformed element.

        Raises:
            NotImplementedError: If no ``map_function`` was supplied.
        """
        if self.map_function is None:
            raise NotImplementedError("MapOperation requires map_function or subclass map()")
        if isinstance(self.map_function, MapTransform):
            return self.map_function.map(element)
        return self.map_function(element)

    def apply_element(
        self,
        element: T_in,
        index: int = 0,
        seed: int | None = None,
        op_seed: int = 0,
    ) -> ElementTree:
        """Apply ``map`` to a single indexed ``element``.

        Args:
            element (T_in): Input element.
            index (int): Logical record index (unused).
            seed (int | None): Base RNG seed (unused).
            op_seed (int): Operation-level seed offset (unused).

        Returns:
            ElementTree: Mapped element.
        """
        del index, seed, op_seed
        return self.map(element)

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_out]]:
        """Apply ``map`` to each non-None record in ``input_iterator``.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[T_out]: Mapped records with ``record_key`` cleared.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                meta = record.metadata.remove_record_key()
                data = self.map(record.data)
                yield Record(metadata=meta, data=data)


class MapWithIndexOperation(Operation[T_in, T_out]):
    """Operation applying an index-aware callable or ``MapWithIndexTransform`` to each record."""

    def __init__(
        self,
        map_function: (Callable[[int, T_in], T_out] | MapWithIndexTransform[T_in, T_out] | None) = None,
    ) -> None:
        """Initialize ``MapWithIndexOperation``.

        Args:
            map_function (Callable[[int, T_in], T_out] | MapWithIndexTransform[T_in, T_out] | None):
                Callable or ``MapWithIndexTransform`` accepting ``(index, element)``.
        """
        super().__init__()
        self.map_function = map_function

    def map_with_index(self, index: int, element: T_in) -> T_out:
        """Apply the index-aware transformation to ``(index, element)``.

        Args:
            index (int): Record index.
            element (T_in): Input element.

        Returns:
            T_out: Transformed element.

        Raises:
            NotImplementedError: If no ``map_function`` was supplied.
        """
        if self.map_function is None:
            raise NotImplementedError("MapWithIndexOperation requires map_function or subclass map_with_index()")
        if isinstance(self.map_function, MapWithIndexTransform):
            return self.map_function.map_with_index(index, element)
        return self.map_function(index, element)

    def apply_element(
        self,
        element: T_in,
        index: int = 0,
        seed: int | None = None,
        op_seed: int = 0,
    ) -> ElementTree:
        """Apply ``map_with_index`` to a single indexed ``element``.

        Args:
            element (T_in): Input element.
            index (int): Logical record index.
            seed (int | None): Base RNG seed (unused).
            op_seed (int): Operation-level seed offset (unused).

        Returns:
            ElementTree: Mapped element.
        """
        del seed, op_seed
        return self.map_with_index(index, element)

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_out]]:
        """Apply ``map_with_index`` to each non-None record in ``input_iterator``.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[T_out]: Mapped records with ``record_key`` cleared.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                idx = record.metadata.index if record.metadata.index is not None else 0
                meta = record.metadata.remove_record_key()
                data = self.map_with_index(idx, record.data)
                yield Record(metadata=meta, data=data)


class RandomMapOperation(Operation[T_in, T_out]):
    """Operation applying a stochastic callable or ``RandomMapTransform`` to each record."""

    def __init__(
        self,
        random_map_function: (
            Callable[[T_in, RNGGeneratorProtocol], T_out] | RandomMapTransform[T_in, T_out] | None
        ) = None,
    ) -> None:
        """Initialize ``RandomMapOperation``.

        Args:
            random_map_function (Callable[[T_in, RNGGeneratorProtocol], T_out] | RandomMapTransform[T_in, T_out] | None):
                Stochastic transformation function or ``RandomMapTransform`` instance.
        """
        super().__init__()
        self.random_map_function = random_map_function

    def random_map(self, element: T_in, rng: RNGGeneratorProtocol) -> T_out:
        """Apply the random transformation to ``element`` with ``rng``.

        Args:
            element (T_in): Input element.
            rng (RNGGeneratorProtocol): Per-record random number generator.

        Returns:
            T_out: Stochastically transformed element.

        Raises:
            NotImplementedError: If no ``random_map_function`` was supplied.
        """
        if self.random_map_function is None:
            raise NotImplementedError("RandomMapOperation requires random_map_function or subclass random_map()")
        if isinstance(self.random_map_function, RandomMapTransform):
            return self.random_map_function.random_map(element, rng)
        return self.random_map_function(element, rng)

    def apply_element(
        self,
        element: T_in,
        index: int = 0,
        seed: int | None = None,
        op_seed: int = 0,
    ) -> ElementTree:
        """Apply ``random_map`` to a single indexed ``element`` with derived RNG.

        Args:
            element (T_in): Input element.
            index (int): Logical record index.
            seed (int | None): Base RNG seed.
            op_seed (int): Operation-level seed offset.

        Returns:
            ElementTree: Stochastically transformed element.
        """
        rng = derive_rng(seed, index, op_seed=op_seed)
        return self.random_map(element, rng)

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_out]]:
        """Apply ``random_map`` to each non-None record in ``input_iterator``.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[T_out]: Stochastically transformed records with ``record_key`` cleared.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                rng = record.metadata.rng
                if rng is None:
                    rng = create_rng(record.metadata.index)
                meta = record.metadata.remove_record_key()
                data = self.random_map(record.data, rng)
                yield Record(metadata=meta, data=data)


class FilterOperation(Operation[T_in, T_in]):
    """Operation filtering records based on a predicate callable or ``FilterTransform``."""

    def __init__(
        self,
        condition_function: (Callable[[T_in], bool] | FilterTransform[T_in] | None) = None,
    ) -> None:
        """Initialize ``FilterOperation``.

        Args:
            condition_function (Callable[[T_in], bool] | FilterTransform[T_in] | None):
                Predicate function or ``FilterTransform`` returning True for kept records.
        """
        super().__init__()
        self.condition_function = condition_function

    def filter(self, element: T_in) -> bool:
        """Evaluate the filter condition on ``element``.

        Args:
            element (T_in): Candidate element.

        Returns:
            bool: True if ``element`` passes the filter.

        Raises:
            NotImplementedError: If no ``condition_function`` was supplied.
        """
        if self.condition_function is None:
            raise NotImplementedError("FilterOperation requires condition_function or subclass filter()")
        if isinstance(self.condition_function, FilterTransform):
            return bool(self.condition_function.filter(element))
        return bool(self.condition_function(element))

    def apply_element(
        self,
        element: T_in,
        index: int = 0,
        seed: int | None = None,
        op_seed: int = 0,
    ) -> ElementTree:
        """Evaluate ``filter`` on ``element``, returning None if filtered out.

        Args:
            element (T_in): Candidate element.
            index (int): Logical record index (unused).
            seed (int | None): Base RNG seed (unused).
            op_seed (int): Operation-level seed offset (unused).

        Returns:
            ElementTree: ``element`` if kept, else ``None``.
        """
        del index, seed, op_seed
        return element if self.filter(element) else None

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_in]]:
        """Yield only records from ``input_iterator`` for which ``filter`` returns True.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[T_in]: Filtered records with ``record_key`` cleared.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                if self.filter(record.data):
                    meta = record.metadata.remove_record_key()
                    yield Record(metadata=meta, data=record.data)


class FlatMapOperation(Operation[T_in, T_out]):
    """Operation expanding each input record into zero or more output records."""

    def __init__(
        self,
        flat_map_function: (Callable[[T_in], Iterable[T_out]] | FlatMapTransform[T_in, T_out] | None) = None,
    ) -> None:
        """Initialize ``FlatMapOperation``.

        Args:
            flat_map_function (Callable[[T_in], Iterable[T_out]] | FlatMapTransform[T_in, T_out] | None):
                Callable or ``FlatMapTransform`` splitting an element into an iterable.
        """
        super().__init__()
        self.flat_map_function = flat_map_function

    def flat_map(self, element: T_in) -> Iterable[T_out]:
        """Expand ``element`` into an iterable of output elements.

        Args:
            element (T_in): Input element.

        Returns:
            Iterable[T_out]: Resulting sequence or generator of sub-elements.

        Raises:
            NotImplementedError: If no ``flat_map_function`` was supplied.
        """
        if self.flat_map_function is None:
            raise NotImplementedError("FlatMapOperation requires flat_map_function or subclass flat_map()")
        if isinstance(self.flat_map_function, FlatMapTransform):
            return self.flat_map_function.flat_map(element)
        return self.flat_map_function(element)

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[T_out]]:
        """Expand each record in ``input_iterator`` into multiple output records.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[T_out]: Flattened output records with ``record_key`` cleared.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                for sub_elem in self.flat_map(record.data):
                    meta = record.metadata.remove_record_key()
                    yield Record(metadata=meta, data=sub_elem)


def _convert_leaf_to_shm(leaf: ElementTree) -> ElementTree:
    """Convert a C-contiguous non-object array leaf to ``SharedMemoryArrayMetadata``.

    Args:
        leaf (ElementTree): Candidate leaf value.

    Returns:
        ElementTree: ``SharedMemoryArrayMetadata`` if eligible, else the original ``leaf``.
    """
    global _SHM_COUNTER
    if is_ndarray_instance(leaf):
        if leaf.dtype.hasobject or not leaf.flags.c_contiguous:
            return leaf
        _SHM_COUNTER += 1
        return SharedMemoryArrayMetadata(
            name=f"shm_{_SHM_COUNTER}",
            dtype=leaf.dtype,
            shape=leaf.shape,
        )
    return leaf


class BatchOperation(Operation[T_in, ElementTree]):
    """Operation collating consecutive records into batches of size ``batch_size``."""

    def __init__(
        self,
        batch_size: int = 1,
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[T_in]], ElementTree] | None = None,
    ) -> None:
        """Initialize ``BatchOperation``.

        Args:
            batch_size (int): Number of records per batch (must be ``> 0``).
            drop_remainder (bool): Whether to drop incomplete final batches.
            batch_fn (Callable[[Sequence[T_in]], ElementTree] | None): Custom batching callable.

        Raises:
            ValueError: If ``batch_size <= 0``.
        """
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_fn = batch_fn
        self._use_shared_memory = False

    def _enable_shared_memory(self) -> None:
        """Enable copying batched arrays into shared memory."""
        self._use_shared_memory = True

    def _collate_tree(self, items: Sequence[T_in]) -> ElementTree:
        """Collate ``items`` into an ``ElementTree`` using ``batch_fn`` or ``_batch_elements``.

        Args:
            items (Sequence[T_in]): Sequence of record payloads.

        Returns:
            ElementTree: Batched PyTree structure.
        """
        if self.batch_fn is not None:
            collated: ElementTree = self.batch_fn(items)
        else:
            collated = _batch_elements(items)
        if self._use_shared_memory:
            return map_tree(_convert_leaf_to_shm, collated)
        return collated

    def __call__(
        self,
        input_iterator: Iterable[Record[T_in] | None],
    ) -> Iterator[Record[ElementTree]]:
        """Group consecutive records from ``input_iterator`` into batches.

        Args:
            input_iterator (Iterable[Record[T_in] | None]): Upstream record iterator.

        Yields:
            Record[ElementTree]: Batched records carrying the last element's metadata.
        """
        batch_records: list[Record[T_in]] = []
        for rec in input_iterator:
            if rec is not None:
                batch_records.append(rec)
            if len(batch_records) == self.batch_size:
                last_meta = batch_records[-1].metadata
                meta = last_meta.remove_record_key() if last_meta is not None else RecordMetadata()
                payloads: list[T_in] = [r.data for r in batch_records]
                yield Record(metadata=meta, data=self._collate_tree(payloads))
                batch_records = []
        if batch_records and not self.drop_remainder:
            last_meta = batch_records[-1].metadata
            meta = last_meta.remove_record_key() if last_meta is not None else RecordMetadata()
            payloads = [r.data for r in batch_records]
            yield Record(metadata=meta, data=self._collate_tree(payloads))


class CopyNumPyArrayToSharedMemoryOperation(Operation[ElementTree, ElementTree]):
    """Operation converting eligible NumPy arrays into ``SharedMemoryArrayMetadata``."""

    def map(self, element: ElementTree) -> ElementTree:
        """Recursively convert eligible arrays in ``element`` to shared memory metadata.

        Args:
            element (ElementTree): Input PyTree or leaf element.

        Returns:
            ElementTree: Transformed PyTree with C-contiguous non-object arrays converted.
        """
        return map_tree(_convert_leaf_to_shm, element)

    def __call__(
        self,
        input_iterator: Iterable[Record[ElementTree] | None],
    ) -> Iterator[Record[ElementTree]]:
        """Apply shared-memory array conversion to each record in ``input_iterator``.

        Args:
            input_iterator (Iterable[Record[ElementTree] | None]): Upstream record iterator.

        Yields:
            Record[ElementTree]: Records with converted shared memory metadata.
        """
        for record in input_iterator:
            if record is not None and record.metadata is not None:
                meta = record.metadata.remove_record_key()
                mapped_data: ElementTree = self.map(record.data)
                yield Record[ElementTree](metadata=meta, data=mapped_data)


CopyNumPyArrayToSharedMemory = CopyNumPyArrayToSharedMemoryOperation


__all__ = [
    "BatchOperation",
    "CopyNumPyArrayToSharedMemory",
    "CopyNumPyArrayToSharedMemoryOperation",
    "FilterOperation",
    "FlatMapOperation",
    "MapOperation",
    "MapWithIndexOperation",
    "Operation",
    "RandomMapOperation",
]
