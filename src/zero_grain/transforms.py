"""Transformation base classes for zero-grain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Generic
from zero_grain._types import ElementTree, RNGGeneratorProtocol, T_in, T_out


class MapTransform(Generic[T_in, T_out]):
    """Base class for deterministic element-wise map transformations."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``MapTransform``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def map(self, element: T_in) -> T_out:
        """Transform a single input ``element``.

        Args:
            element (T_in): Input element to transform.

        Returns:
            T_out: Transformed output element.

        Raises:
            NotImplementedError: If the subclass does not implement ``map``.
        """
        raise NotImplementedError("Subclasses of MapTransform must implement map()")


class MapWithIndex(Generic[T_in, T_out]):
    """Base class for deterministic element-wise transformations receiving record indices."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``MapWithIndex``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def map_with_index(self, index: int, element: T_in) -> T_out:
        """Transform ``element`` along with its sequential ``index``.

        Args:
            index (int): Zero-based record index.
            element (T_in): Input element.

        Returns:
            T_out: Transformed element.

        Raises:
            NotImplementedError: If the subclass does not implement ``map_with_index``.
        """
        raise NotImplementedError("Subclasses of MapWithIndexTransform must implement map_with_index()")


MapWithIndexTransform = MapWithIndex


class RandomMapTransform(Generic[T_in, T_out]):
    """Base class for stochastic element-wise transformations using a per-record RNG."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``RandomMapTransform``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def random_map(self, element: T_in, rng: RNGGeneratorProtocol) -> T_out:
        """Transform ``element`` using the provided deterministic ``rng``.

        Args:
            element (T_in): Input element.
            rng (RNGGeneratorProtocol): Per-record random number generator.

        Returns:
            T_out: Stochastically transformed element.

        Raises:
            NotImplementedError: If the subclass does not implement ``random_map``.
        """
        raise NotImplementedError("Subclasses of RandomMapTransform must implement random_map()")


class TfRandomMapTransform(RandomMapTransform[T_in, T_out]):
    """Alias subclass for TensorFlow-style random map transformations."""


class Filter(Generic[T_in]):
    """Base class for predicate filter transformations."""

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``Filter``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def filter(self, element: T_in) -> bool:
        """Return True if ``element`` should be kept in the dataset stream.

        Args:
            element (T_in): Candidate input element.

        Returns:
            bool: True to retain ``element``, False to filter it out.

        Raises:
            NotImplementedError: If the subclass does not implement ``filter``.
        """
        raise NotImplementedError("Subclasses of FilterTransform must implement filter()")


FilterTransform = Filter


class FlatMapTransform(Generic[T_in, T_out]):
    """Base class for transformations splitting one element into zero or more elements.

    Attributes:
        max_fan_out (int): Maximum number of output elements produced per input element.
    """

    max_fan_out: int = 100

    def __init__(self, *args: ElementTree, **kwargs: ElementTree) -> None:
        """Initialize ``FlatMapTransform``.

        Args:
            *args (ElementTree): Positional arguments.
            **kwargs (ElementTree): Keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs

    def flat_map(self, element: T_in) -> Iterable[T_out]:
        """Split ``element`` into an iterable of output elements.

        Args:
            element (T_in): Input element to expand.

        Returns:
            Iterable[T_out]: Sequence or generator of output elements.

        Raises:
            NotImplementedError: If the subclass does not implement ``flat_map``.
        """
        raise NotImplementedError("Subclasses of FlatMapTransform must implement flat_map()")


class Batch(Generic[T_in, T_out]):
    """Transformation specification that batches consecutive elements together.

    Attributes:
        batch_size (int): Number of elements per batch.
        drop_remainder (bool): Whether to drop a final batch smaller than ``batch_size``.
        batch_fn (Callable[[Sequence[T_in]], T_out] | None): Optional custom collation function.
    """

    def __init__(
        self,
        batch_size: int = 1,
        drop_remainder: bool = False,
        batch_fn: Callable[[Sequence[T_in]], T_out] | None = None,
    ) -> None:
        """Initialize ``Batch``.

        Args:
            batch_size (int): Number of elements to combine per batch (must be ``> 0``).
            drop_remainder (bool): Whether to drop trailing incomplete batches.
            batch_fn (Callable[[Sequence[T_in]], T_out] | None): Custom batching callable.

        Raises:
            ValueError: If ``batch_size <= 0``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_fn = batch_fn


BatchTransform = Batch
Map = MapTransform
RandomMap = RandomMapTransform
TfRandomMap = TfRandomMapTransform
FlatMap = FlatMapTransform


__all__ = [
    "Batch",
    "BatchTransform",
    "Filter",
    "FilterTransform",
    "FlatMap",
    "FlatMapTransform",
    "Map",
    "MapTransform",
    "MapWithIndex",
    "MapWithIndexTransform",
    "RandomMap",
    "RandomMapTransform",
    "TfRandomMap",
    "TfRandomMapTransform",
]
