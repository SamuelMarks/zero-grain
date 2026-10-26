"""Deterministic random number generation and Simon cipher index shuffling utilities for zero-grain."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
import random
from zero_flax.nnx.rnglib import Rngs as FlaxRngs
from zero_grain._types import (
    ArrayProtocol,
    RNGGeneratorProtocol,
    array_to_list,
    get_numpy_module,
)

_MIN_BLOCK_SIZE = 16


@functools.lru_cache(maxsize=256)
def _seed_seq_generate(seed: int, rounds: int) -> tuple[int, ...]:
    """Generate ``rounds`` 32-bit unsigned round keys matching C++ ``std::seed_seq{seed}.generate``.

    Args:
        seed (int): 32-bit unsigned integer seed.
        rounds (int): Number of 32-bit keys to generate (must be positive).

    Returns:
        tuple[int, ...]: Generated 32-bit round keys.
    """
    mask = 0xFFFFFFFF
    v: list[int] = [0x8B8B8B8B] * rounds
    s = 1
    p = (rounds - 1) // 2
    q = p + 1
    m = max(s + 1, rounds)
    for k in range(m):
        x1 = v[k % rounds] ^ v[(k + p) % rounds] ^ v[(k - 1) % rounds]
        t1 = (x1 ^ (x1 >> 27)) & mask
        r1 = (1664525 * t1) & mask
        v[(k + p) % rounds] = (v[(k + p) % rounds] + r1) & mask
        if k == 0:
            r1 = (r1 + s) & mask
        elif k <= s:
            r1 = (r1 + (k % rounds) + (seed & mask)) & mask
        else:
            r1 = (r1 + (k % rounds)) & mask
        v[(k + q) % rounds] = (v[(k + q) % rounds] + r1) & mask
        v[k % rounds] = r1
    for k in range(m, m + rounds):
        x2 = (v[k % rounds] + v[(k + p) % rounds] + v[(k - 1) % rounds]) & mask
        t2 = (x2 ^ (x2 >> 27)) & mask
        r2 = (1566083941 * t2) & mask
        v[(k + p) % rounds] = (v[(k + p) % rounds] ^ r2) & mask
        r2 = (r2 - (k % rounds)) & mask
        v[(k + q) % rounds] = (v[(k + q) % rounds] ^ r2) & mask
        v[k % rounds] = r2
    return tuple(v)


def _simon_encrypt(value: int, round_keys: Sequence[int], word_size: int) -> int:
    """Encrypt ``value`` with a ``2 * word_size``-bit Simon block cipher.

    Args:
        value (int): Block value to encrypt.
        round_keys (Sequence[int]): Sequence of 32-bit round keys (even length).
        word_size (int): Half-block word size in bits (``>= 8``).

    Returns:
        int: Encrypted block value.
    """
    mask = (1 << word_size) - 1
    left = (value >> word_size) & mask
    right = value & mask
    s1 = 1 % word_size
    s2 = 2 % word_size
    s8 = 8 % word_size
    r1 = word_size - s1
    r2 = word_size - s2
    r8 = word_size - s8
    for i in range(0, len(round_keys), 2):
        f_right = (
            (((right << s1) | (right >> r1)) & ((right << s8) | (right >> r8))) ^ ((right << s2) | (right >> r2))
        ) & mask
        left = (left ^ f_right ^ (round_keys[i] & mask)) & mask
        f_left = (
            (((left << s1) | (left >> r1)) & ((left << s8) | (left >> r8))) ^ ((left << s2) | (left >> r2))
        ) & mask
        right = (right ^ f_left ^ (round_keys[i + 1] & mask)) & mask
    return (left << word_size) | right


@functools.lru_cache(maxsize=4096)
def index_shuffle(index: int, max_index: int, seed: int, rounds: int = 4) -> int:
    """Compute the shuffled position of ``index`` in ``[0, max_index]`` in $O(1)$ time.

    Implements Grain's cycle-walking Simon Feistel cipher over ``std::seed_seq``.

    Args:
        index (int): Input index in ``[0, max_index]``.
        max_index (int): Maximum inclusive index.
        seed (int): 32-bit unsigned integer seed.
        rounds (int): Even number of Feistel rounds in ``[4, 1024]``.

    Returns:
        int: Deterministically permuted index in ``[0, max_index]``.

    Raises:
        ValueError: If ``index < 0``, ``index > max_index``, or ``rounds`` is invalid.
    """
    if rounds < 4 or rounds > 1024 or rounds % 2 != 0:
        raise ValueError(f"rounds must be an even integer in [4, 1024], got {rounds}")
    if index < 0 or index > max_index:
        raise ValueError(f"index must be in [0, {max_index}], got {index}")
    if max_index == 0:
        return 0
    block_size = int(math.ceil(math.log2(max_index))) if max_index > 1 else 0
    block_size = max(block_size + (block_size % 2), _MIN_BLOCK_SIZE)
    word_size = block_size // 2
    round_keys = _seed_seq_generate(seed, rounds)
    new_index = index
    while True:
        new_index = _simon_encrypt(new_index, round_keys, word_size)
        if new_index <= max_index:
            return new_index


@dataclasses.dataclass
class DeterministicGenerator(RNGGeneratorProtocol):
    """Pure-Python deterministic random number generator compatible with Grain RNGs."""

    _seed: int | None = None
    _algorithm_name: str = "PCG64"

    def __init__(self, seed: int | None = None, algorithm_name: str = "PCG64") -> None:
        """Initialize the deterministic generator.

        Args:
            seed (int | None): Optional integer seed for reproducibility.
            algorithm_name (str): Bit-generator display name for string formatting.
        """
        self._seed = seed
        self._algorithm_name = algorithm_name
        self._rng = random.Random(seed)

    def random(self) -> float:
        """Return a random floating-point value in ``[0.0, 1.0)``.

        Returns:
            float: Uniformly distributed float in ``[0.0, 1.0)``.
        """
        return self._rng.random()

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return a random floating-point value in ``[low, high)``.

        Args:
            low (float): Lower bound of the interval.
            high (float): Upper bound of the interval.

        Returns:
            float: Uniformly sampled float in ``[low, high)``.
        """
        return self._rng.uniform(low, high)

    def integers(self, low: int, high: int | None = None) -> int:
        """Return a random integer from ``[0, low)`` or ``[low, high)``.

        Args:
            low (int): Inclusive lower bound (or exclusive upper bound if ``high`` is None).
            high (int | None): Exclusive upper bound when provided.

        Returns:
            int: Sampled integer.
        """
        if high is None:
            return self._rng.randrange(0, low)
        return self._rng.randrange(low, high)

    def permutation(self, x: int | Sequence[int]) -> list[int] | ArrayProtocol:
        """Return a randomly permuted list of integers.

        Args:
            x (int | Sequence[int]): Integer count ``n`` or sequence of integers.

        Returns:
            list[int] | ArrayProtocol: Shuffled list of integers.
        """
        items = list(range(x)) if isinstance(x, int) else list(x)
        self._rng.shuffle(items)
        return items

    def __repr__(self) -> str:
        """Return the string representation matching NumPy Generator format.

        Returns:
            str: Formatted generator representation.
        """
        return f"Generator({self._algorithm_name})"


def _extract_array_seed(key_arr: ArrayProtocol) -> int:
    """Extract a deterministic 32-bit integer seed from an ``ArrayProtocol`` PRNG key.

    Args:
        key_arr (ArrayProtocol): PRNG key array (such as ``zero_jax.random.PRNGKey``).

    Returns:
        int: Extracted integer seed.
    """
    raw = array_to_list(key_arr)
    assert isinstance(raw, list)
    if len(raw) == 2 and raw[0] == 0 and isinstance(raw[1], int):
        return int(raw[1])
    acc = 0
    for item in raw:
        acc = ((acc * 1_000_003) ^ (int(item) if isinstance(item, int) else 0)) & 0xFFFFFFFF
    return acc


def _resolve_seed_int(
    seed: int | ArrayProtocol | FlaxRngs | None,
) -> int | None:
    """Resolve an ``int``, ``PRNGKey`` array, or ``zero_flax.nnx.Rngs`` into an integer seed.

    Args:
        seed (int | ArrayProtocol | FlaxRngs | None): Seed value or PRNG key source.

    Returns:
        int | None: Resolved integer seed, or None if ``seed`` is None.
    """
    if seed is None:
        return None
    if isinstance(seed, int):
        return seed
    if isinstance(seed, FlaxRngs):
        return _extract_array_seed(seed())
    return _extract_array_seed(seed)


def create_rng(
    seed: int | ArrayProtocol | FlaxRngs | None = None,
) -> RNGGeneratorProtocol:
    """Create a deterministic RNG using NumPy when loaded or pure Python otherwise.

    Args:
        seed (int | ArrayProtocol | FlaxRngs | None): Optional integer seed, PRNGKey, or Rngs.

    Returns:
        RNGGeneratorProtocol: Deterministic random number generator instance.
    """
    resolved_seed = _resolve_seed_int(seed)
    np_mod = get_numpy_module()
    if np_mod is not None:
        return np_mod.random.default_rng(resolved_seed)
    return DeterministicGenerator(seed=resolved_seed)


def derive_rng(
    base_seed: int | ArrayProtocol | FlaxRngs | None,
    index: int,
    op_seed: int = 0,
) -> DeterministicGenerator:
    """Derive a per-record deterministic RNG from ``base_seed``, ``index``, and ``op_seed``.

    Args:
        base_seed (int | ArrayProtocol | FlaxRngs | JaxArrayLikeSpec | None): Base seed, PRNGKey, or Rngs.
        index (int): Global record index.
        op_seed (int): Operation-level seed offset.

    Returns:
        DeterministicGenerator: Independent deterministic RNG for the given record.
    """
    resolved_base = _resolve_seed_int(base_seed)
    effective_base = 0 if resolved_base is None else resolved_base
    combined_seed = (effective_base * 1_000_003 + index * 9_999_991 + op_seed) & 0xFFFFFFFF
    return DeterministicGenerator(seed=combined_seed, algorithm_name="Philox")


__all__ = [
    "DeterministicGenerator",
    "_extract_array_seed",
    "_resolve_seed_int",
    "create_rng",
    "derive_rng",
    "index_shuffle",
]
