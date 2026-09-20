"""Private training worlds, disjoint from every checked-in evaluation suite."""

import random

from src.benchmarking.config import load_suite


_SEED_SPACE = 2**32


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _SEED_SPACE:
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1.")
    return seed


def training_seeds(seed: int, count: int) -> list[int]:
    """Return a deterministic, unique sequence without touching the global RNG."""
    _validate_seed(seed)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Training seed count must be a positive integer.")
    # load_suite also verifies that quick is a subset of standard.
    excluded = set(load_suite("standard").seeds) | set(load_suite("holdout").seeds)
    if count > _SEED_SPACE - len(excluded):
        raise ValueError("Training seed count exceeds the available non-evaluation worlds.")
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        candidate = rng.getrandbits(32)
        if candidate not in excluded:
            excluded.add(candidate)
            result.append(candidate)
    return result
