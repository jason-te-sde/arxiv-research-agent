"""How far a mean over a few dozen questions can be trusted."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def bootstrap_ci(
    values: Sequence[float], *, level: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    For a difference between two systems on the same questions, pass the
    per-question differences: resampling questions keeps each pair together.
    """
    if not values:
        raise ValueError("no values to bootstrap")
    rng = random.Random(seed)  # noqa: S311 - a seeded resampler, not a secret
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    return means[math.floor(tail * resamples)], means[math.ceil((1 - tail) * resamples) - 1]
