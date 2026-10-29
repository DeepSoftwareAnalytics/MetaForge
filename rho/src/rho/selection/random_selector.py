from __future__ import annotations

import random

from rho.protocols import Task


class RandomSelector:
    """Shuffle-and-truncate selector. seed=None preserves input order
    (matches the prior inline run_evolution behavior)."""

    def select(
        self,
        candidates: list[Task],
        k: int | None,
        *,
        seed: int | None = None,
    ) -> list[Task]:
        pool = list(candidates)
        if seed is not None:
            random.Random(seed).shuffle(pool)
        if k is None:
            return pool
        return pool[:k]
