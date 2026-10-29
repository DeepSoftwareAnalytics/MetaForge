from __future__ import annotations

import hashlib
from collections.abc import Sequence

VAL_TARGET = 100


def split_task_ids(
    task_ids: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministic exact-count val/train split by hash of (seed, id).

    Task IDs are ordered by ``sha256(seed:id)``; the first ``VAL_TARGET`` go to
    ``val`` and the remainder to ``train``. ``test`` is always empty: the
    single-round experiments do not use a held-out test split, and the
    experiment plan fixes val at 100 samples. For ``config=mini`` (~200
    scenarios) this yields ~100 train / 100 val / 0 test. If fewer than
    ``VAL_TARGET`` tasks exist, ``val`` takes all of them and ``train`` is
    empty. The split is order-independent and fully deterministic.
    """
    ordered = sorted(task_ids, key=lambda task_id: _digest(seed, task_id))
    n_val = min(VAL_TARGET, len(ordered))
    return {
        "train": ordered[n_val:],
        "val": ordered[:n_val],
        "test": [],
    }


def apply_max_per_split(
    split_map: dict[str, list[str]],
    *,
    max_per_split: int,
) -> dict[str, list[str]]:
    return {split: ids[:max_per_split] for split, ids in split_map.items()}


def _digest(seed: int, task_id: str) -> str:
    payload = f"{seed}:{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
