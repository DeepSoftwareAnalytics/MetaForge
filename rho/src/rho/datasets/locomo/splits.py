"""Deterministic stratified split of LOCOMO QA pairs.

Drops category 5 (adversarial — excluded; see spec §10). Each of the
four remaining categories is sampled independently with proportions
``0.25 / 0.375 / 0.375`` for train / val / test. Seeded by
``random.Random(seed)``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QARef:
    conv_id: str
    qa_index: int
    category: int


TRAIN_RATIO = 0.25
VAL_RATIO = 0.375
TEST_RATIO = 0.375
_USABLE_CATEGORIES = (1, 2, 3, 4)


def stratified_split(
    qas: Sequence[QARef],
    *,
    seed: int = 0,
) -> dict[str, list[QARef]]:
    filtered = [qa for qa in qas if qa.category in _USABLE_CATEGORIES]
    rng = random.Random(seed)

    groups: dict[int, list[QARef]] = {cat: [] for cat in _USABLE_CATEGORIES}
    for qa in sorted(filtered, key=lambda q: (q.conv_id, q.qa_index)):
        groups[qa.category].append(qa)
    for cat in _USABLE_CATEGORIES:
        rng.shuffle(groups[cat])

    splits: dict[str, list[QARef]] = {"train": [], "val": [], "test": []}
    for cat in _USABLE_CATEGORIES:
        group = groups[cat]
        n = len(group)
        n_train = math.floor(TRAIN_RATIO * n)
        n_val = math.floor(VAL_RATIO * n)
        n_test = n - n_train - n_val
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val : n_train + n_val + n_test])
    return splits


def apply_max_per_split(
    split: list[QARef],
    *,
    max_per_split: int,
) -> list[QARef]:
    """Cap a split to ``max_per_split`` items, keeping category balance.

    For each category, take up to ``ceil(N / 4)`` from the front of its
    sublist (already seeded-shuffled), concatenate in category order,
    then truncate to exactly N.
    """
    if max_per_split >= len(split):
        return list(split)
    per_cat_cap = math.ceil(max_per_split / len(_USABLE_CATEGORIES))
    by_cat: dict[int, list[QARef]] = {cat: [] for cat in _USABLE_CATEGORIES}
    for qa in split:
        by_cat[qa.category].append(qa)
    capped: list[QARef] = []
    for cat in _USABLE_CATEGORIES:
        capped.extend(by_cat[cat][:per_cat_cap])
    return capped[:max_per_split]
