from __future__ import annotations

from rho.datasets.terminal_bench_2.splits import split_task_ids


def test_split_on_89_tasks_is_30_59_0() -> None:
    ids = [f"task-{i:03d}" for i in range(89)]
    splits = split_task_ids(ids, seed=0)
    assert len(splits["train"]) == 30
    assert len(splits["val"]) == 59
    assert len(splits["test"]) == 0
    all_split = set(splits["train"]) | set(splits["val"])
    assert all_split == set(ids)
    assert sum(len(splits[k]) for k in ("train", "val", "test")) == 89


def test_split_is_deterministic_across_runs() -> None:
    ids = [f"task-{i:03d}" for i in range(89)]
    a = split_task_ids(ids, seed=0)
    b = split_task_ids(ids, seed=0)
    assert a == b


def test_split_changes_with_seed() -> None:
    ids = [f"task-{i:03d}" for i in range(89)]
    a = split_task_ids(ids, seed=0)
    b = split_task_ids(ids, seed=1)
    assert a != b


def test_split_empty_input_returns_empty_splits() -> None:
    assert split_task_ids([], seed=0) == {"train": [], "val": [], "test": []}
