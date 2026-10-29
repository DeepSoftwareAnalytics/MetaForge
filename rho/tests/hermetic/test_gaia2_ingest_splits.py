from __future__ import annotations

import pytest

from rho.datasets.gaia2.ingest import parse_payload
from rho.datasets.gaia2.splits import apply_max_per_split, split_task_ids


def test_parse_payload_defaults_to_mini_config() -> None:
    parsed = parse_payload("meta-agents-research-environments/gaia2")

    assert parsed.dataset_spec == "meta-agents-research-environments/gaia2"
    assert parsed.config == "mini"


def test_parse_payload_accepts_explicit_config() -> None:
    parsed = parse_payload("meta-agents-research-environments/gaia2#config=execution")

    assert parsed.dataset_spec == "meta-agents-research-environments/gaia2"
    assert parsed.config == "execution"


def test_parse_payload_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="Unsupported GAIA-2 config"):
        parse_payload("meta-agents-research-environments/gaia2#config=noise")


def test_split_task_ids_is_deterministic_and_complete() -> None:
    task_ids = [f"mini/task-{ix}" for ix in range(20)]

    first = split_task_ids(task_ids, seed=7)
    second = split_task_ids(reversed(task_ids), seed=7)

    assert first == second
    assert set(first) == {"train", "val", "test"}
    assert sorted(first["train"] + first["val"] + first["test"]) == sorted(task_ids)


def test_split_task_ids_fixes_val_at_100_with_empty_test() -> None:
    task_ids = [f"mini/task-{ix}" for ix in range(250)]

    split = split_task_ids(task_ids)

    assert len(split["val"]) == 100
    assert len(split["train"]) == 150
    assert split["test"] == []
    assert sorted(split["train"] + split["val"]) == sorted(task_ids)
    assert not set(split["train"]) & set(split["val"])


def test_split_task_ids_val_takes_all_when_below_target() -> None:
    task_ids = [f"mini/task-{ix}" for ix in range(40)]

    split = split_task_ids(task_ids)

    assert len(split["val"]) == 40
    assert split["train"] == []
    assert split["test"] == []


def test_apply_max_per_split_clamps_each_split() -> None:
    split_map = {
        "train": ["a", "b"],
        "val": ["c", "d"],
        "test": ["e", "f"],
    }

    assert apply_max_per_split(split_map, max_per_split=1) == {
        "train": ["a"],
        "val": ["c"],
        "test": ["e"],
    }
