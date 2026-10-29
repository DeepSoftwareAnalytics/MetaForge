"""Assert LOCOMO split is deterministic at seed=0 and respects spec §10."""

from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.loader import load_dataset
from rho.datasets.locomo import LocomoDataset
from rho.stores.harness import FilesystemHarnessStore

LOCOMO_PATH = Path(__file__).parents[2] / "data" / "locomo10.json"
LOCOMO_HARD_PATH = Path(__file__).parents[2] / "data" / "locomo-hard"


@pytest.fixture
def dataset(tmp_path: Path) -> LocomoDataset:
    hs = FilesystemHarnessStore(tmp_path / "harness")
    return LocomoDataset(LOCOMO_PATH, harness_store=hs, seed=0)


def test_split_sizes_match_spec(dataset: LocomoDataset) -> None:
    # spec §10: 1540 usable QA, train=0.25, val=0.375, test=0.375.
    # Per-category floor accounting: train=384, val=576, test=580.
    assert len(dataset.train) == 384
    assert len(dataset.val) == 576
    assert len(dataset.test) == 580
    assert len(dataset.train) + len(dataset.val) + len(dataset.test) == 1540


def test_splits_are_disjoint(dataset: LocomoDataset) -> None:
    train = {t.id for t in dataset.train}
    val = {t.id for t in dataset.val}
    test = {t.id for t in dataset.test}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_splits_are_bit_stable(tmp_path: Path) -> None:
    hs_a = FilesystemHarnessStore(tmp_path / "harness_a")
    hs_b = FilesystemHarnessStore(tmp_path / "harness_b")
    ds_a = LocomoDataset(LOCOMO_PATH, harness_store=hs_a, seed=0)
    ds_b = LocomoDataset(LOCOMO_PATH, harness_store=hs_b, seed=0)
    assert [t.id for t in ds_a.train] == [t.id for t in ds_b.train]
    assert [t.id for t in ds_a.val] == [t.id for t in ds_b.val]
    assert [t.id for t in ds_a.test] == [t.id for t in ds_b.test]


def test_different_seed_gives_different_split(tmp_path: Path) -> None:
    hs_a = FilesystemHarnessStore(tmp_path / "harness_a")
    hs_b = FilesystemHarnessStore(tmp_path / "harness_b")
    ds_a = LocomoDataset(LOCOMO_PATH, harness_store=hs_a, seed=0)
    ds_b = LocomoDataset(LOCOMO_PATH, harness_store=hs_b, seed=1)
    assert [t.id for t in ds_a.train] != [t.id for t in ds_b.train]


def test_no_cat5_in_any_split(dataset: LocomoDataset) -> None:
    # Drilling into private attrs is fine here — this is a research
    # project and we want to assert the filter actually happens.
    for split_name in ("train", "val", "test"):
        split = getattr(dataset, split_name)
        for task in split:
            assert task._category != 5  # type: ignore[attr-defined]


def test_max_per_split_caps_all_three(tmp_path: Path) -> None:
    hs = FilesystemHarnessStore(tmp_path / "harness")
    ds = LocomoDataset(LOCOMO_PATH, harness_store=hs, seed=0, max_per_split=10)
    assert len(ds.train) == 10
    assert len(ds.val) == 10
    assert len(ds.test) == 10


def test_max_per_split_preserves_some_category_balance(tmp_path: Path) -> None:
    hs = FilesystemHarnessStore(tmp_path / "harness")
    ds = LocomoDataset(LOCOMO_PATH, harness_store=hs, seed=0, max_per_split=10)
    cats = [t._category for t in ds.train]  # type: ignore[attr-defined]
    # All four usable categories should appear (3/3/3/1 by spec §10.1).
    assert set(cats) == {1, 2, 3, 4}


def test_locomo_hard_subset_loads_manifest_order(tmp_path: Path) -> None:
    hs = FilesystemHarnessStore(tmp_path / "harness")
    ds = load_dataset(f"locomo-hard:{LOCOMO_HARD_PATH}", harness_store=hs)

    assert len(ds.train) == 20
    assert len(ds.val) == 20
    assert len(ds.test) == 0
    assert [t.id for t in ds.train][:3] == [
        "conv-42/qa_0060",
        "conv-30/qa_0048",
        "conv-42/qa_0119",
    ]
    assert [t.id for t in ds.val][:3] == [
        "conv-44/qa_0106",
        "conv-42/qa_0080",
        "conv-43/qa_0151",
    ]
