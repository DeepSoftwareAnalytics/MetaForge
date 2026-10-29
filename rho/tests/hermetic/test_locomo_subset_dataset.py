from __future__ import annotations

import json
from pathlib import Path

from rho.datasets.loader import load_dataset
from rho.datasets.locomo import LocomoSubsetDataset
from rho.stores.harness import FilesystemHarnessStore

LOCOMO_HARD_ROOT = Path(__file__).parents[2] / "data" / "locomo-hard"


def test_locomo_hard_subset_loads_manifest_splits(tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    dataset = LocomoSubsetDataset(LOCOMO_HARD_ROOT, harness_store=harness_store)
    manifest = json.loads(
        (LOCOMO_HARD_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert [task.id for task in dataset.train] == manifest["splits"]["train"]
    assert [task.id for task in dataset.val] == manifest["splits"]["val"]
    assert [task.id for task in dataset.test] == []
    assert len(dataset.train) == 20
    assert len(dataset.val) == 20
    assert len(dataset.test) == 0


def test_locomo_hard_loader_and_max_per_split(tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    dataset = load_dataset(
        f"locomo-hard:{LOCOMO_HARD_ROOT}",
        harness_store=harness_store,
        max_per_split=3,
    )

    assert len(dataset.train) == 3
    assert len(dataset.val) == 3
    assert len(dataset.test) == 0

    task = next(iter(dataset.train))
    task_dir = tmp_path / "task"
    task.materialize(task_dir)
    assert "ANSWER:" in (task_dir / "prompt.md").read_text(encoding="utf-8")
