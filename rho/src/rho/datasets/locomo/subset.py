from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rho.datasets.locomo.dataset import LocomoDataset
from rho.protocols import HarnessStore, Task, TaskSet


@dataclass(frozen=True)
class _SubsetTaskSet:
    _split: str
    _tasks: tuple[Task, ...]

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class LocomoSubsetDataset:
    """LOCOMO dataset filtered to explicit task IDs per split."""

    def __init__(
        self,
        root: Path,
        *,
        harness_store: HarnessStore,
        max_per_split: int | None = None,
    ) -> None:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = _resolve_source_path(root, manifest["source_dataset"])
        base = LocomoDataset(source_path, harness_store=harness_store)
        task_by_id = {
            task.id: task
            for split in (base.train, base.val, base.test)
            for task in split
        }

        self._splits: dict[str, TaskSet] = {}
        manifest_splits = manifest.get("splits", {})
        for split_name in ("train", "val", "test"):
            task_ids = list(manifest_splits.get(split_name, []))
            if max_per_split is not None:
                task_ids = task_ids[:max_per_split]
            self._splits[split_name] = _SubsetTaskSet(
                _split=split_name,
                _tasks=tuple(_lookup_task(task_by_id, task_id) for task_id in task_ids),
            )

    @property
    def train(self) -> TaskSet:
        return self._splits["train"]

    @property
    def val(self) -> TaskSet:
        return self._splits["val"]

    @property
    def test(self) -> TaskSet:
        return self._splits["test"]


def _resolve_source_path(root: Path, source_dataset: str) -> Path:
    if ":" in source_dataset:
        scheme, _, payload = source_dataset.partition(":")
        if scheme != "locomo":
            raise ValueError(f"Unsupported LOCOMO subset source scheme: {scheme!r}")
    else:
        payload = source_dataset
    path = Path(payload)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _lookup_task(task_by_id: dict[str, Task], task_id: str) -> Task:
    try:
        return task_by_id[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown LOCOMO task id in subset manifest: {task_id}") from exc
