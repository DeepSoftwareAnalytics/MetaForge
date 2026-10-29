from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rho.datasets.gaia2.ingest import Gaia2Row, load_rows, parse_payload
from rho.datasets.gaia2.prompts import render_prompt, render_query
from rho.datasets.gaia2.splits import apply_max_per_split, split_task_ids
from rho.protocols import Grade, Harness, HarnessStore, Task, TaskSet, Trajectory


@dataclass(frozen=True)
class Gaia2Task:
    _row: Gaia2Row
    _harness: Harness

    @property
    def id(self) -> str:
        return self._row.task_id

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".gaia2").mkdir(exist_ok=True)
        prompt = render_prompt(task_id=self.id, scenario_data=self._row.data)
        (dest / "prompt.md").write_text(prompt, encoding="utf-8")

    def query(self) -> str:
        return render_query(task_id=self.id, scenario_data=self._row.data)

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade:
        from rho.datasets.gaia2.grader import normalize_validation_result
        from rho.datasets.gaia2.runtime import get_active, rpc

        handle = get_active(self.id)
        if handle is not None:
            try:
                response = rpc(handle, {"method": "validate"})
            except (OSError, RuntimeError) as exc:
                # The sidecar process died or its socket became unreachable
                # mid-session. Surface this as a 0-scored task with an error
                # marker rather than letting the exception abort the whole
                # grade_on_split pool and lose every other task's work.
                return Grade(
                    passed=False,
                    score=0.0,
                    details={
                        "error": "runtime_unreachable",
                        "message": f"GAIA-2 validation RPC failed: {exc}",
                        "task_id": self.id,
                    },
                )
            if not response.get("ok", False):
                return Grade(
                    passed=False,
                    score=0.0,
                    details={
                        "error": response.get("error", "validation_failed"),
                        "task_id": self.id,
                    },
                )
            return normalize_validation_result(
                response.get("result"),
                scenario_id=self._row.scenario_id,
                config=self._row.config,
            )
        return Grade(
            passed=False,
            score=0.0,
            details={
                "error": "no_runtime",
                "message": "GAIA-2 grade() was called outside an active runtime session.",
                "task_id": self.id,
            },
        )

    def runtime_session(self, workdir: Path) -> AbstractContextManager[object]:
        from rho.datasets.gaia2.runtime import Gaia2RuntimeSession

        return Gaia2RuntimeSession(self, workdir)

    @property
    def scenario_json(self) -> dict:
        return self._row.data


@dataclass(frozen=True)
class Gaia2TaskSet:
    _split: str
    _tasks: tuple[Gaia2Task, ...]

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class Gaia2Dataset:
    def __init__(
        self,
        payload: str,
        *,
        harness_store: HarnessStore,
        max_per_split: int | None = None,
        seed: int = 0,
    ) -> None:
        parsed = parse_payload(payload)
        rows = load_rows(parsed)
        self._harness = harness_store.empty()
        tasks = [Gaia2Task(_row=row, _harness=self._harness) for row in rows]
        id_to_task = {task.id: task for task in tasks}
        split_ids = split_task_ids(list(id_to_task), seed=seed)
        if max_per_split is not None:
            split_ids = apply_max_per_split(split_ids, max_per_split=max_per_split)
        self._splits = {
            split: Gaia2TaskSet(
                _split=split,
                _tasks=tuple(id_to_task[task_id] for task_id in ids),
            )
            for split, ids in split_ids.items()
        }

    @property
    def train(self) -> TaskSet:
        return self._splits["train"]

    @property
    def val(self) -> TaskSet:
        return self._splits["val"]

    @property
    def test(self) -> TaskSet:
        return self._splits["test"]
