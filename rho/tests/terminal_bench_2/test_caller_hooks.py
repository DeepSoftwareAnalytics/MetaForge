from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

from rho.protocols import Grade, Harness, Trajectory


class _FakeTask:
    """Supports runtime_session and records enter/exit calls."""

    def __init__(self, task_id: str, log: list[str]) -> None:
        self._id = task_id
        self._log = log

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        return _FakeHarness()

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prompt.md").write_text(f"# {self._id}\n", encoding="utf-8")

    def query(self) -> str:
        return ""

    def grade(self, traj, *, artifacts_dir=None) -> Grade:
        self._log.append(f"grade:{self._id}")
        return Grade(passed=True, score=1.0)

    def runtime_session(self, workdir: Path):
        log = self._log
        tid = self._id

        @contextmanager
        def cm() -> Iterator[None]:
            log.append(f"enter:{tid}")
            try:
                yield
            finally:
                log.append(f"exit:{tid}")

        return cm()


class _PlainTask(_FakeTask):
    """Does not implement runtime_session; caller hook must skip gracefully."""

    def __getattribute__(self, name: str):
        if name == "runtime_session":
            raise AttributeError
        return super().__getattribute__(name)


class _FakeHarness:
    @property
    def id(self) -> str:
        return "h"

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)


def test_runtime_session_enter_exit_around_solve_and_grade(monkeypatch, tmp_path) -> None:
    from rho import reporting

    def fake_solve_in(agent, task, harness, ws, **kwargs):
        log = task._log
        log.append(f"solve:{task.id}")
        return Trajectory(
            id=f"tr-{task.id}",
            kind="solve",
            task_id=task.id,
            harness_id="h",
            instructions="",
            events=[],
            final_message="",
            stdout="",
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=0.0,
        )

    monkeypatch.setattr(reporting, "solve_in", fake_solve_in)

    log: list[str] = []
    tasks = [_FakeTask("t1", log)]

    class _Split:
        def __iter__(self):
            return iter(tasks)

        def __len__(self):
            return 1

        @property
        def split(self):
            return "train"

    result = reporting.grade_on_split(MagicMock(), _FakeHarness(), _Split(), workdir=tmp_path)
    assert result[0].grade.passed
    assert log == ["enter:t1", "solve:t1", "grade:t1", "exit:t1"]


def test_plain_task_without_runtime_session_still_works(monkeypatch, tmp_path) -> None:
    from rho import reporting

    def fake_solve_in(agent, task, harness, ws, **kwargs):
        log = task._log
        log.append(f"solve:{task.id}")
        return Trajectory(
            id=f"tr-{task.id}",
            kind="solve",
            task_id=task.id,
            harness_id="h",
            instructions="",
            events=[],
            final_message="",
            stdout="",
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=0.0,
        )

    monkeypatch.setattr(reporting, "solve_in", fake_solve_in)

    log: list[str] = []
    tasks = [_PlainTask("t2", log)]

    class _Split:
        def __iter__(self):
            return iter(tasks)

        def __len__(self):
            return 1

        @property
        def split(self):
            return "train"

    result = reporting.grade_on_split(MagicMock(), _FakeHarness(), _Split(), workdir=tmp_path)
    assert result[0].grade.passed
    assert log == ["solve:t2", "grade:t2"]
