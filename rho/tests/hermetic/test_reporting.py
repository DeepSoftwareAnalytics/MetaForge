from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rho.protocols import Grade, Trajectory, TrajectoryKind
from rho.reporting import grade_on_split

_runtime_state = threading.local()


@dataclass(frozen=True)
class _Harness:
    id: str = "h_test"

    def materialize(self, dest: Path) -> None:
        (dest / "notes.md").write_text("notes\n", encoding="utf-8")


class _Counter:
    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    def exit(self) -> None:
        with self.lock:
            self.current -= 1


class _Task:
    def __init__(
        self,
        task_id: str,
        grade_counter: _Counter,
        runtime_counter: _Counter | None = None,
    ) -> None:
        self._id = task_id
        self._grade_counter = grade_counter
        self._runtime_counter = runtime_counter
        self.harness = _Harness()

    @property
    def id(self) -> str:
        return self._id

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        (dest / "prompt.md").write_text(f"task {self._id}\n", encoding="utf-8")

    @contextmanager
    def runtime_session(self, task_dir: Path) -> Iterator[None]:
        del task_dir
        if self._runtime_counter is not None:
            self._runtime_counter.enter()
        _runtime_state.active_task_id = self._id
        try:
            yield
        finally:
            _runtime_state.active_task_id = None
            if self._runtime_counter is not None:
                self._runtime_counter.exit()

    def query(self) -> str:
        return self._id

    def grade(
        self, trajectory: Trajectory, *, artifacts_dir: Path | None = None
    ) -> Grade:
        del trajectory, artifacts_dir
        assert getattr(_runtime_state, "active_task_id", None) == self._id
        self._grade_counter.enter()
        try:
            time.sleep(0.05)
            return Grade(passed=True, score=1.0, details={})
        finally:
            self._grade_counter.exit()


class _RecordingAgent:
    def __init__(self, solve_counter: _Counter) -> None:
        self._solve_counter = solve_counter

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        del workspace, instructions, output_schema, timeout_s, env
        assert getattr(_runtime_state, "active_task_id", None) == task_id
        self._solve_counter.enter()
        try:
            time.sleep(0.05)
            return Trajectory(
                id=f"traj_{task_id}",
                kind=kind,
                task_id=task_id,
                harness_id=harness_id,
                instructions="",
                events=[],
                final_message="ok",
                stdout="",
                stderr="",
                workspace_diff={},
                workspace_deletions=frozenset(),
                exit_code=0,
                wall_time_s=0.05,
                timed_out=False,
            )
        finally:
            self._solve_counter.exit()


class _FlakyTransportAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        del workspace, instructions, output_schema, timeout_s, env
        self.calls += 1
        exit_code = 1 if self.calls == 1 else 0
        return Trajectory(
            id=f"traj_attempt_{self.calls}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions="",
            events=[],
            final_message="" if self.calls == 1 else "ok",
            stdout=(
                "stream disconnected before completion: response.failed event received"
                if self.calls == 1
                else ""
            ),
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=exit_code,
            wall_time_s=0.05,
            timed_out=False,
        )


def test_grade_workers_limit_grading_but_not_solve_submission(tmp_path: Path) -> None:
    solve_counter = _Counter()
    grade_counter = _Counter()
    tasks = [_Task(f"task_{ix}", grade_counter) for ix in range(3)]
    agent = _RecordingAgent(solve_counter)

    results = grade_on_split(
        agent,
        _Harness(),
        tasks,
        tmp_path / "work",
        max_workers=1,
    )

    assert [result.task.id for result in results] == ["task_0", "task_1", "task_2"]
    assert solve_counter.maximum > 1
    assert grade_counter.maximum == 1


def test_solve_workers_limit_runtime_sessions(tmp_path: Path) -> None:
    solve_counter = _Counter()
    grade_counter = _Counter()
    runtime_counter = _Counter()
    tasks = [
        _Task(f"task_{ix}", grade_counter, runtime_counter=runtime_counter)
        for ix in range(5)
    ]
    agent = _RecordingAgent(solve_counter)

    results = grade_on_split(
        agent,
        _Harness(),
        tasks,
        tmp_path / "work",
        max_workers=1,
        solve_workers=2,
    )

    assert [result.task.id for result in results] == [
        "task_0",
        "task_1",
        "task_2",
        "task_3",
        "task_4",
    ]
    assert runtime_counter.maximum == 2
    assert solve_counter.maximum == 2
    assert grade_counter.maximum == 1


def test_grade_on_split_retries_transient_codex_stream_failures(
    tmp_path: Path,
) -> None:
    grade_counter = _Counter()
    task = _Task("task_0", grade_counter)
    agent = _FlakyTransportAgent()

    results = grade_on_split(
        agent,
        _Harness(),
        [task],
        tmp_path / "work",
        solve_workers=1,
    )

    assert agent.calls == 2
    assert results[0].trajectory.exit_code == 0
    assert results[0].trajectory.final_message == "ok"
