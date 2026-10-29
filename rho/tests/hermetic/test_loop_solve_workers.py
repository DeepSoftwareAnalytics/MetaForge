from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rho.loop import run_round
from rho.protocols import Grade, Harness, Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.conftest import make_fake_agent


class _Counter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


class _RuntimeSession:
    def __init__(self, counter: _Counter) -> None:
        self._counter = counter

    def __enter__(self) -> None:
        self._counter.enter()
        time.sleep(0.02)

    def __exit__(self, *args) -> None:
        self._counter.exit()


@dataclass
class _RuntimeTask:
    _id: str
    prompt: str
    _harness: Harness
    counter: _Counter

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prompt.md").write_text(self.prompt, encoding="utf-8")

    def query(self) -> str:
        return self.prompt

    def grade(self, trajectory: Trajectory, *, artifacts_dir: Path | None = None) -> Grade:
        del trajectory, artifacts_dir
        return Grade(passed=False, score=0.0)

    def runtime_session(self, workdir: Path) -> _RuntimeSession:
        del workdir
        return _RuntimeSession(self.counter)


def test_run_round_solve_workers_limits_runtime_sessions(tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    current = harness_store.empty()
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    counter = _Counter()
    tasks = [
        _RuntimeTask("task_0", "What is the project code name?", current, counter),
        _RuntimeTask("task_1", "What is the oncall rotation?", current, counter),
    ]

    run_round(
        0,
        current,
        tasks,
        make_fake_agent("good"),
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
        optimize_samples=1,
        solve_workers=2,
    )

    assert counter.maximum == 2
