from __future__ import annotations

from pathlib import Path

import pytest

from rho.protocols import Trajectory
from rho.selection.short_solve import short_solve_all


class _FakeAgent:
    def __init__(self, *, raise_on=None):
        self.raise_on = raise_on or set()
        self.calls = []

    def run(self, workspace, instructions, *, task_id="", harness_id="", kind="solve",
            timeout_s=None, output_schema=None, env=None):
        self.calls.append(task_id)
        if task_id in self.raise_on:
            raise RuntimeError(f"infra flake on {task_id}")
        return Trajectory(
            id=f"traj_{task_id}", kind="solve", task_id=task_id, harness_id=harness_id,
            instructions=instructions, events=[
                {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "ok"}}
            ],
            final_message="done", stdout="", stderr="",
            workspace_diff={}, workspace_deletions=frozenset(),
            exit_code=0, wall_time_s=1.0,
        )


class _FakeHarness:
    id = "h_0"
    def materialize(self, dest: Path) -> None: pass


class _FakeTask:
    def __init__(self, tid):
        self.id = tid
        self.harness = _FakeHarness()
        self.agent_timeout_s = None
    def materialize(self, dest): pass
    def query(self): return f"q for {self.id}"
    def grade(self, traj, *, artifacts_dir=None): raise NotImplementedError


class _StoreList:
    """Captures every put() to assert traj_store integration."""
    def __init__(self): self.puts = []
    def put(self, traj): self.puts.append(traj)


def test_short_solve_returns_trajectory_per_task(tmp_path: Path) -> None:
    agent = _FakeAgent()
    harness = _FakeHarness()
    store = _StoreList()
    tasks = [_FakeTask(f"t{i}") for i in range(3)]
    result = short_solve_all(
        tasks, agent=agent, harness=harness, traj_store=store,
        workdir=tmp_path, max_workers=2,
    )
    assert set(result.keys()) == {"t0", "t1", "t2"}
    for tid, traj in result.items():
        assert traj.task_id == tid
        assert traj.stage == "short_solve_for_selection"
    assert len(store.puts) == 3


def test_short_solve_tolerates_single_agent_failure(tmp_path: Path) -> None:
    agent = _FakeAgent(raise_on={"t1"})
    store = _StoreList()
    tasks = [_FakeTask(f"t{i}") for i in range(3)]
    result = short_solve_all(
        tasks, agent=agent, harness=_FakeHarness(), traj_store=store,
        workdir=tmp_path, max_workers=2,
    )
    assert set(result.keys()) == {"t0", "t1", "t2"}
    assert result["t1"].exit_code != 0
    assert result["t1"].events == []
    assert "infra flake" in result["t1"].stderr
    assert result["t1"].final_message == ""
    # Still persisted, so inspect can see it.
    assert any(t.task_id == "t1" for t in store.puts)


def test_short_solve_propagates_keyboard_interrupt(tmp_path: Path) -> None:
    class _InterruptingAgent:
        def run(self, *a, **kw):
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        short_solve_all(
            [_FakeTask("t0")], agent=_InterruptingAgent(),
            harness=_FakeHarness(), traj_store=_StoreList(),
            workdir=tmp_path, max_workers=1,
        )
