from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.terminal_bench_2 import TerminalBench2Dataset
from rho.datasets.terminal_bench_2 import container as cn
from tests.terminal_bench_2.conftest import requires_docker


class _FakeHarness:
    @property
    def id(self):
        return "empty"

    def materialize(self, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)


class _FakeHarnessStore:
    def empty(self):
        return _FakeHarness()

    def capture(self, src):
        return _FakeHarness()

    def get(self, hid):
        return _FakeHarness()


SMALL_TASK_ID = "break-filter-js-from-html"


@pytest.mark.terminal_bench_2
@requires_docker
def test_golden_solution_grade_passes(tb2_repo: Path, tmp_path: Path) -> None:
    """Apply the task reference solve.sh in-container and confirm grading passes."""
    ds = TerminalBench2Dataset(tb2_repo, harness_store=_FakeHarnessStore())
    all_tasks = list(ds.train) + list(ds.val) + list(ds.test)
    task = next((candidate for candidate in all_tasks if candidate.id == SMALL_TASK_ID), None)
    if task is None:
        pytest.skip(f"Task {SMALL_TASK_ID} not present in TB2 repo at pinned SHA")

    workdir = tmp_path / "task"
    workdir.mkdir()
    task.materialize(workdir)
    with task.runtime_session(workdir) as handle:
        cn.cp_to(handle.container_name, str(task._task_dir / "solution") + "/.", "/solution/")
        cn.exec_script(
            handle.container_name,
            "chmod +x /solution/solve.sh && /solution/solve.sh",
            timeout_s=600,
        )
        from rho.protocols import Trajectory

        fake_traj = Trajectory(
            id="fake",
            kind="solve",
            task_id=task.id,
            harness_id="empty",
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
        parent = workdir.parent
        artifacts_dir = parent / "artifacts"
        artifacts_dir.mkdir()
        grade = task.grade(fake_traj, artifacts_dir=artifacts_dir)
    assert grade.passed, f"Expected golden-path pass; got {grade.details}"
