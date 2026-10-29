from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rho.datasets.swebench_pro import SWEbenchProDataset
from rho.orchestrators.solve import solve
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore

pytestmark = [pytest.mark.swebench_pro, pytest.mark.codex]

CANARY_INSTANCE_ID = (
    "instance_internetarchive__openlibrary-e8084193a895d8ee81200f49093389a3887479ce"
    "-ve8c8d62a2b60610a3c4631f5f23ed866bada9818"
)


@pytest.fixture(scope="module")
def docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found on PATH")
    proc = subprocess.run(
        ["docker", "ps"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker daemon is not usable: {proc.stderr or proc.stdout}")


@pytest.fixture(scope="module")
def canary_row() -> dict:
    datasets = pytest.importorskip(
        "datasets",
        reason="SWE-bench Pro tests require `uv run --extra swebench-pro ...`",
    )
    dataset = datasets.load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    for row in dataset:
        if row["instance_id"] == CANARY_INSTANCE_ID:
            return dict(row)
    raise AssertionError(f"canary instance not found: {CANARY_INSTANCE_ID}")


def test_swebench_pro_gold_patch_real_docker_canary(
    docker_available,
    canary_row: dict,
    tmp_path: Path,
) -> None:
    task = _canary_task(canary_row, tmp_path)
    trajectory = _trajectory_from_patch(task.id, canary_row["patch"])

    grade = task.grade(trajectory, artifacts_dir=tmp_path / "gold_grade")

    patch_path = (
        tmp_path
        / "gold_grade"
        / "patch_extract"
        / "prediction.patch"
    )
    assert patch_path.exists()
    assert patch_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "gold_grade" / "docker" / "output.json").exists()
    assert grade.passed is True, grade.details


def test_swebench_pro_real_codex_solve_and_grade_canary(
    docker_available,
    canary_row: dict,
    codex_agent_factory,
    tmp_path: Path,
) -> None:
    task = _canary_task(canary_row, tmp_path)
    handle = codex_agent_factory(cache_mode="off")

    trajectory = solve(
        handle.agent,
        task,
        task.harness,
        workdir=tmp_path / "workdir",
        stage="swebench_pro_real_solve",
    )
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    traj_store.put(trajectory)
    assert trajectory.exit_code == 0

    grade = task.grade(trajectory, artifacts_dir=tmp_path / "codex_grade")
    patch_path = (
        tmp_path
        / "codex_grade"
        / "patch_extract"
        / "prediction.patch"
    )
    assert patch_path.exists()
    assert patch_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "codex_grade" / "docker" / "output.json").exists()
    assert grade.passed is True, (
        f"trajectory_dir={tmp_path / 'trajectories' / trajectory.id}; "
        f"details={grade.details}"
    )


def _canary_task(row: dict, tmp_path: Path):
    dataset = SWEbenchProDataset.from_records(
        [row],
        harness_store=FilesystemHarnessStore(tmp_path / "harness"),
        docker_pull="always",
    )
    return next(iter(dataset.train))


def _trajectory_from_patch(task_id: str, patch: str) -> Trajectory:
    return Trajectory(
        id="traj_gold_patch",
        kind="solve",
        task_id=task_id,
        harness_id="h_empty",
        instructions="gold patch smoke",
        events=[],
        final_message=f"```diff\n{patch.rstrip()}\n```",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )
