from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.terminal_bench_2 import TerminalBench2Dataset


class _FakeHarness:
    @property
    def id(self) -> str:
        return "empty"

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)


class _FakeHarnessStore:
    def empty(self) -> _FakeHarness:
        return _FakeHarness()

    def capture(self, src: Path) -> _FakeHarness:
        return _FakeHarness()

    def get(self, harness_id: str) -> _FakeHarness:
        return _FakeHarness()


def _populate_repo_with_n_tasks(repo: Path, n: int) -> None:
    for i in range(n):
        d = repo / f"task-{i:03d}"
        (d / "environment").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "environment" / "Dockerfile").write_text("FROM alpine\n", encoding="utf-8")
        (d / "tests" / "test.sh").write_text(
            "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
            encoding="utf-8",
        )
        (d / "instruction.md").write_text(f"Task {i}\n", encoding="utf-8")
        difficulty = ("easy", "medium", "hard", "extreme")[i % 4]
        (d / "task.toml").write_text(
            f'''
version = "1.0"
[metadata]
difficulty = "{difficulty}"
category = "misc"
tags = []
[verifier]
timeout_sec = 30.0
[agent]
timeout_sec = 30.0
[environment]
docker_image = "alpine:3.19"
''',
            encoding="utf-8",
        )


def test_dataset_construction_enumerates_tasks(tmp_path: Path) -> None:
    repo = tmp_path / "tb2-repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Terminal-Bench 2\n", encoding="utf-8")
    _populate_repo_with_n_tasks(repo, 20)
    ds = TerminalBench2Dataset(repo, harness_store=_FakeHarnessStore())
    assert len(ds.train) + len(ds.val) + len(ds.test) == 20


def test_dataset_difficulty_filter(tmp_path: Path) -> None:
    repo = tmp_path / "tb2-repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Terminal-Bench 2\n", encoding="utf-8")
    _populate_repo_with_n_tasks(repo, 20)
    ds = TerminalBench2Dataset(
        repo,
        harness_store=_FakeHarnessStore(),
        difficulty_filter=("easy",),
    )
    assert len(ds.train) + len(ds.val) + len(ds.test) == 5


def test_dataset_rejects_bad_path(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        TerminalBench2Dataset(tmp_path / "nope", harness_store=_FakeHarnessStore())


def test_task_materialize_writes_files_no_docker(tmp_path: Path, fake_repo_dir: Path) -> None:
    ds = TerminalBench2Dataset(fake_repo_dir, harness_store=_FakeHarnessStore())
    all_tasks = list(ds.train) + list(ds.val) + list(ds.test)
    task = all_tasks[0]
    dest = tmp_path / "materialize-out"
    dest.mkdir()
    task.materialize(dest)
    assert (dest / "prompt.md").exists()
    assert (dest / ".tb2").exists()


def test_task_query_returns_instruction_md(tmp_path: Path, fake_repo_dir: Path) -> None:
    ds = TerminalBench2Dataset(fake_repo_dir, harness_store=_FakeHarnessStore())
    task = next(iter(ds.train)) if len(ds.train) else next(iter(ds.val))
    text = task.query()
    assert "Fake Task" in text or "fake" in text.lower()


def test_task_grade_without_runtime_returns_failure(
    tmp_path: Path,
    fake_repo_dir: Path,
) -> None:
    from rho.protocols import Trajectory

    ds = TerminalBench2Dataset(fake_repo_dir, harness_store=_FakeHarnessStore())
    task = list(ds.train)[0] if len(ds.train) else list(ds.val)[0]
    dest = tmp_path / "out"
    dest.mkdir()
    task.materialize(dest)
    fake_traj = Trajectory(
        id="tr-0",
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
    grade = task.grade(fake_traj, artifacts_dir=tmp_path / "art")
    assert not grade.passed
    assert grade.details["error"] == "no_runtime"
