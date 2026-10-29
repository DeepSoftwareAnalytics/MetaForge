from __future__ import annotations

import subprocess
from pathlib import Path


def _populate_fake_tb2_repo(repo: Path, n: int = 4) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text("# Terminal-Bench 2\n", encoding="utf-8")
    for i in range(n):
        d = repo / f"task-{i:03d}"
        (d / "environment").mkdir(parents=True)
        (d / "tests").mkdir()
        (d / "environment" / "Dockerfile").write_text("FROM alpine\n", encoding="utf-8")
        (d / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (d / "instruction.md").write_text(f"Task {i}\n", encoding="utf-8")
        (d / "task.toml").write_text(
            '''
version = "1.0"
[metadata]
difficulty = "easy"
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


def test_cli_load_dataset_terminal_bench_2(tmp_path: Path) -> None:
    from rho.datasets.loader import load_dataset

    class _Store:
        class _H:
            @property
            def id(self):
                return "empty"

            def materialize(self, dest: Path):
                dest.mkdir(parents=True, exist_ok=True)

        def empty(self):
            return self._H()

        def capture(self, src):
            return self._H()

        def get(self, hid):
            return self._H()

    repo = tmp_path / "tb2-repo"
    _populate_fake_tb2_repo(repo)
    ds = load_dataset(f"terminal-bench-2:{repo}", harness_store=_Store())
    assert sum(len(split) for split in (ds.train, ds.val, ds.test)) == 4


def test_cli_tb2_cleanup_subcommand_exits_zero() -> None:
    proc = subprocess.run(
        ["uv", "run", "rho", "tb2-cleanup", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "tb2-cleanup" in proc.stdout or "tb2-cleanup" in proc.stderr or "--all" in proc.stdout


def test_cli_evolve_help_includes_difficulty() -> None:
    proc = subprocess.run(
        ["uv", "run", "rho", "evolve", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--difficulty" in proc.stdout
