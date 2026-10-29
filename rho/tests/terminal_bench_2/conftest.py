from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()
TB2_REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"
TB2_PIN_SHA = "53ff2b87d621bdb97b455671f2bd9728b7d86c11"

requires_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="docker daemon not available",
)


@pytest.fixture(scope="session")
def alpine_image() -> str:
    """A tiny public image used for Docker-backed smoke tests."""
    return "alpine:3.19"


@pytest.fixture
def fake_task_dir(tmp_path: Path) -> Path:
    """Minimal TB2-shaped task directory for hermetic dataset tests."""
    d = tmp_path / "tb2-tasks" / "fake-task"
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()
    (d / "environment" / "Dockerfile").write_text("FROM alpine:3.19\n", encoding="utf-8")
    (d / "instruction.md").write_text("# Fake Task\n\nDo nothing.\n", encoding="utf-8")
    (d / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    (d / "solution" / "solve.sh").write_text(":\n", encoding="utf-8")
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
cpus = 1
memory = "128M"
''',
        encoding="utf-8",
    )
    return d


@pytest.fixture
def fake_repo_dir(fake_task_dir: Path) -> Path:
    repo = fake_task_dir.parent
    (repo / "README.md").write_text("# Terminal-Bench 2 fake fixture\n", encoding="utf-8")
    return repo


@pytest.fixture(scope="session")
def tb2_repo(tmp_path_factory) -> Path:
    """Clone the real TB2 repo at the pinned SHA once per session."""
    cache = Path.home() / ".cache" / "rho" / "tb2-repo"
    if not (cache / ".git").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", TB2_REPO_URL, str(cache)],
            check=True,
            timeout=600,
        )
    subprocess.run(
        ["git", "-C", str(cache), "fetch", "--quiet"],
        check=False,
        timeout=300,
    )
    subprocess.run(
        ["git", "-C", str(cache), "checkout", "--quiet", TB2_PIN_SHA],
        check=True,
        timeout=60,
    )
    return cache
