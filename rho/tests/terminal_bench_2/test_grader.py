from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from rho.datasets.terminal_bench_2 import container as cn
from rho.datasets.terminal_bench_2.grader import run_tests
from tests.terminal_bench_2.conftest import requires_docker


def _make_task_dir(tmp_path: Path, test_sh_body: str) -> Path:
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test.sh").write_text(test_sh_body, encoding="utf-8")
    return task_dir


@pytest.fixture
def live_container(alpine_image: str) -> str:
    # Unique name per test so xdist workers don't stomp each other.
    name = f"tbench2-grader-test-{uuid4().hex[:8]}"
    cn.rm_force(name)
    cn.ensure_image(alpine_image, "missing")
    cn.start_container(name, alpine_image)
    try:
        cn.exec_script(name, "apk add --no-cache bash", timeout_s=120)
        yield name
    finally:
        cn.rm_force(name)


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_pass(tmp_path: Path, live_container: str) -> None:
    task_dir = _make_task_dir(
        tmp_path,
        "#!/bin/bash\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n",
    )
    grade = run_tests(
        live_container,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=30.0,
    )
    assert grade.passed
    assert grade.score == 1.0
    assert grade.details["reward"] == "1"


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_fail(tmp_path: Path, live_container: str) -> None:
    task_dir = _make_task_dir(
        tmp_path,
        "#!/bin/bash\nmkdir -p /logs/verifier\necho 0 > /logs/verifier/reward.txt\n",
    )
    grade = run_tests(
        live_container,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=30.0,
    )
    assert not grade.passed
    assert grade.details["reward"] == "0"


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_malformed_reward(tmp_path: Path, live_container: str) -> None:
    task_dir = _make_task_dir(
        tmp_path,
        "#!/bin/bash\nmkdir -p /logs/verifier\necho garbage > /logs/verifier/reward.txt\n",
    )
    grade = run_tests(
        live_container,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=30.0,
    )
    assert not grade.passed
    assert grade.details["error"] == "reward_malformed"


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_missing_reward(tmp_path: Path, live_container: str) -> None:
    task_dir = _make_task_dir(tmp_path, "#!/bin/bash\nexit 0\n")
    grade = run_tests(
        live_container,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=30.0,
    )
    assert not grade.passed
    assert grade.details["error"] in {"reward_missing", "verifier_dir_missing"}


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_container_dead(tmp_path: Path, alpine_image: str) -> None:
    name = "tbench2-grader-dead"
    cn.rm_force(name)
    cn.start_container(name, alpine_image)
    cn.rm_force(name)
    task_dir = _make_task_dir(tmp_path, "#!/bin/bash\n")
    grade = run_tests(
        name,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=30.0,
    )
    assert not grade.passed
    assert grade.details["error"] == "container_dead"


@pytest.mark.terminal_bench_2
@requires_docker
def test_grade_verifier_timeout(tmp_path: Path, live_container: str) -> None:
    task_dir = _make_task_dir(tmp_path, "#!/bin/bash\nsleep 9999\n")
    grade = run_tests(
        live_container,
        task_dir,
        artifacts_dir=tmp_path / "art",
        verifier_timeout_s=2.0,
    )
    assert not grade.passed
    assert grade.details["error"] == "verifier_timeout"
    assert not cn.is_running(live_container)
