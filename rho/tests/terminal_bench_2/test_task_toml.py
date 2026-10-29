from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.terminal_bench_2.task_toml import TaskToml, load_task_toml


SAMPLE_TOML = '''
version = "1.0"
[metadata]
author_name = "jvpoulos"
author_email = "poulos@berkeley.edu"
difficulty = "medium"
category = "scientific-computing"
tags = ["applied-statistics", "Bayesian-inference"]
expert_time_estimate_min = 180.0
junior_time_estimate_min = 480.0
[verifier]
timeout_sec = 900.0
[agent]
timeout_sec = 900.0
[environment]
build_timeout_sec = 600.0
docker_image = "alexgshaw/adaptive-rejection-sampler:20251031"
cpus = 1
memory = "2G"
storage = "10G"
'''


def test_load_task_toml_full_fields(tmp_path: Path) -> None:
    path = tmp_path / "task.toml"
    path.write_text(SAMPLE_TOML, encoding="utf-8")
    toml = load_task_toml(path)
    assert isinstance(toml, TaskToml)
    assert toml.difficulty == "medium"
    assert toml.category == "scientific-computing"
    assert toml.tags == ("applied-statistics", "Bayesian-inference")
    assert toml.agent_timeout_sec == 900.0
    assert toml.verifier_timeout_sec == 900.0
    assert toml.docker_image == "alexgshaw/adaptive-rejection-sampler:20251031"
    assert toml.cpus == 1
    assert toml.memory == "2G"
    assert toml.build_timeout_sec == 600.0


def test_load_task_toml_missing_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "task.toml"
    path.write_text(
        '''
version = "1.0"
[metadata]
difficulty = "easy"
category = "misc"
tags = []
[verifier]
timeout_sec = 60.0
[agent]
timeout_sec = 60.0
[environment]
docker_image = "alpine:3.19"
''',
        encoding="utf-8",
    )
    toml = load_task_toml(path)
    assert toml.difficulty == "easy"
    assert toml.cpus is None
    assert toml.memory is None
    assert toml.build_timeout_sec is None


def test_load_task_toml_rejects_missing_required(tmp_path: Path) -> None:
    path = tmp_path / "task.toml"
    path.write_text(
        '''
version = "1.0"
[metadata]
difficulty = "easy"
[environment]
docker_image = "alpine:3.19"
''',
        encoding="utf-8",
    )
    with pytest.raises((KeyError, ValueError)):
        load_task_toml(path)
