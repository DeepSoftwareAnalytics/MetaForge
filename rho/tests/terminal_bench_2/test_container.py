from __future__ import annotations

import os
import platform
from unittest.mock import patch

import pytest

from rho.datasets.terminal_bench_2 import container
from tests.terminal_bench_2.conftest import requires_docker


def test_container_platform_arm64_forces_linux_amd64() -> None:
    with patch.object(platform, "machine", return_value="arm64"):
        assert container.container_platform() == "linux/amd64"


def test_container_platform_aarch64_forces_linux_amd64() -> None:
    with patch.object(platform, "machine", return_value="aarch64"):
        assert container.container_platform() == "linux/amd64"


def test_container_platform_x86_returns_none() -> None:
    with patch.object(platform, "machine", return_value="x86_64"):
        assert container.container_platform() is None


@pytest.fixture
def unique_name() -> str:
    return f"tb2-test-{os.getpid()}-{id(object())}"


@pytest.mark.terminal_bench_2
@requires_docker
def test_start_and_remove_container(alpine_image: str, unique_name: str) -> None:
    try:
        container.ensure_image(alpine_image, "missing")
        container.start_container(unique_name, alpine_image)
        assert container.is_running(unique_name)
        proc = container.exec_script(unique_name, "echo hi", timeout_s=15)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "hi"
    finally:
        container.rm_force(unique_name)
    assert not container.is_running(unique_name)


@pytest.mark.terminal_bench_2
@requires_docker
def test_rm_force_is_idempotent(unique_name: str) -> None:
    container.rm_force(unique_name)
    container.rm_force(unique_name)


@pytest.mark.terminal_bench_2
@requires_docker
def test_label_filter(alpine_image: str, unique_name: str) -> None:
    try:
        container.start_container(
            unique_name,
            alpine_image,
            labels={"rho-tb2-pid": "99999"},
        )
        found = container.list_with_label("rho-tb2-pid")
        names = [name for name, _ in found]
        assert unique_name in names
        pid_for_ours = dict(found)[unique_name]
        assert pid_for_ours == "99999"
    finally:
        container.rm_force(unique_name)
