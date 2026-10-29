from __future__ import annotations

import os

import pytest

from rho.datasets.terminal_bench_2 import cleanup
from rho.datasets.terminal_bench_2 import container as cn
from tests.terminal_bench_2.conftest import requires_docker


def test_register_unregister_roundtrip() -> None:
    name = "tbench2-dummy-for-register-test"
    cleanup.register(name)
    assert name in cleanup._owned_names()
    cleanup.unregister(name)
    assert name not in cleanup._owned_names()


def test_pid_alive_for_self() -> None:
    assert cleanup._pid_alive(os.getpid()) is True


def test_pid_alive_for_impossible_pid() -> None:
    assert cleanup._pid_alive(2**30) is False


@pytest.mark.terminal_bench_2
@requires_docker
def test_startup_sweep_removes_orphans(alpine_image: str) -> None:
    orphan = "tbench2-orphan-smoke-test"
    cn.rm_force(orphan)
    try:
        cn.start_container(orphan, alpine_image, labels={"rho-tb2-pid": str(2**30)})
        removed = cleanup.startup_sweep()
        assert removed >= 1
        assert not cn.is_running(orphan)
    finally:
        cn.rm_force(orphan)


@pytest.mark.terminal_bench_2
@requires_docker
def test_startup_sweep_preserves_live_pid_containers(alpine_image: str) -> None:
    live = "tbench2-live-smoke-test"
    cn.rm_force(live)
    try:
        cn.start_container(live, alpine_image, labels={"rho-tb2-pid": str(os.getpid())})
        cleanup.startup_sweep()
        assert cn.is_running(live)
    finally:
        cn.rm_force(live)
