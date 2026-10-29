from __future__ import annotations

import time
from pathlib import Path

import pytest

from rho.datasets.terminal_bench_2 import container as cn
from rho.datasets.terminal_bench_2.runtime import TerminalBench2RuntimeSession
from rho.datasets.terminal_bench_2.task_toml import TaskToml
from tests.terminal_bench_2.conftest import requires_docker


class _FakeTask:
    def __init__(self, tid: str, image: str, agent_timeout_sec: float = 300.0) -> None:
        self.id = tid
        self._docker_pull = "missing"
        self._config = TaskToml(
            difficulty="easy",
            category="misc",
            tags=(),
            agent_timeout_sec=agent_timeout_sec,
            verifier_timeout_sec=60.0,
            docker_image=image,
            cpus=None,
            memory="128M",
            build_timeout_sec=None,
        )


@pytest.mark.terminal_bench_2
@requires_docker
def test_runtime_session_starts_and_tears_down(alpine_image: str, tmp_path: Path) -> None:
    task = _FakeTask("test-task-id", alpine_image)
    session = TerminalBench2RuntimeSession(task, tmp_path)
    with session as rt:
        assert rt.container_name.startswith("tbench2-")
        assert cn.is_running(rt.container_name)
        (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
        proc = cn.exec_script(rt.container_name, "cat /host-ws/hello.txt", timeout_s=15)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "hi"
        sidecar = tmp_path / ".tb2" / "container_name"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip() == rt.container_name
        name = rt.container_name
    assert not cn.is_running(name)


@pytest.mark.terminal_bench_2
@requires_docker
def test_runtime_session_watchdog_kills_on_timeout(
    alpine_image: str,
    tmp_path: Path,
) -> None:
    task = _FakeTask("timeout-task", alpine_image, agent_timeout_sec=2.0)
    session = TerminalBench2RuntimeSession(task, tmp_path)
    with session as rt:
        name = rt.container_name
        assert cn.is_running(name)
        time.sleep(4)
        assert not cn.is_running(name)
        assert session.timed_out is True


@pytest.mark.terminal_bench_2
@requires_docker
def test_runtime_session_cleans_on_exception(alpine_image: str, tmp_path: Path) -> None:
    task = _FakeTask("exc-task", alpine_image)
    session = TerminalBench2RuntimeSession(task, tmp_path)
    name_captured: dict[str, str] = {}
    with pytest.raises(RuntimeError):
        with session as rt:
            name_captured["n"] = rt.container_name
            raise RuntimeError("simulated")
    assert not cn.is_running(name_captured["n"])


@pytest.mark.terminal_bench_2
@requires_docker
def test_parallel_sessions_same_task_isolated(alpine_image: str, tmp_path: Path) -> None:
    import threading

    task = _FakeTask("parallel-task", alpine_image)
    results: list[str] = []
    lock = threading.Lock()

    def run_one(slot: int) -> None:
        workdir = tmp_path / f"slot{slot}"
        workdir.mkdir()
        session = TerminalBench2RuntimeSession(task, workdir)
        with session as rt:
            with lock:
                results.append(rt.container_name)

    threads = [threading.Thread(target=run_one, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 3
    for name in results:
        assert not cn.is_running(name)
