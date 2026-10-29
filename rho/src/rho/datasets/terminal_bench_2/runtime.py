from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import uuid4

from rho.datasets.terminal_bench_2 import cleanup, container

if TYPE_CHECKING:
    from rho.datasets.terminal_bench_2.dataset import TerminalBench2Task


logger = logging.getLogger(__name__)


HOST_WS_PATH = "/host-ws"


@dataclass(frozen=True)
class RuntimeHandle:
    container_name: str
    host_scratch: str


_THREAD_LOCAL = threading.local()


def set_active(task_id: str, container_name: str) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        mapping = {}
        _THREAD_LOCAL.mapping = mapping
    mapping[task_id] = container_name


def clear_active(task_id: str) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is not None:
        mapping.pop(task_id, None)


def get_active(task_id: str) -> str | None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        return None
    return mapping.get(task_id)


_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_.-]")


def _safe_task_id(task_id: str) -> str:
    return _NAME_SAFE.sub("_", task_id)[:40]


class TerminalBench2RuntimeSession:
    def __init__(self, task: "TerminalBench2Task", workdir: Path) -> None:
        self._task = task
        self._workdir = Path(workdir)
        self._container_name: str | None = None
        self._watchdog: threading.Timer | None = None
        self._watchdog_fired = threading.Event()
        self.timed_out = False

    def __enter__(self) -> RuntimeHandle:
        suffix = uuid4().hex[:8]
        name = f"tbench2-{_safe_task_id(self._task.id)}-{suffix}"
        self._container_name = name

        cleanup.register(name)

        image = self._task._config.docker_image
        policy = self._task._docker_pull
        container.ensure_image(image, policy)

        volumes = [
            container.Volume(
                host=self._workdir.resolve(),
                container=HOST_WS_PATH,
                mode="rw",
            ),
        ]
        labels = {"rho-tb2-pid": str(os.getpid())}
        try:
            container.start_container(
                name,
                image,
                memory=self._task._config.memory,
                cpus=self._task._config.cpus,
                volumes=volumes,
                labels=labels,
            )
        except Exception:
            cleanup.unregister(name)
            raise

        sidecar_dir = self._workdir / ".tb2"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        (sidecar_dir / "container_name").write_text(name, encoding="utf-8")

        self._watchdog = threading.Timer(self._task._config.agent_timeout_sec, self._on_timeout)
        self._watchdog.daemon = True
        self._watchdog.start()

        return RuntimeHandle(container_name=name, host_scratch=HOST_WS_PATH)

    def _on_timeout(self) -> None:
        self._watchdog_fired.set()
        self.timed_out = True
        name = self._container_name
        if name is not None:
            try:
                container.kill(name)
            except Exception:
                logger.exception("watchdog kill failed for %s", name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        watchdog = self._watchdog
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join(timeout=1)
        name = self._container_name
        if name is not None:
            # Container runs as root; chown mounted files back to the host uid/gid
            # so the outer tempdir cleanup can remove them.
            try:
                if container.is_running(name):
                    container.exec_script(
                        name,
                        f"chown -R {os.getuid()}:{os.getgid()} {HOST_WS_PATH} 2>/dev/null || true",
                        timeout_s=60,
                    )
            except Exception:
                logger.exception("chown-back failed for %s (non-fatal)", name)
            try:
                container.rm_force(name)
            except Exception:
                logger.exception("rm_force failed for %s", name)
            cleanup.unregister(name)
