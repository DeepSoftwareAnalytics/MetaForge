from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from rho.datasets.gaia2.dispatcher import render_dispatcher

if TYPE_CHECKING:
    from rho.datasets.gaia2.dataset import Gaia2Task


@dataclass(frozen=True)
class RuntimeHandle:
    pid: int
    socket_path: str


_THREAD_LOCAL = threading.local()


def set_active(task_id: str, handle: RuntimeHandle) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        mapping = {}
        _THREAD_LOCAL.mapping = mapping
    mapping[task_id] = handle


def clear_active(task_id: str) -> None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is not None:
        mapping.pop(task_id, None)


def get_active(task_id: str) -> RuntimeHandle | None:
    mapping = getattr(_THREAD_LOCAL, "mapping", None)
    if mapping is None:
        return None
    return mapping.get(task_id)


def rpc(handle: RuntimeHandle, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(handle.socket_path)
        sock.sendall(struct.pack(">I", len(raw)) + raw)
        header = _recv_exact(sock, 4)
        size = struct.unpack(">I", header)[0]
        body = _recv_exact(sock, size)
    return json.loads(body.decode("utf-8"))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("GAIA-2 sidecar closed connection early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_handle(handle_file: Path, *, fallback_pid: int) -> RuntimeHandle | None:
    """Parse the sidecar handle once it exists; None until the sidecar is ready.

    The sidecar writes handle.json only after binding and listening, so a
    parseable handle carrying a socket_path means the sidecar is serving.
    The socket itself lives at a short tmp path (AF_UNIX length limit), so
    the path is discovered here rather than assumed from the workdir.
    """
    if not handle_file.exists():
        return None
    try:
        payload = json.loads(handle_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # handle.json is mid-write; retry
    socket_path = payload.get("socket_path")
    if not socket_path:
        return None
    return RuntimeHandle(
        pid=int(payload.get("pid", fallback_pid)),
        socket_path=str(socket_path),
    )


def _sidecar_ready_timeout_s() -> float:
    raw = os.getenv("RHO_GAIA2_SIDECAR_READY_TIMEOUT_S")
    if raw is None:
        return 180.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 180.0


def _sidecar_start_attempts() -> int:
    """How many times to (re)spawn the sidecar before giving up.

    A sidecar that exits early or fails to bind within the ready timeout is
    treated as transient flakiness and respawned, rather than aborting the
    whole run on the first miss.
    """
    raw = os.getenv("RHO_GAIA2_SIDECAR_START_ATTEMPTS")
    if raw is None:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


class Gaia2RuntimeSession:
    def __init__(self, task: "Gaia2Task", workdir: Path) -> None:
        self._task = task
        self._workdir = Path(workdir)
        self._proc: subprocess.Popen[str] | None = None
        self._handle: RuntimeHandle | None = None

    def __enter__(self) -> RuntimeHandle:
        runtime_dir = self._workdir / ".gaia2"
        tools_dir = self._workdir / "tools"
        state_dir = self._workdir / ".gaia2_state"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        tools_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        scenario_file = runtime_dir / "scenario.json"
        scenario_file.write_text(
            json.dumps(self._task.scenario_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dispatcher_path = tools_dir / "are.py"
        dispatcher_path.write_text(render_dispatcher(), encoding="utf-8")
        dispatcher_path.chmod(0o755)
        catalog_path = tools_dir / "catalog.json"
        if not catalog_path.exists():
            catalog_path.write_text("{}\n", encoding="utf-8")

        handle_file = runtime_dir / "handle.json"
        log_path = runtime_dir / "sidecar.log"
        cmd = [
            sys.executable,
            "-m",
            "rho.datasets.gaia2.sidecar",
            "--scenario-file",
            str(scenario_file),
            "--workdir",
            str(self._workdir),
        ]
        attempts = _sidecar_start_attempts()
        for _ in range(attempts):
            handle = self._spawn_and_wait(handle_file, log_path, cmd)
            if handle is not None:
                self._handle = handle
                set_active(self._task.id, self._handle)
                return self._handle
            # Sidecar exited early or never became ready; kill it and retry.
            self._terminate_proc()
        raise RuntimeError(
            f"GAIA-2 sidecar did not become ready after {attempts} attempt(s); "
            f"see {log_path}"
        )

    def _spawn_and_wait(
        self, handle_file: Path, log_path: Path, cmd: list[str]
    ) -> RuntimeHandle | None:
        """Spawn one sidecar and wait for it to bind; None if it failed."""
        try:
            handle_file.unlink()
        except FileNotFoundError:
            pass
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
        finally:
            log_handle.close()
        deadline = time.monotonic() + _sidecar_ready_timeout_s()
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                return None  # exited before becoming ready
            handle = _read_handle(handle_file, fallback_pid=self._proc.pid)
            if handle is not None:
                return handle
            time.sleep(0.1)
        return None  # timed out

    def _terminate_proc(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._proc = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        clear_active(self._task.id)
        handle = self._handle
        if handle is not None:
            try:
                rpc(handle, {"method": "shutdown"})
            except Exception:
                pass
        self._terminate_proc()
