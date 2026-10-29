from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from rho.datasets.terminal_bench_2 import container
from rho.protocols import Grade


logger = logging.getLogger(__name__)


def run_tests(
    container_name: str,
    task_dir: Path,
    *,
    artifacts_dir: Path,
    verifier_timeout_s: float,
) -> Grade:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not container.is_running(container_name):
        return _fail(container_name, artifacts_dir, "container_dead")

    try:
        container.cp_to(
            container_name,
            str(task_dir / "tests") + "/.",
            "/tests/",
        )
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "cp_tests_failed", message=str(exc))

    try:
        container.exec_script(container_name, "mkdir -p /tests /logs/verifier", timeout_s=30)
        container.exec_script(container_name, "chmod +x /tests/test.sh", timeout_s=30)
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "setup_failed", message=str(exc))

    try:
        proc = container.exec_script(
            container_name,
            "/tests/test.sh",
            timeout_s=verifier_timeout_s,
        )
    except subprocess.TimeoutExpired:
        try:
            container.kill(container_name)
        except Exception:
            pass
        return _fail(container_name, artifacts_dir, "verifier_timeout")
    except Exception as exc:
        return _fail(container_name, artifacts_dir, "verifier_crashed", message=str(exc))

    _write_log(artifacts_dir / "test_stdout.log", proc.stdout)
    _write_log(artifacts_dir / "test_stderr.log", proc.stderr)

    try:
        container.cp_from(container_name, "/logs/verifier/.", artifacts_dir / "verifier")
    except Exception as exc:
        return _fail(
            container_name,
            artifacts_dir,
            "verifier_dir_missing",
            message=str(exc),
            docker_exit_code=proc.returncode,
        )

    reward_path = artifacts_dir / "verifier" / "reward.txt"
    if not reward_path.exists():
        return _fail(
            container_name,
            artifacts_dir,
            "reward_missing",
            docker_exit_code=proc.returncode,
        )

    reward = reward_path.read_text(encoding="utf-8").strip()
    details: dict[str, object] = {
        "reward": reward,
        "artifacts_dir": str(artifacts_dir),
        "container_name": container_name,
        "docker_exit_code": proc.returncode,
    }

    if reward not in {"0", "1"}:
        details["error"] = "reward_malformed"
        return Grade(passed=False, score=0.0, details=details)

    ctrf_path = artifacts_dir / "verifier" / "ctrf.json"
    if ctrf_path.exists():
        details["ctrf_summary"] = _summarize_ctrf(ctrf_path)

    passed = reward == "1"
    return Grade(passed=passed, score=1.0 if passed else 0.0, details=details)


def _fail(
    container_name: str,
    artifacts_dir: Path,
    error: str,
    *,
    message: str | None = None,
    docker_exit_code: int | None = None,
) -> Grade:
    details: dict[str, object] = {
        "error": error,
        "artifacts_dir": str(artifacts_dir),
        "container_name": container_name,
    }
    if message is not None:
        details["message"] = message
    if docker_exit_code is not None:
        details["docker_exit_code"] = docker_exit_code
    return Grade(passed=False, score=0.0, details=details)


def _write_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def _summarize_ctrf(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "ctrf_parse_failed"}
    results = data.get("results") or {}
    tests = results.get("tests") or []
    return {
        "total": len(tests),
        "passed": sum(1 for test in tests if test.get("status") == "passed"),
        "failed": sum(1 for test in tests if test.get("status") == "failed"),
        "names": [test.get("name") for test in tests][:50],
    }
