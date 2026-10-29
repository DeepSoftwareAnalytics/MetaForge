from __future__ import annotations

import threading
from types import SimpleNamespace

from rho.datasets.gaia2.dataset import Gaia2Task
from rho.datasets.gaia2.grader import normalize_validation_result
from rho.datasets.gaia2.ingest import Gaia2Row
from rho.datasets.gaia2.runtime import (
    RuntimeHandle,
    _sidecar_ready_timeout_s,
    clear_active,
    get_active,
    set_active,
)
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore


def test_active_runtime_handles_are_thread_local(tmp_path) -> None:
    handle = RuntimeHandle(pid=123, socket_path="/tmp/sidecar.sock")
    set_active("mini/a", handle)
    seen_elsewhere = []

    def read_in_thread() -> None:
        seen_elsewhere.append(get_active("mini/a"))

    thread = threading.Thread(target=read_in_thread)
    thread.start()
    thread.join()

    assert get_active("mini/a") == handle
    assert seen_elsewhere == [None]
    clear_active("mini/a")
    assert get_active("mini/a") is None


def test_sidecar_ready_timeout_defaults_to_cold_start_budget(monkeypatch) -> None:
    monkeypatch.delenv("RHO_GAIA2_SIDECAR_READY_TIMEOUT_S", raising=False)

    assert _sidecar_ready_timeout_s() == 180.0


def test_sidecar_ready_timeout_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("RHO_GAIA2_SIDECAR_READY_TIMEOUT_S", "12.5")

    assert _sidecar_ready_timeout_s() == 12.5


def test_normalize_validation_bool_result() -> None:
    grade = normalize_validation_result(True, scenario_id="scenario-a", config="mini")

    assert grade.passed is True
    assert grade.score == 1.0
    assert grade.details["scenario_id"] == "scenario-a"
    assert grade.details["validation"] is True


def test_normalize_validation_structured_object() -> None:
    result = SimpleNamespace(success=False, reason="missing final message")

    grade = normalize_validation_result(result, scenario_id="scenario-a", config="mini")

    assert grade.passed is False
    assert grade.score == 0.0
    assert grade.details["validation"]["success"] is False
    assert grade.details["validation"]["reason"] == "missing final message"


def test_gaia2_task_grade_reports_no_runtime(tmp_path) -> None:
    harness = FilesystemHarnessStore(tmp_path / "harness_store").empty()
    task = Gaia2Task(
        _row=Gaia2Row(
            id="row-1",
            scenario_id="scenario-a",
            config="mini",
            data={"metadata": {"definition": {"hints": ["Send the note."]}}},
        ),
        _harness=harness,
    )
    trajectory = Trajectory(
        id="traj-1",
        kind="solve",
        task_id=task.id,
        harness_id=harness.id,
        instructions="",
        events=[],
        final_message="",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.0,
    )

    grade = task.grade(trajectory)

    assert grade.passed is False
    assert grade.score == 0.0
    assert grade.details["error"] == "no_runtime"
