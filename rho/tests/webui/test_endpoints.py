from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from rho.webui.data import RunRepository
from rho.webui.server import create_app

from .helpers import find_trajectory_id, require_run


def test_new_rho_inspector_endpoints_smoke() -> None:
    run_dir = require_run("exp-rho-tb2")
    app = create_app(runs_root="runs")

    runs_response = asgi_get_json(app, "/api/runs")
    assert runs_response["status"] == 200
    assert any(run["id"] == "exp-rho-tb2" for run in runs_response["json"])

    selection_response = asgi_get_json(app, "/api/runs/exp-rho-tb2/selection")
    assert selection_response["status"] == 200
    selection = selection_response["json"]
    assert len(selection["selected_task_ids"]) == 10
    assert len(selection["candidates"]) == 30
    assert sum(1 for candidate in selection["candidates"] if candidate["selected"]) == 10

    tasks_response = asgi_get_json(app, "/api/runs/exp-rho-tb2/tasks")
    assert tasks_response["status"] == 200
    tasks = tasks_response["json"]
    assert tasks
    assert all(task["task_id"] != "*" for task in tasks)
    assert any(task["task_id"] == "make-mips-interpreter" for task in tasks)

    task_response = asgi_get_json(app, "/api/runs/exp-rho-tb2/tasks/make-mips-interpreter")
    assert task_response["status"] == 200
    task = task_response["json"]
    assert task["task_id"] == "make-mips-interpreter"
    assert task["trajectories"]
    assert all("score" in trajectory for trajectory in task["trajectories"])
    assert task["selection"]["selected"] is True

    harness_response = asgi_get_json(app, "/api/runs/exp-rho-tb2/harness/h_edccd200e74a")
    assert harness_response["status"] == 200
    assert any(entry["path"] == "README.md" for entry in harness_response["json"]["files"])

    composite_harness_response = asgi_get_json(app, "/api/runs/exp-rho-tb2/harness/a->b")
    assert composite_harness_response["status"] == 404

    score_traj_id = find_trajectory_id(run_dir, stage="final_val_grade")
    score_response = asgi_get_json(app, f"/api/runs/exp-rho-tb2/trajectories/{score_traj_id}/score")
    assert score_response["status"] == 200
    assert score_response["json"] == RunRepository(run_dir.parent).get_trajectory_score(run_dir.name, score_traj_id)

    trace_traj_id = find_trajectory_id(run_dir, stage="round_solve_before")
    trace_response = asgi_get_json(app, f"/api/runs/exp-rho-tb2/trajectories/{trace_traj_id}/trace")
    assert trace_response["status"] == 200
    assert trace_response["json"]["summary"]["step_count"] > 0


def test_run_and_harness_ids_reject_path_traversal() -> None:
    run_dir = require_run("exp-rho-tb2")
    repository = RunRepository(run_dir.parent)

    assert repository.get_run("exp-rho-tb2")["id"] == "exp-rho-tb2"
    with pytest.raises(KeyError):
        repository.get_run("..")
    with pytest.raises(KeyError):
        repository.get_harness("exp-rho-tb2", "..")


def asgi_get_json(app: Any, path: str) -> dict[str, Any]:
    async def request() -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return {"status": status, "json": json.loads(body.decode("utf-8")) if body else None}

    return asyncio.run(request())
