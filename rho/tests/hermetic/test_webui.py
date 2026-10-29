from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rho.webui.data import RunRepository
from rho.webui.server import create_app
from rho.webui.trace import build_trace_payload


def test_build_trace_payload_summarizes_large_command_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = "\n".join(
        [
            "harness/conv-48/session_29.md:13:We tried a scuba diving lesson last Friday",
            "harness/conv-48/session_29.md:14:We found a cool dive spot we can explore together",
            "task/prompt.md:2:Where did Jolene and her partner find a cool diving spot?",
        ]
    )
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.started",
            "item": {
                "id": "cmd_1",
                "type": "command_execution",
                "command": "/bin/bash -lc 'rg -n \"diving\" harness task -S'",
                "aggregated_output": "",
                "status": "in_progress",
                "exit_code": None,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "cmd_1",
                "type": "command_execution",
                "command": "/bin/bash -lc 'rg -n \"diving\" harness task -S'",
                "aggregated_output": output,
                "status": "completed",
                "exit_code": 0,
            },
        },
        {
            "type": "item.started",
            "item": {
                "id": "todo_1",
                "type": "todo_list",
                "items": [
                    {"text": "Inspect prompt", "completed": False},
                    {"text": "Compare traces", "completed": False},
                ],
            },
        },
        {
            "type": "item.updated",
            "item": {
                "id": "todo_1",
                "type": "todo_list",
                "items": [
                    {"text": "Inspect prompt", "completed": True},
                    {"text": "Compare traces", "completed": False},
                ],
            },
        },
        {
            "type": "item.started",
            "item": {
                "id": "files_1",
                "type": "file_change",
                "status": "in_progress",
                "changes": [
                    {
                        "path": str(run_dir / "workdir" / "opt_123" / "harness" / "ANSWERING_GUIDELINES.md"),
                        "kind": "add",
                    },
                    {
                        "path": str(run_dir / "workdir" / "opt_123" / "harness" / "conv-48" / "INDEX.md"),
                        "kind": "update",
                    },
                ],
            },
        },
        {"type": "raw_stderr", "line": "Reading additional input from stdin..."},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "cached_input_tokens": 25, "output_tokens": 50},
        },
    ]

    payload = build_trace_payload(events, run_dir=run_dir)

    assert payload["summary"]["command_count"] == 1
    command_step = next(step for step in payload["steps"] if step["kind"] == "command")
    assert command_step["preview"]["mode"] == "search_matches"
    assert command_step["summary"] == "3 matches across 2 files."
    assert "full_output" not in command_step
    todo_step = next(step for step in payload["steps"] if step["kind"] == "todo")
    assert todo_step["metrics"]["completed_count"] == 1
    file_change_step = next(step for step in payload["steps"] if step["kind"] == "file_change")
    assert file_change_step["preview"]["paths"][0]["path"].startswith("workdir/")
    usage_step = next(step for step in payload["steps"] if step["kind"] == "usage")
    assert usage_step["metrics"]["output_tokens"] == 50


def test_run_repository_fallbacks_fill_missing_manifest_usage_and_round_diff(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260414-000000-demo"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "rounds" / "round_0").mkdir(parents=True)
    (run_dir / "trajectories" / "traj_demo").mkdir(parents=True)
    (run_dir / "harness" / "h_before").mkdir(parents=True)
    (run_dir / "harness" / "h_after").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"dataset_spec": "demo"}), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"cwd": "/tmp/demo"}), encoding="utf-8")
    (run_dir / "reports" / "summary.json").write_text(
        json.dumps(
            {
                "initial_harness_id": "h_before",
                "final_harness_id": "h_after",
                "final_val": {"mean_score": 0.75, "n": 2},
                "rounds": [{"round_ix": 0}],
            }
        ),
        encoding="utf-8",
    )
    round_dir = run_dir / "rounds" / "round_0"
    (round_dir / "input_harness_id").write_text("h_before", encoding="utf-8")
    (round_dir / "candidate_harness_id").write_text("h_after", encoding="utf-8")
    (round_dir / "accepted").write_text("true", encoding="utf-8")
    (round_dir / "mean_score").write_text("1.2500", encoding="utf-8")
    (round_dir / "solve_before_traj_ids.json").write_text(json.dumps([["traj_demo"]]), encoding="utf-8")
    (round_dir / "solve_after_traj_ids.json").write_text(json.dumps([]), encoding="utf-8")
    (round_dir / "eval_traj_ids.json").write_text(json.dumps([]), encoding="utf-8")
    (round_dir / "scores.json").write_text(json.dumps([]), encoding="utf-8")
    (run_dir / "harness" / "h_before" / "README.md").write_text("before\n", encoding="utf-8")
    (run_dir / "harness" / "h_after" / "README.md").write_text("after\n", encoding="utf-8")

    traj_dir = run_dir / "trajectories" / "traj_demo"
    (traj_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": "traj_demo",
                "kind": "solve",
                "task_id": "task_001",
                "harness_id": "h_before",
                "exit_code": 0,
                "timed_out": False,
                "wall_time_s": 12.5,
                "stage": "round_solve_before",
            }
        ),
        encoding="utf-8",
    )
    (traj_dir / "instructions.md").write_text("Use the harness.\n", encoding="utf-8")
    (traj_dir / "final_message.txt").write_text("ANSWER:\nA demo answer.\n", encoding="utf-8")
    (traj_dir / "stdout.log").write_text("", encoding="utf-8")
    (traj_dir / "stderr.log").write_text("", encoding="utf-8")
    (traj_dir / "workspace_diff").mkdir()
    (traj_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "cmd_1",
                            "type": "command_execution",
                            "command": "/bin/bash -lc 'sed -n \"1,20p\" task/prompt.md'",
                            "aggregated_output": "prompt line 1\nprompt line 2\n",
                            "status": "completed",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 5},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 2, "cached_input_tokens": 1, "output_tokens": 7},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    repository = RunRepository(run_dir.parent)
    run_payload = repository.get_run(run_dir.name)
    round_payload = repository.get_round(run_dir.name, 0)
    trace_payload = repository.get_trace(run_dir.name, "traj_demo")

    assert run_payload["manifest"]["generated_fallback"] is True
    assert run_payload["usage_summary"]["generated_fallback"] is True
    assert run_payload["usage_summary"]["overall"]["input_tokens"] == 12
    assert run_payload["usage_summary"]["overall"]["cached_input_tokens"] == 4
    assert run_payload["usage_summary"]["overall"]["output_tokens"] == 12
    assert round_payload["strategy"] == "diagnosis"
    assert round_payload["diagnoses"] == []
    assert round_payload["diagnose_trajectories"] == []
    assert "a/README.md" in round_payload["candidate_harness_diff"]
    assert trace_payload["summary"]["command_count"] == 1


def test_create_app_serves_api_without_static_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20260414-000000-demo"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"dataset_spec": "demo"}), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"cwd": "/tmp/demo"}), encoding="utf-8")
    (run_dir / "reports" / "summary.json").write_text(json.dumps({"rounds": []}), encoding="utf-8")
    app = create_app(runs_root=run_dir.parent)
    client = TestClient(app)

    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.json()[0]["id"] == run_dir.name

    root = client.get("/")
    assert root.status_code in {200, 503}
