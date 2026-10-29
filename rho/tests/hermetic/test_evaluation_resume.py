from pathlib import Path
import runpy
import threading
import time
from types import SimpleNamespace

import pytest

from rho.agent.codex import CodexAgent
from rho.protocols import Trajectory
from rho.protocols import Grade


def test_empty_submission_is_terminal_but_infrastructure_errors_still_retry():
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run-swebench-v1-evaluation.py"))
    normalize = module["normalize_terminal_grade"]
    empty = Grade(False, 0.0, {"error": "patch_extraction_failed",
        "message": "trajectory did not modify task/repo and final_message contains no git patch"})
    result = normalize(empty)
    assert not result.passed and result.score == 0
    assert result.details["outcome"] == "no_patch" and "error" not in result.details
    assert empty.details["error"] == "patch_extraction_failed"
    failed = Grade(False, 0.0, {"error": "patch_extraction_failed", "message": "git failed: disk full"})
    assert normalize(failed) is failed


def test_independent_arm_completion_preserves_other_arm(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run-swebench-v1-evaluation.py"))
    import json
    write = module["write_json"]
    publish = module["publish_overall_status"]
    complete = module["baseline_grading_complete"]
    assert not complete(tmp_path)
    write(tmp_path / "baseline/status.json", {"completed": 100})
    write(tmp_path / "baseline/summary.json", {"n": 100, "mean_score": .5})
    write(tmp_path / "final/status.json", {"completed": 3})
    publish(tmp_path, ["baseline", "final"])
    assert complete(tmp_path)
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["phase"] == "running" and state["arms"]["final"]["completed"] == 3
    write(tmp_path / "final/summary.json", {"n": 100, "mean_score": .6})
    publish(tmp_path, ["baseline", "final"])
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "complete"
    assert set(json.loads((tmp_path / "summary.json").read_text())) == {"baseline", "final"}


@pytest.mark.parametrize("stream_logs", [False, True])
def test_none_timeout_reaches_subprocess(tmp_path, monkeypatch, stream_logs):
    config = tmp_path / "config.toml"
    config.write_text('model="test"\n')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("rho.agent.codex.subprocess.run", run)
    agent = CodexAgent(config, default_timeout_s=None, stream_logs=stream_logs,
                       sandbox="danger-full-access", codex_auth_home=tmp_path / "empty-auth")
    tr = agent.run(workspace, "solve", timeout_s=None)
    assert calls[0]["timeout"] is None
    assert not tr.timed_out


def test_resume_preserves_prior_patch_and_new_patch(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run-swebench-v1-evaluation.py"))

    class Task:
        id = "test-task"
        agent_timeout_s = None
        def _prompt(self): return "Test task"
        def materialize(self, dest):
            (dest / "repo").mkdir(parents=True)
            (dest / "prompt.md").write_text(self._prompt())
            (dest / "repo/a.txt").write_text("original")
            (dest / "repo/deleted.txt").write_text("delete me")

    class Harness:
        id = "test-harness"
        def materialize(self, dest):
            (dest / "README.md").write_text("harness")

    task, harness = Task(), Harness()
    ws = tmp_path / "retained"
    (ws / "harness").mkdir(parents=True)
    harness.materialize(ws / "harness")
    task.materialize(ws / "task")
    (ws / "task/repo/a.txt").write_text("previous fix")
    (ws / "task/repo/deleted.txt").unlink()
    (ws / ".rho").mkdir()
    (ws / ".rho/live.stdout.jsonl").write_text('{"type":"prior_event"}\n')

    def run(workspace, instructions, **kwargs):
        assert kwargs["timeout_s"] is None
        assert "has not reset" in instructions
        (workspace / "task/repo/b.txt").write_text("new fix")
        return Trajectory(id="tr", kind="solve", task_id=task.id, harness_id=harness.id,
            instructions=instructions, events=[], final_message="done", stdout="", stderr="",
            workspace_diff={}, workspace_deletions=frozenset(), exit_code=0, wall_time_s=0)

    tr = module["resume_interrupted_solve"](SimpleNamespace(run=run), task, harness, ws, tmp_path / "work", "heldout_baseline")
    assert tr.workspace_diff["task/repo/a.txt"] == b"previous fix"
    assert tr.workspace_diff["task/repo/b.txt"] == b"new fix"
    assert "task/repo/deleted.txt" in tr.workspace_deletions
    assert tr.events[0]["type"] == "prior_event"
    assert tr.events[1]["request_budget_reset"] is False


def test_eight_blocked_graders_do_not_consume_eight_solver_slots():
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run-swebench-v1-evaluation.py"))
    release = threading.Event()
    all_solved = threading.Event()
    eight_graders = threading.Event()
    lock = threading.Lock()
    counts = {"solving": 0, "grading": 0, "max_solving": 0, "max_grading": 0, "solved": 0}
    output = {}

    def solve_one(task):
        with lock:
            counts["solving"] += 1
            counts["max_solving"] = max(counts["max_solving"], counts["solving"])
        time.sleep(0.03)
        with lock:
            counts["solving"] -= 1
            counts["solved"] += 1
            if counts["solved"] == 24:
                all_solved.set()

    def grade_one(task):
        with lock:
            counts["grading"] += 1
            counts["max_grading"] = max(counts["max_grading"], counts["grading"])
            if counts["grading"] == 8:
                eight_graders.set()
        if not release.wait(10):
            raise TimeoutError("test did not release graders")
        with lock:
            counts["grading"] -= 1
        return task

    errors = []
    def run():
        output["records"] = module["run_evaluation_pipeline"](list(range(24)), solve_one, grade_one,
            solve_workers=8, grade_workers=8, on_error=lambda *args: errors.append(args))
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert eight_graders.wait(5)
        assert all_solved.wait(5), "All solver work must finish even while every grader is blocked"
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    assert sorted(output["records"]) == list(range(24))
    assert counts["max_solving"] == counts["max_grading"] == 8


def test_pipeline_continues_after_individual_solve_and_grade_errors():
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run-swebench-v1-evaluation.py"))
    errors = []
    def solve_one(task):
        if task == 0: raise OSError("solve failed")
    def grade_one(task):
        if task == 1: raise OSError("grade failed")
        return task
    records = module["run_evaluation_pipeline"](list(range(4)), solve_one, grade_one,
        solve_workers=2, grade_workers=2, on_error=lambda task, stage, exc: errors.append((task, stage)))
    assert sorted(records) == [2, 3]
    assert set(errors) == {(0, "solve"), (1, "grade")}
