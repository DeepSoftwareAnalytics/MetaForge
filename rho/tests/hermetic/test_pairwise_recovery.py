import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery_module():
    path = Path(__file__).resolve().parents[2] / "scripts/resume-swebench-pairwise.py"
    spec = importlib.util.spec_from_file_location("pairwise_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_fixture():
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return SimpleNamespace(write_json=write_json, log=lambda message: None)


def test_rate_limited_pair_waits_without_blocking_other_pairs_or_becoming_zero(recovery_module, tmp_path):
    runtime = runtime_fixture()
    pairs = [{"key": key} for key in ("a", "b", "kept")]
    kept = {"key": "kept", "usable": True, "value": 7}
    runtime.write_json(tmp_path / "pairs/kept.json", kept)
    reductions, calls = [], []
    gate = SimpleNamespace(limit=1, reduce=reductions.append)

    def attempt(pair, number):
        key = pair["key"]
        calls.append((key, number))
        if key == "a" and number == 1:
            return {**pair, "usable": False, "value": 0, "rate_limited": True, "eval_trajectory_id": "limited"}
        if key == "b":
            assert not (tmp_path / "pairs/a.json").exists()
            state = json.loads((tmp_path / "retry_queue.json").read_text())["a"]
            assert state["state"] == "waiting" and state["not_before"] > 0
        return {**pair, "usable": True, "value": 3, "rate_limited": False}

    records = recovery_module.run_retry_queue(pairs, attempt, tmp_path, runtime, gate, 1, retry_base_s=0.05)
    assert calls == [("a", 1), ("b", 1), ("a", 2)]
    assert reductions and records["a"]["value"] == 3 and records["kept"] == kept
    assert json.loads((tmp_path / "status.json").read_text())["completed"] == 3


def test_retry_queue_resumes_attempt_count_and_preserves_success_after_internal_retry(recovery_module, tmp_path):
    runtime = runtime_fixture()
    runtime.write_json(tmp_path / "retry_queue.json", {
        "a": {"state": "running", "attempts": 3, "not_before": 0, "rate_limited_attempts": 2}})
    calls = []
    gate = SimpleNamespace(limit=1, reduce=lambda reason: pytest.fail("Successful result must not be rerun"))

    def attempt(pair, number):
        calls.append(number)
        return {**pair, "usable": True, "value": 2, "rate_limited": True}

    recovery_module.run_retry_queue([{"key": "a"}], attempt, tmp_path, runtime, gate, 1)
    assert calls == [4]
    assert json.loads((tmp_path / "retry_queue.json").read_text())["a"]["state"] == "complete"


def test_rate_limit_detection_ignores_quoted_solver_output(recovery_module):
    pattern = re.compile("rate limit exceeded", re.I)
    tr = SimpleNamespace(stderr="", events=[{
        "type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "rate limit exceeded"}}])
    assert not recovery_module.has_rate_limit(tr, pattern)
    tr.events.append({"type": "error", "message": "Reconnecting: rate limit exceeded"})
    assert recovery_module.has_rate_limit(tr, pattern)


def test_finished_attempt_survives_scheduler_interruption_and_worker_error_does_not_block_queue(recovery_module, tmp_path):
    runtime = runtime_fixture()
    recovered = {"key": "kept", "usable": True, "value": 6}
    runtime.write_json(tmp_path / "attempts/kept/0001.json", recovered)
    gate = SimpleNamespace(limit=1, reduce=lambda reason: None)
    calls = []

    def attempt(pair, number):
        calls.append((pair["key"], number))
        if pair["key"] == "a" and number == 1:
            raise OSError("temporary workspace failure")
        return {**pair, "usable": True, "value": 4}

    records = recovery_module.run_retry_queue([{"key": k} for k in ("a", "b", "kept")],
        attempt, tmp_path, runtime, gate, 1, retry_base_s=0.05)
    assert calls == [("a", 1), ("b", 1), ("a", 2)]
    assert records["kept"] == recovered
