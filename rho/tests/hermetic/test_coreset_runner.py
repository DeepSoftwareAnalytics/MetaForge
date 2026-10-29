import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import pytest

from rho.agent.codex import CodexAgent


@pytest.mark.parametrize("message", [
    "Reconnecting... 1/3 (rate limit exceeded: Request rate increased too quickly.)",
    "Unexpected status code: 429",
    "Too many requests",
    "RateLimitError",
])
def test_rate_limit_provider_messages(message):
    assert load_runner().RATE_LIMIT.search(message)


def test_retry_retains_completed_probes_and_resumes_without_resetting(tmp_path):
    runner = load_runner()
    source, dest = tmp_path / "source", tmp_path / "dest"
    for task_id, timed_out in [("done", False), ("timeout", True)]:
        path = source / "trajectories" / task_id
        path.mkdir(parents=True)
        (path / "meta.json").write_text(json.dumps({"timed_out": timed_out, "exit_code": -1 if timed_out else 0}))
        (path / "evidence.txt").write_text(task_id)
    (source / "probe_index.json").write_text(json.dumps({"task1": "done", "task2": "timeout"}))
    runner.seed_timeout_retry(source, dest, ["task1", "task2"])
    assert json.loads((dest / "probe_index.json").read_text()) == {"task1": "done"}
    assert (dest / "trajectories/done/evidence.txt").read_text() == "done"
    assert not (dest / "trajectories/timeout").exists()
    plan = json.loads((dest / "retry_plan.json").read_text())
    assert plan["retry_previous_trajectory_ids"] == {"task2": "timeout"}
    assert (source / "trajectories/timeout/evidence.txt").read_text() == "timeout"
    updated = {"task1": "done", "task2": "new_attempt"}
    (dest / "probe_index.json").write_text(json.dumps(updated))
    runner.seed_timeout_retry(source, dest, ["task1", "task2"])
    assert json.loads((dest / "probe_index.json").read_text()) == updated


def load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts/select-swebench-v1-coreset.py"
    spec = importlib.util.spec_from_file_location("coreset_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reuse_probes_requires_complete_pool_and_preserves_source(tmp_path):
    runner = load_runner()
    source, dest = tmp_path / "source", tmp_path / "destination"
    tr = source / "trajectories" / "tr1"
    tr.mkdir(parents=True)
    (tr / "events.jsonl").write_text('evidence\n')
    (source / "probe_index.json").write_text('{"task1":"tr1"}')
    with pytest.raises(AssertionError, match="All candidate"):
        runner.seed_existing_probes(source, dest, ["task1", "task2"])
    assert not (dest / "probe_index.json").exists()
    runner.seed_existing_probes(source, dest, ["task1"])
    assert (dest / "trajectories/tr1/events.jsonl").read_text() == 'evidence\n'
    assert json.loads((dest / "probe_index.json").read_text()) == {"task1": "tr1"}
    assert (tr / "events.jsonl").read_text() == 'evidence\n'


def test_exports_use_manifest_language_when_solver_row_omits_it(tmp_path, monkeypatch):
    runner = load_runner()
    run, splits = tmp_path / "run", tmp_path / "splits"
    (run / "selector_calls").mkdir(parents=True)
    splits.mkdir()
    monkeypatch.setattr(runner, "RUN", run)
    monkeypatch.setattr(runner, "SPLITS", splits)
    (splits / "evolution_pool_100.csv").write_text("instance_id,repo_language\nt1,python\n")
    (run / "selector_calls/dpp_trace.json").write_text("[]")
    np.save(run / "selector_calls/similarity.npy", np.eye(1))
    record = {"selected_task_ids": ["t1"], "all_candidate_ids": ["t1"],
        "selector": "dpp", "k": 1, "seed": 0, "theta": 0.7,
        "difficulty_scores": {"t1": 6.0}, "fingerprints": {"t1": "fingerprint"},
        "provenance": {"judge_backend": "codex", "dataset_spec": "fixture"}}
    task = SimpleNamespace(id="t1", row={"repo": "repo"}, query=lambda: "task")
    runner.finish_exports(record, [task])
    csv_path = splits / "coreset_ds_v4_astra_dpp_theta07_seed0.csv"
    assert "python" in csv_path.read_text()
    assert (run / "selection_report.md").is_file()
    assert (run / "exports_complete.json").is_file()
    runner.finish_exports(record, [task])  # Report-only resume is safe.


def test_probe_checkout_cannot_read_future_fix_commit(tmp_path):
    runner = load_runner()
    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source, check=True):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@localhost")
    (source / "code.py").write_text("bug\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    (source / "code.py").write_text("secret fix\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "future fix")
    fix = git("rev-parse", "HEAD").stdout.strip()
    dest = tmp_path / "task" / "repo"
    runner.HistoryFreeRepoCache(root=tmp_path / "cache").materialize(
        {"repo_path": str(source), "base_commit": base}, dest)
    assert (dest / "code.py").read_text() == "bug\n"
    assert git("rev-list", "--all", "--count", cwd=dest).stdout.strip() == "1"
    assert git("cat-file", "-e", fix, cwd=dest, check=False).returncode != 0
    assert git("remote", "-v", cwd=dest).stdout == ""
    assert (source / "code.py").read_text() == "secret fix\n"


@pytest.mark.parametrize("timeout", [False, True])
def test_live_logs_preserve_output_and_timeout_evidence(tmp_path, timeout):
    config = tmp_path / "config.toml"
    config.write_text('model = "test"\n')
    binary = tmp_path / "fake-codex"
    binary.write_text('''#!/usr/bin/env python3
import sys, time
from pathlib import Path
if "--version" in sys.argv:
    print("fake 1")
    raise SystemExit(0)
print('{"type":"turn.started"}', flush=True)
print("diagnostic", file=sys.stderr, flush=True)
if "--simulate-timeout" in sys.argv:
    time.sleep(10)
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("ok")
''')
    binary.chmod(0o755)
    agent = CodexAgent(config, binary=str(binary), sandbox="read-only", stream_logs=True,
                       codex_auth_home=tmp_path / "auth",
                       extra_flags=("--simulate-timeout",) if timeout else ())
    ws = tmp_path / "ws"
    ws.mkdir()
    result = agent.run(ws, "test", timeout_s=0.5 if timeout else 5)
    assert result.timed_out == timeout
    assert result.events == [{"type": "turn.started"}]
    assert "diagnostic" in result.stderr
    assert (ws / ".rho/live.stdout.jsonl").read_text() == result.stdout
    assert not any(path.startswith(".rho/") for path in result.workspace_diff)


def test_rate_limit_reduces_admissions_without_repeated_collapse(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "RUN", tmp_path)
    gate = runner.AdaptiveGate()
    gate.reduce("test rate limit")
    assert gate.limit == 6
    gate.reduce("same burst")
    assert gate.limit == 6


def test_isolated_codex_forwards_proxy_without_other_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.2:7896")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("UNRELATED_SECRET", "keep-out")
    config = tmp_path / "config.toml"
    config.write_text('model = "test"\n')
    agent = CodexAgent(config, sandbox="read-only", codex_auth_home=tmp_path / "auth")
    env = agent._subprocess_env()
    assert env["HTTPS_PROXY"] == "http://127.0.0.2:7896"
    assert env["NO_PROXY"] == "localhost,127.0.0.1"
    assert "UNRELATED_SECRET" not in env
