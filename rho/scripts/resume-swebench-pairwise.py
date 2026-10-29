#!/usr/bin/env python3
"""Resume only native RHO pairwise evaluation, with per-pair durable checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import statistics
import tempfile
import threading
import time
from types import SimpleNamespace

from rho.agent.codex import CodexAgent, EVAL_TIMEOUT_S
from rho.agent.codex_pool import configure_global_codex_pool
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset
from rho.loop import CandidateResult, _serialize_candidate_pool, _write_harness_diff
from rho.orchestrators.evaluate import EVAL_INSTRUCTIONS, evaluate
from rho.protocols import Score
from rho.stores.harness import FilesystemHarnessStore, _tree_digest
from rho.stores.trajectory import FilesystemTrajectoryStore

ROOT = Path(__file__).resolve().parents[1]
PINNED_CODEX = ROOT.parent / ".codex/packages/standalone/releases/0.163.0-x86_64-unknown-linux-musl/bin/codex"


def has_rate_limit(trajectory, pattern) -> bool:
    # Inspect actual evaluator errors, never quoted solver/tool output containing 429.
    for event in trajectory.events:
        if event.get("type") in ("error", "turn.failed") or event.get("item", {}).get("type") == "error":
            if pattern.search(json.dumps(event)):
                return True
    return bool(pattern.search(trajectory.stderr))


def run_retry_queue(pairs, attempt, recovery, runtime, gate, workers, *, retry_base_s=60, retry_cap_s=900):
    """Persist retry deadlines and release workers while rate-limited pairs cool down."""
    checkpoints = recovery / "pairs"
    queue_path = recovery / "retry_queue.json"
    states = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    records = {}
    by_key = {p["key"]: p for p in pairs}
    for key, pair in by_key.items():
        saved = checkpoints / f"{key}.json"
        if not saved.exists():
            # An exception/interruption in the scheduler must not discard a finished attempt.
            for attempt_path in sorted((recovery / "attempts" / key).glob("*.json"), reverse=True):
                completed_attempt = json.loads(attempt_path.read_text())
                assert all(completed_attempt[k] == v for k, v in pair.items())
                if completed_attempt["usable"]:
                    runtime.write_json(saved, completed_attempt)
                    break
        if saved.exists():
            record = json.loads(saved.read_text())
            assert all(record[k] == v for k, v in pair.items())
            records[key] = record
            states[key] = {**states.get(key, {}), "state": "complete"}
        else:
            state = states.setdefault(key, {"attempts": 0, "not_before": 0})
            state["state"] = "waiting"  # Interrupted active attempts are safe to resubmit.

    def persist():
        runtime.write_json(queue_path, states)
        runtime.write_json(recovery / "status.json", {
            "completed": len(records), "total": len(pairs),
            "usable": sum(r["usable"] for r in records.values()),
            "phase": "pairwise_evaluation", "concurrency_limit": gate.limit,
            "active": sum(s["state"] == "running" for s in states.values()),
            "waiting": sum(s["state"] == "waiting" for s in states.values()),
            "retry_waiting": sum(s["state"] == "waiting" and s.get("attempts", 0) > 0 for s in states.values()),
            "infrastructure_failures": {k: s["last_error"] for k, s in states.items() if s.get("last_error")},
            "rate_limited_attempts": sum(s.get("rate_limited_attempts", 0) for s in states.values())})

    persist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while len(records) + sum(s["state"] == "failed" for s in states.values()) < len(pairs):
            now = time.time()
            ready = sorted((k for k, s in states.items()
                            if s["state"] == "waiting" and s.get("not_before", 0) <= now),
                           key=lambda k: (states[k].get("attempts", 0), states[k].get("not_before", 0), k))
            for key in ready[:max(0, min(workers, gate.limit) - len(active))]:
                state = states[key]
                state.update(state="running", attempts=state.get("attempts", 0) + 1)
                persist()  # Reserve attempt number durably before starting the API call.
                active[pool.submit(attempt, by_key[key], state["attempts"])] = key
            if not active:
                deadline = min(s.get("not_before", 0) for s in states.values() if s["state"] == "waiting")
                time.sleep(min(5, max(0.01, deadline - time.time())))
                continue
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                key = active.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    state = states[key]
                    count = state.get("infrastructure_errors", 0) + 1
                    state.update(infrastructure_errors=count,
                                 last_error=f"{type(exc).__name__}: {exc}",
                                 state="waiting" if count < 3 else "failed",
                                 not_before=time.time() + retry_base_s * count)
                    runtime.log(f"Infrastructure error: {key}: {state['last_error']}; state={state['state']}")
                    persist()
                    continue
                if record.get("rate_limited") and not record["usable"]:
                    state = states[key]
                    count = state.get("rate_limited_attempts", 0) + 1
                    delay = min(retry_cap_s, retry_base_s * 2 ** min(count - 1, 10))
                    state.update(state="waiting", rate_limited_attempts=count,
                                 not_before=time.time() + delay,
                                 last_eval_trajectory_id=record["eval_trajectory_id"])
                    gate.reduce("Pairwise API rate limit; queued for retry")
                    runtime.log(f"Retry queued: {key}, attempt={state['attempts']}, wait={delay}s")
                else:
                    runtime.write_json(checkpoints / f"{key}.json", record)
                    records[key] = record
                    states[key]["state"] = "complete"
                    runtime.log(f"Pair {len(records)}/{len(pairs)} saved: {key} score={record['value']} usable={record['usable']}")
                persist()
    failures = {k: s["last_error"] for k, s in states.items() if s["state"] == "failed"}
    if failures:
        runtime.write_json(recovery / "infrastructure_failures.json", failures)
        raise RuntimeError(f"Unresolved infrastructure failures: {failures}")
    return records


def seed_timeout_retry(old, recovery, plan, runtime):
    """Copy only successful prior judgments; pin the exact timed-out subset."""
    old_plan = json.loads((old / "plan.json").read_text())
    for key in ("source_run", "pairs", "codex_config_sha256", "eval_prompt_sha256", "model"):
        assert old_plan[key] == plan[key], f"Retry source differs: {key}"
    retried, preserved = [], []
    for pair in plan["pairs"]:
        record = json.loads((old / "pairs" / f"{pair['key']}.json").read_text())
        assert all(record[k] == v for k, v in pair.items())
        assert (Path(plan["source_run"]) / "trajectories" / record["eval_trajectory_id"] / "meta.json").is_file()
        if record["timed_out"]:
            retried.append(pair["key"])
        else:
            assert record["usable"], "Only usable prior evaluations may be preserved"
            preserved.append(pair["key"])
            dest = recovery / "pairs" / f"{pair['key']}.json"
            if dest.exists():
                assert json.loads(dest.read_text()) == record
            else:
                runtime.write_json(dest, record)
    return {"retry_from": str(old), "retry_keys": retried, "preserved_keys": preserved,
            "preserved_eval_timeout_seconds": old_plan["eval_timeout_seconds"]}


def load_runner():
    spec = importlib.util.spec_from_file_location("pairwise_runtime", ROOT / "scripts/select-swebench-v1-coreset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_candidate_id(initial, trajectory, known_ids: set[str], scratch: Path) -> str:
    with tempfile.TemporaryDirectory(dir=scratch, prefix="candidate_check_") as tmp:
        dest = Path(tmp)
        initial.materialize(dest)
        for rel in trajectory.workspace_deletions:
            if rel.startswith("harness/"):
                sub = Path(rel.removeprefix("harness/"))
                assert not sub.is_absolute() and ".." not in sub.parts
                (dest / sub).unlink(missing_ok=True)
        for rel, content in trajectory.workspace_diff.items():
            if not rel.startswith("harness/"):
                continue
            sub = Path(rel.removeprefix("harness/"))
            assert not sub.is_absolute() and ".." not in sub.parts
            assert not content.startswith(b"HASH:"), "Cannot reconstruct a hash-only harness file"
            target = dest / sub
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        candidate_id = "h_" + _tree_digest(dest)[:12]
        assert candidate_id in known_ids, "Reconstructed proposal must match a saved candidate"
        return candidate_id


def prepare_plan(source: Path, recovery: Path) -> dict:
    rd = source / "rounds/round_0"
    config = json.loads((source / "config.json").read_text())
    ids = json.loads((source / "selection.json").read_text())["selected_task_ids"]
    groups = json.loads((rd / "solve_before_traj_ids.json").read_text())
    assert len(ids) == len(groups) == 10 and all(len(g) == 3 for g in groups)
    metadata = {m["id"]: m for p in (source / "trajectories").glob("*/meta.json")
                for m in [json.loads(p.read_text())]}
    before = {}
    for task_id, group in zip(ids, groups):
        assert all(metadata[tr]["task_id"] == task_id and metadata[tr]["stage"] == "round_solve_before" for tr in group)
        before[task_id] = group[0]  # Upstream run_round uses sample zero, not best-of-three.
    after = {}
    for m in metadata.values():
        if m.get("stage") == "round_solve_after" and m.get("round_ix") == 0:
            key = (m["harness_id"], m["task_id"])
            assert key not in after
            after[key] = m["id"]
    known = {h for h, _ in after}
    assert len(known) == 3 and len(after) == 30
    hs = FilesystemHarnessStore(source / "harness")
    initial_id = (rd / "input_harness_id").read_text().strip()
    initial = hs.get(initial_id)
    store = FilesystemTrajectoryStore(source / "trajectories")
    samples = []
    opts = sorted((m for m in metadata.values() if m.get("stage") == "round_optimize" and m.get("round_ix") == 0),
                  key=lambda m: m["sample_index"])
    assert [m["sample_index"] for m in opts] == [0, 1, 2]
    for m in opts:
        cid = recover_candidate_id(initial, store.get(m["id"]), known, recovery)
        samples.append({"sample_index": m["sample_index"], "optimize_traj_id": m["id"], "candidate_harness_id": cid})
    candidates = list(dict.fromkeys(s["candidate_harness_id"] for s in samples))
    assert set(candidates) == known
    pairs = [{"key": f"c{ci}_t{ti:02d}", "candidate_harness_id": cid, "task_id": tid,
              "before_trajectory_id": before[tid], "after_trajectory_id": after[cid, tid]}
             for ci, cid in enumerate(candidates) for ti, tid in enumerate(ids)]
    return {"source_run": str(source), "task_ids": ids, "initial_harness_id": initial_id,
            "samples": samples, "candidate_ids": candidates, "pairs": pairs,
            "dataset_spec": config["dataset_spec"], "dataset_digest": config["dataset_digest"],
            "model": config["model"], "reasoning_effort": config["reasoning_effort"],
            "codex_binary": str(PINNED_CODEX), "codex_version": "0.163.0",
            "codex_config_sha256": hashlib.sha256((source / "codex_config.toml").read_bytes()).hexdigest(),
            "eval_timeout_seconds": EVAL_TIMEOUT_S,
            "eval_prompt_sha256": hashlib.sha256(EVAL_INSTRUCTIONS.encode()).hexdigest(),
            "rule": "native evaluate; candidate shown as A, initial as B; negate visible A->B score; choose max mean, accept only mean>0",
            "old_unpersisted_calls_reused": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-timeout-seconds", type=float, default=EVAL_TIMEOUT_S)
    parser.add_argument("--retry-timeouts-from", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.eval_timeout_seconds <= 0:
        parser.error("--eval-timeout-seconds must be positive")
    source, recovery = args.source_run.resolve(), args.recovery_dir.resolve()
    recovery.mkdir(parents=True, exist_ok=True)
    lock = (recovery / "process.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime = load_runner()
    runtime.RUN = recovery
    plan = prepare_plan(source, recovery)
    plan["eval_timeout_seconds"] = args.eval_timeout_seconds
    if args.retry_timeouts_from:
        assert args.retry_timeouts_from.resolve() != recovery
        plan.update(seed_timeout_retry(args.retry_timeouts_from.resolve(), recovery, plan, runtime))
    plan_path = recovery / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Resume inputs changed"
    else:
        runtime.write_json(plan_path, plan)
    if args.prepare_only:
        print(json.dumps({"pairs": len(plan["pairs"]), "candidate_ids": plan["candidate_ids"],
                          "model": plan["model"], "eval_timeout_seconds": plan["eval_timeout_seconds"],
                          "retry_count": len(plan.get("retry_keys", [])),
                          "preserved_count": len(plan.get("preserved_keys", []))}), flush=True)
        return 0
    if (recovery / "summary.json").exists():
        runtime.log("Pairwise recovery already complete")
        return 0
    assert PINNED_CODEX.is_file()
    launcher = runpy.run_path(str(ROOT / "scripts/rho-ds.py"))
    launcher["configure_environment"]()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = "http://127.0.0.2:7896"
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = "localhost,127.0.0.1,127.0.0.2"
    dataset_path = Path(plan["dataset_spec"].split(":", 1)[1])
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == plan["dataset_digest"]
    hs = FilesystemHarnessStore(source / "harness")
    store = FilesystemTrajectoryStore(source / "trajectories")
    ds = SWEbenchProDataset(str(dataset_path), harness_store=hs, seed=0)
    tasks = {t.id: runtime.SelectionTask(_row=t.row, _harness=t.harness,
             _repo_cache=runtime.HistoryFreeRepoCache()) for t in ds.train}
    configure_global_codex_pool(args.workers)
    gate = runtime.AdaptiveGate()
    cp = recovery / "concurrency.json"
    gate.limit = min(args.workers, json.loads(cp.read_text())["limit"]) if cp.exists() else args.workers
    runtime.write_json(cp, {"limit": gate.limit, "reason": "Resumed pairwise evaluation"})
    agent = CodexAgent(source / "codex_config.toml", model=plan["model"],
        reasoning_effort=plan["reasoning_effort"], binary=str(PINNED_CODEX),
        sandbox="danger-full-access", codex_auth_home=recovery / "empty-auth", stream_logs=True)
    initial = hs.get(plan["initial_harness_id"])
    checkpoints = recovery / "pairs"
    checkpoints.mkdir(exist_ok=True)

    def monitor():
        # Old interrupted workspaces remain for inspection; do not replay their errors.
        offsets = {str(p): p.stat().st_size for p in (recovery / "workdir").glob("eval_*/.rho/live.*")}
        while not gate.stop.wait(5):
            for p in (recovery / "workdir").glob("eval_*/.rho/live.*"):
                try:
                    with p.open(errors="replace") as f:
                        f.seek(offsets.get(str(p), 0)); lines = f.readlines(); offsets[str(p)] = f.tell()
                    for line in lines:
                        if p.name.endswith("jsonl"):
                            try: e = json.loads(line)
                            except ValueError: continue
                            if e.get("type") not in ("error", "turn.failed") and e.get("item", {}).get("type") != "error": continue
                        if runtime.RATE_LIMIT.search(line): gate.reduce("Pairwise API rate limit")
                except OSError: pass

    threading.Thread(target=monitor, daemon=True).start()

    def one(pair, attempt_number):
        with gate.slot():
            before = store.get(pair["before_trajectory_id"])
            after = store.get(pair["after_trajectory_id"])
            tr, score = evaluate(agent, tasks[pair["task_id"]], before, after,
                harness_before=initial, harness_after=hs.get(pair["candidate_harness_id"]),
                workdir=recovery / "workdir", stage="round_evaluate", round_ix=0,
                timeout_s=plan["eval_timeout_seconds"])
            store.put(tr)
            record = {**pair, "eval_trajectory_id": tr.id, "value": score.value,
                "rationale": score.rationale, "exit_code": tr.exit_code,
                "timed_out": tr.timed_out, "usable": score.rationale != "(eval trajectory unusable)",
                "rate_limited": has_rate_limit(tr, runtime.RATE_LIMIT), "attempt": attempt_number,
                "eval_timeout_seconds": plan["eval_timeout_seconds"]}
            runtime.write_json(recovery / "attempts" / pair["key"] / f"{attempt_number:04d}.json", record)
        return record

    runtime.log(f"Resuming native DeepSeek pairwise evaluation; {len(list(checkpoints.glob('*.json')))}/30 checkpointed")
    try:
        records = run_retry_queue(plan["pairs"], one, recovery, runtime, gate, args.workers)
        candidate_pool = []
        ordered_tasks = [tasks[tid] for tid in plan["task_ids"]]
        for cid in plan["candidate_ids"]:
            pairs = [p for p in plan["pairs"] if p["candidate_harness_id"] == cid]
            samples = [s for s in plan["samples"] if s["candidate_harness_id"] == cid]
            candidate = CandidateResult(hs.get(cid), [s["sample_index"] for s in samples],
                [s["optimize_traj_id"] for s in samples])
            candidate.after_trajs = [SimpleNamespace(id=p["after_trajectory_id"]) for p in pairs]
            candidate.eval_trajs = [SimpleNamespace(id=records[p["key"]]["eval_trajectory_id"]) for p in pairs]
            candidate.scores = [Score(records[p["key"]]["value"], records[p["key"]]["rationale"]) for p in pairs]
            candidate.mean_score = statistics.mean(s.value for s in candidate.scores)
            candidate_pool.append(candidate)
        winner = max(candidate_pool, key=lambda c: c.mean_score)
        accepted = winner.mean_score > 0
        rd = source / "rounds/round_0"
        backup = recovery / "previous_round_outputs"
        if not backup.exists():
            shutil.copytree(rd, backup)
        runtime.write_json(rd / "optimize_traj_ids.json", [s["optimize_traj_id"] for s in plan["samples"]])
        (rd / "optimize_traj_id").write_text(winner.representative_optimize_traj_id)
        (rd / "candidate_harness_id").write_text(winner.candidate.id)
        _write_harness_diff(initial, winner.candidate, rd / "candidate_harness_diff.patch")
        runtime.write_json(rd / "solve_after_traj_ids.json", [t.id for t in winner.after_trajs])
        runtime.write_json(rd / "eval_traj_ids.json", [t.id for t in winner.eval_trajs])
        runtime.write_json(rd / "scores.json", [{"task_id": tid, "value": s.value, "rationale": s.rationale}
                                               for tid, s in zip(plan["task_ids"], winner.scores)])
        (rd / "mean_score").write_text(f"{winner.mean_score:.4f}")
        (rd / "accepted").write_text("true" if accepted else "false")
        runtime.write_json(rd / "optimize_candidates.json", _serialize_candidate_pool(
            ordered_tasks, plan["samples"], candidate_pool, winner, accepted))
        summary = {"completed": 30, "usable": sum(r["usable"] for r in records.values()),
            "winner_candidate_harness_id": winner.candidate.id, "mean_score": winner.mean_score,
            "accepted": accepted, "final_harness_id": winner.candidate.id if accepted else initial.id,
            "candidate_means": {c.candidate.id: c.mean_score for c in candidate_pool},
            "formal_grading_started": False, "model": plan["model"],
            "eval_timeout_seconds": plan["eval_timeout_seconds"],
            "retried_count": len(plan.get("retry_keys", []))}
        runtime.write_json(recovery / "summary.json", summary)
        runtime.write_json(recovery / "status.json", {**summary, "phase": "complete"})
        runtime.log(f"COMPLETE: mean={winner.mean_score} accepted={accepted} final={summary['final_harness_id']}")
        return 0
    finally:
        gate.stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
