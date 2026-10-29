#!/usr/bin/env python3
"""Resumable, eight-worker execution of RHO's native V1 DPP selection."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import runpy

import numpy as np

from rho.agent.codex import CodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset, SWEbenchProTask
from rho.datasets.swebench_pro.repo_cache import RepoCache
from rho.datasets.swebench_pro.util import git
from rho.selection.difficulty_selector import TaskJudge
from rho.selection.codex_judge_client import CodexJudgeClient
from rho.selection.dpp_selector import DPPSelector
from rho.selection.llm_client import LiteLLMClient
from rho.selection.local_embedder import LocalEmbedder
from rho.selection.report import write_selection_report
from rho.selection.short_solve import short_solve_one
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.local_settings import codex_binary, codex_config, configure_network

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/swebench-v1-dpp-ds-v4-seed0-clean-proxy"
SPLITS = Path(os.environ.get('RHO_SPLITS_DIR', str(ROOT / 'data/splits/swebench-pro-v1-seed0'))).expanduser().resolve()
SOURCE = Path(os.environ.get('RHO_DATASET_PATH', str(ROOT / 'data/swebench-pro/test.jsonl'))).expanduser().resolve()
WORKERS = 8


class HistoryFreeRepoCache(RepoCache):
    def materialize(self, row, dest):
        super().materialize(row, dest)
        # Only this disposable task checkout is changed. Preserve the shared
        # download cache and all other repositories.
        shutil.rmtree(dest / ".git")
        git(["init", "--quiet"], cwd=dest)
        git(["add", "-A"], cwd=dest)
        git(["-c", "user.name=RHO snapshot", "-c", "user.email=rho@localhost",
             "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
             "commit", "--quiet", "--allow-empty", "-m", "Task base snapshot"], cwd=dest)


class SelectionTask(SWEbenchProTask):
    def _prompt(self):
        prompt = super()._prompt()
        # V1 instance IDs encode the fixing commit; retain IDs in metadata,
        # but do not disclose them to the solver.
        prompt = "\n".join(line for line in prompt.splitlines()
                           if not line.startswith("Instance ID:"))
        return prompt + (
            "\n\nExperimental constraints: solve independently from the provided base snapshot. "
            "Do not use the network, fetch repository history, search for upstream fixes, "
            "or access files outside this task workspace. Do not read benchmark reference "
            "patches, hidden tests, or previous experiment results. The local Git repository "
            "contains only the base snapshot for diff inspection.\n"
        )


RATE_LIMIT = re.compile(r"(?i)(unexpected status(?: code)?[ :=]+429|(?:HTTP|status)[ :=]+429|too many requests|RateLimitError|rate[ _-]?limit(?:ed| exceeded| reached)|request rate increased too quickly)")


class AdaptiveGate:
    def __init__(self):
        self.condition = threading.Condition()
        self.active = 0
        self.limit = WORKERS
        self.last_reduction = 0.0
        self.stop = threading.Event()

    @contextmanager
    def slot(self):
        with self.condition:
            while self.active >= self.limit:
                self.condition.wait(timeout=5)
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def reduce(self, reason):
        with self.condition:
            now = time.monotonic()
            if self.limit == 1 or now - self.last_reduction < 60:
                return
            self.limit = max(1, self.limit - 2)
            self.last_reduction = now
            write_json(RUN / "concurrency.json", {"limit": self.limit, "reason": reason})
            log(f"API rate limit detected; reducing admissions to {self.limit}")

    def monitor(self):
        offsets = {}
        while not self.stop.wait(5):
            paths = list((RUN / "workdir/short_solve").glob("solve_*/.rho/live.*"))
            paths += list((RUN / "judge_transport_calls").glob("call_*/.rho/live.*"))
            for path in paths:
                try:
                    with path.open(errors="replace") as f:
                        f.seek(offsets.get(str(path), 0))
                        lines = f.readlines()
                        offsets[str(path)] = f.tell()
                    for line in lines:
                        if path.name.endswith("jsonl"):
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if (event.get("type") not in {"error", "turn.failed"}
                                    and event.get("item", {}).get("type") != "error"):
                                continue
                        if RATE_LIMIT.search(line):
                            self.reduce("429/rate-limit signal in live Codex logs")
                except OSError:
                    pass


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def log(message: str) -> None:
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, flush=True)


def finish_exports(record: dict, tasks: list) -> None:
    """Resume report/export generation without repeating any model calls."""
    by_id = {task.id: task for task in tasks}
    # The local solver JSONL omits language; use the verified split metadata.
    with (SPLITS / "evolution_pool_100.csv").open() as f:
        languages = {row["instance_id"]: row["repo_language"] for row in csv.DictReader(f)}
    selected_ids = record["selected_task_ids"]
    export_name = "coreset_ds_v4_astra_dpp_theta07_seed0" if record["provenance"].get("judge_backend") == "codex" else "coreset_ds_v4_dpp_theta07_seed0"
    write_json(SPLITS / f"{export_name}.json", record)
    (SPLITS / f"{export_name}.txt").write_text("\n".join(selected_ids) + "\n")
    with (SPLITS / f"{export_name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "instance_id", "repo", "language", "difficulty", "fingerprint"])
        writer.writeheader()
        for i, task_id in enumerate(selected_ids, 1):
            writer.writerow({"rank": i, "instance_id": task_id, "repo": by_id[task_id].row["repo"],
                "language": languages[task_id], "difficulty": record["difficulty_scores"][task_id],
                "fingerprint": record["fingerprints"][task_id]})
    write_selection_report(run_dir=RUN, selection=record, queries={t.id: t.query() for t in tasks},
        dataset_spec=record["provenance"]["dataset_spec"], split="train", scores=record["difficulty_scores"],
        fingerprints=record["fingerprints"],
        gain_trace=json.loads((RUN / "selector_calls/dpp_trace.json").read_text()),
        similarity=np.load(RUN / "selector_calls/similarity.npy"), candidate_ids=record["all_candidate_ids"])
    write_json(RUN / "exports_complete.json", {"selected_count": len(selected_ids), "export_name": export_name})


def seed_timeout_retry(source: Path, destination: Path, task_ids: list[str]) -> None:
    """Preserve completed probes; leave only previously timed-out tasks missing."""
    if (destination / "probe_index.json").exists():
        return
    source_index = json.loads((source / "probe_index.json").read_text())
    assert set(source_index) == set(task_ids), "Retry source must cover the exact candidate pool"
    retained, retry = {}, {}
    for task_id in task_ids:
        trajectory_id = source_index[task_id]
        original = source / "trajectories" / trajectory_id
        meta = json.loads((original / "meta.json").read_text())
        if meta["timed_out"]:
            retry[task_id] = trajectory_id
        else:
            assert meta["exit_code"] == 0, "Non-timeout failures require a separate retry plan"
            target = destination / "trajectories" / trajectory_id
            if not target.exists():
                shutil.copytree(original, target)
            retained[task_id] = trajectory_id
    assert retry, "Source has no timed-out probes"
    write_json(destination / "retry_plan.json", {
        "source_run": str(source), "retained_trajectory_ids": retained,
        "retry_previous_trajectory_ids": retry,
        "retained_count": len(retained), "retry_count": len(retry),
    })
    write_json(destination / "probe_index.json", retained)


def seed_existing_probes(source: Path, destination: Path, task_ids: list[str]) -> None:
    if (destination / "probe_index.json").exists():
        return
    index = json.loads((source / "probe_index.json").read_text())
    assert set(index) == set(task_ids), "All candidate trajectories must be available before judging"
    for trajectory_id in index.values():
        target = destination / "trajectories" / trajectory_id
        if not target.exists():
            shutil.copytree(source / "trajectories" / trajectory_id, target)
    write_json(destination / "reused_probes.json", {"source_run": str(source), "count": len(index)})
    write_json(destination / "probe_index.json", index)


def main() -> None:
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--probe-timeout-seconds", type=float, default=3600)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--retry-timeouts-from", type=Path)
    reuse.add_argument("--reuse-probes-from", type=Path)
    parser.add_argument("--judge-backend", choices=("litellm", "codex"), default="litellm")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-config", type=Path, default=ROOT / "configs/codex.judge-astra.toml")
    parser.add_argument("--judge-auth-home", type=Path, default=Path.home() / ".codex")
    args = parser.parse_args()
    if args.probe_timeout_seconds <= 0:
        parser.error("--probe-timeout-seconds must be positive")
    RUN = args.run_dir.resolve()
    retry_source = args.retry_timeouts_from.resolve() if args.retry_timeouts_from else None
    reuse_source = args.reuse_probes_from.resolve() if args.reuse_probes_from else None
    if retry_source == RUN or reuse_source == RUN:
        parser.error("Use a new run directory to preserve the original trajectories")
    launcher = runpy.run_path(str(ROOT / "scripts/rho-ds.py"))
    launcher["configure_environment"]()
    # Scoped to this selection process and children. Reuse the existing local
    # proxy to avoid the host's intermittent upstream DNS resolution failures.
    configure_network()
    manifest = json.loads((SPLITS / "manifest.json").read_text())
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_sha == manifest["verified_local_source"]["sha256"], "Source dataset changed"
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "selection.json").exists():
        record = json.loads((RUN / "selection.json").read_text())
        dataset = SWEbenchProDataset(str(SOURCE), harness_store=FilesystemHarnessStore(RUN / "harness"), seed=0)
        finish_exports(record, list(dataset.train))
        log(f"COMPLETE: preserved selection and repaired exports: {RUN / 'selection.json'}")
        return
    config = {
        "dataset_spec": f"swebench-pro:{SOURCE}", "dataset_sha256": source_sha,
        "model": launcher["MODEL"],
        "judge_model": args.judge_model or ("gpt-6-astra" if args.judge_backend == "codex" else "openai/" + launcher["MODEL"]),
        "judge_backend": args.judge_backend,
        "judge_config_sha256": hashlib.sha256(args.judge_config.read_bytes()).hexdigest() if args.judge_backend == "codex" else None,
        "judge_auth_mode": "isolated Codex login copy" if args.judge_backend == "codex" else "provider API key",
        "judge_network": "direct, isolated child environment" if args.judge_backend == "codex" else "selection process proxy",
        "judge_output_limit": "Codex schema and 300s timeout; no enforced max_tokens/temperature" if args.judge_backend == "codex" else "16384 tokens",
        "reasoning_effort": "high", "selector": "dpp", "theta": 0.7,
        "k": 10, "split": "train", "split_seed": 0, "workers": WORKERS,
        "embedding_model": "local:BAAI/bge-large-en-v1.5",
        "codex_config_sha256": hashlib.sha256(codex_config(ROOT).read_bytes()).hexdigest(),
        "method": "Native RHO short_solve_one, TaskJudge, LocalEmbedder, DPPSelector; resumable orchestration.",
        "grader_used": False,
        "probe_isolation": "Base snapshot only; no upstream refs/remotes; fixing ID omitted; prompt forbids network and outside-workspace reads. Prompt restrictions are not OS isolation.",
        "discarded_attempt": "runs/swebench-v1-dpp-ds-v4-seed0 (upstream-fix contamination)",
        "network_failed_attempt": "runs/swebench-v1-dpp-ds-v4-seed0-clean-history",
        "https_proxy": "configured externally" if os.environ.get('HTTPS_PROXY') else None,
        "probe_timeout_seconds": args.probe_timeout_seconds,
        "retry_timeouts_from": str(retry_source) if retry_source else None,
        "reuse_probes_from": str(reuse_source) if reuse_source else None,
        "timeout_policy": "Inherited from reuse_probes_from; see source config" if reuse_source else ("Only timed-out probes rerun from a fresh base; completed source probes retained with their original budget" if retry_source else "Uniform probe timeout"),
    }
    config_path = RUN / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Resume configuration mismatch"
    else:
        write_json(config_path, config)
    hs = FilesystemHarnessStore(RUN / "harness")
    dataset = SWEbenchProDataset(str(SOURCE), harness_store=hs, seed=0)
    repo_cache = HistoryFreeRepoCache()
    tasks = [SelectionTask(_row=t.row, _harness=t.harness, _repo_cache=repo_cache)
             for t in dataset.train]
    assert [t.id for t in tasks] == manifest["groups"]["evolution_pool_100"]
    assert len(tasks) == 100
    evaluation_ids = set(manifest["groups"]["evaluation_100"])
    assert not evaluation_ids.intersection(t.id for t in tasks)
    if retry_source:
        source_config = json.loads((retry_source / "config.json").read_text())
        for key in ("dataset_sha256", "model", "codex_config_sha256", "split_seed"):
            assert source_config[key] == config[key], f"Retry source mismatch: {key}"
        seed_timeout_retry(retry_source, RUN, [task.id for task in tasks])
    if reuse_source:
        source_config = json.loads((reuse_source / "config.json").read_text())
        for key in ("dataset_sha256", "model", "codex_config_sha256", "split_seed"):
            assert source_config[key] == config[key], f"Reused probes mismatch: {key}"
        seed_existing_probes(reuse_source, RUN, [task.id for task in tasks])
    configure_global_codex_pool(WORKERS)
    agent = CodexAgent(codex_config(ROOT), model=launcher["MODEL"],
                       binary=str(codex_binary()), sandbox="workspace-write",
                       codex_auth_home=RUN / "empty-auth", stream_logs=True,
                       default_timeout_s=args.probe_timeout_seconds)
    gate = AdaptiveGate()
    threading.Thread(target=gate.monitor, daemon=True).start()
    store = FilesystemTrajectoryStore(RUN / "trajectories")
    index_path = RUN / "probe_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    trajectories = {task_id: store.get(traj_id) for task_id, traj_id in index.items()}
    missing = [task for task in tasks if task.id not in trajectories]
    log(f"Probes: {len(trajectories)}/100 already saved; launching {len(missing)} with {WORKERS} workers")
    def probe(task):
        with gate.slot():
            return short_solve_one(task, agent=agent, harness=task.harness,
                                   workdir=RUN / "workdir/short_solve")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(probe, task): task for task in missing}
        for future in as_completed(futures):
            task = futures[future]
            tr = future.result()
            store.put(tr)
            index[task.id] = tr.id
            trajectories[task.id] = tr
            write_json(index_path, index)
            log(f"Probe {len(index)}/100 exit={tr.exit_code} timeout={tr.timed_out} events={len(tr.events)} {task.id}")
    # Empty infrastructure placeholders contain no evidence for trajectory-based
    # selection. Keep their logs, but stop rather than silently ranking prompts.
    failed = [tid for tid, tr in trajectories.items() if tr.exit_code != 0 and not tr.events]
    if failed:
        write_json(RUN / "infra_failed_task_ids.json", failed)
        raise RuntimeError(f"{len(failed)} probes lack trajectory evidence; inspect infra_failed_task_ids.json")
    judge_cache = ROOT / "data/cache"
    if args.judge_backend == "codex":
        judge_agent = CodexAgent(args.judge_config, model=config["judge_model"],
            reasoning_effort="high", binary=str(codex_binary()),
            sandbox="read-only", network_access=False,
            codex_auth_home=args.judge_auth_home, stream_logs=True)
        direct_env = {key: "" for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                                         "http_proxy", "https_proxy", "all_proxy")}
        direct_env.update(NO_PROXY="*", no_proxy="*")
        judge_client = CodexJudgeClient(judge_agent, RUN / "judge_transport_calls", request_env=direct_env)
        judge_cache = judge_cache / ("codex_judge_" + config["judge_config_sha256"][:16])
    else:
        judge_client = LiteLLMClient(use_azure_foundry=False)
    judge = TaskJudge(client=judge_client, model=config["judge_model"],
                      reasoning_effort="high", workdir=RUN / "selector_calls",
                      trajectories=trajectories, cache_root=judge_cache)
    # Populate the judge's native cache with a bounded pool; DPP's subsequent
    # judge_many call then reuses these exact results without extra API calls.
    failures = {}
    def judge_one(task):
        for attempt in range(1, 4):
            try:
                with gate.slot():
                    return judge.judge(task)
            except Exception as exc:
                if RATE_LIMIT.search(str(exc)):
                    gate.reduce("Judge API rate limit")
                log(f"Judge attempt {attempt}/3 failed for {task.id}: {type(exc).__name__}: {str(exc)[:250]}")
                if attempt == 3:
                    raise
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(judge_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                log(f"Judge {i}/100 difficulty={result.difficulty} {task.id}")
            except Exception as exc:
                failures[task.id] = str(exc)
    if failures:
        write_json(RUN / "judge_failures.json", failures)
        raise RuntimeError(f"{len(failures)} judge calls failed; rerun to reuse successful cached results")
    log("Embedding 100 fingerprints and running native greedy DPP")
    selector = DPPSelector(judge=judge, embedder=LocalEmbedder(cache_root=ROOT / "data/cache"),
                           theta=0.7, workdir=RUN / "selector_calls")
    selected = selector.select(tasks, k=10, seed=0)
    selected_ids = [t.id for t in selected]
    assert len(set(selected_ids)) == 10, "DPP returned fewer than 10 distinct tasks"
    assert set(selected_ids) <= set(index) and not evaluation_ids.intersection(selected_ids)
    results = selector.results()
    record = {"selector": "dpp", "k": 10, "seed": 0, "theta": 0.7,
              "all_candidate_ids": [t.id for t in tasks], "selected_task_ids": selected_ids,
              "difficulty_scores": {tid: r.difficulty for tid, r in results.items()},
              "fingerprints": {tid: r.fingerprint for tid, r in results.items()},
              "short_solve_trajectory_ids": index,
              "judge_input_token_estimate": {tid: r.digest_token_estimate for tid, r in results.items()},
              "provenance": config, "run_dir": str(RUN)}
    write_json(RUN / "selection.json", record)
    finish_exports(record, tasks)
    log(f"COMPLETE: {RUN / 'selection.json'}")
    gate.stop.set()


if __name__ == "__main__":
    main()
