#!/usr/bin/env python3
"""Run upstream rho evolve with the selected V1 coreset and local runtime settings."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import threading
import time
from contextlib import nullcontext

from rho.agent.codex import CodexAgent
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset, SWEbenchProTask
from rho.local_settings import codex_binary, configure_network

ROOT = Path(__file__).resolve().parents[1]


def container_task_prompt(task):
    prompt = SWEbenchProTask._prompt(task)
    prompt = "\n".join(line for line in prompt.splitlines() if not line.startswith("Instance ID:"))
    return prompt + "\nSolve independently in this task container. Installed Python/Go/Node/system dependencies are available. Package managers may download dependencies through the configured audited proxy. Do not search the web for answers, upstream fixes, reference patches or newer versions of this project. Search sites and code-hosting sites are blocked; repository history, other tasks, hidden grading tests and prior experiment outputs are unavailable.\n"


def load_selection_runner():
    spec = importlib.util.spec_from_file_location("rho_local_selection", ROOT / "scripts/select-swebench-v1-coreset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evolution_arguments(selection: Path, run: Path, source: Path) -> list[str]:
    # Match scripts/run-table1.sh: one round, diagnosis strategy, three proposals,
    # ten core tasks, held-out val[:100], no agent response cache.
    return ["evolve", "--dataset", f"swebench-pro:{source}", "--rounds", "1",
            "--selection-json", str(selection), "--max-evolve-tasks", "10",
            "--max-grading-tasks", "100", "--optimize-strategy", "diagnosis",
            "--optimize-samples", "3", "--cache", "off", "--seed", "0",
            "--docker-pull", "missing", "--grade-workers", "1",
            "--codex-concurrency", "8", "--run-dir", str(run),
            "--judge-model", "gpt-6-astra"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--container-images", type=Path)
    parser.add_argument("--max-model-requests", type=int,
                        help="Per solve/sample limit; diagnosis, optimize and rank retain their time budgets")
    parser.add_argument("--solve-timeout-seconds", type=float, default=3600,
                        help="Solve wall-clock limit; 0 disables it")
    parser.add_argument("--resume-initial", action="store_true")
    parser.add_argument("--resume-round", type=Path)
    args = parser.parse_args()
    if args.max_model_requests is not None and (args.max_model_requests < 1 or not args.container_images):
        parser.error('--max-model-requests requires a positive limit and isolated containers')
    if args.solve_timeout_seconds < 0:
        parser.error('--solve-timeout-seconds must be nonnegative')
    solve_timeout = args.solve_timeout_seconds or None
    selection, run = args.selection_json.resolve(), args.run_dir.resolve()
    if (run / "config.json").exists() and not (args.resume_initial or args.resume_round):
        raise RuntimeError("Evolution run already started; refusing to overwrite or restart it")
    recovery = None
    if args.resume_round:
        from rho.agent.round_recovery import RoundRecovery
        recovery = RoundRecovery(run, args.resume_round)
    if args.resume_initial:
        if not args.container_images:
            raise RuntimeError('Initial recovery requires isolated task containers')
        from rho.agent.initial_solve_recovery import InitialSolveRecovery
        recovery = InitialSolveRecovery(run, selection, args.container_images.resolve())
    runner = load_selection_runner()
    local_manifest = run.parent / 'split_manifest.json'
    manifest = json.loads((local_manifest if local_manifest.exists() else runner.SPLITS / 'manifest.json').read_text())
    selected = json.loads(selection.read_text())
    ids = selected["selected_task_ids"]
    assert len(ids) == len(set(ids)) == 10
    assert set(ids) <= set(manifest["groups"]["evolution_pool_100"])
    assert not set(ids) & set(manifest["groups"]["evaluation_100"])
    selection_judge = selected.get('provenance', {}).get('judge_model', 'not-recorded')
    assert hashlib.sha256(runner.SOURCE.read_bytes()).hexdigest() == manifest["verified_local_source"]["sha256"]
    launcher = runpy.run_path(str(ROOT / "scripts/rho-ds.py"))
    launcher["configure_environment"]()
    configure_network()
    run.mkdir(parents=True, exist_ok=True)
    runner.RUN = run
    gate = runner.AdaptiveGate()
    # Keep the established sustainable DeepSeek limit from the probe run.
    gate.limit = 8 if args.container_images else 4
    runner.write_json(run / "concurrency.json", {"limit": gate.limit,
        "reason": "Requested concurrency; reduce on provider rate limiting"})
    event_lock = threading.Lock()
    from rho.agent.retry_queue import ExecutionRetryQueue
    retry_queue = ExecutionRetryQueue(run / 'retry_queue')
    runner.write_json(run / 'retry_policy.json', {
        'enabled': True, 'scope': 'evolution solve/diagnose/optimize/pairwise',
        'retry_on': ['timeout', 'nonzero_exit', 'execution_exception'],
        'initial_delay_seconds': 60, 'maximum_delay_seconds': 600,
        'attempt_limit': None, 'retry_normal_empty_patch': False,
        'retry_budget_exhausted': False, 'max_model_requests_per_solve': args.max_model_requests,
        'concurrency_limit': 8, 'fresh_container_each_attempt': True})

    class ExperimentTask(runner.SelectionTask):
        def materialize(self, dest):
            if not args.container_images:
                return super().materialize(dest)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "prompt.md").write_text(self._prompt())
            (dest / "repo").mkdir(exist_ok=True)

        def _prompt(self):
            if not args.container_images:
                return super()._prompt()
            return container_task_prompt(self)

        @property
        def agent_timeout_s(self):
            return solve_timeout

    class ExperimentDataset(SWEbenchProDataset):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, repo_cache=runner.HistoryFreeRepoCache(), **kwargs)
            for split, taskset in self._splits.items():
                tasks = tuple(ExperimentTask(**{f.name: getattr(t, f.name)
                    for f in fields(SWEbenchProTask)}) for t in taskset)
                self._splits[split] = replace(taskset, _tasks=tasks)
            assert [t.id for t in self.train] == manifest["groups"]["evolution_pool_100"]
            assert [t.id for t in list(self.val)[:100]] == manifest["groups"]["evaluation_100"]

    if args.container_images:
        from rho.agent.container_codex import ContainerCodexAgent
    AgentBase = ContainerCodexAgent if args.container_images else CodexAgent

    class ExperimentAgent(AgentBase):
        def __init__(self, *positional, **kwargs):
            kwargs.update(default_timeout_s=solve_timeout, stream_logs=True,
                          codex_auth_home=run / "empty-auth")
            if args.container_images:
                kwargs.update(images_path=args.container_images.resolve(), audit_root=run / "runtime",
                              solve_maximum=args.max_model_requests)
            super().__init__(*positional, **kwargs)

        def run(self, *positional, **kwargs):
            if recovery is not None:
                recovered = recovery.take(*positional, **kwargs)
                if recovered is not None:
                    return recovered
            trajectory = retry_queue.run(self._admitted_run, *positional, **kwargs)
            with event_lock:
                with (run / "completed_calls.jsonl").open("a") as f:
                    f.write(json.dumps({"trajectory_id": trajectory.id,
                        "kind": trajectory.kind, "task_id": trajectory.task_id,
                        "exit_code": trajectory.exit_code, "timed_out": trajectory.timed_out,
                        "wall_time_s": trajectory.wall_time_s}) + "\n")
            return trajectory

        def _admitted_run(self, *positional, **kwargs):
            if kwargs.get('kind', 'solve') != 'solve' and kwargs.get('timeout_s') is None:
                kwargs['timeout_s'] = 3600
            with gate.slot():
                with recovery.slot() if args.resume_round else nullcontext():
                    return super().run(*positional, **kwargs)

    def monitor():
        offsets = {}
        if args.resume_round:
            for p in (run / 'runtime/calls').glob('*/*'):
                if p.name in {'stdout.jsonl', 'stderr.log'}:
                    offsets[str(p)] = p.stat().st_size
        last_control = None
        while not gate.stop.wait(5):
            control_path = run / 'concurrency_control.json'
            if control_path.exists():
                control = json.loads(control_path.read_text())
                if control.get('request_id') != last_control:
                    limit = int(control['limit'])
                    if not 1 <= limit <= 8: raise ValueError('Invalid concurrency limit')
                    with gate.condition:
                        gate.limit = limit
                        gate.condition.notify_all()
                    runner.write_json(run / 'concurrency.json', {'limit': limit, 'reason': 'User runtime concurrency request'})
                    last_control = control.get('request_id')
            paths = list((run / "workdir").glob("*/.rho/live.*"))
            if args.container_images:
                paths += list((run / "runtime/calls").glob("*/stdout.jsonl"))
                paths += list((run / "runtime/calls").glob("*/stderr.log"))
            for path in paths:
                try:
                    with path.open(errors="replace") as f:
                        f.seek(offsets.get(str(path), 0))
                        lines = f.readlines()
                        offsets[str(path)] = f.tell()
                    for line in lines:
                        if path.name.endswith("jsonl"):
                            try:
                                e = json.loads(line)
                            except ValueError:
                                continue
                            if e.get("type") not in ("error", "turn.failed") and e.get("item", {}).get("type") != "error":
                                continue
                        if runner.RATE_LIMIT.search(line):
                            gate.reduce("DeepSeek evolution rate limit")
                except OSError:
                    pass

    argv = launcher["command_arguments"](evolution_arguments(selection, run, runner.SOURCE))
    if args.container_images:
        argv[argv.index("--max-grading-tasks") + 1] = "0"
        argv[argv.index("--docker-pull") + 1] = "never"
        argv[argv.index("--codex-binary") + 1] = str(codex_binary('0.163.0'))
    runner.write_json(run / "launch.json", {"argv": argv, "selection_json": str(selection),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "solver_model": launcher["MODEL"], "selection_judge_model": selection_judge,
        "evolution_diagnosis_and_pairwise_model": launcher["MODEL"],
        "rounds": 1, "group_rollouts_per_task": 3, "optimize_samples": 3,
        "solve_timeout_seconds": solve_timeout, "max_model_requests_per_solve": args.max_model_requests,
        "request_budget_scope": "task/harness/sample, shared across retries",
        "initial_admission_limit": gate.limit,
        "execution_runtime": "isolated_task_container" if args.container_images else "host",
        "container_images": str(args.container_images) if args.container_images else None,
        "upstream_loop": "rho.cli._cmd_evolve -> rho.loop.run_evolution (unchanged)",
        "runtime_adaptations": ["isolated DeepSeek config/auth", "base snapshots without future history",
                                "fixing ID omitted from solver prompt", "explicit solve budget", "adaptive concurrency/live logs"],
        "held_out_grading": "Original RHO evaluator, val[:100], after the evolution round only"})
    # Runtime-local adapters; no global Codex configuration or CLI sources change.
    import rho.cli as cli
    import rho.datasets.loader as loader
    cli.CodexAgent = ExperimentAgent
    if args.resume_round:
        from rho.agent.round_recovery import ReplayTrajectoryStore
        cli.FilesystemTrajectoryStore = ReplayTrajectoryStore
        original_evolution = cli.run_evolution
        def resumed_evolution(**kwargs):
            # Extra coordinator threads can wait for old workers. Actual
            # model/container admission remains globally capped at eight.
            kwargs['solve_workers'] = 16
            return original_evolution(**kwargs)
        cli.run_evolution = resumed_evolution
    loader.SWEbenchProDataset = ExperimentDataset
    threading.Thread(target=monitor, daemon=True).start()
    runner.log("Starting native RHO evolution: 10 tasks x 3 rollouts, 3 harness proposals, 1 round")
    try:
        result = cli.main(argv)
        runner.write_json(run / "launcher_exit.json", {"exit_code": result})
        return result
    finally:
        gate.stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
