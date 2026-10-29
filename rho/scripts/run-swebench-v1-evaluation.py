#!/usr/bin/env python3
"""Prepare/run matched held-out baseline and final-harness evaluation budgets."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

from rho.agent.codex_pool import configure_global_codex_pool
from rho.agent.codex import _snapshot, _diff
from rho.agent.request_budget import RequestLimitedCodexAgent, write_json
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset
from rho.orchestrators.solve import SOLVE_INSTRUCTIONS, solve, solve_workspace
from rho.observability import annotate_trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.local_settings import codex_binary, configure_network

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/swebench-v1-heldout-evaluation.json"


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, flush=True)


def normalize_terminal_grade(grade):
    """An empty submission is a failed solve, not a retryable grader outage."""
    if (grade.details.get("error") == "patch_extraction_failed"
            and grade.details.get("message") == "trajectory did not modify task/repo and final_message contains no git patch"):
        details = dict(grade.details)
        details["original_error"] = details.pop("error")
        details["outcome"] = "no_patch"
        return replace(grade, passed=False, score=0.0, details=details)
    return grade


def publish_overall_status(output_root, arms):
    """Merge independent arm checkpoints without overwriting the other arm."""
    with (output_root / "status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states, summaries = {}, {}
        for arm in arms:
            for name, destination in (("status.json", states), ("summary.json", summaries)):
                path = output_root / arm / name
                if path.exists():
                    destination[arm] = json.loads(path.read_text())
        write_json(output_root / "summary.json", summaries)
        write_json(output_root / "status.json", {
            "phase": "complete" if all(a in summaries for a in arms) else "running",
            "arms": states, "summaries": summaries,
        })


def baseline_grading_complete(output_root):
    path = output_root / "baseline/summary.json"
    return path.exists() and json.loads(path.read_text()).get("n") == 100


def run_evaluation_pipeline(tasks, solve_one, grade_one, *, solve_workers, grade_workers, on_error):
    """Separate solve and grade capacities; grading never occupies a solver slot."""
    records = []
    with ThreadPoolExecutor(max_workers=solve_workers) as solvers, ThreadPoolExecutor(max_workers=grade_workers) as graders:
        pending = {solvers.submit(solve_one, task): ("solve", task) for task in tasks}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                stage, task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    on_error(task, stage, exc)
                    continue
                if stage == "solve":
                    pending[graders.submit(grade_one, task)] = ("grade", task)
                else:
                    records.append(result)
    return records


def resume_interrupted_solve(agent, task, harness, workspace, workdir, stage):
    """Continue retained code and logs without losing pre-interruption patch edits."""
    if (workspace / "task/prompt.md").read_text() != task._prompt():
        raise ValueError("Retained workspace does not match evaluation task")
    with solve_workspace(task, harness, workdir) as original:
        baseline = _snapshot(original, exclude={".rho"})
    meta = workspace / ".rho"
    archive = meta / f"before_resume_{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True)
    for name in ("live.stdout.jsonl", "live.stderr.log", "last_message.txt", "instructions.md"):
        if (meta / name).exists():
            shutil.copy2(meta / name, archive / name)
    history = archive / "live.stdout.jsonl"
    instructions = SOLVE_INSTRUCTIONS + (
        "\nThis evaluation was interrupted by an operator to update evaluation runtime settings. "
        "Your existing repository edits are preserved in this workspace. Continue the same task "
        "from those edits; inspect git diff and the prior session log if needed. "
        f"Prior session log: {history.relative_to(workspace)}. "
        "The model-request allowance remains 120 total including requests already used; it has not reset.\n"
    )
    tr = agent.run(workspace, instructions, task_id=task.id, harness_id=harness.id,
                   kind="solve", timeout_s=task.agent_timeout_s)
    changes, deleted = _diff(baseline, _snapshot(workspace, exclude={".rho"}))
    previous = []
    if history.exists():
        for line in history.read_text(errors="replace").splitlines():
            try:
                previous.append(json.loads(line))
            except ValueError:
                pass
    tr = replace(tr, workspace_diff=changes, workspace_deletions=frozenset(deleted),
        events=previous + [{"type": "operator_resume", "reason": "evaluation_runtime_update",
                           "workspace_preserved": True, "request_budget_reset": False}] + tr.events)
    return annotate_trajectory(tr, agent=agent, stage=stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--arm", choices=("baseline", "final", "both"), default="both")
    parser.add_argument("--grade-after-baseline", action="store_true",
                        help="Start final solves immediately, but wait for baseline grading before final grading")
    parser.add_argument("--execute", action="store_true", help="Without this flag, only prepare the evaluation plan")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    source = ROOT / policy["source_run"]
    config = json.loads((source / "config.json").read_text())
    manifest = json.loads((ROOT / policy["split_manifest"]).read_text())
    ids = manifest["groups"]["evaluation_100"]
    assert len(ids) == policy["max_grading_tasks"] == 100
    assert not set(ids) & set(manifest["groups"]["evolution_pool_100"])
    if args.execute and policy.get("container_images"):
        images_path = ROOT / policy["container_images"]
        checks_root = images_path.parent / "environment_preflight/results"
        while True:
            pinned = json.loads(images_path.read_text())
            pending = []
            for task_id in ids:
                report = checks_root / (hashlib.sha256(task_id.encode()).hexdigest() + ".json")
                if not report.exists():
                    pending.append(task_id)
                    continue
                check = json.loads(report.read_text())
                if not check["passed"]:
                    raise RuntimeError(f"Environment preflight failed: {task_id}: {check.get('error')}")
                assert check["task_id"] == task_id and check["image_id"] == pinned[task_id]["image_id"]
                assert check["model_requests"] == 0
                assert check["network_checks"]["network_policy"] == "dependencies-only-v1" and check["base_comparison"]["passed"]
            if not pending:
                break
            if subprocess.run(["systemctl", "--user", "is-active", "--quiet", "rho-clean-environment-checks.service"]).returncode:
                raise RuntimeError(f"{len(pending)} evaluation environments remain unverified")
            log(f"Waiting for {len(pending)} evaluation environment preflights before any evaluation model calls")
            time.sleep(15)
    dataset_path = Path(config["dataset_spec"].split(":", 1)[1])
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == config["dataset_digest"]
    assert policy["model"] == config["model"] and policy["reasoning_effort"] == config["reasoning_effort"]
    assert policy["solve_attempts"] == 1 and policy["cache"] == "off"
    binary = codex_binary(policy['codex_version'])
    assert binary.is_file()
    result_path = source / policy["pairwise_results"]
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if result and policy.get("pairwise_results_format") == "rho_summary":
        assert len(result["rounds"]) == 1
    elif result:
        assert result["completed"] == 30
    plan = {"policy": policy, "evaluation_task_ids": ids, "dataset_sha256": config["dataset_digest"],
            "codex_config_sha256": hashlib.sha256((source / "codex_config.toml").read_bytes()).hexdigest(),
            "baseline_harness_id": "h_empty", "final_harness_id": result["final_harness_id"] if result else None,
            "pairwise_results": str(result_path), "codex_binary": str(binary),
            "execution_started": args.execute}
    output_root = ROOT / policy["output_root"]
    write_json(output_root / "evaluation_plan.json", plan)
    if not args.execute:
        print(json.dumps({"prepared": True, "tasks_per_arm": len(ids),
            "max_model_requests_per_task": policy["max_model_requests_per_task"],
            "final_harness_ready": result is not None, "execution_started": False}))
        return 0
    arms = policy["arms"] if args.arm == "both" else [args.arm]
    if "final" in arms and not result:
        raise RuntimeError("Final pairwise evaluation must finish before final-harness grading")
    if args.grade_after_baseline and args.arm != "final":
        raise ValueError("--grade-after-baseline requires --arm final")
    lock = (output_root / ("final.process.lock" if args.arm == "final" else "process.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launcher = runpy.run_path(str(ROOT / "scripts/rho-ds.py"))
    launcher["configure_environment"]()
    configure_network()
    spec = importlib.util.spec_from_file_location("heldout_runtime", ROOT / "scripts/select-swebench-v1-coreset.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    class EvaluationTask(runtime.SelectionTask):
        def materialize(self, dest):
            if not policy.get("container_images"):
                return super().materialize(dest)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "prompt.md").write_text(self._prompt())
            (dest / "repo").mkdir(exist_ok=True)

        def _prompt(self):
            if not policy.get("container_images"):
                return super()._prompt()
            from rho.datasets.swebench_pro.dataset import SWEbenchProTask
            prompt = SWEbenchProTask._prompt(self)
            prompt = "\n".join(line for line in prompt.splitlines() if not line.startswith("Instance ID:"))
            return prompt + "\nSolve independently in this task container. Installed Python/Go/Node/system dependencies are available. Package managers may download dependencies through the configured audited proxy. Do not search the web for answers, upstream fixes, reference patches or newer versions of this project. Search sites and code-hosting sites are blocked; repository history, other tasks, hidden grading tests and prior experiment outputs are unavailable.\n"

        @property
        def agent_timeout_s(self):
            return policy["solve_timeout_seconds"]

    source_hs = FilesystemHarnessStore(source / "harness")
    dataset = SWEbenchProDataset(str(dataset_path), harness_store=source_hs, seed=manifest["split_seed"])
    selected = list(dataset.val)[:100]
    assert [t.id for t in selected] == ids
    images = json.loads((ROOT / policy["container_images"]).read_text()) if policy.get("container_images") else {}
    dependency_policies = {}
    if images:
        for task_id in ids:
            report = (ROOT / policy["container_images"]).parent / "environment_preflight/results" / (hashlib.sha256(task_id.encode()).hexdigest()+".json")
            dependency_policies[task_id] = json.loads(report.read_text())["dependency_policy"]
    tasks = [EvaluationTask(_row=dict(t.row, dockerhub_tag="", image_name=images[t.id]["image_id"], _rho_network_disabled=True,
                    _rho_dependency_policy=dependency_policies[t.id],
                    _rho_grading_environment=policy.get('grading_environment', {}).get(t.id)) if images else t.row,
             _harness=t.harness, _repo_cache=runtime.HistoryFreeRepoCache(),
             _docker_pull="never" if images else "missing") for t in selected]
    configure_global_codex_pool(policy["solve_workers"])
    summaries = {}
    for arm in arms:
        final_lock = None
        if arm == "final" and args.arm == "both":
            final_lock = (output_root / "final.process.lock").open("a")
            fcntl.flock(final_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = output_root / arm
        run.mkdir(parents=True, exist_ok=True)
        harness_id = plan[f"{arm}_harness_id"]
        arm_plan = {"arm": arm, "harness_id": harness_id, "task_ids": ids, "policy": policy,
                    "dataset_sha256": plan["dataset_sha256"], "codex_config_sha256": plan["codex_config_sha256"]}
        if (run / "config.json").exists():
            assert json.loads((run / "config.json").read_text()) == arm_plan, "Evaluation inputs changed"
        else:
            write_json(run / "config.json", arm_plan)
        store = FilesystemTrajectoryStore(run / "trajectories")
        harness = source_hs.get(harness_id)
        agent_options = {}
        AgentClass = RequestLimitedCodexAgent
        if policy.get("container_images"):
            from rho.agent.container_codex import ContainerCodexAgent
            AgentClass = ContainerCodexAgent
            agent_options.update(images_path=ROOT / policy["container_images"], audit_root=run / "runtime")
        agent = AgentClass(source / "codex_config.toml", **agent_options,
            maximum=policy["max_model_requests_per_task"], budgets_root=run / "request_budgets",
            model=policy["model"], reasoning_effort=policy["reasoning_effort"], binary=str(binary),
            default_timeout_s=policy["solve_timeout_seconds"], sandbox="danger-full-access",
            stream_logs=True, codex_auth_home=run / "empty-auth")
        status_lock = threading.Lock()
        resume_path = run / "resume_workspaces.json"
        resume_workspaces = json.loads(resume_path.read_text()) if resume_path.exists() else {}
        active_tasks = {}
        failures = {}

        def status():
            with status_lock:
                payload = {"arm": arm, "total": len(tasks),
                    "solved": len(list((run / "solves").glob("*.json"))),
                    "completed": len(list((run / "grades").glob("*.json"))),
                    "active_tasks": dict(active_tasks), "failures": dict(failures),
                    "solve_workers": policy["solve_workers"], "grade_workers": policy["grade_workers"],
                    "max_model_requests_per_task": policy["max_model_requests_per_task"]}
                payload["solve_timeout_seconds"] = policy["solve_timeout_seconds"]
                payload["concurrency_mode"] = "independent_solve_and_grade_pools"
                payload["active_solves"] = sum(p == "solving" for p in active_tasks.values())
                payload["active_grades"] = sum(p == "grading" for p in active_tasks.values())
                payload["waiting_for_grade"] = sum(p == "waiting_for_grade" for p in active_tasks.values())
                write_json(run / "status.json", payload)
                publish_overall_status(output_root, policy["arms"])

        status()
        log(f"Starting {arm}: {len(tasks)} tasks, harness={harness_id}, model requests/task={policy['max_model_requests_per_task']}, solve_workers={policy['solve_workers']}, grade_workers={policy['grade_workers']} (independent pools)")

        def solve_one(task):
            key = hashlib.sha256(task.id.encode()).hexdigest()
            checkpoint = run / "grades" / f"{key}.json"
            if checkpoint.exists():
                existing = json.loads(checkpoint.read_text())
                if not existing["grade"]["details"].get("error"):
                    return
                failed = run / "failed_grade_attempts" / f"{key}-{uuid.uuid4().hex[:8]}.json"
                failed.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.rename(failed)
            with status_lock:
                active_tasks[task.id] = "solving"
            status()
            saved_solve = run / "solves" / f"{key}.json"
            if not saved_solve.exists():
                retained = resume_workspaces.get(task.id)
                if retained and Path(retained).exists():
                    log(f"{arm} resuming preserved workspace: {task.id}")
                    trajectory = resume_interrupted_solve(agent, task, harness, Path(retained),
                        run / "workdir", f"heldout_{arm}")
                else:
                    trajectory = solve(agent, task, harness, workdir=run / "workdir", stage=f"heldout_{arm}")
                store.put(trajectory)
                write_json(saved_solve, {"task_id": task.id, "trajectory_id": trajectory.id})
                log(f"{arm} solve saved: {task.id} exit={trajectory.exit_code} timeout={trajectory.timed_out}")
            with status_lock:
                active_tasks[task.id] = "waiting_for_grade"
            status()

        def grade_one(task):
            key = hashlib.sha256(task.id.encode()).hexdigest()
            checkpoint = run / "grades" / f"{key}.json"
            if checkpoint.exists():
                return json.loads(checkpoint.read_text())
            if args.grade_after_baseline:
                while not baseline_grading_complete(output_root):
                    time.sleep(5)
            with status_lock:
                active_tasks[task.id] = "grading"
            status()
            saved_solve = run / "solves" / f"{key}.json"
            trajectory = store.get(json.loads(saved_solve.read_text())["trajectory_id"])
            grade = normalize_terminal_grade(task.grade(trajectory, artifacts_dir=run / "grade_artifacts" / key))
            record = {"task_id": task.id, "trajectory_id": trajectory.id, "harness_id": harness_id,
                      "grade": asdict(grade), "model_request_budget": next(
                          (e for e in reversed(trajectory.events) if e.get("type") == "model_request_budget"), None)}
            if grade.details.get("error"):
                write_json(run / "failed_grade_attempts" / f"{key}-{uuid.uuid4().hex[:8]}.json", record)
                raise RuntimeError(f"Grading infrastructure failure: {grade.details['error']}: {grade.details.get('message', '')}")
            write_json(checkpoint, record)
            with status_lock:
                active_tasks.pop(task.id, None)
            status()
            log(f"{arm} grade saved: {task.id} score={grade.score} error={grade.details.get('error')}")
            return record

        def on_error(task, stage, exc):
            tid = task.id
            with status_lock:
                failures[tid] = f"{stage}: {type(exc).__name__}: {exc}"
                active_tasks.pop(tid, None)
            status()
            log(f"{arm} task failed: {tid}: {failures[tid]}")

        records = run_evaluation_pipeline(tasks, solve_one, grade_one,
            solve_workers=policy["solve_workers"], grade_workers=policy["grade_workers"], on_error=on_error)
        if failures:
            raise RuntimeError(f"{arm}: {len(failures)} tasks failed; saved checkpoints will be reused on restart")
        summaries[arm] = {"n": len(records), "mean_score": sum(x["grade"]["score"] for x in records) / len(records),
                          "harness_id": harness_id, "max_model_requests_per_task": policy["max_model_requests_per_task"]}
        write_json(run / "summary.json", summaries[arm])
        publish_overall_status(output_root, policy["arms"])
        log(f"Completed {arm}: mean_score={summaries[arm]['mean_score']}")
        if final_lock is not None:
            final_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
