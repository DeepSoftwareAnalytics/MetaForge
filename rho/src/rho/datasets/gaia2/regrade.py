"""Re-grade existing GAIA-2 trajectories with a different judge model.

Reads `<run-dir>/reports/final_val_grades.json`, and for each entry replays
the agent's tool call sequence on a fresh sidecar configured with the chosen
judge provider/model, then writes a `regrade_<provider>_<model>.json` next
to the original grade file.

No codex calls are made — only ARE tool replays + judge LLM calls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rho.datasets.gaia2.dispatcher import render_dispatcher
from rho.datasets.gaia2.ingest import load_rows, parse_payload
from rho.datasets.gaia2.runtime import (
    RuntimeHandle,
    _read_handle,
    _sidecar_ready_timeout_s,
    rpc,
)


def parse_commands(events_jsonl: Path) -> list[str]:
    """Extract codex `command_execution` shell strings in order.

    Returns the verbatim bash commands the agent emitted. These are replayed
    by shell-exec so that heredoc Python blocks (which loop over IDs and
    invoke `are.py` via subprocess) execute the same multi-call sequence
    they did in the original run — a structured parse would silently lose
    every batched call inside such a heredoc.
    """
    cmds: list[str] = []
    for line in events_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "item.completed":
            continue
        item = d.get("item", {})
        if item.get("type") != "command_execution":
            continue
        cmd = item.get("command", "")
        if cmd:
            cmds.append(cmd)
    return cmds


@dataclass
class RegradeResult:
    task_id: str
    trajectory_id: str
    old_score: float
    old_passed: bool
    new_score: float
    new_passed: bool
    new_rationale: str
    new_validation: dict[str, Any]
    replay_steps: int
    error: str | None = None


def _spawn_sidecar(
    workdir: Path,
    scenario_file: Path,
    judge_provider: str,
    judge_model: str,
    log_path: Path,
    fix_flags: dict[str, bool] | None = None,
) -> tuple[subprocess.Popen, RuntimeHandle]:
    env = os.environ.copy()
    env["RHO_GAIA2_ENABLE_JUDGE"] = "1"
    env["RHO_GAIA2_JUDGE_PROVIDER"] = judge_provider
    env["RHO_GAIA2_JUDGE_MODEL"] = judge_model
    if fix_flags:
        if fix_flags.get("filter_env_events"):
            env["RHO_GAIA2_FILTER_ENV_EVENTS"] = "1"
        if fix_flags.get("relax_aui_judge"):
            env["RHO_GAIA2_RELAX_AUI_JUDGE"] = "1"
        if fix_flags.get("filter_fs_reads"):
            env["RHO_GAIA2_FILTER_FS_READS"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "rho.datasets.gaia2.sidecar",
        "--scenario-file",
        str(scenario_file),
        "--workdir",
        str(workdir),
    ]
    log = log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True
        )
    finally:
        log.close()
    handle_file = workdir / ".gaia2" / "handle.json"
    deadline = time.monotonic() + _sidecar_ready_timeout_s()
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"sidecar exited early (rc={proc.returncode}); see {log_path}"
            )
        handle = _read_handle(handle_file, fallback_pid=proc.pid)
        if handle is not None:
            return proc, handle
        time.sleep(0.1)
    proc.terminate()
    raise TimeoutError(f"sidecar not ready within timeout; see {log_path}")


def _shutdown(proc: subprocess.Popen, handle: RuntimeHandle | None) -> None:
    if handle is not None:
        try:
            rpc(handle, {"method": "shutdown"})
        except Exception:
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def regrade_one(
    *,
    scenario_data: dict,
    task_id: str,
    trajectory_id: str,
    old_score: float,
    old_passed: bool,
    trajectory_dir: Path,
    judge_provider: str,
    judge_model: str,
    fix_flags: dict[str, bool] | None = None,
) -> RegradeResult:
    """Replay one trajectory and validate with the new judge."""
    proc: subprocess.Popen | None = None
    handle: RuntimeHandle | None = None
    n_steps = 0
    tmpdir: str | None = None
    try:
        commands = parse_commands(trajectory_dir / "events.jsonl")
        n_steps = len(commands)
        tmpdir = tempfile.mkdtemp(prefix=f"regrade-{trajectory_id}-")
        agent_cwd = Path(tmpdir)
        task_dir = agent_cwd / "task"
        runtime_dir = task_dir / ".gaia2"
        tools_dir = task_dir / "tools"
        state_dir = task_dir / ".gaia2_state"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        tools_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        scenario_file = runtime_dir / "scenario.json"
        scenario_file.write_text(
            json.dumps(scenario_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (tools_dir / "are.py").write_text(render_dispatcher(), encoding="utf-8")
        (tools_dir / "are.py").chmod(0o755)
        (tools_dir / "catalog.json").write_text("{}\n", encoding="utf-8")
        log_path = runtime_dir / "sidecar.log"
        proc, handle = _spawn_sidecar(
            task_dir, scenario_file, judge_provider, judge_model, log_path, fix_flags
        )
        # Shell-exec each command from agent_cwd so `task/tools/are.py` resolves
        # and so heredoc Python blocks that invoke `are.py` via subprocess
        # execute every call they originally did.
        for cmd in commands:
            try:
                subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(agent_cwd),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=180,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                pass
        resp = rpc(handle, {"method": "validate"})
        result = resp.get("result") if resp.get("ok") else None
        new_passed = False
        new_rationale = ""
        if isinstance(result, dict):
            new_passed = bool(result.get("success", False))
            new_rationale = str(result.get("rationale") or "")
        return RegradeResult(
            task_id=task_id,
            trajectory_id=trajectory_id,
            old_score=old_score,
            old_passed=old_passed,
            new_score=1.0 if new_passed else 0.0,
            new_passed=new_passed,
            new_rationale=new_rationale,
            new_validation=result if isinstance(result, dict) else {},
            replay_steps=n_steps,
        )
    except Exception as exc:
        return RegradeResult(
            task_id=task_id,
            trajectory_id=trajectory_id,
            old_score=old_score,
            old_passed=old_passed,
            new_score=0.0,
            new_passed=False,
            new_rationale="",
            new_validation={},
            replay_steps=n_steps,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if proc is not None:
            _shutdown(proc, handle)
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def regrade_run(
    *,
    run_dir: Path,
    judge_provider: str,
    judge_model: str,
    limit: int | None = None,
    workers: int = 4,
    output_path: Path | None = None,
    task_filter: str | None = None,
    fix_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Regrade all tasks in a run's final_val_grades.json with a new judge."""
    grade_file = run_dir / "reports" / "final_val_grades.json"
    if not grade_file.exists():
        raise FileNotFoundError(f"No final_val_grades.json at {grade_file}")
    entries = json.loads(grade_file.read_text(encoding="utf-8"))

    dataset_spec = _resolve_dataset_spec(run_dir)
    payload = parse_payload(dataset_spec)
    rows = load_rows(payload)
    task_id_to_scenario = {f"{r.config}/{r.scenario_id}": r.data for r in rows}

    if task_filter:
        entries = [e for e in entries if task_filter in e.get("task_id", "")]
    if limit is not None:
        entries = entries[:limit]

    jobs = []
    skipped: list[str] = []
    for e in entries:
        sd = task_id_to_scenario.get(e["task_id"])
        if sd is None:
            skipped.append(e["task_id"])
            continue
        jobs.append((e, sd))

    results: list[RegradeResult] = []
    print(f"regrade: {len(jobs)} task(s), {workers} workers, judge={judge_provider}/{judge_model}", flush=True)
    if skipped:
        print(f"skipped (no scenario in dataset): {len(skipped)}", flush=True)
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                regrade_one,
                scenario_data=sd,
                task_id=e["task_id"],
                trajectory_id=e["trajectory_id"],
                old_score=float(e.get("score") or 0.0),
                old_passed=bool((e.get("score") or 0.0) >= 1.0),
                trajectory_dir=run_dir / "trajectories" / e["trajectory_id"],
                judge_provider=judge_provider,
                judge_model=judge_model,
                fix_flags=fix_flags,
            ): e
            for e, sd in jobs
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            flip = ""
            if r.old_passed and not r.new_passed:
                flip = " [P->F]"
            elif not r.old_passed and r.new_passed:
                flip = " [F->P]"
            err = f" err={r.error}" if r.error else ""
            print(
                f"[{i}/{len(jobs)}] {r.task_id} old={r.old_score:.1f} new={r.new_score:.1f}{flip} steps={r.replay_steps}{err}",
                flush=True,
            )
    dur = time.monotonic() - t0

    summary = _summarize(results, duration_s=dur)
    output = {
        "run_dir": str(run_dir),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output


def _summarize(results: list[RegradeResult], *, duration_s: float) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0, "duration_s": duration_s}
    old_pass = sum(1 for r in results if r.old_passed)
    new_pass = sum(1 for r in results if r.new_passed)
    became_pass = sum(1 for r in results if not r.old_passed and r.new_passed)
    became_fail = sum(1 for r in results if r.old_passed and not r.new_passed)
    errors = sum(1 for r in results if r.error)
    return {
        "n": n,
        "old_pass": old_pass,
        "new_pass": new_pass,
        "old_rate": round(old_pass / n, 4),
        "new_rate": round(new_pass / n, 4),
        "fail_to_pass": became_pass,
        "pass_to_fail": became_fail,
        "delta": new_pass - old_pass,
        "errors": errors,
        "duration_s": round(duration_s, 1),
    }


def _resolve_dataset_spec(run_dir: Path) -> str:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    spec = cfg.get("dataset_spec")
    if not spec or not spec.startswith("gaia2:"):
        raise ValueError(f"run {run_dir} is not a gaia2 run: dataset_spec={spec!r}")
    return spec[len("gaia2:") :]


def main(argv: list[str] | None = None) -> int:
    from rho.datasets.gaia2.sidecar import (
        _DEFAULT_JUDGE_MODEL_AZURE,
        _DEFAULT_JUDGE_MODEL_OPENROUTER,
    )

    p = argparse.ArgumentParser(
        description="Re-grade existing GAIA-2 trajectories with a different judge."
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "--judge-provider", choices=["azure", "openrouter"], default="openrouter"
    )
    p.add_argument(
        "--judge-model", default=None,
        help="Override default judge model for the chosen provider.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task-filter", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--filter-env-events", action="store_true",
        help="Drop ENV/USER events from oracle/agent tool-count check.",
    )
    p.add_argument(
        "--relax-aui-judge", action="store_true",
        help="Accept agent AUI message when oracle content is short/null and substring-matches.",
    )
    p.add_argument(
        "--filter-fs-reads", action="store_true",
        help="Drop SandboxLocalFileSystem read-only ops from agent counter.",
    )
    args = p.parse_args(argv)

    fix_flags = {
        "filter_env_events": bool(args.filter_env_events),
        "relax_aui_judge": bool(args.relax_aui_judge),
        "filter_fs_reads": bool(args.filter_fs_reads),
    }

    if args.judge_model is None:
        args.judge_model = (
            _DEFAULT_JUDGE_MODEL_OPENROUTER
            if args.judge_provider == "openrouter"
            else _DEFAULT_JUDGE_MODEL_AZURE
        )
    if args.output is None:
        safe_model = args.judge_model.replace("/", "_")
        suffix_parts = []
        if fix_flags["filter_env_events"]:
            suffix_parts.append("filtenv")
        if fix_flags["relax_aui_judge"]:
            suffix_parts.append("relaxaui")
        if fix_flags["filter_fs_reads"]:
            suffix_parts.append("filtfs")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        args.output = args.run_dir / "reports" / (
            f"regrade_{args.judge_provider}_{safe_model}{suffix}.json"
        )

    result = regrade_run(
        run_dir=args.run_dir,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        limit=args.limit,
        workers=args.workers,
        output_path=args.output,
        task_filter=args.task_filter,
        fix_flags=fix_flags,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
