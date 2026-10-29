"""Clean diagnosis-content ablation.

Reuses diagnoses from a prior diagnosis-strategy evolve run (e.g.
runs/exp-rho-swebench), drops one or more Diagnosis fields, then re-runs the
optimize -> solve_after -> evaluate -> val_grade pipeline using the FULL
OPTIMIZE_INSTRUCTIONS prompt. Compare to the existing diagnosis-no-consistency
/ diagnosis-no-validation ablations which also modified the diagnose and
optimize prompts; this script changes only the data shown to the optimize
agent.

Variants (always also drop harness_improvement_direction):
  no-consistency: drop inconsistency_analysis
  no-validation:  drop trajectory_analyses + failure_mode_analysis

Output run-dir mirrors `rho evolve` so the result can be compared
directly to runs in runs/exp-abl-diag-*.

Usage:
    uv run --extra swebench-pro scripts/clean_ablation_diagnosis.py \\
        --source-run runs/exp-rho-swebench \\
        --variant no-consistency \\
        --run-dir runs/exp-abl-clean-noconsis-swebench
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import statistics
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rho.agent.base import Agent
from rho.agent.cache import build_default_agent
from rho.agent.codex import CodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.cli import _capture_environment
from rho.datasets.loader import load_dataset
from rho.loop import (
    CandidateResult,
    _parallel_map,
    _serialize_candidate_pool,
    _write_harness_diff,
    _write_json,
)
from rho.observability import extract_usage, usage_summary
from rho.orchestrators.evaluate import evaluate
from rho.orchestrators.solve import solve_in, solve_workspace
from rho.protocols import Diagnosis, Harness, Task, Trajectory, TrajectoryAnalysis
from rho.reporting import GradedSolve, grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies._common import optimize_agent_call, parallel_map
from rho.strategies.diagnose import OPTIMIZE_INSTRUCTIONS, _dump_diagnosis

VARIANTS = ("no-consistency", "no-validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True, type=Path,
                   help="Prior evolve run dir with diagnoses to reuse (e.g. runs/exp-rho-swebench)")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--codex-config", type=Path,
                   default=Path("configs/codex.azure-foundry.toml"),
                   help="Codex config TOML. Default: configs/codex.azure-foundry.toml.")
    p.add_argument("--model", default=None,
                   help="Codex model. Default: inherit from source-run config.json.")
    p.add_argument("--reasoning-effort", default=None,
                   help="Codex reasoning effort. Default: inherit from source-run.")
    p.add_argument("--codex-concurrency", type=int, default=10)
    p.add_argument("--grade-workers", type=int, default=10)
    p.add_argument("--optimize-samples", type=int, default=3)
    p.add_argument("--max-grading-tasks", type=int, default=None,
                   help="Cap val tasks. Default: inherit max_grading_tasks from source-run.")
    p.add_argument("--max-evolve-tasks", type=int, default=None,
                   help="Cap train tasks (for smoke tests). Default: all selected tasks.")
    p.add_argument("--check-only", action="store_true",
                   help="Render diagnosis previews then exit. Skips optimize / solve / grade.")
    return p.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ablate(diagnoses: list[Diagnosis], variant: str) -> list[Diagnosis]:
    """Drop diagnosis fields per variant. harness_improvement_direction is
    always blanked so the optimize agent cannot lean on whatever bullet the
    diagnose agent suggested (which often paraphrases the dropped section)."""
    if variant == "no-consistency":
        return [
            replace(d, inconsistency_analysis="", harness_improvement_direction="")
            for d in diagnoses
        ]
    if variant == "no-validation":
        return [
            replace(
                d,
                trajectory_analyses=[],
                failure_mode_analysis="",
                harness_improvement_direction="",
            )
            for d in diagnoses
        ]
    raise ValueError(f"unknown variant: {variant}")


def _reconstruct_diagnosis(raw: dict[str, Any]) -> Diagnosis:
    analyses = [
        TrajectoryAnalysis(
            trajectory=str(a.get("trajectory", "")),
            successful=int(a.get("successful", 0)),
            quality_analysis=str(a.get("quality_analysis", "")),
            issues=str(a.get("issues", "")),
        )
        for a in raw.get("trajectory_analyses", []) or []
    ]
    return Diagnosis(
        task_id=str(raw["task_id"]),
        trajectory_analyses=analyses,
        failure_mode_analysis=str(raw.get("failure_mode_analysis", "")),
        inconsistency_analysis=str(raw.get("inconsistency_analysis", "")),
        harness_improvement_direction=str(raw.get("harness_improvement_direction", "")),
        severity=float(raw.get("severity", 1.0)),
    )


def _build_agent(args: argparse.Namespace, source_config: dict[str, Any], run_dir: Path) -> Agent:
    model = args.model or source_config["model"]
    reasoning_effort = args.reasoning_effort or source_config["reasoning_effort"]
    codex_config_path = args.codex_config.expanduser().resolve()
    if not codex_config_path.is_file():
        print(f"--codex-config not found: {codex_config_path}", file=sys.stderr)
        raise SystemExit(2)
    # Persist resolved config in run-dir for audit (matches cli._audit_codex_config).
    shutil.copy(codex_config_path, run_dir / "codex_config.toml")
    configure_global_codex_pool(args.codex_concurrency)
    return build_default_agent(
        CodexAgent(
            codex_config_path=codex_config_path,
            model=model,
            reasoning_effort=reasoning_effort,
            isolate_codex_home=True,
            ephemeral=True,
        ),
        mode="off",
        cache_dir=None,
    )


def _copy_harness(
    src_store: FilesystemHarnessStore,
    src_id: str,
    dst_store: FilesystemHarnessStore,
) -> Harness:
    src_harness = src_store.get(src_id)
    with tempfile.TemporaryDirectory(prefix="abl_copy_") as tmp:
        tmp_dir = Path(tmp) / "h"
        tmp_dir.mkdir()
        src_harness.materialize(tmp_dir)
        return dst_store.capture(tmp_dir)


def _copy_trajectory(
    src_store: FilesystemTrajectoryStore,
    dst_store: FilesystemTrajectoryStore,
    traj_id: str,
) -> Trajectory:
    traj = src_store.get(traj_id)
    try:
        dst_store.put(traj)
    except FileExistsError:
        pass
    return traj


def _build_workspace_fn(train_tasks: list[Task], diagnoses: list[Diagnosis]):
    """Return a workspace_builder that dumps ablated diagnoses for the
    optimize agent. Ranks tasks by severity DESC, matching DiagnoseStrategy."""
    def build(ws: Path) -> None:
        diagnoses_dir = ws / "diagnoses"
        diagnoses_dir.mkdir()
        ranked = sorted(
            enumerate(zip(train_tasks, diagnoses)),
            key=lambda item: (-item[1][1].severity, item[0]),
        )
        for task_ix, (_orig_ix, (task, diag)) in enumerate(ranked):
            _dump_diagnosis(
                diagnoses_dir / f"task_{task_ix:04d}",
                task,
                diag,
                # Fields have already been cleared in `diagnoses`; we just
                # need to also suppress the harness_improvement_direction
                # section. Keep the section toggles ON so empty fields drop
                # naturally rather than via the section flag (this lets the
                # SAME _dump_diagnosis logic handle every variant uniformly).
                include_consistency=True,
                include_validation=True,
                include_direction=False,
            )
    return build


def _preview_ablated(train_tasks: list[Task], diagnoses: list[Diagnosis], run_dir: Path) -> None:
    """Render the first task's diagnosis.md so we can eyeball that the
    ablation actually removed the intended sections."""
    preview = run_dir / "diagnoses_preview"
    preview.mkdir(parents=True, exist_ok=True)
    # Severity-ranked top task (same ordering the optimize agent sees).
    ranked = sorted(
        enumerate(zip(train_tasks, diagnoses)),
        key=lambda item: (-item[1][1].severity, item[0]),
    )
    _orig_ix, (task, diag) = ranked[0]
    _dump_diagnosis(
        preview / "task_0000",
        task,
        diag,
        include_consistency=True,
        include_validation=True,
        include_direction=False,
    )
    md = (preview / "task_0000" / "diagnosis.md").read_text(encoding="utf-8")
    sections = [line for line in md.splitlines() if line.startswith("## ")]
    print(f"[preview] task_0000 severity={diag.severity:.2f}", file=sys.stderr)
    print(f"[preview] sections present: {sections}", file=sys.stderr)
    print(f"[preview] markdown chars: {len(md)}", file=sys.stderr)
    print(f"[preview] full markdown at {preview / 'task_0000' / 'diagnosis.md'}", file=sys.stderr)


def _build_candidate_pool(
    optimize_samples: list[tuple[Trajectory, Harness | None]],
) -> tuple[list[CandidateResult], list[dict[str, Any]]]:
    candidates_by_id: dict[str, CandidateResult] = {}
    pool: list[CandidateResult] = []
    sample_payload: list[dict[str, Any]] = []
    for sample_index, (opt_traj, candidate) in enumerate(optimize_samples):
        candidate_id = candidate.id if candidate is not None else None
        sample_payload.append({
            "sample_index": sample_index,
            "optimize_traj_id": opt_traj.id,
            "candidate_harness_id": candidate_id,
        })
        if candidate is None:
            continue
        existing = candidates_by_id.get(candidate.id)
        if existing is None:
            cr = CandidateResult(
                candidate=candidate,
                sample_indices=[sample_index],
                optimize_traj_ids=[opt_traj.id],
            )
            candidates_by_id[candidate.id] = cr
            pool.append(cr)
        else:
            existing.sample_indices.append(sample_index)
            existing.optimize_traj_ids.append(opt_traj.id)
    return pool, sample_payload


def _format_summary(summary: dict[str, Any]) -> str:
    """Mirror cli._format_summary so summary.txt matches existing runs."""
    lines = [
        f"initial harness: {summary.get('initial_harness_id', '(unknown)')}",
        f"final harness: {summary.get('final_harness_id', '(unknown)')}",
        "",
        "rounds:",
    ]
    for r in summary.get("rounds", []):
        lines.append(
            f"  round {r['round_ix']}: mean_score={r['mean_score']:.2f} "
            f"accepted={str(r['accepted']).lower()} candidate={r['candidate_harness_id']} "
            f"winner_sample={r['winner_sample_index']} "
            f"unique_candidates={r['unique_candidate_count']}"
        )
    final_val = summary.get("final_val") or {}
    if final_val.get("mean_score") is not None:
        lines += [
            "",
            "val:",
            f"  final: mean_score={final_val['mean_score']:.2f} (n={final_val['n']})",
        ]
    else:
        lines += ["", "val: skipped"]
    return "\n".join(lines) + "\n"


def _serialize_grades(grades: list[GradedSolve]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": r.task.id,
            "harness_id": r.trajectory.harness_id,
            "trajectory_id": r.trajectory.id,
            "stage": r.stage,
            "prediction": r.grade.details.get("prediction", r.trajectory.final_message),
            "score": r.grade.score,
            "details": r.grade.details,
        }
        for r in grades
    ]


def _counts_by_attr(items, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = getattr(item, attr) or "(none)"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    source_run = args.source_run.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "rounds" / "round_0").mkdir(parents=True, exist_ok=True)
    (run_dir / "workdir").mkdir(exist_ok=True)

    source_round = source_run / "rounds" / "round_0"
    source_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))

    # ---- 1. Load source artifacts -----------------------------------------
    selection = json.loads((source_run / "selection.json").read_text(encoding="utf-8"))
    train_task_ids: list[str] = list(selection["selected_task_ids"])
    if args.max_evolve_tasks is not None:
        train_task_ids = train_task_ids[: args.max_evolve_tasks]

    raw_diags = json.loads((source_round / "diagnoses.json").read_text(encoding="utf-8"))
    # source diagnoses are ordered by task_id matching selection order; the
    # task_id field on each diagnosis lets us verify we line them up correctly.
    diagnoses_by_id = {d["task_id"]: _reconstruct_diagnosis(d) for d in raw_diags}
    missing = [tid for tid in train_task_ids if tid not in diagnoses_by_id]
    if missing:
        raise RuntimeError(
            f"selection has tasks with no diagnosis in source run: {missing[:3]}"
        )
    diagnoses = [diagnoses_by_id[tid] for tid in train_task_ids]

    before_traj_groups = json.loads(
        (source_round / "solve_before_traj_ids.json").read_text(encoding="utf-8")
    )
    if len(before_traj_groups) != len(selection["selected_task_ids"]):
        raise RuntimeError(
            f"solve_before_traj_ids has {len(before_traj_groups)} groups; "
            f"expected {len(selection['selected_task_ids'])} (one per selected task)"
        )
    # Trim to match maybe-truncated train_task_ids (smoke-test mode).
    before_traj_groups = before_traj_groups[: len(train_task_ids)]

    initial_harness_id = (source_round / "input_harness_id").read_text(encoding="utf-8").strip()

    # ---- 2. Stores --------------------------------------------------------
    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    src_harness_store = FilesystemHarnessStore(source_run / "harness")
    src_traj_store = FilesystemTrajectoryStore(source_run / "trajectories")

    initial = _copy_harness(src_harness_store, initial_harness_id, harness_store)
    # Copy all 3 before trajectories per task (matches the 3 solve_before
    # samples evolve does). The eval step only uses group[0], but downstream
    # tooling (webui) expects the full 3-element groups on disk.
    for group in before_traj_groups:
        for traj_id in group:
            _copy_trajectory(src_traj_store, traj_store, traj_id)
    before_trajs = [src_traj_store.get(group[0]) for group in before_traj_groups]

    # Copy diagnose trajectories from the source so the round_0 record can
    # link back to the original diagnoses (we do NOT re-run diagnose).
    diagnose_traj_ids = json.loads(
        (source_round / "diagnose_traj_ids.json").read_text(encoding="utf-8")
    )
    diagnose_traj_ids = diagnose_traj_ids[: len(train_task_ids)]
    for traj_id in diagnose_traj_ids:
        _copy_trajectory(src_traj_store, traj_store, traj_id)

    # ---- 3. Dataset & tasks ----------------------------------------------
    dataset = load_dataset(
        source_config["dataset_spec"],
        harness_store=harness_store,
        docker_pull=source_config.get("docker_pull", "missing"),
    )
    train_by_id = {t.id: t for t in dataset.train}
    val_tasks_for_grade = dataset.val
    missing = [tid for tid in train_task_ids if tid not in train_by_id]
    if missing:
        raise RuntimeError(f"train tasks missing from dataset: {missing[:3]}")
    train_tasks = [train_by_id[tid] for tid in train_task_ids]

    # ---- 4. Ablate diagnoses ---------------------------------------------
    diagnoses = _ablate(diagnoses, args.variant)
    _preview_ablated(train_tasks, diagnoses, run_dir)

    # ---- 5. Persist run-level metadata (config / selection / environment)
    codex_config_path = args.codex_config.expanduser().resolve()
    codex_config_sha = (
        hashlib.sha256(codex_config_path.read_bytes()).hexdigest()
        if codex_config_path.is_file()
        else ""
    )
    config_payload = {
        "argv": sys.argv,
        "source_run": str(source_run),
        "source_selection_path": str(source_run / "selection.json"),
        "variant": args.variant,
        "dataset_spec": source_config["dataset_spec"],
        "optimize_strategy": f"clean-{args.variant}",
        "optimize_samples": args.optimize_samples,
        "n_rounds": 1,
        "model": args.model or source_config["model"],
        "reasoning_effort": args.reasoning_effort or source_config["reasoning_effort"],
        "codex_concurrency": args.codex_concurrency,
        "grade_workers": args.grade_workers,
        "max_grading_tasks": args.max_grading_tasks,
        "max_evolve_tasks": args.max_evolve_tasks,
        "docker_pull": source_config.get("docker_pull", "missing"),
        "cache_mode": "off",
        "codex_config_path": str(codex_config_path),
        "codex_config_sha256": codex_config_sha,
        "start_timestamp": _now_iso(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config_payload)

    # selection.json — mirrors cli._cmd_evolve so webui / inspect can read it.
    _write_json(run_dir / "selection.json", {
        "source_selection": str(source_run / "selection.json"),
        "inherited_from_source": True,
        "selected_task_ids": train_task_ids,
        "all_candidate_ids": [t.id for t in dataset.train],
    })

    # environment.json — codex version + git sha + uname for reproducibility.
    _write_json(run_dir / "environment.json", _capture_environment())

    if args.check_only:
        print("[abl] --check-only set, exiting after preview render", file=sys.stderr)
        return 0

    # ---- 6. Build agent ---------------------------------------------------
    agent = _build_agent(args, source_config, run_dir)

    # ---- 7. Optimize ------------------------------------------------------
    workspace_builder = _build_workspace_fn(train_tasks, diagnoses)
    print(
        f"[abl] running {args.optimize_samples} optimize sample(s) with full "
        f"OPTIMIZE_INSTRUCTIONS (variant={args.variant})",
        file=sys.stderr,
    )
    optimize_results = parallel_map(
        lambda i: optimize_agent_call(
            agent,
            initial,
            harness_store,
            workspace_builder=workspace_builder,
            instructions=OPTIMIZE_INSTRUCTIONS,
            workdir=run_dir / "workdir",
            stage="round_optimize",
            round_ix=0,
            sample_index=i,
        ),
        list(range(args.optimize_samples)),
    )
    optimize_trajs = [r[0] for r in optimize_results]
    for traj in optimize_trajs:
        try:
            traj_store.put(traj)
        except FileExistsError:
            pass

    candidate_pool, sample_payload = _build_candidate_pool(optimize_results)
    optimize_traj_ids = [t.id for t in optimize_trajs]
    round_dir = run_dir / "rounds" / "round_0"
    (round_dir / "input_harness_id").write_text(initial.id, encoding="utf-8")
    _write_json(
        round_dir / "diagnoses.json",
        [dataclasses.asdict(d) for d in diagnoses],
    )
    (round_dir / "optimize_instructions.txt").write_text(
        OPTIMIZE_INSTRUCTIONS, encoding="utf-8"
    )
    _write_json(
        round_dir / "optimize_input_tokens.json",
        [
            {
                "sample_index": i,
                "input_tokens": (
                    usage["input_tokens"]
                    if (usage := extract_usage(t.events)) is not None
                    else None
                ),
            }
            for i, t in enumerate(optimize_trajs)
        ],
    )

    if not candidate_pool:
        # No candidate harness produced; bail and write empty round artifacts.
        _write_json(round_dir / "optimize_traj_ids.json", optimize_traj_ids)
        (round_dir / "candidate_harness_id").write_text("(none)", encoding="utf-8")
        (round_dir / "accepted").write_text("false", encoding="utf-8")
        (round_dir / "mean_score").write_text("0.0000", encoding="utf-8")
        _write_json(round_dir / "scores.json", [])
        _write_json(
            round_dir / "optimize_candidates.json",
            {"samples": sample_payload, "unique_candidates": [],
             "winner_candidate_harness_id": None, "winner_sample_index": None,
             "reason": "optimize produced no candidate harness"},
        )
        print("[abl] optimize produced no candidate harness; aborting", file=sys.stderr)
        return 1

    # ---- 8. solve_after (each unique candidate × each train task) --------
    after_items = [
        (c_ix, task)
        for c_ix in range(len(candidate_pool))
        for task in train_tasks
    ]

    def _solve_after(item):
        c_ix, task = item
        with solve_workspace(task, candidate_pool[c_ix].candidate, run_dir / "workdir") as ws:
            return solve_in(
                agent, task, candidate_pool[c_ix].candidate, ws,
                stage="round_solve_after", round_ix=0,
            )

    after_results = _parallel_map(_solve_after, after_items, max_workers=args.codex_concurrency)
    for traj in after_results:
        try:
            traj_store.put(traj)
        except FileExistsError:
            pass
    grouped_after: list[list[Trajectory]] = [[] for _ in candidate_pool]
    for (c_ix, _task), traj in zip(after_items, after_results):
        grouped_after[c_ix].append(traj)
    for cr, after in zip(candidate_pool, grouped_after):
        cr.after_trajs = after

    # ---- 9. evaluate (each unique candidate × each task, pairwise) -------
    eval_items = [
        (c_ix, task_ix, task)
        for c_ix in range(len(candidate_pool))
        for task_ix, task in enumerate(train_tasks)
    ]

    def _eval_one(item):
        c_ix, task_ix, task = item
        return evaluate(
            agent, task,
            before_trajs[task_ix],
            candidate_pool[c_ix].after_trajs[task_ix],
            harness_before=initial,
            harness_after=candidate_pool[c_ix].candidate,
            workdir=run_dir / "workdir",
            stage="round_evaluate",
            round_ix=0,
        )

    eval_results = _parallel_map(_eval_one, eval_items, max_workers=args.codex_concurrency)
    grouped_eval: list[list[Trajectory]] = [[] for _ in candidate_pool]
    grouped_scores: list[list] = [[] for _ in candidate_pool]
    for (c_ix, _t_ix, _t), (eval_traj, score) in zip(eval_items, eval_results):
        grouped_eval[c_ix].append(eval_traj)
        grouped_scores[c_ix].append(score)
        try:
            traj_store.put(eval_traj)
        except FileExistsError:
            pass
    for cr, evs, scs in zip(candidate_pool, grouped_eval, grouped_scores):
        cr.eval_trajs = evs
        cr.scores = scs
        cr.mean_score = statistics.mean(s.value for s in scs) if scs else 0.0

    # ---- 10. Pick winner --------------------------------------------------
    winner = max(candidate_pool, key=lambda cr: cr.mean_score)
    accepted = winner.mean_score > 0

    # ---- 11. Persist round_0 artifacts (matches loop.run_round) ----------
    _write_json(round_dir / "optimize_traj_ids.json", optimize_traj_ids)
    representative_opt_id = winner.representative_optimize_traj_id or optimize_traj_ids[0]
    (round_dir / "optimize_traj_id").write_text(representative_opt_id, encoding="utf-8")
    (round_dir / "candidate_harness_id").write_text(winner.candidate.id, encoding="utf-8")
    _write_harness_diff(initial, winner.candidate, round_dir / "candidate_harness_diff.patch")
    _write_json(
        round_dir / "solve_before_traj_ids.json",
        before_traj_groups,
    )
    _write_json(round_dir / "diagnose_traj_ids.json", diagnose_traj_ids)
    _write_json(round_dir / "solve_after_traj_ids.json", [t.id for t in winner.after_trajs])
    _write_json(round_dir / "eval_traj_ids.json", [t.id for t in winner.eval_trajs])
    _write_json(
        round_dir / "scores.json",
        [{"task_id": task.id, "value": s.value, "rationale": s.rationale}
         for task, s in zip(train_tasks, winner.scores)],
    )
    (round_dir / "mean_score").write_text(f"{winner.mean_score:.4f}", encoding="utf-8")
    (round_dir / "accepted").write_text("true" if accepted else "false", encoding="utf-8")
    _write_json(
        round_dir / "optimize_candidates.json",
        _serialize_candidate_pool(train_tasks, sample_payload, candidate_pool, winner, accepted),
    )

    # ---- 12. Val grade ----------------------------------------------------
    max_grading = args.max_grading_tasks
    if max_grading is None:
        max_grading = source_config.get("max_grading_tasks")
    skip_val = max_grading == 0
    if skip_val:
        final_grades: list[GradedSolve] = []
        final_summary = {"mean_score": None, "n": 0}
    else:
        print(f"[abl] grading on val (max_tasks={max_grading})", file=sys.stderr)
        final_grades = grade_on_split(
            agent,
            winner.candidate,
            val_tasks_for_grade,
            run_dir / "workdir",
            max_tasks=max_grading,
            traj_store=traj_store,
            stage="final_val_grade",
            artifacts_root=run_dir / "workdir" / "grade_artifacts",
            max_workers=args.grade_workers,
            solve_workers=args.codex_concurrency,
        )
        final_summary = summarize(final_grades)

    # ---- 13. Final reports -----------------------------------------------
    report_dir = run_dir / "reports"
    _write_json(report_dir / "final_val_grades.json", _serialize_grades(final_grades))

    summary = {
        "initial_harness_id": initial.id,
        "final_harness_id": winner.candidate.id,
        "optimize_strategy": f"clean-{args.variant}",
        "final_val": final_summary,
        "rounds": [{
            "round_ix": 0,
            "optimize_samples": args.optimize_samples,
            "unique_candidate_count": len(candidate_pool),
            "candidate_harness_id": winner.candidate.id,
            "accepted": accepted,
            "mean_score": winner.mean_score,
            "winner_sample_index": winner.winner_sample_index,
            "candidates": [
                {
                    "candidate_harness_id": cr.candidate.id,
                    "sample_indices": cr.sample_indices,
                    "mean_score": cr.mean_score,
                    "accepted": accepted and cr.candidate.id == winner.candidate.id,
                }
                for cr in candidate_pool
            ],
            "scores": [
                {"task_id": task.id, "value": s.value, "rationale": s.rationale}
                for task, s in zip(train_tasks, winner.scores)
            ],
        }],
        "end_timestamp": _now_iso(),
    }
    _write_json(report_dir / "summary.json", summary)
    all_trajs = list(traj_store._iter_all())
    _write_json(report_dir / "usage_summary.json", usage_summary(all_trajs))
    _write_json(report_dir / "manifest.json", {
        "run_dir": str(run_dir),
        "report_files": sorted(
            p.relative_to(run_dir).as_posix() for p in report_dir.iterdir() if p.is_file()
        ),
        "round_dirs": ["rounds/round_0"],
        "trajectory_count": len(all_trajs),
        "trajectory_ids": [t.id for t in all_trajs],
        "trajectory_counts_by_kind": _counts_by_attr(all_trajs, "kind"),
        "trajectory_counts_by_stage": _counts_by_attr(all_trajs, "stage"),
    })
    summary_text = _format_summary(summary)
    (report_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
