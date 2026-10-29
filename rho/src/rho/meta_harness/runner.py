from __future__ import annotations

import json
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rho.agent.base import Agent
from rho.meta_harness.history import build_history_dir
from rho.meta_harness.prompts import render_proposer_instructions
from rho.meta_harness.store import (
    CandidateRecord,
    append_record,
    best_record,
    load_records,
    write_frontier,
)
from rho.observability import annotate_trajectory
from rho.protocols import Harness, HarnessStore, Task, Trajectory, TrajectoryStore
from rho.reporting import GradedSolve, grade_on_split


@dataclass
class MetaHarnessResult:
    records: list[CandidateRecord]
    best: CandidateRecord
    test_grades: list[GradedSolve]


def _evaluate_candidate(
    *,
    agent: Agent,
    harness: Harness,
    search_tasks: list[Task],
    workdir: Path,
    traj_store: TrajectoryStore,
    search_trials: int,
    solve_workers: int,
    iteration: int,
    name: str,
    hypothesis: str,
    parent: str | None,
) -> CandidateRecord:
    """Solve every search-set task `search_trials` times and grade with ground truth.

    `search_trials` is the number of independent evaluation samples per task.
    grade_on_split's own `solve_attempts` (left at its default) is an orthogonal
    transport-failure retry, not an extra trial, and is intentionally not changed.
    """
    per_task_scores: dict[str, list[float]] = {task.id: [] for task in search_tasks}
    per_task_passed: dict[str, list[bool]] = {task.id: [] for task in search_tasks}
    solve_traj_ids: list[str] = []
    for trial in range(search_trials):
        graded = grade_on_split(
            agent,
            harness,
            search_tasks,
            workdir,
            traj_store=traj_store,
            stage=f"meta_harness_eval:iter{iteration}:trial{trial}",
            solve_workers=solve_workers,
        )
        for solved in graded:
            per_task_scores[solved.task.id].append(solved.grade.score)
            per_task_passed[solved.task.id].append(solved.grade.passed)
            solve_traj_ids.append(solved.trajectory.id)

    per_task = {
        task_id: statistics.mean(scores)
        for task_id, scores in per_task_scores.items()
        if scores
    }
    all_passed = [ok for column in per_task_passed.values() for ok in column]
    mean_score = statistics.mean(per_task.values()) if per_task else 0.0
    pass_rate = (sum(all_passed) / len(all_passed)) if all_passed else 0.0
    return CandidateRecord(
        iteration=iteration,
        harness_id=harness.id,
        name=name,
        hypothesis=hypothesis,
        parent=parent,
        per_task=per_task,
        mean_score=mean_score,
        pass_rate=pass_rate,
        solve_traj_ids=solve_traj_ids,
    )


def _propose(
    *,
    agent: Agent,
    ws: Path,
    instructions: str,
    harness_store: HarnessStore,
) -> tuple[Trajectory, list[tuple[Harness, dict]]]:
    """Run one proposer session; capture every candidate listed in proposed/manifest.json."""
    traj = agent.run(ws, instructions, task_id="*", harness_id="", kind="optimize")
    traj = annotate_trajectory(traj, agent=agent, stage="meta_harness_propose")

    captured: list[tuple[Harness, dict]] = []
    manifest_path = ws / "proposed" / "manifest.json"
    if traj.exit_code != 0 or traj.timed_out or not manifest_path.exists():
        return traj, captured

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return traj, captured

    for entry in manifest.get("candidates", []):
        cand_dir = ws / "proposed" / str(entry.get("dir", ""))
        if not cand_dir.is_dir():
            continue
        captured.append((harness_store.capture(cand_dir), entry))
    return traj, captured


def _persist_reports(src_reports: Path, dest_reports: Path) -> None:
    """Copy newly written proposer post-mortems out of the per-iteration workspace."""
    if not src_reports.exists():
        return
    dest_reports.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_reports.glob("*.md")):
        target = dest_reports / path.name
        if not target.exists():
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def run_meta_harness(
    *,
    agent: Agent,
    search_tasks: list[Task],
    test_tasks: list[Task],
    seed_harness: Harness,
    harness_store: HarnessStore,
    traj_store: TrajectoryStore,
    run_meta_dir: Path,
    workdir: Path,
    iterations: int,
    candidates_per_iter: int,
    search_trials: int,
    solve_workers: int,
) -> MetaHarnessResult:
    """Faithful Meta-Harness loop: Phase 0 seed eval, N propose/evaluate iterations,
    Phase Final test eval on the best harness over the whole population."""
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    summary_path = run_meta_dir / "summary.jsonl"
    frontier_path = run_meta_dir / "frontier.json"
    reports_dir = run_meta_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    instructions = render_proposer_instructions(candidates_per_iter)

    # Phase 0: evaluate the seed harness so iteration 1 has real traces to read.
    seed_record = _evaluate_candidate(
        agent=agent,
        harness=seed_harness,
        search_tasks=search_tasks,
        workdir=workdir,
        traj_store=traj_store,
        search_trials=search_trials,
        solve_workers=solve_workers,
        iteration=0,
        name="seed",
        hypothesis="dataset built-in harness",
        parent=None,
    )
    append_record(summary_path, seed_record)
    records = load_records(summary_path)
    write_frontier(frontier_path, records)

    # Phase 1..N: each iteration is one Meta-Harness iteration.
    for iteration in range(1, iterations + 1):
        iter_ws = Path(
            tempfile.mkdtemp(dir=str(workdir), prefix=f"mh_iter{iteration}_")
        )
        try:
            (iter_ws / "proposed").mkdir()
            build_history_dir(
                dest=iter_ws / "history",
                summary_path=summary_path,
                frontier_path=frontier_path,
                reports_dir=reports_dir,
                harness_store=harness_store,
                traj_store=traj_store,
            )
            traj, captured = _propose(
                agent=agent,
                ws=iter_ws,
                instructions=instructions,
                harness_store=harness_store,
            )
            traj_store.put(traj)
            _persist_reports(iter_ws / "proposed" / "reports", reports_dir)
            if len(captured) != candidates_per_iter:
                print(
                    f"[meta-harness] iteration {iteration}: proposer produced "
                    f"{len(captured)} candidate(s), expected {candidates_per_iter}"
                )

            for harness, entry in captured:
                record = _evaluate_candidate(
                    agent=agent,
                    harness=harness,
                    search_tasks=search_tasks,
                    workdir=workdir,
                    traj_store=traj_store,
                    search_trials=search_trials,
                    solve_workers=solve_workers,
                    iteration=iteration,
                    name=str(entry.get("name", f"cand_iter{iteration}")),
                    hypothesis=str(entry.get("hypothesis", "")),
                    parent=entry.get("parent"),
                )
                append_record(summary_path, record)
                records = load_records(summary_path)
                write_frontier(frontier_path, records)
        finally:
            shutil.rmtree(iter_ws, ignore_errors=True)

    best = best_record(records)
    assert best is not None  # Phase 0 always appends at least the seed record.
    test_grades: list[GradedSolve] = []
    if test_tasks:
        test_grades = grade_on_split(
            agent,
            harness_store.get(best.harness_id),
            test_tasks,
            workdir,
            traj_store=traj_store,
            stage="meta_harness_test",
            solve_workers=solve_workers,
        )
    return MetaHarnessResult(records=records, best=best, test_grades=test_grades)
