from __future__ import annotations

import dataclasses
import difflib
import json
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rho.agent.base import Agent
from rho.observability import extract_usage, is_runtime_scratch
from rho.orchestrators.evaluate import evaluate
from rho.orchestrators.solve import solve_in, solve_workspace
from rho.protocols import (
    Harness,
    HarnessStore,
    OptimizeStrategy,
    Score,
    Task,
    TaskSet,
    Trajectory,
    TrajectoryStore,
)


def _parallel_map(fn, items, *, max_workers: int | None = None):
    """Run fn over items concurrently, return results in input order."""
    if not items:
        return []
    if max_workers is None:
        worker_count = len(items)
    else:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        worker_count = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(fn, items))


@dataclass
class CandidateResult:
    candidate: Harness
    sample_indices: list[int]
    optimize_traj_ids: list[str]
    mean_score: float = 0.0
    scores: list[Score] = field(default_factory=list)
    after_trajs: list[Trajectory] = field(default_factory=list)
    eval_trajs: list[Trajectory] = field(default_factory=list)

    @property
    def winner_sample_index(self) -> int | None:
        return self.sample_indices[0] if self.sample_indices else None

    @property
    def representative_optimize_traj_id(self) -> str | None:
        return self.optimize_traj_ids[0] if self.optimize_traj_ids else None


@dataclass
class RoundResult:
    round_ix: int
    candidate: Harness
    accepted: bool
    mean_score: float
    scores: list[Score]
    score_task_ids: list[str]
    before_trajs: list[Trajectory]
    after_trajs: list[Trajectory]
    winner_sample_index: int | None = None
    candidate_pool: list[CandidateResult] = field(default_factory=list)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _serialize_candidate_pool(
    tasks: list[Task],
    samples: list[dict[str, object]],
    candidate_pool: list[CandidateResult],
    winner: CandidateResult | None,
    accepted: bool,
) -> dict[str, object]:
    winner_id = winner.candidate.id if winner is not None else None
    return {
        "samples": samples,
        "unique_candidates": [
            {
                "candidate_harness_id": candidate_result.candidate.id,
                "sample_indices": candidate_result.sample_indices,
                "optimize_traj_ids": candidate_result.optimize_traj_ids,
                "solve_after_traj_ids": [traj.id for traj in candidate_result.after_trajs],
                "eval_traj_ids": [traj.id for traj in candidate_result.eval_trajs],
                "scores": [
                    {
                        "task_id": task.id,
                        "value": score.value,
                        "rationale": score.rationale,
                    }
                    for task, score in zip(tasks, candidate_result.scores)
                ],
                "mean_score": candidate_result.mean_score,
                "winner": candidate_result.candidate.id == winner_id,
                "accepted": accepted and candidate_result.candidate.id == winner_id,
            }
            for candidate_result in candidate_pool
        ],
        "winner_candidate_harness_id": winner_id,
        "winner_sample_index": winner.winner_sample_index if winner is not None else None,
    }


def _write_empty_candidate_outputs(
    round_dir: Path,
    *,
    optimize_traj_ids: list[str],
    optimize_samples_payload: list[dict[str, object]],
    reason: str,
) -> None:
    _write_json(round_dir / "optimize_traj_ids.json", optimize_traj_ids)
    if optimize_traj_ids:
        (round_dir / "optimize_traj_id").write_text(optimize_traj_ids[0], encoding="utf-8")
    else:
        (round_dir / "optimize_traj_id").write_text("(none)", encoding="utf-8")
    (round_dir / "candidate_harness_id").write_text("(none)", encoding="utf-8")
    (round_dir / "accepted").write_text("false", encoding="utf-8")
    (round_dir / "mean_score").write_text("0.0000", encoding="utf-8")
    (round_dir / "candidate_harness_diff.patch").write_text(
        f"({reason}; no candidate harness)\n",
        encoding="utf-8",
    )
    _write_json(round_dir / "solve_after_traj_ids.json", [])
    _write_json(round_dir / "eval_traj_ids.json", [])
    _write_json(round_dir / "scores.json", [])
    _write_json(
        round_dir / "optimize_candidates.json",
        {
            "samples": optimize_samples_payload,
            "unique_candidates": [],
            "winner_candidate_harness_id": None,
            "winner_sample_index": None,
            "reason": reason,
        },
    )


def _write_harness_diff(before: Harness, after: Harness, dest: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="harness_diff_") as tmp:
        root = Path(tmp)
        before_dir = root / "before"
        after_dir = root / "after"
        before_dir.mkdir()
        after_dir.mkdir()
        before.materialize(before_dir)
        after.materialize(after_dir)

        patch_lines: list[str] = []
        before_files = {
            path.relative_to(before_dir).as_posix(): path
            for path in before_dir.rglob("*")
            if path.is_file() and not is_runtime_scratch(path.relative_to(before_dir))
        }
        after_files = {
            path.relative_to(after_dir).as_posix(): path
            for path in after_dir.rglob("*")
            if path.is_file() and not is_runtime_scratch(path.relative_to(after_dir))
        }
        for rel in sorted(set(before_files) | set(after_files)):
            before_path = before_files.get(rel)
            after_path = after_files.get(rel)
            before_text = (
                before_path.read_bytes().decode("utf-8", errors="replace")
                if before_path
                else ""
            )
            after_text = (
                after_path.read_bytes().decode("utf-8", errors="replace")
                if after_path
                else ""
            )
            if before_text == after_text:
                continue
            diff = difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            patch_lines.extend(diff)
        if not patch_lines:
            patch_lines = ["(no diff)\n"]
        dest.write_text("".join(patch_lines), encoding="utf-8")


def run_round(
    round_ix: int,
    current: Harness,
    tasks: list[Task],
    agent: Agent,
    harness_store: HarnessStore,
    traj_store: TrajectoryStore,
    workdir: Path,
    round_dir: Path,
    *,
    strategy: OptimizeStrategy,
    optimize_samples: int = 3,
    solve_workers: int | None = None,
) -> RoundResult:
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "input_harness_id").write_text(current.id, encoding="utf-8")

    # Step 1: solve each task 3 times, optionally limiting solve/runtime concurrency.
    solve_items = [(task, i) for task in tasks for i in range(3)]

    def _solve_one(item: tuple[Task, int]) -> Trajectory:
        task_, sample_ix = item
        with solve_workspace(task_, current, workdir) as ws:
            return solve_in(
                agent,
                task_,
                current,
                ws,
                sample_index=sample_ix,
                stage="round_solve_before",
                round_ix=round_ix,
            )

    all_trajs = _parallel_map(
        _solve_one,
        solve_items,
        max_workers=solve_workers,
    )
    for traj in all_trajs:
        traj_store.put(traj)
    task_trajs = [all_trajs[i * 3 : (i + 1) * 3] for i in range(len(tasks))]
    before = [group[0] for group in task_trajs]
    _write_json(
        round_dir / "solve_before_traj_ids.json",
        [[traj.id for traj in group] for group in task_trajs],
    )

    # Step 2: strategy-specific analysis + optimize sampling.
    result = strategy.propose_candidates(
        agent=agent,
        harness=current,
        tasks_with_trajectories=list(zip(tasks, task_trajs)),
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=workdir,
        n_samples=optimize_samples,
        round_ix=round_ix,
    )
    if result.diagnose_trajectories is not None:
        _write_json(
            round_dir / "diagnose_traj_ids.json",
            [trajectory.id for trajectory in result.diagnose_trajectories],
        )
    if result.diagnoses is not None:
        _write_json(
            round_dir / "diagnoses.json",
            [dataclasses.asdict(diagnosis) for diagnosis in result.diagnoses],
        )
    optimize_instructions = (
        result.samples[0].optimize_trajectory.instructions if result.samples else ""
    )
    (round_dir / "optimize_instructions.txt").write_text(
        optimize_instructions,
        encoding="utf-8",
    )
    _write_json(
        round_dir / "optimize_input_tokens.json",
        [
            {
                "sample_index": sample.sample_index,
                "input_tokens": (
                    usage["input_tokens"]
                    if (usage := extract_usage(sample.optimize_trajectory.events)) is not None
                    else None
                ),
            }
            for sample in result.samples
        ],
    )
    for sample in result.samples:
        try:
            traj_store.put(sample.optimize_trajectory)
        except FileExistsError:
            pass

    optimize_traj_ids = [sample.optimize_trajectory.id for sample in result.samples]
    candidate_pool: list[CandidateResult] = []
    candidates_by_id: dict[str, CandidateResult] = {}
    sample_payload: list[dict[str, object]] = []
    for sample in result.samples:
        sample_index = sample.sample_index
        opt_traj = sample.optimize_trajectory
        candidate = sample.candidate
        candidate_id = candidate.id if candidate is not None else None
        sample_payload.append(
            {
                "sample_index": sample_index,
                "optimize_traj_id": opt_traj.id,
                "candidate_harness_id": candidate_id,
            }
        )
        if candidate is None:
            continue
        existing = candidates_by_id.get(candidate.id)
        if existing is None:
            existing = CandidateResult(
                candidate=candidate,
                sample_indices=[sample_index],
                optimize_traj_ids=[opt_traj.id],
            )
            candidates_by_id[candidate.id] = existing
            candidate_pool.append(existing)
        else:
            existing.sample_indices.append(sample_index)
            existing.optimize_traj_ids.append(opt_traj.id)

    if not candidate_pool:
        _write_empty_candidate_outputs(
            round_dir,
            optimize_traj_ids=optimize_traj_ids,
            optimize_samples_payload=sample_payload,
            reason="optimize produced no candidate harness",
        )
        return RoundResult(
            round_ix=round_ix,
            candidate=current,
            accepted=False,
            mean_score=0.0,
            scores=[],
            score_task_ids=[],
            before_trajs=before,
            after_trajs=[],
        )

    # Step 4: evaluate every unique candidate and keep the best one.
    after_items = [
        (candidate_ix, task)
        for candidate_ix, candidate_result in enumerate(candidate_pool)
        for task in tasks
    ]

    def _solve_after(item: tuple[int, Task]) -> Trajectory:
        candidate_ix, task_ = item
        with solve_workspace(task_, candidate_pool[candidate_ix].candidate, workdir) as ws:
            return solve_in(
                agent,
                task_,
                candidate_pool[candidate_ix].candidate,
                ws,
                stage="round_solve_after",
                round_ix=round_ix,
            )

    after_results = _parallel_map(
        _solve_after,
        after_items,
        max_workers=solve_workers,
    )
    for traj in after_results:
        traj_store.put(traj)
    grouped_after: list[list[Trajectory]] = [[] for _ in candidate_pool]
    for (candidate_ix, _), traj in zip(after_items, after_results):
        grouped_after[candidate_ix].append(traj)
    for candidate_result, after_trajs in zip(candidate_pool, grouped_after):
        candidate_result.after_trajs = after_trajs

    eval_items = [
        (candidate_ix, task_ix, task)
        for candidate_ix, _candidate_result in enumerate(candidate_pool)
        for task_ix, task in enumerate(tasks)
    ]
    eval_results = _parallel_map(
        lambda item: evaluate(
            agent,
            item[2],
            before[item[1]],
            candidate_pool[item[0]].after_trajs[item[1]],
            harness_before=current,
            harness_after=candidate_pool[item[0]].candidate,
            workdir=workdir,
            stage="round_evaluate",
            round_ix=round_ix,
        ),
        eval_items,
    )
    for eval_traj, _ in eval_results:
        traj_store.put(eval_traj)
    grouped_eval_trajs: list[list[Trajectory]] = [[] for _ in candidate_pool]
    grouped_scores: list[list[Score]] = [[] for _ in candidate_pool]
    for (candidate_ix, _task_ix, _task), (eval_traj, score) in zip(eval_items, eval_results):
        grouped_eval_trajs[candidate_ix].append(eval_traj)
        grouped_scores[candidate_ix].append(score)
    for candidate_result, eval_trajs, scores in zip(candidate_pool, grouped_eval_trajs, grouped_scores):
        candidate_result.eval_trajs = eval_trajs
        candidate_result.scores = scores
        candidate_result.mean_score = statistics.mean(score.value for score in scores) if scores else 0.0

    winner = max(candidate_pool, key=lambda candidate_result: candidate_result.mean_score)
    accepted = winner.mean_score > 0
    _write_json(round_dir / "optimize_traj_ids.json", optimize_traj_ids)
    representative_optimize_traj_id = winner.representative_optimize_traj_id or optimize_traj_ids[0]
    (round_dir / "optimize_traj_id").write_text(representative_optimize_traj_id, encoding="utf-8")
    (round_dir / "candidate_harness_id").write_text(winner.candidate.id, encoding="utf-8")
    _write_harness_diff(current, winner.candidate, round_dir / "candidate_harness_diff.patch")
    _write_json(round_dir / "solve_after_traj_ids.json", [traj.id for traj in winner.after_trajs])
    _write_json(round_dir / "eval_traj_ids.json", [traj.id for traj in winner.eval_trajs])
    _write_json(
        round_dir / "scores.json",
        [
            {"task_id": task.id, "value": score.value, "rationale": score.rationale}
            for task, score in zip(tasks, winner.scores)
        ],
    )
    (round_dir / "mean_score").write_text(f"{winner.mean_score:.4f}", encoding="utf-8")
    (round_dir / "accepted").write_text("true" if accepted else "false", encoding="utf-8")
    _write_json(
        round_dir / "optimize_candidates.json",
        _serialize_candidate_pool(tasks, sample_payload, candidate_pool, winner, accepted),
    )
    return RoundResult(
        round_ix=round_ix,
        candidate=winner.candidate,
        accepted=accepted,
        mean_score=winner.mean_score,
        scores=winner.scores,
        score_task_ids=[task.id for task in tasks],
        before_trajs=before,
        after_trajs=winner.after_trajs,
        winner_sample_index=winner.winner_sample_index,
        candidate_pool=candidate_pool,
    )


def run_evolution(
    *,
    train: TaskSet | list[Task],
    n_rounds: int,
    agent: Agent,
    harness_store: HarnessStore,
    traj_store: TrajectoryStore,
    workdir: Path,
    rounds_dir: Path,
    initial: Harness | None = None,
    strategy: OptimizeStrategy,
    optimize_samples: int = 3,
    solve_workers: int | None = None,
) -> tuple[Harness, list[RoundResult]]:
    rounds_dir.mkdir(parents=True, exist_ok=True)
    current = initial if initial is not None else harness_store.empty()
    history: list[RoundResult] = []
    tasks = list(train)
    for round_ix in range(n_rounds):
        result = run_round(
            round_ix,
            current,
            tasks,
            agent,
            harness_store,
            traj_store,
            workdir,
            rounds_dir / f"round_{round_ix}",
            strategy=strategy,
            optimize_samples=optimize_samples,
            solve_workers=solve_workers,
        )
        if result.accepted:
            current = result.candidate
        history.append(result)
    return current, history
