from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rho.agent.base import Agent
from rho.agent.codex_pool import DEFAULT_CODEX_CONCURRENCY
from rho.orchestrators.solve import solve_in, solve_workspace
from rho.protocols import Grade, Harness, Task, TaskSet, Trajectory, TrajectoryStore


@dataclass(frozen=True)
class GradedSolve:
    task: Task
    grade: Grade
    trajectory: Trajectory
    stage: str


def grade_on_split(
    agent: Agent,
    harness: Harness,
    split: TaskSet,
    workdir,
    *,
    max_tasks: int | None = None,
    traj_store: TrajectoryStore | None = None,
    stage: str = "grade",
    artifacts_root: Path | None = None,
    max_workers: int | None = None,
    solve_workers: int | None = DEFAULT_CODEX_CONCURRENCY,
    solve_attempts: int = 2,
) -> list[GradedSolve]:
    workdir = Path(workdir)
    tasks = list(split)[:max_tasks]

    if not tasks:
        return []
    if solve_attempts <= 0:
        raise ValueError("solve_attempts must be positive")

    grade_workers = len(tasks) if max_workers is None else min(max_workers, len(tasks))
    grade_gate = threading.Semaphore(grade_workers)
    solve_worker_count = (
        len(tasks) if solve_workers is None else min(solve_workers, len(tasks))
    )
    if solve_worker_count <= 0:
        raise ValueError("solve_workers must be positive")

    def _solve_and_grade(task: Task) -> GradedSolve:
        for attempt_ix in range(solve_attempts):
            with solve_workspace(task, harness, workdir) as ws:
                traj = solve_in(agent, task, harness, ws, stage=stage)
                if traj_store is not None:
                    traj_store.put(traj)
                should_retry = (
                    attempt_ix + 1 < solve_attempts
                    and _retryable_transport_failure(traj)
                )
                if should_retry:
                    continue
                grade_artifacts_dir = _grade_artifacts_dir(
                    artifacts_root, stage, task, traj
                )
                with grade_gate:
                    grade = task.grade(traj, artifacts_dir=grade_artifacts_dir)
                return GradedSolve(
                    task=task,
                    grade=grade,
                    trajectory=traj,
                    stage=stage,
                )
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=solve_worker_count) as pool:
        return list(pool.map(_solve_and_grade, tasks))


def _grade_artifacts_dir(
    artifacts_root: Path | None,
    stage: str,
    task: Task,
    trajectory: Trajectory,
) -> Path | None:
    if artifacts_root is None:
        return None
    safe_task_id = task.id.replace("/", "__")
    return artifacts_root / stage / safe_task_id / trajectory.id


def _retryable_transport_failure(trajectory: Trajectory) -> bool:
    if trajectory.exit_code == 0 or trajectory.timed_out:
        return False
    haystack = f"{trajectory.stdout}\n{trajectory.stderr}"
    return (
        "stream disconnected before completion" in haystack
        or "response.failed event received" in haystack
    )


def summarize(grades: list[GradedSolve]) -> dict:
    total = len(grades)
    mean_score = sum(record.grade.score for record in grades) / total if total else 0.0
    return {
        "n": total,
        "mean_score": mean_score,
    }
