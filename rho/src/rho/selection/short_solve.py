"""Parallel short-solve probe stage used before trajectory-aware selection.

For each candidate task, runs the configured agent once under the initial
harness, producing one Trajectory. Infra-level agent failures are caught and
materialised as an `exit_code != 0`, `events=[]` Trajectory so the selection
pipeline survives single-task flakes (spec §4.3 short-solve failure contract).
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from rho.agent.base import Agent
from rho.orchestrators.solve import solve_in, solve_workspace
from rho.protocols import Harness, Task, Trajectory, TrajectoryStore


def _failure_trajectory(task: Task, harness: Harness, exc: BaseException) -> Trajectory:
    summary = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
    return Trajectory(
        id=f"shortfail_{task.id}",
        kind="solve",
        task_id=task.id,
        harness_id=harness.id,
        instructions="",
        events=[],
        final_message="",
        stdout="",
        stderr=summary + "\n" + traceback.format_exc()[:2000],
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=-1,
        wall_time_s=0.0,
        timed_out=False,
        stage="short_solve_for_selection",
    )


def short_solve_one(
    task: Task, *, agent: Agent, harness: Harness, workdir: Path
) -> Trajectory:
    """Run one short-solve attempt. Catches infra errors and returns a
    placeholder Trajectory; only BaseException (KeyboardInterrupt) propagates.
    """
    try:
        with solve_workspace(task, harness, workdir) as ws:
            return solve_in(
                agent, task, harness, ws,
                sample_index=None,
                stage="short_solve_for_selection",
                round_ix=None,
            )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        return _failure_trajectory(task, harness, exc)


def short_solve_all(
    tasks: Iterable[Task],
    *,
    agent: Agent,
    harness: Harness,
    traj_store: TrajectoryStore,
    workdir: Path,
    max_workers: int | None = None,
) -> dict[str, Trajectory]:
    """Run short-solve on every task in parallel. Always returns one Trajectory
    per task (real or failure placeholder), tagged with
    `stage="short_solve_for_selection"`, and persists each through `traj_store`.
    """
    tasks_list = list(tasks)
    if not tasks_list:
        return {}
    workdir.mkdir(parents=True, exist_ok=True)
    workers = max_workers if max_workers and max_workers > 0 else len(tasks_list)
    out: dict[str, Trajectory] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(short_solve_one, t, agent=agent, harness=harness, workdir=workdir): t
            for t in tasks_list
        }
        for fut, task in list(futures.items()):
            traj = fut.result()
            out[task.id] = traj
            traj_store.put(traj)
    return out
