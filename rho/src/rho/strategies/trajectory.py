from __future__ import annotations

from pathlib import Path

from rho.agent.base import Agent
from rho.orchestrators._util import HARNESS_DESCRIPTION, dump_trajectory
from rho.protocols import (
    Harness,
    HarnessStore,
    OptimizeSample,
    OptimizeStrategyResult,
    Task,
    Trajectory,
    TrajectoryStore,
)
from rho.strategies._common import optimize_agent_call, parallel_map

TRAJECTORY_INSTRUCTIONS = f"""
Based on the per-task solve trajectories in tasks/, analyze and optimize the current harness/ to improve performance on future tasks. "Better performance" means the agent's final answer more directly and correctly answers what each task asks, with fewer wasted steps.

{HARNESS_DESCRIPTION}

Workspace layout:
  harness/              — the current harness; you may directly modify it (add/remove/edit any files)
  tasks/task_XXXX/      — one subdirectory per task, each containing:
    - prompt.md                        — the original task
    - trajectory_N/events.jsonl        — agent events (actions, file reads, reasoning)
    - trajectory_N/final_message.txt   — the agent's final answer
    - trajectory_N/workspace_diff/     — files the agent created or modified

## Steps

1. For each task, read prompt.md and each trajectory_N/ (events.jsonl, final_message.txt, workspace_diff/).
2. Make surgical improvements to harness/ based on what the trajectories reveal about how the agent solves these tasks.

When done, send the changes and your rationale as your final message. If you made no changes, explain why the current harness is already sufficient.
"""


class TrajectoryStrategy:
    def __init__(self, *, trajectories_per_task: int) -> None:
        self.trajectories_per_task = trajectories_per_task

    def propose_candidates(
        self,
        *,
        agent: Agent,
        harness: Harness,
        tasks_with_trajectories: list[tuple[Task, list[Trajectory]]],
        harness_store: HarnessStore,
        traj_store: TrajectoryStore,
        workdir: Path,
        n_samples: int,
        round_ix: int,
    ) -> OptimizeStrategyResult:
        del traj_store

        def build_workspace(ws: Path) -> None:
            tasks_dir = ws / "tasks"
            tasks_dir.mkdir()
            for task_ix, (task, trajectories) in enumerate(tasks_with_trajectories):
                task_dir = tasks_dir / f"task_{task_ix:04d}"
                task.materialize(task_dir)
                selected = trajectories[: self.trajectories_per_task]
                if self.trajectories_per_task == 1:
                    if selected:
                        dump_trajectory(task_dir / "trajectory", None, selected[0])
                    continue
                for traj_ix, trajectory in enumerate(selected):
                    dump_trajectory(task_dir / f"trajectory_{traj_ix}", None, trajectory)

        optimize_results = parallel_map(
            lambda sample_index: optimize_agent_call(
                agent,
                harness,
                harness_store,
                workspace_builder=build_workspace,
                instructions=TRAJECTORY_INSTRUCTIONS,
                workdir=workdir,
                stage="round_optimize",
                round_ix=round_ix,
                sample_index=sample_index,
            ),
            list(range(n_samples)),
        )
        return OptimizeStrategyResult(
            samples=[
                OptimizeSample(
                    sample_index=sample_index,
                    optimize_trajectory=optimize_trajectory,
                    candidate=candidate,
                )
                for sample_index, (optimize_trajectory, candidate) in enumerate(optimize_results)
            ]
        )
