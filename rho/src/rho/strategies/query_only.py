from __future__ import annotations

from pathlib import Path

from rho.agent.base import Agent
from rho.orchestrators._util import HARNESS_DESCRIPTION
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

QUERY_ONLY_INSTRUCTIONS = f"""
Based on the per-task prompts in tasks/, analyze and optimize the current harness/ to improve performance on future tasks. "Better performance" means the agent's final answer more directly and correctly answers what each task asks, with fewer wasted steps.

{HARNESS_DESCRIPTION}

Workspace layout:
  harness/              — the current harness; you may directly modify it (add/remove/edit any files)
  tasks/task_XXXX/      — one subdirectory per task, each containing:
    - prompt.md  — the original task

## Steps

1. Read each prompt in tasks/task_XXXX/prompt.md.
2. Infer what reusable information, workflows, or guardrails a good harness should contain to help future agents answer such tasks.
3. Look for patterns across prompts — are multiple tasks asking about similar topics, referencing similar structures, or requiring similar workflows?
4. Prioritize recurring patterns across multiple tasks.
5. Single-task-specific knowledge usually should not cause a harness edit by itself unless the same motif recurs across tasks.
6. Make surgical improvements to harness/ that address the high-level patterns you identified.

When done, send the changes and your rationale as your final message. If you made no changes, explain why the current harness is already sufficient.
"""


class QueryOnlyStrategy:
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
            for task_ix, (task, _trajectories) in enumerate(tasks_with_trajectories):
                task.materialize(tasks_dir / f"task_{task_ix:04d}")

        optimize_results = parallel_map(
            lambda sample_index: optimize_agent_call(
                agent,
                harness,
                harness_store,
                workspace_builder=build_workspace,
                instructions=QUERY_ONLY_INSTRUCTIONS,
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
