from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rho.agent.base import Agent
from rho.orchestrators._util import HARNESS_DESCRIPTION
from rho.orchestrators.diagnose import diagnose
from rho.protocols import (
    Diagnosis,
    Harness,
    HarnessStore,
    OptimizeSample,
    OptimizeStrategyResult,
    Task,
    Trajectory,
    TrajectoryStore,
)
from rho.strategies._common import optimize_agent_call, parallel_map

_OPTIMIZE_PREAMBLE = f"""
Based on the per-task diagnoses in diagnoses/, analyze and optimize the current harness/ to improve performance on future tasks. "Better performance" means the agent's final answer more directly and correctly answers what each task asks, with fewer wasted steps.

{HARNESS_DESCRIPTION}
"""


OPTIMIZE_INSTRUCTIONS = _OPTIMIZE_PREAMBLE + """

Workspace layout:
  harness/       — the current harness; you may directly modify it (add/remove/edit any files)
  diagnoses/     — one subdirectory per task, each containing:
    - diagnosis.md  — structured trajectory analysis, severity, failure modes, inconsistency analysis, and a high-level harness improvement direction
    - prompt.md     — the original task for context

## Steps

1. Read each diagnosis in diagnoses/task_XXXX/diagnosis.md and the corresponding prompt.md.
2. Use Severity as a soft attention weight from 0.00 to 1.00, not as ground truth. Higher severity means the diagnosis should influence optimization more strongly.
3. Look for patterns across diagnoses — are multiple tasks failing for similar reasons?
4. Prioritize high-severity recurring failure modes and high-severity recurring inconsistency root causes.
5. Low-severity tasks usually should not cause a harness edit by themselves unless the same issue motif recurs across tasks.
6. Make surgical improvements to harness/ that address the high-level diagnosed issues.

When done, send the changes and your rationale as your final message. If you made no changes, explain why the current harness is already sufficient.
"""


OPTIMIZE_NO_CONSISTENCY_INSTRUCTIONS = _OPTIMIZE_PREAMBLE + """
Workspace layout:
  harness/       — the current harness; you may directly modify it (add/remove/edit any files)
  diagnoses/     — one subdirectory per task, each containing:
    - diagnosis.md  — structured trajectory analysis, severity, failure modes, and a high-level harness improvement direction
    - prompt.md     — the original task for context

## Steps

1. Read each diagnosis in diagnoses/task_XXXX/diagnosis.md and the corresponding prompt.md.
2. Use Severity as a soft attention weight from 0.00 to 1.00, not as ground truth. Higher severity means the diagnosis should influence optimization more strongly.
3. Look for patterns across diagnoses — are multiple tasks failing for similar reasons?
4. Prioritize high-severity recurring failure modes.
5. Low-severity tasks usually should not cause a harness edit by themselves unless the same issue motif recurs across tasks.
6. Make surgical improvements to harness/ that address the high-level diagnosed issues.

When done, send the changes and your rationale as your final message. If you made no changes, explain why the current harness is already sufficient.
"""


OPTIMIZE_NO_VALIDATION_INSTRUCTIONS = _OPTIMIZE_PREAMBLE + """
Workspace layout:
  harness/       — the current harness; you may directly modify it (add/remove/edit any files)
  diagnoses/     — one subdirectory per task, each containing:
    - diagnosis.md  — severity, inconsistency analysis, and a high-level harness improvement direction
    - prompt.md     — the original task for context

## Steps

1. Read each diagnosis in diagnoses/task_XXXX/diagnosis.md and the corresponding prompt.md.
2. Use Severity as a soft attention weight from 0.00 to 1.00, not as ground truth. Higher severity means the diagnosis should influence optimization more strongly.
3. Look for patterns across diagnoses — do multiple tasks show similar trajectory divergence?
4. Prioritize high-severity recurring inconsistency root causes.
5. Low-severity tasks usually should not cause a harness edit by themselves unless the same issue motif recurs across tasks.
6. Make surgical improvements to harness/ that address the high-level diagnosed issues.

When done, send the changes and your rationale as your final message. If you made no changes, explain why the current harness is already sufficient.
"""


class DiagnoseStrategy:
    def __init__(
        self,
        *,
        include_consistency: bool = True,
        include_validation: bool = True,
    ) -> None:
        self.include_consistency = include_consistency
        self.include_validation = include_validation
        if include_validation and include_consistency:
            self.optimize_instructions = OPTIMIZE_INSTRUCTIONS
        elif include_validation and not include_consistency:
            self.optimize_instructions = OPTIMIZE_NO_CONSISTENCY_INSTRUCTIONS
        elif not include_validation and include_consistency:
            self.optimize_instructions = OPTIMIZE_NO_VALIDATION_INSTRUCTIONS
        else:
            raise ValueError("DiagnoseStrategy requires validation, consistency, or both")

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
        diag_results: list[tuple[Trajectory, Diagnosis]] = parallel_map(
            lambda args: diagnose(
                agent,
                args[0],
                args[1],
                harness,
                workdir=workdir,
                stage="round_diagnose",
                round_ix=round_ix,
                include_consistency=self.include_consistency,
                include_validation=self.include_validation,
            ),
            tasks_with_trajectories,
        )
        for diagnose_trajectory, _diagnosis in diag_results:
            traj_store.put(diagnose_trajectory)
        diagnoses = [diagnosis for _trajectory, diagnosis in diag_results]
        if not self.include_consistency:
            diagnoses = [
                replace(diagnosis, inconsistency_analysis="")
                for diagnosis in diagnoses
            ]
        if not self.include_validation:
            diagnoses = [
                replace(diagnosis, trajectory_analyses=[], failure_mode_analysis="")
                for diagnosis in diagnoses
            ]
        task_diagnoses = [
            (task, diagnosis)
            for (task, _trajs), diagnosis in zip(tasks_with_trajectories, diagnoses)
        ]

        def build_workspace(ws: Path) -> None:
            diagnoses_dir = ws / "diagnoses"
            diagnoses_dir.mkdir()
            ranked_task_diagnoses = sorted(
                enumerate(task_diagnoses),
                key=lambda item: (-item[1][1].severity, item[0]),
            )
            for task_ix, (_original_ix, (task, diagnosis)) in enumerate(ranked_task_diagnoses):
                _dump_diagnosis(
                    diagnoses_dir / f"task_{task_ix:04d}",
                    task,
                    diagnosis,
                    include_consistency=self.include_consistency,
                    include_validation=self.include_validation,
                )

        optimize_results = parallel_map(
            lambda sample_index: optimize_agent_call(
                agent,
                harness,
                harness_store,
                workspace_builder=build_workspace,
                instructions=self.optimize_instructions,
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
            ],
            diagnose_trajectories=[trajectory for trajectory, _diagnosis in diag_results],
            diagnoses=diagnoses,
        )


def _dump_diagnosis(
    dest: Path,
    task: Task,
    diagnosis: Diagnosis,
    *,
    include_consistency: bool = True,
    include_validation: bool = True,
    include_direction: bool = True,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    task.materialize(dest)
    lines = [
        f"# Diagnosis: {diagnosis.task_id}",
        "",
        f"**Severity:** {diagnosis.severity:.2f}",
        "",
    ]
    if include_validation and diagnosis.trajectory_analyses:
        lines += ["## Per-trajectory analysis", ""]
        for analysis in diagnosis.trajectory_analyses:
            lines += [
                f"### {analysis.trajectory}",
                "",
                f"**Successful:** {analysis.successful}",
                "",
                "**Quality analysis:**",
                "",
                analysis.quality_analysis,
                "",
            ]
            if analysis.issues:
                lines += ["**Issues:**", "", analysis.issues, ""]
    if include_validation and diagnosis.failure_mode_analysis:
        lines += ["## Failure mode analysis", "", diagnosis.failure_mode_analysis, ""]
    if include_consistency and diagnosis.inconsistency_analysis:
        lines += ["## Inconsistency analysis", "", diagnosis.inconsistency_analysis, ""]
    if include_direction and diagnosis.harness_improvement_direction:
        lines += [
            "## Harness improvement direction",
            "",
            diagnosis.harness_improvement_direction,
            "",
        ]
    (dest / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
