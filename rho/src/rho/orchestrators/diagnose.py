from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from rho.agent.base import Agent
from rho.observability import annotate_trajectory
from rho.orchestrators._util import HARNESS_DESCRIPTION, dump_trajectory
from rho.orchestrators._util import extract_json_object as _extract_json
from rho.protocols import Diagnosis, Harness, Task, Trajectory, TrajectoryAnalysis

_DIAGNOSE_PREAMBLE = """
Analyze three solve trajectories for the same task.

""" + HARNESS_DESCRIPTION + """

Workspace layout:
  task/              - the original task. Read task/prompt.md to understand the question.
  harness/           - the shared harness used by all three trajectories.
  trajectory_0/      - first solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.
  trajectory_1/      - second solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.
  trajectory_2/      - third solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.

Your job is to analyze and evaluate the three trajectories. Follow this workflow:

## Step 1: Inspect each trajectory

For each of trajectory_0, trajectory_1, and trajectory_2:

1. Inspect final_message.txt and events.jsonl to understand the action and decision process.
2. Evaluate whether the trajectory accurately and efficiently completed the task.
3. Set successful to 1 if the trajectory accurately completed the task, otherwise set it to 0.
4. In quality_analysis, note what evidence, files, tools, or reasoning steps the trajectory relied on, and whether there was unnecessary work, missed information, misleading evidence, or an incorrect decision.

## Step 2: Analyze failure modes

If all three trajectories accurately and efficiently completed the task, this section can be brief. Otherwise, analyze why one or more trajectories failed or performed poorly. Make this analysis faithful and actionable. Ground it in what the trajectories actually did.
"""


DIAGNOSE_INSTRUCTIONS = _DIAGNOSE_PREAMBLE + """

## Step 3: Analyze inconsistency

Compare the three event sequences and final answers. Identify whether there are inconsistencies among them: where and why the trajectories diverged, and how those differences affected the behavior.

## Step 4: Summarize harness improvement direction

Suggest one high-level, simple direction for improving the harness. This should be a general improvement direction based on the trajectory analysis, not a detailed edit plan and not a task-specific hardcoded fix.

## Step 5: Assign severity

Set severity to a float from 0.0 to 1.0 for how strongly this task should influence the next harness optimization:

- 0.0: no meaningful issue; all trajectories answered accurately and efficiently.
- 0.1-0.3: minor inefficiency or weak concern; do not optimize from this alone.
- 0.4-0.7: mixed success, inconsistency, or a plausible harness gap.
- 0.8-1.0: clear failure, missing information, or a high-confidence harness issue.

## Output format

Your final message must be exactly one JSON object (no markdown fences, no other text):
{
  "task_id": "<task id, if available from the task prompt or context>",
  "severity": 0.0,
  "trajectory_analyses": [
    {
      "trajectory": "trajectory_0",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    },
    {
      "trajectory": "trajectory_1",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    },
    {
      "trajectory": "trajectory_2",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    }
  ],
  "failure_mode_analysis": "<actionable analysis in markdown of why any trajectory failed or performed poorly; brief if none failed>",
  "inconsistency_analysis": "<root-cause analysis in markdown of where and why the trajectories diverged, and how that affected quality>",
  "harness_improvement_direction": "<one high-level, simple direction for improving the harness>"
}
"""


DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS = _DIAGNOSE_PREAMBLE + """
## Step 3: Summarize harness improvement direction

Suggest one high-level, simple direction for improving the harness. This should be a general improvement direction based on the trajectory analysis, not a detailed edit plan and not a task-specific hardcoded fix.

## Step 4: Assign severity

Set severity to a float from 0.0 to 1.0 for how strongly this task should influence the next harness optimization:

- 0.0: no meaningful issue; all trajectories answered accurately and efficiently.
- 0.1-0.3: minor inefficiency or weak concern; do not optimize from this alone.
- 0.4-0.7: mixed success, inefficient behavior, or a plausible harness gap.
- 0.8-1.0: clear failure, missing information, or a high-confidence harness issue.

## Output format

Your final message must be exactly one JSON object (no markdown fences, no other text):
{
  "task_id": "<task id, if available from the task prompt or context>",
  "severity": 0.0,
  "trajectory_analyses": [
    {
      "trajectory": "trajectory_0",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    },
    {
      "trajectory": "trajectory_1",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    },
    {
      "trajectory": "trajectory_2",
      "successful": 1,
      "quality_analysis": "<faithful analysis of whether this trajectory completed the task accurately and efficiently>",
      "issues": "<any missed information, misleading evidence, inefficiency, or incorrect decision; empty string if none>"
    }
  ],
  "failure_mode_analysis": "<actionable analysis in markdown of why any trajectory failed or performed poorly; brief if none failed>",
  "harness_improvement_direction": "<one high-level, simple direction for improving the harness>"
}
"""


DIAGNOSE_NO_VALIDATION_INSTRUCTIONS = """
Analyze three solve trajectories for the same task.

""" + HARNESS_DESCRIPTION + """

Workspace layout:
  task/              - the original task. Read task/prompt.md to understand the question.
  harness/           - the shared harness used by all three trajectories.
  trajectory_0/      - first solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.
  trajectory_1/      - second solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.
  trajectory_2/      - third solve attempt. Contains events.jsonl, final_message.txt, and workspace_diff/.

Your job is to analyze how the three trajectories relate to each other. Follow this workflow:

## Step 1: Read each trajectory

For each of trajectory_0, trajectory_1, and trajectory_2:

1. Inspect final_message.txt and events.jsonl to understand the action and decision process.
2. Note the evidence, files, tools, and reasoning steps the trajectory relied on so the trajectories can be compared.
3. Do not judge whether an individual trajectory completed the task accurately or efficiently.

## Step 2: Analyze inconsistency

Compare the three event sequences and final answers. Identify whether there are inconsistencies among them: where and why the trajectories diverged, and how those differences affected the behavior.

## Step 3: Summarize harness improvement direction

Suggest one high-level, simple direction for improving the harness. This should be a general improvement direction based on the trajectory analysis, not a detailed edit plan and not a task-specific hardcoded fix.

## Step 4: Assign severity

Set severity to a float from 0.0 to 1.0 for how strongly this task should influence the next harness optimization:

- 0.0: no meaningful issue; the trajectories agree, with no notable divergence.
- 0.1-0.3: minor divergence or weak concern; do not optimize from this alone.
- 0.4-0.7: consequential divergence among trajectories, or a plausible harness gap.
- 0.8-1.0: severe or contradictory divergence, or a high-confidence harness issue.

## Output format

Your final message must be exactly one JSON object (no markdown fences, no other text):
{
  "task_id": "<task id, if available from the task prompt or context>",
  "severity": 0.0,
  "inconsistency_analysis": "<root-cause analysis in markdown of where and why the trajectories diverged, and how that affected behavior>",
  "harness_improvement_direction": "<one high-level, simple direction for improving the harness>"
}
"""


def _diagnosis_schema(*, include_validation: bool, include_consistency: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "task_id": {"type": "string"},
        "severity": {"type": "number", "minimum": 0, "maximum": 1},
        "harness_improvement_direction": {"type": "string"},
    }
    if include_validation:
        analysis = {
            "trajectory": {"type": "string", "enum": ["trajectory_0", "trajectory_1", "trajectory_2"]},
            "successful": {"type": "integer", "enum": [0, 1]},
            "quality_analysis": {"type": "string"},
            "issues": {"type": "string"},
        }
        properties["trajectory_analyses"] = {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "object", "properties": analysis,
                      "required": list(analysis), "additionalProperties": False},
        }
        properties["failure_mode_analysis"] = {"type": "string"}
    if include_consistency:
        properties["inconsistency_analysis"] = {"type": "string"}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def diagnose(
    agent: Agent,
    task: Task,
    trajectories: list[Trajectory],  # exactly 3
    harness: Harness,
    *,
    workdir: Path,
    stage: str | None = None,
    round_ix: int | None = None,
    include_consistency: bool = True,
    include_validation: bool = True,
) -> tuple[Trajectory, Diagnosis]:
    workdir.mkdir(parents=True, exist_ok=True)
    if include_validation and include_consistency:
        instructions = DIAGNOSE_INSTRUCTIONS
    elif include_validation and not include_consistency:
        instructions = DIAGNOSE_NO_CONSISTENCY_INSTRUCTIONS
    elif not include_validation and include_consistency:
        instructions = DIAGNOSE_NO_VALIDATION_INSTRUCTIONS
    else:
        raise ValueError("diagnose requires validation, consistency, or both")
    with tempfile.TemporaryDirectory(
        dir=str(workdir), prefix="diag_", ignore_cleanup_errors=True
    ) as tmp:
        ws = Path(tmp)
        (ws / "task").mkdir()
        task.materialize(ws / "task")
        (ws / "harness").mkdir()
        harness.materialize(ws / "harness")
        for ix, traj in enumerate(trajectories):
            dump_trajectory(ws / f"trajectory_{ix}", None, traj)
        tr = agent.run(
            ws,
            instructions,
            task_id=task.id,
            harness_id=harness.id,
            kind="diagnose",
            output_schema=_diagnosis_schema(
                include_validation=include_validation,
                include_consistency=include_consistency,
            ),
        )
        tr = annotate_trajectory(tr, agent=agent, stage=stage, round_ix=round_ix)
        diag = _parse_diagnosis(task.id, tr)
        return tr, diag


def _parse_diagnosis(task_id: str, tr: Trajectory) -> Diagnosis:
    """Parse a Diagnosis from the trajectory's final_message, with fallback."""
    try:
        parsed = json.loads(_extract_json(tr.final_message))
        analyses = _parse_trajectory_analyses(parsed.get("trajectory_analyses"))
        failure_mode_analysis = str(parsed.get("failure_mode_analysis", ""))
        inconsistency_analysis = str(parsed.get("inconsistency_analysis", ""))
        harness_improvement_direction = str(parsed.get("harness_improvement_direction", ""))
        severity = _parse_severity(
            parsed.get("severity"),
            analyses,
            failure_mode_analysis,
            inconsistency_analysis,
            harness_improvement_direction,
        )
        return Diagnosis(
            task_id=task_id,
            trajectory_analyses=analyses,
            failure_mode_analysis=failure_mode_analysis,
            inconsistency_analysis=inconsistency_analysis,
            harness_improvement_direction=harness_improvement_direction,
            severity=severity,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return Diagnosis(
            task_id=task_id,
            trajectory_analyses=[],
            failure_mode_analysis=tr.final_message,
            inconsistency_analysis="",
            harness_improvement_direction="",
            severity=1.0,
        )


def _parse_trajectory_analyses(raw: Any) -> list[TrajectoryAnalysis]:
    if not isinstance(raw, list):
        return []
    analyses: list[TrajectoryAnalysis] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        analyses.append(
            TrajectoryAnalysis(
                trajectory=str(item.get("trajectory", "")),
                successful=int(item.get("successful", 0)),
                quality_analysis=str(item.get("quality_analysis", "")),
                issues=str(item.get("issues", "")),
            )
        )
    return analyses


def _parse_severity(
    raw: Any,
    analyses: list[TrajectoryAnalysis],
    failure_mode_analysis: str,
    inconsistency_analysis: str,
    harness_improvement_direction: str,
) -> float:
    if raw is None:
        return 1.0 if _has_diagnosed_issue(
            analyses,
            failure_mode_analysis,
            inconsistency_analysis,
            harness_improvement_direction,
        ) else 0.0
    try:
        severity = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(severity):
        return 1.0
    return max(0.0, min(1.0, severity))


def _has_diagnosed_issue(
    analyses: list[TrajectoryAnalysis],
    failure_mode_analysis: str,
    inconsistency_analysis: str,
    harness_improvement_direction: str,
) -> bool:
    return (
        bool(failure_mode_analysis.strip())
        or bool(inconsistency_analysis.strip())
        or bool(harness_improvement_direction.strip())
        or any(analysis.successful == 0 or analysis.issues.strip() for analysis in analyses)
    )
