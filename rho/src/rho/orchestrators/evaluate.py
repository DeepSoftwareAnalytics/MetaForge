from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rho.agent.base import Agent
from rho.agent.codex import EVAL_TIMEOUT_S
from rho.observability import annotate_trajectory
from rho.orchestrators._util import HARNESS_DESCRIPTION, dump_trajectory
from rho.orchestrators._util import extract_json_object as _extract_json
from rho.protocols import Harness, Score, Task, Trajectory

EVAL_INSTRUCTIONS = """
Analyze the difference in performance between harness A and harness B on the same task. You will see one task/ directory (containing prompt.md) and two trajectory_*/ directories (each containing events.jsonl and final_message.txt). Produce a JSON object scoring the A -> B transition on an integer scale from -10 to +10 with a short rationale.

""" + HARNESS_DESCRIPTION + """

Scoring rubric:

- +10: A -> B is a change from unacceptable to excellent; B's trajectory is efficient and its answer is correct.
- 0: A and B perform comparably, or it is not possible to determine which is better.
- -10: A -> B is a severe regression; B's trajectory is inefficient and its answer is wrong.

Workspace layout:
  task/                 — the original task (prompt.md is the question)
  harness_A/            — the harness used by trajectory A
  harness_B/            — the harness used by trajectory B
  trajectory_A/         — trajectory from harness A (final_message.txt is the answer)
  trajectory_B/         — trajectory from harness B (final_message.txt is the answer)

## Evaluation steps

1. Read task/prompt.md to understand what the task requires.
2. Read and compare trajectory_A and trajectory_B.

Your final reply must be exactly one JSON object (no markdown fences, no other text):
{"value": <integer in [-10, 10]>, "rationale": "<one-sentence rationale>"}
"""


def evaluate(
    agent: Agent,
    task: Task,
    before: Trajectory,
    after: Trajectory,
    *,
    harness_before: Harness | None = None,
    harness_after: Harness | None = None,
    workdir: Path,
    stage: str | None = None,
    round_ix: int | None = None,
    timeout_s: float = EVAL_TIMEOUT_S,
) -> tuple[Trajectory, Score]:
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(workdir), prefix="eval_", ignore_cleanup_errors=True
    ) as tmp:
        ws = Path(tmp)
        (ws / "task").mkdir()
        task.materialize(ws / "task")
        # Present the candidate first to reduce later-option preference in blind evals.
        dump_trajectory(ws / "trajectory_A", None, after)
        dump_trajectory(ws / "trajectory_B", None, before)
        # Provide harnesses so the evaluator can see what changed.
        if harness_after is not None:
            (ws / "harness_A").mkdir()
            harness_after.materialize(ws / "harness_A")
        if harness_before is not None:
            (ws / "harness_B").mkdir()
            harness_before.materialize(ws / "harness_B")
        tr = agent.run(
            ws,
            EVAL_INSTRUCTIONS,
            task_id=task.id,
            harness_id=f"{before.harness_id}->{after.harness_id}",
            kind="evaluate",
            timeout_s=timeout_s,
        )
        tr = annotate_trajectory(tr, agent=agent, stage=stage, round_ix=round_ix)
        score: Score | None = None
        if tr.exit_code == 0 and not tr.timed_out:
            try:
                parsed = json.loads(_extract_json(tr.final_message))
                raw_value = int(parsed["value"])
                # The prompt scores visible A -> B; callers expect before -> after.
                score = Score(
                    value=-raw_value,
                    rationale=str(parsed.get("rationale", "")),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                score = None
        return tr, score or Score(value=0, rationale="(eval trajectory unusable)")
