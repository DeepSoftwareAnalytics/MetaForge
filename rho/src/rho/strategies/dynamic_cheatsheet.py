from __future__ import annotations

import dataclasses
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rho.agent.base import Agent
from rho.observability import annotate_trajectory, extract_usage
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

CHEATSHEET_FILENAME = "cheatsheet.md"

CHEATSHEET_INITIAL_CONTENT = (
    "<cheatsheet>\n"
    "Version: 0\n"
    "(empty — no entries yet)\n"
    "</cheatsheet>\n"
)


CURATOR_INSTRUCTIONS = f"""\
# Cheatsheet Curator

You are a Cheatsheet Curator. Your job is to maintain a single evolving cheatsheet that helps a problem-solving agent tackle future tasks. The cheatsheet lives at `harness/{CHEATSHEET_FILENAME}` and will be visible to the agent at solve time as part of the harness.

{HARNESS_DESCRIPTION}

Workspace layout:
  harness/                       — the current harness; you may modify ONLY `harness/{CHEATSHEET_FILENAME}`
  transcript/prompt.md           — the task the agent just attempted
  transcript/trajectory_N/       — one directory per solve attempt for this task
    events.jsonl                 — actions, file reads, reasoning (can be large; use `head`, `tail`, or `grep` to skim)
    final_message.txt            — the agent's final answer
    workspace_diff/              — files the agent created or modified

## Goals
- Curate and preserve knowledge: keep useful, generalizable solutions, strategies, and insights from past tasks. Preserve previous cheatsheet entries that are still useful.
- Refine and update content: integrate new insights from this experience into the cheatsheet, removing redundancies and trivial entries.
- Stay practical and concise: prefer actionable code snippets, reusable patterns, and meta-strategies. Keep the total length under ~2000 words.

## Principles
- Only add solutions and strategies that look correct and useful. Do not blindly copy mistakes from the trajectory.
- Prefer reusable, general patterns over narrow task-specific knowledge.
- Group related entries; refine wording rather than duplicating.

## Required output structure
Wrap the entire cheatsheet inside `<cheatsheet>...</cheatsheet>`. Inside the wrapper:
- Bump the `Version: N` line by 1 from the previous cheatsheet (e.g. `Version: 3` → `Version: 4`).
- Organize entries as `<memory_item>` blocks. Each item should contain a short `<description>` and a concrete `<example>` (code snippet, worked solution, or rule). Append `Count: N` to track how many times the item has been useful (start at 1 for new items; increment when an existing item helped on this task).

Skeleton:
```
<cheatsheet>
Version: <bumped>

<memory_item>
<description>...</description>
<example>...</example>
Count: <n>
</memory_item>

<memory_item>
...
</memory_item>
</cheatsheet>
```

## Preservation rule (IMPORTANT)
When you overwrite `harness/{CHEATSHEET_FILENAME}`, anything not explicitly copied forward is permanently lost. Always restate every prior `<memory_item>` that is still relevant — do not assume any history is retained outside this file.

## Output
Edit `harness/{CHEATSHEET_FILENAME}` in place to reflect the updated cheatsheet. Do NOT modify any other file. When done, send a short summary of what you changed (or "no changes" if this experience added nothing new).
"""


class DynamicCheatsheetStrategy:
    """Dynamic Cheatsheet baseline (Suzgun et al., arXiv:2504.07952).

    Single-stream semantics: the round's M train tasks are processed
    sequentially; each task triggers one curator agent.run that sees the
    cheatsheet accumulated from prior tasks plus the task's solve
    trajectories, and rewrites `harness/cheatsheet.md`. The strategy
    produces exactly one candidate harness — the cheatsheet's terminal
    state after task M-1.

    DC has no parallel-restart concept (curator is a deterministic stream
    update), so n_samples must be 1; passing anything else raises. For
    full byte-faithfulness with the original DC paper, also run with
    `--max-evolve-tasks 1` so the next round's solver sees the curator's
    update (otherwise all M solves in a round share the round's starting
    cheatsheet).
    """

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
        if n_samples != 1:
            raise ValueError(
                "dynamic-cheatsheet is single-stream and produces exactly one "
                f"candidate per round; require n_samples=1, got {n_samples}. "
                "Pass --optimize-samples 1."
            )
        if not tasks_with_trajectories:
            raise ValueError("dynamic-cheatsheet requires at least one task trajectory group")

        current = harness
        curator_trajs: list[Trajectory] = []
        # Initialize to seed content so metrics on an all-failure round still
        # reflect what the next solver would see (the placeholder cheatsheet),
        # not a misleading 0-byte / version=None reading.
        final_cheatsheet = CHEATSHEET_INITIAL_CONTENT
        workdir.mkdir(parents=True, exist_ok=True)
        for task_ix, (task, trajectories) in enumerate(tasks_with_trajectories):
            with tempfile.TemporaryDirectory(
                dir=str(workdir),
                prefix=f"dc_t{task_ix:04d}_",
                ignore_cleanup_errors=True,
            ) as tmp:
                ws = Path(tmp)
                harness_dir = ws / "harness"
                harness_dir.mkdir()
                current.materialize(harness_dir)
                _ensure_cheatsheet(harness_dir)
                transcript_dir = ws / "transcript"
                _dump_task_transcript(transcript_dir, task, trajectories)
                traj = agent.run(
                    ws,
                    CURATOR_INSTRUCTIONS,
                    task_id=task.id,
                    harness_id=current.id,
                    kind="optimize",
                )
                traj = annotate_trajectory(
                    traj,
                    agent=agent,
                    stage=f"round_optimize:dc_t{task_ix:04d}",
                    round_ix=round_ix,
                    sample_index=0,
                )
                curator_trajs.append(traj)
                if traj.exit_code != 0 or traj.timed_out:
                    continue
                new_harness = harness_store.capture(harness_dir)
                if new_harness.id != current.id:
                    current = new_harness
                final_cheatsheet = (harness_dir / CHEATSHEET_FILENAME).read_text(
                    encoding="utf-8"
                )

        assert curator_trajs
        # Put the first M-1 trajs as-is; persist the last with a metrics event
        # summarising the whole stream (M curator runs, total tokens, final
        # cheatsheet stats). Round-level artifacts only surface the strategy's
        # last trajectory, so attaching metrics there is the only way to read
        # the full DC stream cost without inspecting every per-task traj.
        for traj in curator_trajs[:-1]:
            traj_store.put(traj)
        last_traj = _with_dc_metrics(
            curator_trajs[-1],
            curator_trajs=curator_trajs,
            final_cheatsheet=final_cheatsheet,
        )
        traj_store.put(last_traj)
        candidate = current if current.id != harness.id else None
        return OptimizeStrategyResult(
            samples=[
                OptimizeSample(
                    sample_index=0,
                    optimize_trajectory=last_traj,
                    candidate=candidate,
                )
            ]
        )


def _ensure_cheatsheet(harness_dir: Path) -> None:
    target = harness_dir / CHEATSHEET_FILENAME
    if not target.exists():
        target.write_text(CHEATSHEET_INITIAL_CONTENT, encoding="utf-8")


def _dump_task_transcript(
    dest: Path,
    task: Task,
    trajectories: list[Trajectory],
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    task.materialize(dest)
    for traj_ix, trajectory in enumerate(trajectories):
        dump_trajectory(dest / f"trajectory_{traj_ix}", None, trajectory)


_VERSION_RE = re.compile(r"Version:\s*(\d+)", re.IGNORECASE)


def _with_dc_metrics(
    traj: Trajectory,
    *,
    curator_trajs: Iterable[Trajectory],
    final_cheatsheet: str,
) -> Trajectory:
    metrics = _dc_metrics(curator_trajs, final_cheatsheet=final_cheatsheet)
    events = list(traj.events)
    events.append({"type": "dc_cheatsheet.metrics", "metrics": metrics})
    return dataclasses.replace(traj, events=events)


def _dc_metrics(
    curator_trajs: Iterable[Trajectory],
    *,
    final_cheatsheet: str,
) -> dict[str, Any]:
    trajs = list(curator_trajs)
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for traj in trajs:
        usage = extract_usage(traj.events)
        if usage is None:
            continue
        for key in totals:
            totals[key] += usage.get(key, 0)
    cheatsheet_text = final_cheatsheet
    word_count = len(cheatsheet_text.split())
    byte_size = len(cheatsheet_text.encode("utf-8"))
    # Take the LAST Version match: curators that paste the previous cheatsheet
    # forward leave a stale older `Version: N` earlier in the file; the bumped
    # version is whichever appears most recently.
    version_matches = _VERSION_RE.findall(cheatsheet_text)
    final_version = int(version_matches[-1]) if version_matches else None
    return {
        "dc_curator_traj_ids": [t.id for t in trajs],
        "dc_curator_calls": len(trajs),
        "dc_curator_failures": sum(
            1 for t in trajs if t.exit_code != 0 or t.timed_out
        ),
        "dc_total_input_tokens": totals["input_tokens"],
        "dc_total_cached_input_tokens": totals["cached_input_tokens"],
        "dc_total_output_tokens": totals["output_tokens"],
        "dc_cheatsheet_word_count": word_count,
        "dc_cheatsheet_byte_size": byte_size,
        "dc_cheatsheet_version": final_version,
    }
