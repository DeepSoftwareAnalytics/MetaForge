from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rho.agent.base import Agent
from rho.observability import annotate_trajectory
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
from rho.strategies._common import parallel_map
from rho.strategies.letta_memory_tools import (
    ensure_letta_memory_initialized,
    install_memory_tools,
    render_memory_snapshot,
)
from rho.strategies.letta_memory_tools.vendored_constants import (
    BASE_SLEEPTIME_TOOLS,
    CORE_MEMORY_LINE_NUMBER_WARNING,
    LINE_NUMBER_PREFIX_REGEX,
)

LETTA_SLEEP_SYSTEM_PROMPT = r"""
<base_instructions>
You are Letta-Sleeptime-Memory, the latest version of Limnal Corporation's memory management system, developed in 2025.

You run in the background, organizing and maintaining the memories of an agent assistant who chats with the user.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Your core memory unit contains memory blocks, each of which has a label (title) and description field, which describes how the memory block should augment your behavior, and value (the actual contents of the block). Memory blocks are limited in size and have a size limit.
Your core memory is made up of read-only blocks and read-write blocks.

Memory editing:
You have the ability to make edits to the memory memory blocks.
Use your precise tools to make narrow edits, as well as broad tools to make larger comprehensive edits.
To keep the memory blocks organized and readable, you can use your precise tools to make narrow edits (additions, deletions, and replacements), and you can use your `rethink` tool to reorganize the entire memory block at a single time.
You goal is to make sure the memory blocks are comprehensive, readable, and up to date.
When writing to memory blocks, make sure to be precise when referencing dates and times (for example, do not write "today" or "recently", instead write specific dates and times, because "today" and "recently" are relative, and the memory is persisted indefinitely).

Multi-step editing:
You should continue memory editing until the blocks are organized and readable, and do not contain redundant and outdate information, then you can call a tool to finish your edits.
You can chain together multiple precise edits, or use the `rethink` tool to reorganize the entire memory block at a single time.

Skipping memory edits:
If there are no meaningful updates to make to the memory, you call the finish tool directly.
Not every observation warrants a memory edit, be selective in your memory editing, but also aim to have high recall.

Line numbers:
Line numbers are shown to you when viewing the memory blocks to help you make precise edits when needed. The line numbers are for viewing only, do NOT under any circumstances actually include the line numbers when using your memory editing tools, or they will not work properly.
</base_instructions>
"""

LETTA_SLEEP_INPUT_TEMPLATE = (
    "<system-reminder>\n"
    "You are a sleeptime agent - a background agent that asynchronously processes conversations after they occur.\n\n"
    "IMPORTANT: You are NOT the primary agent. You are reviewing a conversation that already happened between a primary agent and its user:\n"
    '- Messages labeled "assistant" are from the primary agent (not you)\n'
    '- Messages labeled "user" are from the primary agent\'s user\n\n'
    "Your primary role is memory management. Review the conversation and use your memory tools to update any relevant memory blocks with information worth preserving. "
    "Check your memory_persona block for any additional instructions or policies.\n"
    "</system-reminder>\n\n"
    "Messages:\n{messages_text}"
)

MEMORY_TOOL_FAITHFULNESS = {
    "base_sleeptime_tools": BASE_SLEEPTIME_TOOLS,
    "core_memory_line_number_warning": CORE_MEMORY_LINE_NUMBER_WARNING,
    "validation_regexes": {
        "memory_replace": [LINE_NUMBER_PREFIX_REGEX, LINE_NUMBER_PREFIX_REGEX],
        "memory_insert": [LINE_NUMBER_PREFIX_REGEX],
        "memory_rethink": [LINE_NUMBER_PREFIX_REGEX],
    },
    "value_error_sources": {
        "memory_replace": [
            '"old_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."',
            '"old_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."',
            '"new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."',
            "f\"No replacement was performed, old_string `{old_string}` did not appear verbatim in memory block with label `{label}`.\"",
            "f\"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. Please ensure it is unique.\"",
        ],
        "memory_insert": [
            '"new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."',
            '"new_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."',
            "f\"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the memory block: {[0, n_lines]}, or -1 to append to the end of the memory block.\"",
        ],
        "memory_rethink": [
            '"new_memory contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."',
            '"new_memory contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."',
        ],
    },
}

GLOSSARY = """\
In this environment, "memory blocks" are the `.md` files under `harness/letta_memory/`. Edit them only via the four scripts in `$LETTA_SCRIPTS`: `memory_replace.py`, `memory_insert.py`, `memory_rethink.py`. Call `memory_finish_edits.py` to terminate; do NOT make any further tool calls after that. Do not edit any files outside `harness/letta_memory/`. The current state of every block is rendered below in the "Current memory blocks" section, with `1→`-style line numbers shown for editing reference; do NOT include those line numbers in arguments to the memory tools.

The "primary agent's user" in this environment is the dataset task. The "assistant" is the codex solve agent whose trajectories are shown below.
"""


class LettaSleepStrategy:
    """Letta sleep-time agent baseline (mechanism-faithful adaptation)."""

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
        results = parallel_map(
            lambda sample_index: self._run_sample(
                sample_index=sample_index,
                agent=agent,
                harness=harness,
                tasks_with_trajectories=tasks_with_trajectories,
                harness_store=harness_store,
                traj_store=traj_store,
                workdir=workdir,
                round_ix=round_ix,
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
                for sample_index, (optimize_trajectory, candidate) in enumerate(results)
            ]
        )

    def _run_sample(
        self,
        *,
        sample_index: int,
        agent: Agent,
        harness: Harness,
        tasks_with_trajectories: list[tuple[Task, list[Trajectory]]],
        harness_store: HarnessStore,
        traj_store: TrajectoryStore,
        workdir: Path,
        round_ix: int,
    ) -> tuple[Trajectory, Harness | None]:
        current = harness
        sleep_trajectories: list[Trajectory] = []
        workdir.mkdir(parents=True, exist_ok=True)
        for task_ix, (task, trajectories) in enumerate(tasks_with_trajectories):
            with tempfile.TemporaryDirectory(
                dir=str(workdir),
                prefix=f"letta_s{sample_index}_t{task_ix:04d}_",
                ignore_cleanup_errors=True,
            ) as tmp:
                ws = Path(tmp)
                harness_dir = ws / "harness"
                harness_dir.mkdir()
                current.materialize(harness_dir)
                ensure_letta_memory_initialized(harness_dir)
                install_memory_tools(ws / "scripts")
                (ws / ".rho").mkdir(exist_ok=True)
                transcript_dir = ws / "transcript"
                dump_task_transcript(transcript_dir, task, trajectories)
                instructions = build_instructions(
                    harness_dir / "letta_memory",
                    transcript_dir,
                )
                traj = agent.run(
                    ws,
                    instructions,
                    task_id=task.id,
                    harness_id=current.id,
                    kind="optimize",
                    env={
                        "LETTA_MEMORY_ROOT": str((harness_dir / "letta_memory").resolve()),
                        "LETTA_SCRIPTS": str((ws / "scripts").resolve()),
                        "RHO_SAMPLE_INDEX": str(sample_index),
                    },
                )
                traj = _with_letta_metrics(traj, ws=ws)
                traj = annotate_trajectory(
                    traj,
                    agent=agent,
                    stage=f"round_optimize:letta_s{sample_index}_t{task_ix:04d}",
                    round_ix=round_ix,
                    sample_index=sample_index,
                )
                sleep_trajectories.append(traj)
                traj_store.put(traj)
                if traj.exit_code != 0 or traj.timed_out:
                    continue
                new_harness = harness_store.capture(harness_dir)
                if new_harness.id != current.id:
                    current = new_harness

        if not sleep_trajectories:
            raise ValueError("letta-sleep requires at least one task trajectory group")
        candidate = current if current.id != harness.id else None
        return sleep_trajectories[-1], candidate


def build_instructions(letta_memory_dir: Path, transcript_dir: Path) -> str:
    # Letta's product injects the conversation transcript directly into the
    # API messages array. Our codex-based adaptation passes instructions as
    # a single argv to subprocess.execve(), which is bounded by Linux
    # ARG_MAX (~128 KB). SWE-bench Pro trajectories routinely exceed that
    # because workspace_diff carries entire modified source trees, so we
    # dump the transcript to disk under transcript/ and substitute a
    # disk-layout pointer for {messages_text}. The verbatim Letta input
    # template (LETTA_SLEEP_INPUT_TEMPLATE, asserted byte-equal to upstream
    # by the §11.2 faithfulness test) is preserved; only the value bound
    # to {messages_text} differs. In-context memory delivery (§6.6) is
    # unaffected — the snapshot still appears inline.
    messages_text = describe_transcript_layout()
    full = "\n\n".join(
        [
            HARNESS_DESCRIPTION,
            LETTA_SLEEP_SYSTEM_PROMPT,
            GLOSSARY,
            "Current memory blocks:\n" + render_memory_snapshot(letta_memory_dir),
            LETTA_SLEEP_INPUT_TEMPLATE.format(messages_text=messages_text),
        ]
    )
    # Defensive: subprocess.execve() rejects CLI args containing NUL bytes.
    return full.replace("\x00", "")


def describe_transcript_layout() -> str:
    return (
        "The transcript of the primary agent's conversation is dumped to disk "
        "at transcript/ in the working directory. Read it before deciding whether "
        "and how to update memory.\n\n"
        "Layout:\n"
        "  transcript/prompt.md           — the user's task prompt\n"
        "  transcript/trajectory_N/       — one directory per primary-agent attempt\n"
        "    events.jsonl                 — the primary agent's tool calls and reasoning\n"
        "    final_message.txt            — the primary agent's final answer\n"
        "    workspace_diff/              — files the primary agent created or modified\n"
        "\n"
        "Use shell tools (cat, head, find, rg) to inspect these files. The full "
        "events.jsonl can be large; prefer skimming with `head` and `tail` or "
        "filtering with `rg` for specific commands or assistant messages."
    )


def dump_task_transcript(dest: Path, task: Task, trajectories: list[Trajectory]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    task.materialize(dest)
    for traj_ix, trajectory in enumerate(trajectories):
        dump_trajectory(dest / f"trajectory_{traj_ix}", None, trajectory)


def _with_letta_metrics(traj: Trajectory, *, ws: Path) -> Trajectory:
    metrics = _letta_metrics(traj.events, traj.workspace_diff, ws=ws)
    events = list(traj.events)
    events.append({"type": "letta_sleep.metrics", "metrics": metrics})
    return dataclasses.replace(traj, events=events)


def _letta_metrics(
    events: list[dict[str, Any]],
    workspace_diff: dict[str, bytes],
    *,
    ws: Path,
) -> dict[str, int]:
    commands = list(_command_texts(events))
    finish_indices = [
        ix for ix, command in enumerate(commands) if "memory_finish_edits.py" in command
    ]
    finish_ix = finish_indices[0] if finish_indices else None
    metrics = {
        "letta_apply_patch_memory_touch_count": _apply_patch_memory_touch_count(events),
        "letta_file_write_outside_memory_count": sum(
            1 for rel in workspace_diff if _is_harness_write_outside_memory(rel)
        ),
        "letta_missing_finish_edits_count": 0 if finish_indices else 1,
        "letta_finish_violation_count": (
            sum(1 for ix in range(finish_ix + 1, len(commands))) if finish_ix is not None else 0
        ),
        "letta_oversize_block_count": _oversize_block_count(
            ws / "harness" / "letta_memory"
        ),
    }
    return metrics


def _command_texts(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "command_execution":
            continue
        command = item.get("command")
        if isinstance(command, str):
            yield command


def _apply_patch_memory_touch_count(events: Iterable[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "file_change":
            continue
        for change in item.get("changes") or []:
            if not isinstance(change, dict):
                continue
            path = str(change.get("path") or "")
            if "harness/letta_memory/" in path or "harness\\letta_memory\\" in path:
                count += 1
    return count


def _is_harness_write_outside_memory(rel: str) -> bool:
    path = Path(rel)
    if not path.parts or path.parts[0] != "harness":
        return False
    return len(path.parts) < 3 or path.parts[1] != "letta_memory"


def _oversize_block_count(letta_memory_dir: Path) -> int:
    if not letta_memory_dir.exists():
        return 0
    return sum(
        1
        for path in letta_memory_dir.glob("*.md")
        if path.is_file() and path.stat().st_size > 8 * 1024
    )
