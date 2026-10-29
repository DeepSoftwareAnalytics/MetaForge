from __future__ import annotations

import json
import re
from pathlib import Path

from rho.protocols import Task, Trajectory

HARNESS_DESCRIPTION = """\
The harness (harness/) is a toolkit of resources and guidance that helps \
the agent solve tasks. It can contain any type of file — helper scripts, \
artifacts, environment setup, documentation with relevant context, and \
workflows to follow."""


def extract_json_object(text: str) -> str:
    """Accept a JSON reply, a fenced reply, or prose followed by one JSON object.

    Never repair JSON or select a later object after a malformed/ambiguous first
    one. Return the original text on failure so the caller's fallback remains
    visible in the recorded result.
    """
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    text = text.strip()
    start = text.find("{")
    if start <= 0:
        return text
    candidate = text[start:]
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return text
    if isinstance(value, dict) and not candidate[end:].strip():
        return candidate
    return text


def dump_trajectory(dest: Path, task: Task | None, traj: Trajectory) -> None:
    """Dump a trajectory to disk for LLM inspection.

    Writes events.jsonl, final_message.txt, workspace_diff/, and optionally
    task/prompt.md. Used by optimize, diagnose, and evaluate orchestrators.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in traj.events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (dest / "final_message.txt").write_text(traj.final_message, encoding="utf-8")
    if task is not None:
        task.materialize(dest / "task")
    diff_dir = dest / "workspace_diff"
    diff_dir.mkdir()
    for rel, content in traj.workspace_diff.items():
        target = diff_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if traj.workspace_deletions:
        (dest / "_deletions.txt").write_text(
            "\n".join(sorted(traj.workspace_deletions)),
            encoding="utf-8",
        )
