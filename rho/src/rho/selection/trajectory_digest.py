"""Render a Trajectory into a compact text digest for the judge LLM.

The digest is plain text bounded to ~10k cl100k_base tokens, designed to
preserve the agent's prose / commands / file edits while discarding codex's
high-volume `raw_stderr` noise. See spec §4.4 for full rules.
"""
from __future__ import annotations

import re
from typing import Any

import tiktoken

from rho.protocols import Trajectory

_BASH_LC_WRAPPER = re.compile(r"""^/bin/bash\s+-lc\s+(['"])(.*)\1\s*$""", re.DOTALL)
# A command that reads task/prompt.md or task/expected.json leaks the task
# description (or the ground-truth answer) verbatim into the digest, which
# defeats the fingerprint's "abstract these out" goal. Replace the output of
# such commands with a structural marker. See spec §13.1 and the §10.5
# grounding self-check finding documented in
# docs/experiments/2026-05-20-trajectory-aware-selection-validation.md.
_TASK_CONTENT_PATH_RE = re.compile(r"(?<![\w./-])task/(?:prompt\.md|expected\.json)(?![\w])")
_ENCODING = tiktoken.get_encoding("cl100k_base")

_GLOBAL_TOKEN_BUDGET = 10_000
_CMD_OUTPUT_HEAD_TOK = 150
_CMD_OUTPUT_TAIL_TOK = 150
_AGENT_MSG_CAP_TOK = 500


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _truncate_head_tail(
    text: str, *, head_tok: int, tail_tok: int, marker_fmt: str = "[{elided} tokens elided]"
) -> str:
    ids = _ENCODING.encode(text)
    if len(ids) <= head_tok + tail_tok:
        return text
    head = _ENCODING.decode(ids[:head_tok])
    tail = _ENCODING.decode(ids[-tail_tok:])
    elided = len(ids) - head_tok - tail_tok
    marker = "  ... " + marker_fmt.format(elided=elided) + " ..."
    return head + "\n" + marker + "\n" + tail


def _cap_tokens(text: str, *, max_tok: int) -> str:
    ids = _ENCODING.encode(text)
    if len(ids) <= max_tok:
        return text
    kept = _ENCODING.decode(ids[:max_tok])
    return kept + " ...[truncated]"


def _enforce_global_budget(
    full_text: str,
    header_lines: list[str],
    unmatched_lines: list[str],
    trailer_lines: list[str],
) -> str:
    """If full_text exceeds budget, shrink the body section using head 60% + tail 40%.

    Preserves header (always at top) and the unmatched+trailer block (always at
    bottom). Spec §4.4: avoid uniform-stride sampling — it shreds multi-step
    reasoning chains.
    """
    if _count_tokens(full_text) <= _GLOBAL_TOKEN_BUDGET:
        return full_text
    header_block = "\n".join(header_lines + [""])
    bottom_block = "\n".join(unmatched_lines + trailer_lines) + "\n"
    overhead = _count_tokens(header_block) + _count_tokens(bottom_block)
    marker_slack = _count_tokens("\n\n... [000000 body tokens elided to fit budget] ...\n\n")
    body_budget = max(_GLOBAL_TOKEN_BUDGET - overhead - marker_slack, 200)

    # Reconstruct body from full_text by stripping header + bottom blocks.
    body_text = full_text[len(header_block):-len(bottom_block)] if bottom_block else full_text[len(header_block):]
    body_ids = _ENCODING.encode(body_text)
    if len(body_ids) <= body_budget:
        return full_text  # nothing to do; overshoot was in header/trailer
    head_n = int(body_budget * 0.6)
    tail_n = body_budget - head_n
    head = _ENCODING.decode(body_ids[:head_n])
    tail = _ENCODING.decode(body_ids[-tail_n:])
    elided = len(body_ids) - head_n - tail_n
    body_clipped = head + f"\n\n... [{elided} body tokens elided to fit budget] ...\n\n" + tail
    return header_block + body_clipped + bottom_block


def _strip_bash_wrapper(command: str) -> str:
    m = _BASH_LC_WRAPPER.match(command.strip())
    # group(1) is the quote character, group(2) is the wrapped body.
    return m.group(2) if m else command


def render_digest(traj: Trajectory) -> tuple[str, int]:
    """Return (digest_text, token_count). See spec §4.4."""
    completed, unmatched = _collect_signal_items(traj.events)

    agent_msgs = sum(1 for it in completed if it["type"] == "agent_message")
    cmds = sum(1 for it in completed if it["type"] == "command_execution")
    edits = sum(len(it.get("changes", [])) for it in completed if it["type"] == "file_change")
    searches = sum(1 for it in completed if it["type"] == "web_search")

    header_lines = [
        f"## Summary: {agent_msgs} agent msgs, {cmds} cmds, {edits} file edits, "
        f"{searches} searches | exit={traj.exit_code} "
        f"timed_out={'true' if traj.timed_out else 'false'}"
    ]
    file_summary = ", ".join(
        f"{ch['kind']}:{ch['path']}"
        for it in completed if it["type"] == "file_change"
        for ch in it.get("changes", [])
    )
    if file_summary:
        header_lines.append(f"## Files changed: {file_summary}")
    final_todo = _final_todo_line(completed)
    if final_todo:
        header_lines.append(final_todo)

    body_lines = _render_body(completed)
    unmatched_lines: list[str] = []
    if unmatched:
        unmatched_lines.append("")
        unmatched_lines.append("## Unmatched in-flight commands")
        for it in unmatched:
            if it.get("type") == "command_execution":
                cmd = _strip_bash_wrapper(it.get("command", ""))
                unmatched_lines.append(f"[CMD in-flight] {cmd}")
            else:
                unmatched_lines.append(f"[{it.get('type','item')} in-flight]")
    final_message = traj.final_message if traj.final_message else "<empty>"
    trailer_lines = ["", "## Final message", final_message]

    text = (
        "\n".join(header_lines + [""] + body_lines + unmatched_lines + trailer_lines) + "\n"
    )
    text = _enforce_global_budget(text, header_lines, unmatched_lines, trailer_lines)
    return text, _count_tokens(text)


def _collect_signal_items(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (completed_items_in_order, unmatched_started_items).

    - `item.started` is tracked for unmatched detection; the started form
      is otherwise ignored.
    - `item.updated` is ignored EXCEPT for `todo_list`, where the latest
      seen state (from updated OR completed) is the rendered final state
      (spec §4.4: "Intermediate item.updated states for todo_list; keep
      only the final state.").
    - `item.completed` items are kept in source order.
    """
    completed: list[dict[str, Any]] = []
    started_by_id: dict[str, dict[str, Any]] = {}
    completed_ids: set[str] = set()
    last_todo: dict[str, Any] | None = None
    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "item.started":
            item = ev.get("item", {})
            iid = item.get("id")
            if iid is not None:
                started_by_id[iid] = item
            continue
        if ev_type == "item.updated":
            item = ev.get("item", {})
            if item.get("type") == "todo_list":
                last_todo = item
            continue
        if ev_type != "item.completed":
            continue
        item = ev.get("item", {})
        iid = item.get("id")
        if iid is not None:
            completed_ids.add(iid)
        item_type = item.get("type")
        if item_type == "todo_list":
            last_todo = item
            continue
        completed.append(item)
    if last_todo is not None:
        completed.append(last_todo)
    unmatched = [it for iid, it in started_by_id.items() if iid not in completed_ids]
    return completed, unmatched


def _final_todo_line(items: list[dict[str, Any]]) -> str | None:
    for it in items:
        if it.get("type") != "todo_list":
            continue
        entries = it.get("items", [])
        rendered = ", ".join(
            f"[{'✓' if entry.get('completed') else ' '}] {entry.get('text', '')}"
            for entry in entries
        )
        return f"## Final todo: {rendered}" if rendered else None
    return None


def _render_body(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for it in items:
        kind = it.get("type")
        if kind == "agent_message":
            text = _cap_tokens(it.get("text", ""), max_tok=_AGENT_MSG_CAP_TOK)
            lines.append(f"[AGENT] {text}")
        elif kind == "command_execution":
            cmd = _strip_bash_wrapper(it.get("command", ""))
            exit_code = it.get("exit_code", "?")
            lines.append(f"[CMD exit={exit_code}] {cmd}")
            output = it.get("aggregated_output", "") or ""
            if output.strip():
                if _TASK_CONTENT_PATH_RE.search(cmd):
                    output_tokens = _count_tokens(output)
                    lines.append(
                        f"  <task content read, {output_tokens} tokens elided>"
                    )
                else:
                    truncated = _truncate_head_tail(
                        output, head_tok=_CMD_OUTPUT_HEAD_TOK, tail_tok=_CMD_OUTPUT_TAIL_TOK
                    )
                    lines.append("  " + truncated.replace("\n", "\n  "))
        elif kind == "file_change":
            for ch in it.get("changes", []):
                lines.append(f"[FILE {ch.get('kind', 'update')}:{ch.get('path', '')}]")
        elif kind == "web_search":
            lines.append(f"[SEARCH \"{it.get('query', '')}\"]")
        elif kind == "todo_list":
            continue  # rendered in header
    return lines
