from __future__ import annotations

import json
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rho.observability import extract_usage

MAX_PREVIEW_LINES = 80
MAX_PREVIEW_BYTES = 32 * 1024
MAX_GROUPS = 12
MAX_GROUP_SAMPLES = 4

_SEARCH_LINE_RE = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<text>.*)$")


def load_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            events.append(json.loads(raw_line))
        except json.JSONDecodeError:
            events.append({"type": "raw_stdout", "line": raw_line})
    return events


def build_trace_payload(events: list[dict[str, Any]], *, run_dir: Path) -> dict[str, Any]:
    steps = _build_trace_steps(events, run_dir=run_dir)
    usage = extract_usage(events) or {}
    command_count = sum(1 for step in steps if step["kind"] == "command")
    failed_command_count = sum(
        1
        for step in steps
        if step["kind"] == "command" and step.get("metrics", {}).get("exit_code") not in (None, 0)
    )
    summary = {
        "step_count": len(steps),
        "event_count": len(events),
        "counts_by_kind": dict(Counter(step["kind"] for step in steps)),
        "command_count": command_count,
        "failed_command_count": failed_command_count,
        "stderr_count": sum(1 for step in steps if step["kind"] == "stderr"),
        "usage": usage,
    }
    return {"summary": summary, "steps": [_public_step(step) for step in steps]}


def read_command_output_chunk(
    events: list[dict[str, Any]],
    *,
    run_dir: Path,
    step_id: str,
    start_line: int,
    max_lines: int,
) -> dict[str, Any]:
    for step in _build_trace_steps(events, run_dir=run_dir):
        if step["id"] != step_id:
            continue
        output = step.get("full_output")
        if not isinstance(output, str):
            raise KeyError(step_id)
        lines = output.splitlines()
        total_lines = len(lines)
        start = max(0, start_line)
        end = min(total_lines, start + max(1, max_lines))
        return {
            "step_id": step_id,
            "start_line": start,
            "end_line": end,
            "total_lines": total_lines,
            "has_more": end < total_lines,
            "lines": lines[start:end],
        }
    raise KeyError(step_id)


def _build_trace_steps(events: list[dict[str, Any]], *, run_dir: Path) -> list[dict[str, Any]]:
    assembler = _TraceAssembler(run_dir=run_dir)
    return assembler.consume(events)


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    public = dict(step)
    public.pop("full_output", None)
    return public


class _TraceAssembler:
    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._steps: list[dict[str, Any]] = []
        self._open_items: dict[str, int] = {}

    def consume(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event_index, event in enumerate(events):
            event_type = str(event.get("type") or "")
            if event_type in {"thread.started", "turn.started"}:
                self._steps.append(_build_boundary_step(event_index, event))
                continue
            if event_type == "turn.completed":
                self._steps.append(_build_usage_step(event_index, event))
                continue
            if event_type in {"raw_stderr", "raw_stdout"}:
                self._steps.append(_build_raw_stream_step(event_index, event))
                continue
            if event_type == "sandbox_fallback":
                self._steps.append(_build_sandbox_step(event_index, event))
                continue
            if event_type not in {"item.started", "item.completed", "item.updated"}:
                self._steps.append(_build_unknown_step(event_index, event))
                continue

            item = event.get("item")
            if not isinstance(item, dict):
                self._steps.append(_build_unknown_step(event_index, event))
                continue
            item_id = str(item.get("id") or f"event-{event_index}")
            item_type = str(item.get("type") or "unknown")
            if item_type == "agent_message":
                self._steps.append(_build_agent_message_step(event_index, item))
                continue

            existing_index = self._open_items.get(item_id)
            if existing_index is None:
                step = _build_item_step(event_index, item, self.run_dir)
                self._steps.append(step)
                existing_index = len(self._steps) - 1
                if event_type != "item.completed":
                    self._open_items[item_id] = existing_index
            else:
                _merge_item_step(self._steps[existing_index], item, self.run_dir)
            if event_type == "item.completed":
                self._open_items.pop(item_id, None)

        for item_id, step_index in list(self._open_items.items()):
            step = self._steps[step_index]
            if step["status"] == "in_progress":
                step["summary"] = step["summary"] or "Still in progress when the trace ended."
            self._open_items.pop(item_id, None)

        for index, step in enumerate(self._steps):
            step["index"] = index
        return self._steps


def _build_boundary_step(event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("type") or "")
    title = "Thread started" if event_type == "thread.started" else "Turn started"
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "kind": "boundary",
        "status": "completed",
        "title": title,
        "subtitle": event_type,
        "summary": "",
        "metrics": {},
        "preview": None,
        "raw_available": False,
    }


def _build_usage_step(event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    summary = "Input {input_tokens:,} | Cached {cached_input_tokens:,} | Output {output_tokens:,}".format(
        input_tokens=int(usage.get("input_tokens", 0)),
        cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "kind": "usage",
        "status": "completed",
        "title": "Turn usage",
        "subtitle": "turn.completed",
        "summary": summary,
        "metrics": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        },
        "preview": None,
        "raw_available": False,
    }


def _build_raw_stream_step(event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    stream_kind = "stderr" if event.get("type") == "raw_stderr" else "stdout"
    line = str(event.get("line") or "")
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "kind": stream_kind,
        "status": "completed",
        "title": f"Raw {stream_kind}",
        "subtitle": "",
        "summary": line,
        "metrics": {"byte_count": len(line.encode('utf-8'))},
        "preview": {"mode": "text", "lines": [line], "truncated": False},
        "raw_available": False,
    }


def _build_sandbox_step(event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "kind": "warning",
        "status": "completed",
        "title": "Sandbox fallback",
        "subtitle": "",
        "summary": f"Retried from {event.get('from')} to {event.get('to')}.",
        "metrics": {},
        "preview": None,
        "raw_available": False,
    }


def _build_unknown_step(event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"event-{event_index}",
        "index": event_index,
        "kind": "unknown",
        "status": "completed",
        "title": str(event.get("type") or "unknown"),
        "subtitle": "",
        "summary": "Unrecognized event payload.",
        "metrics": {},
        "preview": {"mode": "json", "value": event},
        "raw_available": False,
    }


def _build_agent_message_step(event_index: int, item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text") or "")
    lines = text.splitlines() or [text]
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "agent_message",
        "status": "completed",
        "title": "Agent message",
        "subtitle": "",
        "summary": lines[0] if lines else "",
        "metrics": {"line_count": len(lines), "byte_count": len(text.encode("utf-8"))},
        "preview": {"mode": "text", "lines": lines[:8], "truncated": len(lines) > 8},
        "raw_available": False,
        "text": text,
    }


def _build_item_step(event_index: int, item: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    item_type = str(item.get("type") or "unknown")
    if item_type == "command_execution":
        return _build_command_step(event_index, item)
    if item_type == "todo_list":
        return _build_todo_step(event_index, item)
    if item_type == "file_change":
        return _build_file_change_step(event_index, item, run_dir)
    if item_type == "web_search":
        return _build_web_search_step(event_index, item)
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "unknown",
        "status": str(item.get("status") or "completed"),
        "title": item_type,
        "subtitle": "",
        "summary": "",
        "metrics": {},
        "preview": {"mode": "json", "value": item},
        "raw_available": False,
    }


def _merge_item_step(step: dict[str, Any], item: dict[str, Any], run_dir: Path) -> None:
    item_type = str(item.get("type") or "unknown")
    if item_type == "command_execution":
        merged = _build_command_step(step["index"], item)
    elif item_type == "todo_list":
        merged = _build_todo_step(step["index"], item)
    elif item_type == "file_change":
        merged = _build_file_change_step(step["index"], item, run_dir)
    elif item_type == "web_search":
        merged = _build_web_search_step(step["index"], item)
    else:
        merged = _build_item_step(step["index"], item, run_dir)
    preserved_index = step["index"]
    step.clear()
    step.update(merged)
    step["index"] = preserved_index


def _build_command_step(event_index: int, item: dict[str, Any]) -> dict[str, Any]:
    command = _unwrap_shell_command(str(item.get("command") or ""))
    output = str(item.get("aggregated_output") or "")
    line_count = len(output.splitlines()) if output else 0
    exit_code = item.get("exit_code")
    mode, summary, preview = _summarize_command_output(command, output)
    subtitle_parts = [mode]
    if exit_code is not None:
        subtitle_parts.append(f"exit {exit_code}")
    if line_count:
        subtitle_parts.append(f"{line_count} lines")
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "command",
        "status": str(item.get("status") or "completed"),
        "title": _truncate(command, 160),
        "subtitle": " | ".join(subtitle_parts),
        "summary": summary,
        "metrics": {
            "exit_code": exit_code,
            "line_count": line_count,
            "byte_count": len(output.encode("utf-8")),
        },
        "preview": preview,
        "raw_available": bool(output),
        "full_output": output,
        "command": command,
    }


def _build_todo_step(event_index: int, item: dict[str, Any]) -> dict[str, Any]:
    raw_items = item.get("items")
    items = raw_items if isinstance(raw_items, list) else []
    completed_count = sum(1 for entry in items if isinstance(entry, dict) and entry.get("completed"))
    total_count = len(items)
    summary = f"{completed_count}/{total_count} tasks completed" if total_count else "No checklist items"
    preview_items = [
        {
            "text": str(entry.get("text") or ""),
            "completed": bool(entry.get("completed")),
        }
        for entry in items
        if isinstance(entry, dict)
    ]
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "todo",
        "status": "completed" if completed_count == total_count and total_count else "in_progress",
        "title": "Todo list",
        "subtitle": "",
        "summary": summary,
        "metrics": {
            "completed_count": completed_count,
            "total_count": total_count,
        },
        "preview": {"mode": "todo", "items": preview_items},
        "raw_available": False,
    }


def _build_file_change_step(event_index: int, item: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    raw_changes = item.get("changes")
    changes = raw_changes if isinstance(raw_changes, list) else []
    counts = Counter()
    normalized_paths: list[dict[str, str]] = []
    for entry in changes:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "update")
        counts[kind] += 1
        normalized_paths.append(
            {
                "kind": kind,
                "path": _normalize_display_path(str(entry.get("path") or ""), run_dir),
            }
        )
    summary_bits = [f"{kind} {value}" for kind, value in sorted(counts.items())]
    summary = ", ".join(summary_bits) if summary_bits else "No file changes recorded."
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "file_change",
        "status": str(item.get("status") or "completed"),
        "title": "File changes",
        "subtitle": f"{len(normalized_paths)} files",
        "summary": summary,
        "metrics": dict(counts),
        "preview": {
            "mode": "file_change",
            "paths": normalized_paths[:MAX_GROUPS],
            "truncated": len(normalized_paths) > MAX_GROUPS,
        },
        "raw_available": False,
    }


def _build_web_search_step(event_index: int, item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action")
    action_type = ""
    if isinstance(action, dict):
        action_type = str(action.get("type") or "")
    query = str(item.get("query") or "")
    queries = action.get("queries") if isinstance(action, dict) else None
    preview_queries = [str(entry) for entry in queries if isinstance(entry, str)] if isinstance(queries, list) else []
    summary = query or "Search action with no query text."
    return {
        "id": str(item.get("id") or f"event-{event_index}"),
        "index": event_index,
        "kind": "web_search",
        "status": "completed",
        "title": "Web search",
        "subtitle": action_type or "search",
        "summary": summary,
        "metrics": {"query_count": len(preview_queries) or (1 if query else 0)},
        "preview": {
            "mode": "queries",
            "query": query,
            "queries": preview_queries[:MAX_GROUPS],
            "truncated": len(preview_queries) > MAX_GROUPS,
        },
        "raw_available": False,
    }


def _unwrap_shell_command(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if len(parts) >= 3 and parts[0].endswith("bash") and parts[1] == "-lc":
        return parts[2]
    return command


def _summarize_command_output(command: str, output: str) -> tuple[str, str, dict[str, Any] | None]:
    if not output:
        return "command", "No output.", {"mode": "text", "lines": [], "truncated": False}

    lines = output.splitlines()
    if _looks_like_search_output(command, lines):
        groups: dict[str, list[str]] = defaultdict(list)
        for line in lines:
            match = _SEARCH_LINE_RE.match(line)
            if not match:
                continue
            path = match.group("path")
            snippet = f"{match.group('line')}: {match.group('text')}".strip()
            groups[path].append(snippet)
        if groups:
            ordered_paths = sorted(groups, key=lambda path: (-len(groups[path]), path))
            preview_groups = [
                {
                    "path": path,
                    "count": len(groups[path]),
                    "snippets": groups[path][:MAX_GROUP_SAMPLES],
                }
                for path in ordered_paths[:MAX_GROUPS]
            ]
            total_matches = sum(len(matches) for matches in groups.values())
            summary = f"{total_matches} matches across {len(groups)} files."
            return (
                "search",
                summary,
                {
                    "mode": "search_matches",
                    "groups": preview_groups,
                    "truncated": len(ordered_paths) > MAX_GROUPS,
                },
            )

    if _looks_like_file_list(command, lines):
        groups: dict[str, list[str]] = defaultdict(list)
        for line in lines:
            normalized = line.strip()
            if not normalized:
                continue
            key = normalized.split("/", 1)[0] if "/" in normalized else "."
            groups[key].append(normalized)
        ordered_groups = sorted(groups, key=lambda key: (-len(groups[key]), key))
        preview_groups = [
            {
                "name": key,
                "count": len(groups[key]),
                "sample": groups[key][:MAX_GROUP_SAMPLES],
            }
            for key in ordered_groups[:MAX_GROUPS]
        ]
        summary = f"{sum(len(paths) for paths in groups.values())} paths across {len(groups)} groups."
        return (
            "file listing",
            summary,
            {
                "mode": "file_list",
                "groups": preview_groups,
                "truncated": len(ordered_groups) > MAX_GROUPS,
            },
        )

    preview_lines, truncated = _take_preview_lines(lines)
    summary = f"{len(lines)} lines of text output."
    return (
        "text",
        summary,
        {
            "mode": "text",
            "lines": preview_lines,
            "truncated": truncated,
        },
    )


def _looks_like_search_output(command: str, lines: list[str]) -> bool:
    lowered = command.lower()
    if "rg " in lowered or lowered.startswith("rg") or "grep" in lowered:
        return any(_SEARCH_LINE_RE.match(line) for line in lines)
    return any(_SEARCH_LINE_RE.match(line) for line in lines[:10])


def _looks_like_file_list(command: str, lines: list[str]) -> bool:
    lowered = command.lower()
    if "rg --files" in lowered or lowered.startswith("find "):
        return True
    if not lines:
        return False
    sample = lines[: min(20, len(lines))]
    path_like = 0
    for line in sample:
        if ":" in line:
            continue
        if "/" in line or line.endswith((".md", ".json", ".txt", ".py", ".log", ".patch")):
            path_like += 1
    return path_like >= max(3, len(sample) // 2)


def _take_preview_lines(lines: list[str]) -> tuple[list[str], bool]:
    preview: list[str] = []
    used_bytes = 0
    for line in lines:
        encoded = len((line + "\n").encode("utf-8"))
        if len(preview) >= MAX_PREVIEW_LINES or used_bytes + encoded > MAX_PREVIEW_BYTES:
            return preview, True
        preview.append(line)
        used_bytes += encoded
    return preview, False


def _normalize_display_path(path: str, run_dir: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(run_dir).as_posix()
        except ValueError:
            return candidate.as_posix()
    return path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
