from __future__ import annotations

from typing import Any


def render_prompt(*, task_id: str, scenario_data: dict[str, Any]) -> str:
    scenario_text = _scenario_text(scenario_data)
    return f"""\
# GAIA-2 Task

Task id: `{task_id}`

{scenario_text}

## Operating Protocol

- Treat `harness/` as the only persistent, evolvable working memory.
- Treat files under this `task/` directory as the current external environment.
- From the workspace root, use `python task/tools/are.py list`, `schema`, and `call` to interact with ARE apps once the runtime is active.
- Do not read `task/.gaia2/scenario.json`; it is runtime-private sidecar data.
- Use `task/tools/catalog.json` for tool names instead of dumping the full `python task/tools/are.py list` output.
- For user interaction, only use AgentUserInterface message tools: read the user's request with a get_*_messages tool, and reply with `send_message_to_user`, for example:
  `python task/tools/are.py call AgentUserInterface get_all_messages --json '{{}}'`
  `python task/tools/are.py call AgentUserInterface send_message_to_user --json '{{"content":"2700 sqft"}}'`
- The Codex final message is only a run summary. It is not sent to the GAIA-2 user and does not count for grading.
- Use `python task/tools/are.py poll` after actions that may trigger environment replies.
- Use `python task/tools/are.py wait --timeout-seconds <n>` only when explicitly waiting for an event or deadline.
- Treat `.gaia2_state/` as read-only. Editing it does not affect grading.
"""


def render_query(*, task_id: str, scenario_data: dict[str, Any]) -> str:
    """Task-discriminative retrieval/selection key — never shown to the solver.

    Unlike `render_prompt` (which keeps the task generic because the GAIA-2
    protocol requires the agent to read the request at runtime via
    AgentUserInterface), the query embeds the actual user request so that
    similarity-based retrieval and task selection can tell scenarios apart.
    """
    task_text = _user_request_text(scenario_data) or _scenario_text(scenario_data)
    return f"""\
# GAIA-2 Task

Task id: `{task_id}`

{task_text}
"""


def _user_request_text(data: dict[str, Any]) -> str:
    """Concatenate the user's request messages from the scenario events.

    The task is delivered as USER `send_message_to_agent` events; the static
    scenario metadata carries no description. AGENT/ENV events are skipped so
    the reference solution never leaks into a retrieval key.
    """
    events = data.get("events")
    if not isinstance(events, list):
        return ""
    messages: list[str] = []
    for event in events:
        if not isinstance(event, dict) or event.get("event_type") != "USER":
            continue
        action = event.get("action")
        if not isinstance(action, dict):
            continue
        if action.get("function") != "send_message_to_agent":
            continue
        for arg in action.get("args") or []:
            if (
                isinstance(arg, dict)
                and arg.get("name") == "content"
                and isinstance(arg.get("value"), str)
                and arg["value"].strip()
            ):
                messages.append(arg["value"].strip())
    return "\n\n".join(messages)


def _scenario_text(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        definition = metadata.get("definition")
        if isinstance(definition, dict):
            hints = definition.get("hints")
            if isinstance(hints, list):
                text = "\n".join(f"- {hint}" for hint in hints if hint)
                if text:
                    return text
            for key in ("prompt", "query", "task", "description"):
                value = definition.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "Complete the scenario using the available ARE apps."
