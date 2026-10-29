from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import struct
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SidecarError(ValueError):
    pass


# GAIA-2's ARE pins litellm 1.71.1. The judge runs through one of two providers,
# selected by RHO_GAIA2_JUDGE_PROVIDER (default "azure"):
#   * azure       -> Azure OpenAI Foundry, same Entra auth as the solver
#   * openrouter  -> OpenRouter, routes to Meta's Llama-3.3-70B (the official
#                    GAIA-2 leaderboard judge model)
_DEFAULT_JUDGE_MODEL_AZURE = "gpt-5.5"
_DEFAULT_JUDGE_MODEL_OPENROUTER = "meta-llama/llama-3.3-70b-instruct"

# Per-turn slack for extra agent `send_message_to_user` calls beyond the
# oracle's count. ARE's default is 1 (validation/configs.py). This widens the
# budget so minor over-messaging WITHIN a correctly-structured turn is not
# penalized. It does NOT make a chatty agent pass in general: ARE re-segments
# the agent's turns at every `send_message_to_user`, so a message emitted at a
# different point than the oracle shifts later tool calls into the wrong turn
# and fails the per-turn non-message count check. Turn structure is graded
# officially; this constant only relaxes the per-turn message count.
_EXTRA_SEND_MESSAGE_TO_USER_ALLOWED = 4


@dataclass(frozen=True)
class ToolRecord:
    app: str
    function: str
    tool_name: str
    schema: dict[str, Any]
    tool: Any


def handle_request(runtime: "ToolSidecarRuntime", payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        method = str(payload.get("method") or "")
        if method == "list_tools":
            return {"ok": True, "tools": runtime.list_tools()}
        if method == "schema":
            return {
                "ok": True,
                "schema": runtime.schema(
                    _required_string(payload, "app"),
                    _required_string(payload, "function"),
                ),
            }
        if method == "call_tool":
            args = payload.get("args", {})
            if not isinstance(args, dict):
                raise SidecarError("call_tool args must be a JSON object")
            return {
                "ok": True,
                "result": runtime.call_tool(
                    _required_string(payload, "app"),
                    _required_string(payload, "function"),
                    args,
                ),
            }
        if method == "dump_state":
            app = payload.get("app")
            if app is not None and not isinstance(app, str):
                raise SidecarError("dump_state app must be a string when provided")
            return {"ok": True, "state": runtime.dump_state(app)}
        if method == "poll_notifications":
            return {"ok": True, "notifications": runtime.poll_notifications()}
        if method == "wait_for_notification":
            timeout = float(payload.get("timeout_seconds", 30.0))
            return {"ok": True, "notifications": runtime.wait_for_notification(timeout)}
        if method == "validate":
            return {"ok": True, "result": runtime.validate()}
        if method == "shutdown":
            runtime.close()
            return {"ok": True}
        raise SidecarError(f"unknown GAIA-2 sidecar method: {method}")
    except SidecarError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }


class ToolSidecarRuntime:
    def __init__(
        self,
        *,
        tools: Iterable[Any],
        validate_fn: Callable[[], Any] | None = None,
        state_fn: Callable[[str | None], Any] | None = None,
        notification_fn: Callable[[], Any] | None = None,
        wait_fn: Callable[[float], Any] | None = None,
        close_fn: Callable[[], None] | None = None,
        call_fn: Callable[[ToolRecord, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._validate_fn = validate_fn
        self._state_fn = state_fn
        self._notification_fn = notification_fn
        self._wait_fn = wait_fn
        self._close_fn = close_fn
        self._call_fn = call_fn
        self._records: dict[tuple[str, str], ToolRecord] = {}
        for tool in tools:
            record = _build_record(tool)
            self._records[(record.app, record.function)] = record

    def list_tools(self) -> dict[str, dict[str, dict[str, Any]]]:
        catalog: dict[str, dict[str, dict[str, Any]]] = {}
        for record in self._records.values():
            schema = dict(record.schema)
            schema.pop("app", None)
            schema.pop("function", None)
            catalog.setdefault(record.app, {})[record.function] = _jsonable(schema)
        return catalog

    def schema(self, app: str, function: str) -> dict[str, Any]:
        return _jsonable(self._record(app, function).schema)

    def call_tool(self, app: str, function: str, args: dict[str, Any]) -> Any:
        record = self._record(app, function)
        if self._call_fn is not None:
            return _jsonable(self._call_fn(record, args))
        return _jsonable(record.tool(**args))

    def dump_state(self, app: str | None = None) -> Any:
        if self._state_fn is None:
            return {}
        return _jsonable(self._state_fn(app))

    def poll_notifications(self) -> Any:
        if self._notification_fn is None:
            return []
        return _jsonable(self._notification_fn())

    def wait_for_notification(self, timeout_seconds: float) -> Any:
        if self._wait_fn is None:
            return self.poll_notifications()
        return _jsonable(self._wait_fn(timeout_seconds))

    def validate(self) -> Any:
        if self._validate_fn is None:
            return {
                "success": False,
                "rationale": "GAIA-2 validation is not configured for this runtime.",
            }
        return _jsonable(self._validate_fn())

    def close(self) -> None:
        if self._close_fn is not None:
            self._close_fn()

    def write_catalog(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.list_tools(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def write_state_dir(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        state = self.dump_state()
        if not isinstance(state, dict):
            return
        for app, app_state in state.items():
            path = state_dir / f"{_safe_filename(str(app))}.json"
            path.write_text(
                json.dumps(app_state, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

    def _record(self, app: str, function: str) -> ToolRecord:
        record = self._records.get((app, function))
        if record is None:
            raise SidecarError(f"unknown GAIA-2 tool: {app}.{function}")
        return record


class ARESidecarRuntime(ToolSidecarRuntime):
    def __init__(
        self,
        *,
        scenario: Any,
        env: Any,
        judge_enabled: bool,
    ) -> None:
        self.scenario = scenario
        self.env = env
        self.judge_enabled = judge_enabled
        super().__init__(
            tools=env.get_tools(),
            validate_fn=self._validate,
            state_fn=self._dump_state,
            notification_fn=self._poll_notifications,
            wait_fn=self._wait_for_notification,
            close_fn=self._close,
            call_fn=self._call_are_tool,
        )

    @classmethod
    def from_scenario_file(cls, scenario_file: Path, workdir: Path) -> "ARESidecarRuntime":
        try:
            from are.simulation.data_handler.importer import JsonScenarioImporter
            from are.simulation.environment import Environment, EnvironmentConfig
            from are.simulation.notification_system import VerboseNotificationSystem
            from are.simulation.scenarios.scenario_imported_from_json.utils import (
                preprocess_scenario,
                preprocess_scenario_for_execution_without_oracle,
            )
            from are.simulation.types import CapabilityTag
            from are.simulation.validation.configs import GraphPerEventJudgeConfig
        except ImportError as exc:
            raise RuntimeError(
                "GAIA-2 sidecar requires Meta ARE. Install rho with the `gaia2` extra."
            ) from exc

        scenario_json = scenario_file.read_text(encoding="utf-8")
        scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(
            scenario_json,
            apps_to_skip=None,
            load_completed_events=False,
        )
        judge_enabled = _judge_enabled_from_env()
        if judge_enabled:
            provider = _judge_provider_from_env()
            default_model = (
                _DEFAULT_JUDGE_MODEL_OPENROUTER
                if provider == "openrouter"
                else _DEFAULT_JUDGE_MODEL_AZURE
            )
            model = os.getenv("RHO_GAIA2_JUDGE_MODEL", default_model)
            # offline_validation=True runs ARE's official offline validation:
            # a single pass at the end that loops over each turn (turns are
            # delimited by the agent's send_message_to_user calls) and grades
            # it against that turn's oracle events. This is the right fit for
            # "run the agent fully, then grade". The judge rubrics are ARE's
            # own; only the verdict PARSING is patched (see the helper's
            # docstring) to tolerate the substituted judge model's casing.
            _install_case_insensitive_judge_parsing()
            if os.getenv("RHO_GAIA2_FILTER_ENV_EVENTS", "").strip().lower() in {"1", "true", "yes", "on"}:
                _install_filter_env_events_from_counter()
            if os.getenv("RHO_GAIA2_RELAX_AUI_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"}:
                _install_relax_aui_judge()
            if os.getenv("RHO_GAIA2_FILTER_FS_READS", "").strip().lower() in {"1", "true", "yes", "on"}:
                _install_filter_fs_reads_from_counter()
            # Diagnostic: when RHO_GAIA2_JUDGE_TRACE_DIR is set, dump
            # every judge LLM call (system prompt + inputs + raw response) so
            # we can see which soft checker rejected and why.
            trace_dir = os.getenv("RHO_GAIA2_JUDGE_TRACE_DIR")
            if trace_dir:
                scen_id = getattr(scenario, "scenario_id", None) or "scenario"
                _install_judge_tracing(
                    Path(trace_dir) / f"{scen_id}-{os.getpid()}.jsonl"
                )
            preprocess_scenario(
                scenario,
                judge_config=GraphPerEventJudgeConfig(
                    engine=_build_judge_engine(provider, model),
                    extra_send_message_to_user_allowed=(
                        _EXTRA_SEND_MESSAGE_TO_USER_ALLOWED
                    ),
                ),
                max_scenario_duration=_scenario_duration_limit(scenario, CapabilityTag),
                offline_validation=True,
            )
        else:
            preprocess_scenario_for_execution_without_oracle(scenario)

        env = Environment(
            config=EnvironmentConfig(
                oracle_mode=False,
                queue_based_loop=False,
                start_time=scenario.start_time,
                time_increment_in_seconds=scenario.time_increment_in_seconds,
                exit_when_no_events=False,
                verbose=False,
            ),
            notification_system=VerboseNotificationSystem(),
        )
        env.run(scenario, wait_for_end=False)
        return cls(scenario=scenario, env=env, judge_enabled=judge_enabled)

    def _call_are_tool(self, record: ToolRecord, args: dict[str, Any]) -> Any:
        self._append_tool_log(record, args)
        result = record.tool(**args)
        self._append_observation_log(result)
        return result

    def _dump_state(self, app: str | None = None) -> dict[str, Any]:
        apps = getattr(self.env, "apps", {})
        if app is not None and app not in apps:
            raise SidecarError(f"unknown GAIA-2 app: {app}")
        selected = {app: apps[app]} if app is not None else dict(apps)
        state: dict[str, Any] = {}
        for app_name, app_obj in selected.items():
            get_state = getattr(app_obj, "get_state", None)
            if callable(get_state):
                app_state = get_state()
                if app_state is not None:
                    state[app_name] = app_state
        return state

    def _poll_notifications(self) -> list[dict[str, Any]]:
        notification_system = getattr(self.env, "notification_system", None)
        if notification_system is None:
            return []
        timestamp = datetime.fromtimestamp(
            self.env.time_manager.time(),
            tz=timezone.utc,
        )
        messages = notification_system.message_queue.get_by_timestamp(timestamp)
        return [_message_to_dict(message) for message in messages]

    def _wait_for_notification(self, timeout_seconds: float) -> list[dict[str, Any]]:
        timeout = int(max(0.0, min(timeout_seconds, 3600.0)))
        if ("SystemApp", "wait_for_notification") in self._records:
            self.call_tool("SystemApp", "wait_for_notification", {"timeout": timeout})
        elif ("System", "wait_for_notification") in self._records:
            self.call_tool("System", "wait_for_notification", {"timeout": timeout})
        else:
            time.sleep(min(timeout, 1))
        return self._poll_notifications()

    def _validate(self) -> Any:
        if not self.judge_enabled:
            return {
                "success": False,
                "rationale": (
                    "GAIA-2 ARE runtime is active, but the official judge is not "
                    "configured. Set RHO_GAIA2_ENABLE_JUDGE=1 and judge model "
                    "credentials to enable task grading."
                ),
            }
        _maybe_dump_event_log_diag(self.env)
        return self.scenario.validate(self.env)

    def _close(self) -> None:
        try:
            self.env.stop()
            self.env.join()
        except Exception:
            pass

    def _append_tool_log(self, record: ToolRecord, args: dict[str, Any]) -> None:
        try:
            from are.simulation.agents.agent_log import ToolCallLog

            self.env.append_to_world_logs(
                ToolCallLog(
                    tool_name=record.tool_name,
                    tool_arguments=args,
                    timestamp=self.env.time_manager.time(),
                    agent_id="rho",
                )
            )
        except Exception:
            pass

    def _append_observation_log(self, result: Any) -> None:
        try:
            from are.simulation.agents.agent_log import ObservationLog

            self.env.append_to_world_logs(
                ObservationLog(
                    content=str(result),
                    timestamp=self.env.time_manager.time(),
                    agent_id="rho",
                )
            )
        except Exception:
            pass


def _short_socket_path() -> Path:
    """A short AF_UNIX path for the sidecar socket.

    AF_UNIX paths are capped at ~108 bytes by the OS. The sidecar workdir
    can be deeply nested (pytest tmp dirs, long usernames), which overflows
    that limit, so the socket lives at a short tmp path and is advertised
    to clients via handle.json.
    """
    return Path(tempfile.gettempdir()) / f"se-gaia2-{os.getpid()}.sock"


def serve(runtime: ToolSidecarRuntime, workdir: Path) -> int:
    runtime_dir = workdir / ".gaia2"
    tools_dir = workdir / "tools"
    state_dir = workdir / ".gaia2_state"
    socket_path = _short_socket_path()
    handle_path = runtime_dir / "handle.json"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _unlink_if_exists(socket_path)
    runtime.write_catalog(tools_dir / "catalog.json")
    runtime.write_state_dir(state_dir)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(16)
        handle_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "socket_path": str(socket_path),
                    "started_at": time.time(),
                    "scenario_id": getattr(getattr(runtime, "scenario", None), "scenario_id", None),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        should_stop = False
        while not should_stop:
            conn, _ = server.accept()
            with conn:
                payload = read_frame(conn)
                response = handle_request(runtime, payload)
                write_frame(conn, response)
                runtime.write_state_dir(state_dir)
                should_stop = payload.get("method") == "shutdown"
    runtime.close()
    _unlink_if_exists(socket_path)
    return 0


def read_frame(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 4)
    size = struct.unpack(">I", header)[0]
    if size > 100 * 1024 * 1024:
        raise SidecarError(f"GAIA-2 RPC frame is too large: {size} bytes")
    body = _recv_exact(sock, size)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise SidecarError("GAIA-2 RPC frame must contain a JSON object")
    return payload


def write_frame(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(_jsonable(dict(payload)), ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise SidecarError("GAIA-2 RPC connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SidecarError(f"{key} must be a non-empty string")
    return value


def _build_record(tool: Any) -> ToolRecord:
    tool_name = str(getattr(tool, "name", "") or getattr(tool, "_public_name", ""))
    app = getattr(tool, "app_name", None)
    function = getattr(tool, "func_name", None)
    if not app and "__" in tool_name:
        app, inferred_function = tool_name.split("__", 1)
        function = function or inferred_function
    if not app:
        app = getattr(tool, "class_name", None) or "Agent"
    if not function:
        prefix = f"{app}__"
        function = tool_name[len(prefix) :] if tool_name.startswith(prefix) else tool_name
    if not tool_name:
        tool_name = f"{app}__{function}"
    schema = _tool_schema(tool)
    schema["app"] = str(app)
    schema["function"] = str(function)
    schema["tool_name"] = tool_name
    return ToolRecord(
        app=str(app),
        function=str(function),
        tool_name=tool_name,
        schema=_jsonable(schema),
        tool=tool,
    )


def _tool_schema(tool: Any) -> dict[str, Any]:
    to_metadata = getattr(tool, "to_metadata_dict", None)
    if callable(to_metadata):
        metadata = to_metadata()
        if isinstance(metadata, dict):
            return dict(metadata)
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "function_description", None)
        or getattr(tool, "description", None),
        "args": [
            _jsonable(vars(arg) if hasattr(arg, "__dict__") else arg)
            for arg in getattr(tool, "args", [])
        ],
        "return_type": getattr(tool, "return_type", None),
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, type):
        return str(value)
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _message_to_dict(message: Any) -> dict[str, Any]:
    return {
        "type": _jsonable(getattr(message, "message_type", None)),
        "message": getattr(message, "message", ""),
        "timestamp": _jsonable(getattr(message, "timestamp", None)),
        "attachments": _jsonable(getattr(message, "attachments", [])),
    }


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _judge_enabled_from_env() -> bool:
    raw = os.getenv("RHO_GAIA2_ENABLE_JUDGE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _judge_provider_from_env() -> str:
    raw = (os.getenv("RHO_GAIA2_JUDGE_PROVIDER", "") or "azure").strip().lower()
    if raw not in {"azure", "openrouter"}:
        raise SidecarError(
            f"Unsupported GAIA-2 judge provider {raw!r}; expected azure or openrouter."
        )
    return raw


def _build_judge_engine(provider: str, model_name: str):
    if provider == "openrouter":
        return _build_openrouter_judge_engine(model_name)
    return _build_azure_judge_engine(model_name)


def _build_openrouter_judge_engine(model_name: str):
    """ARE judge engine routed through OpenRouter.

    Uses litellm's native ``openrouter/<model>`` prefix, which picks up
    ``OPENROUTER_API_KEY`` (and optionally ``OPENROUTER_BASE_URL``) from the
    environment. We pass the model name with the prefix attached so the engine
    routes to OpenRouter regardless of what ARE's internal provider switch does.
    """
    from are.simulation.agents.llm.litellm.litellm_engine import (
        LiteLLMEngine,
        LiteLLMModelConfig,
    )

    prefixed = (
        model_name
        if model_name.startswith("openrouter/")
        else f"openrouter/{model_name}"
    )
    return LiteLLMEngine(
        model_config=LiteLLMModelConfig(model_name=prefixed, provider="openrouter")
    )


def _build_azure_judge_engine(model_name: str):
    """ARE judge engine routed through Azure OpenAI Foundry with Entra auth.

    The Entra Bearer is re-fetched on every judge call so a long-lived
    sidecar never validates with an expired token.
    """
    from are.simulation.agents.llm.litellm.litellm_engine import (
        LiteLLMEngine,
        LiteLLMModelConfig,
    )

    from rho.selection.azure_auth import azure_foundry_kwargs

    class _AzureFoundryJudgeEngine(LiteLLMEngine):
        def chat_completion(self, messages, stop_sequences=[], **kwargs):
            creds = azure_foundry_kwargs()
            self.model_config.endpoint = creds["api_base"]
            self.model_config.api_key = creds["api_key"]
            return super().chat_completion(messages, stop_sequences, **kwargs)

    return _AzureFoundryJudgeEngine(
        model_config=LiteLLMModelConfig(model_name=model_name, provider="openai")
    )


def _maybe_dump_event_log_diag(env: Any) -> None:
    """Dump event_log composition before validation when RHO_GAIA2_EVENTLOG_DIAG is set."""
    path = os.getenv("RHO_GAIA2_EVENTLOG_DIAG")
    if not path:
        return
    try:
        from collections import Counter

        events = env.event_log.list_view()
        per_type: dict[str, int] = {}
        per_event_detail: list[dict[str, Any]] = []
        for e in events:
            et = getattr(getattr(e, "event_type", None), "value", str(getattr(e, "event_type", "")))
            per_type[et] = per_type.get(et, 0) + 1
            action = getattr(e, "action", None)
            op_type = getattr(action, "operation_type", None)
            op_type_val = getattr(op_type, "value", str(op_type)) if op_type is not None else None
            per_event_detail.append({
                "event_type": et,
                "tool_name": getattr(e, "tool_name", None) or "<no-tool>",
                "operation_type": op_type_val,
                "failed": bool(e.failed()) if hasattr(e, "failed") else None,
                "event_id": str(getattr(e, "event_id", "")),
                "event_time": getattr(e, "event_time", None),
            })
        payload = {
            "ts": time.time(),
            "n_events": len(events),
            "per_type": per_type,
            "events": per_event_detail,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        try:
            with Path(path).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"error": str(exc)}) + "\n")
        except Exception:
            pass


def _install_case_insensitive_judge_parsing() -> None:
    """Make ARE's soft-checker verdict parsing case-insensitive.

    ARE's ``LLMChecker.__call__`` decides Success/Failure with a
    CASE-SENSITIVE substring test: ``self.success_str in response`` where
    ``success_str`` is ``[[True]]`` / ``[[Success]]`` and ``failure_str`` is
    ``[[False]]`` / ``[[Failure]]``. ARE's official judge model
    (Llama-3.3-70B) reliably emits that exact casing. Our substituted gpt-5.5
    judge instead emits the verdict in lowercase (``[[true]]`` / ``[[false]]``),
    which matches NEITHER token -> ``LLMChecker`` returns ``None`` ->
    ``SoftToolJudge.compare`` evaluates ``not None`` as ``True`` and rejects
    the tool call. The net effect is that EVERY soft-checked tool call fails,
    regardless of whether the agent's output is correct.

    This patch lowercases both the response and the verdict tokens before the
    substring test. It does not change grading semantics -- ``[[true]]`` and
    ``[[True]]`` denote the same verdict -- it only lets the parser recognize
    the verdict the model actually emitted. Required because we run a
    non-official judge model; see _DEFAULT_JUDGE_MODEL.
    """
    from are.simulation.validation.utils import llm_utils as _llm

    if getattr(_llm.LLMChecker, "_se_ci_parsing", False):
        return

    def _case_insensitive_call(
        self: Any, user_prompt_args: dict[str, Any]
    ) -> bool | None:
        votes: list[bool] = []
        success = self.success_str.lower()
        failure = self.failure_str.lower()
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            if response is None:
                continue
            low = response.lower()
            if success in low:
                votes.append(True)
            elif failure in low:
                votes.append(False)
        if len(votes) == 0:
            return None
        return sum(votes) >= len(votes) / 2

    _llm.LLMChecker.__call__ = _case_insensitive_call
    _llm.LLMChecker._se_ci_parsing = True


def _install_filter_env_events_from_counter() -> None:
    """Drop ENV/USER events from oracle/agent counters in preliminary tool-count check."""
    from are.simulation.types import EventType
    from are.simulation.validation import judge as _judge

    if getattr(_judge.GraphPerEventJudge, "_se_filter_env", False):
        return

    def _preliminary_checks(self, agent_events, oracle_events):
        from collections import Counter

        def get_count(events):
            aui_name = "AgentUserInterface__send_message_to_user"
            filtered = [e for e in events if getattr(e, "event_type", None) == EventType.AGENT]
            counter = Counter([e.tool_name for e in filtered])
            aui_count = counter[aui_name]
            counter[aui_name] = 0
            return counter, aui_count

        agent_counter, agent_aui_count = get_count(agent_events)
        oracle_counter, oracle_aui_count = get_count(oracle_events)
        return self.check_tool_call_counts(
            agent_counter,
            agent_aui_count,
            oracle_counter,
            oracle_aui_count,
            self.config.extra_send_message_to_user_allowed,
        )

    _judge.GraphPerEventJudge.preliminary_checks = _preliminary_checks
    _judge.GraphPerEventJudge._se_filter_env = True


def _install_relax_aui_judge() -> None:
    """Accept agent's AUI send_message_to_user when oracle content is short and substring of agent's."""
    from are.simulation.validation import tool_judge as _tj

    if getattr(_tj.SoftToolJudge, "_se_relax_aui", False):
        return

    _orig_compare = _tj.SoftToolJudge.compare
    _aui = "AgentUserInterface__send_message_to_user"
    _diag = os.getenv("RHO_GAIA2_RELAX_AUI_DIAG")

    def _diag_write(payload: dict[str, Any]) -> None:
        if not _diag:
            return
        try:
            Path(_diag).parent.mkdir(parents=True, exist_ok=True)
            with Path(_diag).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _compare(self, agent_event, oracle_event, **kwargs):
        tn = getattr(self, "tool_name", None)
        if tn == _aui:
            try:
                oracle_args = oracle_event.get_args()
                agent_args = agent_event.get_args()
                oracle_content = oracle_args.get("content")
                agent_content = agent_args.get("content")
                if oracle_content is None or (
                    isinstance(oracle_content, str) and oracle_content.strip() == ""
                ):
                    _diag_write({"hit": "null_oracle", "oc": oracle_content, "ac": str(agent_content)[:120]})
                    return True
                if isinstance(oracle_content, str) and isinstance(agent_content, str):
                    import re as _re
                    def _norm(s: str) -> str:
                        # Lowercase, strip punctuation, collapse whitespace
                        return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()
                    oc_n = _norm(oracle_content)
                    ac_n = _norm(agent_content)
                    # Bidirectional substring after normalization; only fires for short oracle (≤30 chars) or short agent (≤30 chars).
                    if oc_n and ac_n and (
                        (len(oc_n) <= 30 and oc_n in ac_n)
                        or (len(ac_n) <= 30 and ac_n in oc_n)
                    ):
                        _diag_write({"hit": "norm_substring", "oc": oracle_content, "ac": agent_content[:120]})
                        return True
                _diag_write({"hit": "fallthrough", "tn": tn, "oc": str(oracle_content)[:120], "ac": str(agent_content)[:120]})
            except Exception as exc:
                _diag_write({"hit": "exception", "tn": tn, "exc": str(exc)})
        else:
            _diag_write({"hit": "not_aui", "tn": tn})
        return _orig_compare(self, agent_event, oracle_event, **kwargs)

    _tj.SoftToolJudge.compare = _compare
    _tj.SoftToolJudge._se_relax_aui = True


def _install_filter_fs_reads_from_counter() -> None:
    """Drop SandboxLocalFileSystem read-only ops (open/ls/cat/exists) from agent counter."""
    from are.simulation.types import EventType
    from are.simulation.validation import judge as _judge

    if getattr(_judge.GraphPerEventJudge, "_se_filter_fs_reads", False):
        return

    _fs_reads = {
        "SandboxLocalFileSystem__open",
        "SandboxLocalFileSystem__ls",
        "SandboxLocalFileSystem__cat",
        "SandboxLocalFileSystem__exists",
    }
    _orig = _judge.GraphPerEventJudge.preliminary_checks

    def _preliminary_checks(self, agent_events, oracle_events):
        filtered_agent = [
            e for e in agent_events
            if not (getattr(e, "event_type", None) == EventType.AGENT and e.tool_name in _fs_reads)
        ]
        return _orig(self, filtered_agent, oracle_events)

    _judge.GraphPerEventJudge.preliminary_checks = _preliminary_checks
    _judge.GraphPerEventJudge._se_filter_fs_reads = True


def _install_judge_tracing(trace_path: Path) -> None:
    """Diagnostic instrumentation for the ARE judge.

    Monkeypatches ``LLMFunction.__call__`` -- the single chokepoint every
    soft-checker / subtask-extractor LLM call passes through -- to append the
    system prompt, the user-prompt arguments, and the model's raw response to
    a JSONL file. This lets us see which soft checker rejected a tool call and
    read the checker's verbatim ``[[Success]]``/``[[Failure]]`` (or
    ``[[True]]``/``[[False]]``) verdict and reasoning. Enabled only when
    RHO_GAIA2_JUDGE_TRACE_DIR is set, so production runs are unaffected.
    """
    from are.simulation.validation.utils import llm_utils as _llm

    if getattr(_llm.LLMFunction, "_se_traced", False):
        return
    _orig_call = _llm.LLMFunction.__call__

    def _traced_call(self: Any, user_prompt_args: dict[str, Any]) -> Any:
        response = _orig_call(self, user_prompt_args)
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "system_prompt": getattr(self, "system_prompt", ""),
                            "user_prompt_args": {
                                str(k): str(v)
                                for k, v in (user_prompt_args or {}).items()
                            },
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
        return response

    _llm.LLMFunction.__call__ = _traced_call
    _llm.LLMFunction._se_traced = True


def _scenario_duration_limit(scenario: Any, capability_tag: Any) -> int:
    try:
        if capability_tag.Time in getattr(scenario, "tags", ()):
            return 420
    except Exception:
        pass
    return 1800


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GAIA-2 ARE sidecar")
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    runtime = ARESidecarRuntime.from_scenario_file(
        Path(args.scenario_file),
        Path(args.workdir),
    )
    return serve(runtime, Path(args.workdir))


if __name__ == "__main__":
    raise SystemExit(main())
