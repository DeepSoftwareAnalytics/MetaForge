from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

from rho.datasets.gaia2.sidecar import (
    _DEFAULT_JUDGE_MODEL,
    ToolSidecarRuntime,
    _judge_enabled_from_env,
    handle_request,
)


class FakeMessageType(Enum):
    USER = "USER"


class FakeTool:
    app_name = "Mail"
    name = "Mail__send"
    func_name = "send"
    function_description = "Send a message."
    args = [
        SimpleNamespace(
            name="to",
            arg_type="str",
            description="Recipient address",
            has_default=False,
            default=None,
            type_obj=str,
        )
    ]

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"accepted": True, "args": kwargs}

    def to_metadata_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.function_description,
            "args": [
                {
                    "name": "to",
                    "arg_type": "str",
                    "description": "Recipient address",
                    "type_obj": str,
                }
            ],
        }


def test_sidecar_runtime_lists_schemas_calls_and_validates() -> None:
    tool = FakeTool()
    runtime = ToolSidecarRuntime(
        tools=[tool],
        validate_fn=lambda: SimpleNamespace(success=True, rationale="ok"),
        state_fn=lambda app=None: {"Mail": {"calls": tool.calls}},
    )

    listed = handle_request(runtime, {"method": "list_tools"})
    assert listed["ok"] is True
    assert listed["tools"]["Mail"]["send"]["tool_name"] == "Mail__send"

    schema = handle_request(
        runtime,
        {"method": "schema", "app": "Mail", "function": "send"},
    )
    assert schema["ok"] is True
    assert schema["schema"]["args"][0]["name"] == "to"
    assert schema["schema"]["args"][0]["type_obj"] == "<class 'str'>"

    called = handle_request(
        runtime,
        {
            "method": "call_tool",
            "app": "Mail",
            "function": "send",
            "args": {"to": "team@example.test"},
        },
    )
    assert called["ok"] is True
    assert called["result"]["accepted"] is True
    assert tool.calls == [{"to": "team@example.test"}]

    state = handle_request(runtime, {"method": "dump_state", "app": "Mail"})
    assert state["ok"] is True
    assert state["state"]["Mail"]["calls"] == [{"to": "team@example.test"}]

    validation = handle_request(runtime, {"method": "validate"})
    assert validation["ok"] is True
    assert validation["result"]["success"] is True
    assert validation["result"]["rationale"] == "ok"


def test_sidecar_runtime_reports_unknown_tool() -> None:
    runtime = ToolSidecarRuntime(tools=[])

    response = handle_request(
        runtime,
        {"method": "schema", "app": "Mail", "function": "send"},
    )

    assert response["ok"] is False
    assert "unknown GAIA-2 tool" in response["error"]


def test_judge_enabled_only_via_explicit_env(monkeypatch) -> None:
    monkeypatch.delenv("RHO_GAIA2_ENABLE_JUDGE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-irrelevant")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf-irrelevant")
    assert _judge_enabled_from_env() is False

    monkeypatch.setenv("RHO_GAIA2_ENABLE_JUDGE", "1")
    assert _judge_enabled_from_env() is True

    monkeypatch.setenv("RHO_GAIA2_ENABLE_JUDGE", "off")
    assert _judge_enabled_from_env() is False


def test_default_judge_model_is_azure_foundry_gpt() -> None:
    assert _DEFAULT_JUDGE_MODEL == "gpt-5.5"


def test_sidecar_serializes_enum_values_in_notifications() -> None:
    runtime = ToolSidecarRuntime(
        tools=[],
        notification_fn=lambda: [
            SimpleNamespace(message_type=FakeMessageType.USER, message="hello")
        ],
    )

    response = handle_request(runtime, {"method": "poll_notifications"})

    assert response["ok"] is True
    assert response["notifications"][0]["message_type"] == "USER"


def test_turn_collapse_monkeypatch_is_removed() -> None:
    """GAIA-2 must use ARE's official per-turn scoring — the single-turn
    collapse monkeypatch must not be reintroduced."""
    from rho.datasets.gaia2 import sidecar

    assert not hasattr(sidecar, "_collapse_turns_to_single")


def test_extra_send_message_budget_is_relaxed() -> None:
    """The per-turn extra-send_message_to_user budget is widened beyond
    ARE's default of 1, tolerating minor within-turn over-messaging."""
    from rho.datasets.gaia2.sidecar import (
        _EXTRA_SEND_MESSAGE_TO_USER_ALLOWED,
    )

    assert isinstance(_EXTRA_SEND_MESSAGE_TO_USER_ALLOWED, int)
    assert _EXTRA_SEND_MESSAGE_TO_USER_ALLOWED > 1


def test_soft_judge_rubrics_are_not_overridden() -> None:
    """We do NOT relax ARE's soft-judge rubrics. The 0/5 smoke regression was
    a verdict-PARSING bug, not rubric over-strictness, so the rubric-override
    helper and its prompt constants must not be reintroduced."""
    from rho.datasets.gaia2 import sidecar

    assert not hasattr(sidecar, "_install_lenient_soft_judge_prompts")
    for name in (
        "_SE_CONTENT_SYSTEM",
        "_SE_EMAIL_SYSTEM",
        "_SE_MESSAGE_SYSTEM",
        "_SE_USER_MESSAGE_SYSTEM",
    ):
        assert not hasattr(sidecar, name)


def test_case_insensitive_judge_parsing_helper_defined() -> None:
    """The case-insensitive verdict-parsing patch exists. ARE's LLMChecker
    matches [[True]]/[[Success]] case-sensitively; the substituted gpt-5.5
    judge emits lowercase verdicts, so without this patch every soft-checked
    tool call is rejected."""
    from rho.datasets.gaia2 import sidecar

    assert hasattr(sidecar, "_install_case_insensitive_judge_parsing")
    assert hasattr(sidecar, "_install_judge_tracing")
