from types import SimpleNamespace
import json

import pytest

from rho.selection.codex_judge_client import CodexJudgeClient, JUDGE_SCHEMA


class StubAgent:
    model = "gpt-6-astra"
    reasoning_effort = "high"

    def __init__(self, **changes):
        self.result = SimpleNamespace(exit_code=0, timed_out=False, wall_time_s=1,
            events=[], final_message='{"difficulty":4,"abstract_fingerprint":"example"}')
        self.result.__dict__.update(changes)

    def run(self, workspace, instructions, **kwargs):
        self.request = (workspace, instructions, kwargs)
        return self.result


def test_judge_preserves_native_prompt_and_isolates_network(tmp_path):
    agent = StubAgent()
    client = CodexJudgeClient(agent, tmp_path, request_env={"NO_PROXY": "*"})
    text = client.complete("native task and trajectory", model="gpt-6-astra",
                           reasoning_effort="high", max_tokens=16384)
    assert json.loads(text)["difficulty"] == 4
    ws, instructions, kwargs = agent.request
    assert instructions.endswith("native task and trajectory")
    assert kwargs["output_schema"] == JUDGE_SCHEMA
    assert kwargs["env"] == {"NO_PROXY": "*"}
    assert kwargs["timeout_s"] == 300
    assert json.loads((ws / "metadata.json").read_text())["max_tokens_enforced"] is False


@pytest.mark.parametrize("changes", [
    {"exit_code": 1}, {"timed_out": True}, {"final_message": ""},
    {"events": [{"type": "item.completed", "item": {"type": "command_execution"}}]},
])
def test_judge_rejects_failed_or_tool_using_calls(tmp_path, changes):
    agent = StubAgent(**changes)
    with pytest.raises((RuntimeError, ValueError)):
        CodexJudgeClient(agent, tmp_path).complete("task", model="gpt-6-astra")
    assert list(tmp_path.glob("*/metadata.json"))


def test_judge_does_not_silently_change_model(tmp_path):
    with pytest.raises(ValueError, match="model"):
        CodexJudgeClient(StubAgent(), tmp_path).complete("task", model="other")
