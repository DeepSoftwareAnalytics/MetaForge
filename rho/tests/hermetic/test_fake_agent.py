import pytest

from rho.agent.fake import FakeAgent, FakeResponse


def test_fake_agent_mode_dispatch_and_schema_validation(tmp_path) -> None:
    def solve_script(workspace, instructions, output_schema):
        del workspace, instructions, output_schema
        return FakeResponse(final_message='{"value": 1, "rationale": "ok"}')

    agent = FakeAgent({"solve": solve_script})
    trajectory = agent.run(
        tmp_path,
        "MODE: solve\n\nhello",
        output_schema={
            "type": "object",
            "properties": {
                "value": {"type": "integer", "minimum": -10, "maximum": 10},
                "rationale": {"type": "string"},
            },
            "required": ["value", "rationale"],
        },
    )
    assert trajectory.final_message


def test_fake_agent_missing_mode_raises(tmp_path) -> None:
    agent = FakeAgent({})
    with pytest.raises(KeyError):
        agent.run(tmp_path, "MODE: solve\n\nhello")
