import json

import pytest

from rho.orchestrators.evaluate import evaluate
from rho.protocols import Trajectory


def _trajectory(final_message: str, harness_id: str) -> Trajectory:
    return Trajectory(
        id=f"{harness_id}_traj",
        kind="solve",
        task_id="task_001",
        harness_id=harness_id,
        instructions="MODE: solve",
        events=[],
        final_message=final_message,
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )


def test_evaluate_primitive_parses_score(fake_agent_default_scripts, toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask
    from rho.stores.harness import FilesystemHarnessStore

    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    before = _trajectory("I don't know project code name", "h_before")
    after = _trajectory("team project code name is Phoenix", "h_after")
    eval_traj, score = evaluate(fake_agent_default_scripts, task, before, after, workdir=tmp_path)

    assert eval_traj.kind == "evaluate"
    assert score.value > 0


def test_evaluate_presents_after_as_a_and_preserves_positive_candidate_score(
    toy_dataset_root, tmp_path
) -> None:
    from rho.agent.fake import FakeAgent, FakeResponse
    from rho.datasets.directory import DirectoryTask
    from rho.stores.harness import FilesystemHarnessStore

    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    before = _trajectory("I don't know project code name", "h_before")
    after = _trajectory("team project code name is Phoenix", "h_after")
    seen: dict[str, str] = {}

    def evaluate_script(workspace, instructions, output_schema):
        del instructions, output_schema
        seen["a"] = (workspace / "trajectory_A" / "final_message.txt").read_text(
            encoding="utf-8"
        )
        seen["b"] = (workspace / "trajectory_B" / "final_message.txt").read_text(
            encoding="utf-8"
        )
        return FakeResponse(
            final_message=json.dumps(
                {"value": -7, "rationale": "A is stronger than B"}
            )
        )

    agent = FakeAgent({"evaluate": evaluate_script})
    eval_traj, score = evaluate(agent, task, before, after, workdir=tmp_path)

    assert seen == {"a": after.final_message, "b": before.final_message}
    assert eval_traj.harness_id == "h_before->h_after"
    assert score.value == 7


def test_evaluate_primitive_falls_back_on_invalid_json(toy_dataset_root, tmp_path) -> None:
    from rho.agent.fake import FakeAgent, FakeResponse
    from rho.datasets.directory import DirectoryTask
    from rho.stores.harness import FilesystemHarnessStore

    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    before = _trajectory("I don't know project code name", "h_before")
    after = _trajectory("team project code name is Phoenix", "h_after")
    agent = FakeAgent(
        {
            "evaluate": lambda workspace, instructions, output_schema: FakeResponse(
                final_message="not json"
            )
        }
    )

    eval_traj, score = evaluate(agent, task, before, after, workdir=tmp_path)
    assert score.value == 0
    assert "unusable" in score.rationale


def test_evaluate_passes_one_hour_timeout_to_agent(fake_agent_default_scripts, toy_dataset_root, tmp_path, monkeypatch):
    from rho.datasets.directory import DirectoryTask
    from rho.stores.harness import FilesystemHarnessStore

    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    original_run = fake_agent_default_scripts.run
    timeouts = []

    def recording_run(*args, **kwargs):
        timeouts.append(kwargs["timeout_s"])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(fake_agent_default_scripts, "run", recording_run)
    evaluate(fake_agent_default_scripts, task, _trajectory("before", "h_before"),
             _trajectory("after", "h_after"), workdir=tmp_path, timeout_s=3600)
    assert timeouts == [3600]
