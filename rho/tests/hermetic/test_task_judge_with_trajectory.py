from __future__ import annotations

import json
from pathlib import Path

import pytest

from rho.protocols import Trajectory
from rho.selection.difficulty_selector import TaskJudge
from rho.selection.llm_client import FakeLLMClient


def _traj(task_id: str) -> Trajectory:
    return Trajectory(
        id=f"traj_{task_id}", kind="solve", task_id=task_id, harness_id="h",
        instructions="", events=[
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message",
                                                 "text": f"thinking about {task_id}"}},
        ],
        final_message=f"final {task_id}", stdout="", stderr="",
        workspace_diff={}, workspace_deletions=frozenset(),
        exit_code=0, wall_time_s=1.0,
    )


class _FakeTask:
    def __init__(self, tid): self.id = tid
    @property
    def harness(self): return None
    def materialize(self, _): pass
    def query(self): return f"please solve {self.id}"
    def grade(self, traj, *, artifacts_dir=None): raise NotImplementedError
    @property
    def agent_timeout_s(self): return None


def _judge_completion(difficulty: float = 4.0) -> str:
    return json.dumps({
        "difficulty": difficulty,
        "abstract_fingerprint": (
            "Failure mode partial propagation across modules with "
            "scattered invariants requires contextual tracing of a "
            "shared contract across multiple call sites making the "
            "change non-local and mechanically subtle to verify while "
            "preserving ordering assumptions, boundary reconciliation, "
            "and consistent state transitions across dependent branches."
        ),
    })


def test_judge_prompt_contains_query_and_digest(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_complete(prompt, model):
        captured["prompt"] = prompt
        return _judge_completion()

    judge = TaskJudge(
        client=FakeLLMClient(fake_complete),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories={"t0": _traj("t0")},
        cache_root=None,
    )
    judge.judge(_FakeTask("t0"))
    assert "please solve t0" in captured["prompt"]
    assert "thinking about t0" in captured["prompt"]
    assert "Observed agent run" in captured["prompt"]
    assert "## Final message" in captured["prompt"]


def test_judge_missing_trajectory_raises(tmp_path: Path) -> None:
    judge = TaskJudge(
        client=FakeLLMClient(lambda p, m: _judge_completion()),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories={},  # nothing
        cache_root=None,
    )
    with pytest.raises(KeyError, match="t_missing"):
        judge.judge(_FakeTask("t_missing"))


def test_judge_emits_digest_sidecar(tmp_path: Path) -> None:
    judge = TaskJudge(
        client=FakeLLMClient(lambda p, m: _judge_completion()),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories={"task/with/slash": _traj("task/with/slash")},
        cache_root=None,
    )
    judge.judge(_FakeTask("task/with/slash"))
    digest_dir = tmp_path / "calls" / "short_solve"
    assert (digest_dir / "task__with__slash.txt").exists()
    assert (tmp_path / "calls" / "task__with__slash.json").exists()


def test_judge_result_includes_token_estimate(tmp_path: Path) -> None:
    judge = TaskJudge(
        client=FakeLLMClient(lambda p, m: _judge_completion()),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories={"t0": _traj("t0")},
        cache_root=None,
    )
    result = judge.judge(_FakeTask("t0"))
    assert result.digest_token_estimate > 0
