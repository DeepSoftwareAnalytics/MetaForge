from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from rho.agent.cache import build_default_agent
from rho.agent.codex import CodexAgent
from rho.datasets.directory import DirectoryTask
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies.letta_sleep import LettaSleepStrategy

pytestmark = pytest.mark.codex


def test_letta_sleep_real_codex_one_task_one_sample(
    toy_dataset_root,
    tmp_path: Path,
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    agent = _real_codex_agent(default_timeout_s=480.0)

    result = LettaSleepStrategy().propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[(task, _sleep_feed_trajectories(task.id, harness.id))],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.optimize_trajectory.kind == "optimize"
    assert sample.optimize_trajectory.stage == "round_optimize:letta_s0_t0000"
    assert any(
        event.get("type") == "letta_sleep.metrics"
        for event in sample.optimize_trajectory.events
    )
    persisted = list(traj_store.list_for_task(task.id))
    assert [traj.id for traj in persisted] == [sample.optimize_trajectory.id]


def _real_codex_agent(*, default_timeout_s: float):
    binary = shutil.which("codex")
    if binary is None:
        pytest.skip("codex CLI not found on PATH")
    return build_default_agent(
        CodexAgent(
            codex_config_path=_codex_config_for_tests(),
            binary=binary,
            fallback_sandbox="danger-full-access",
            default_timeout_s=default_timeout_s,
        ),
        mode="off",
    )


def _codex_config_for_tests() -> Path:
    override = os.environ.get("RHO_CODEX_CONFIG")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
    repo_config = Path(__file__).resolve().parents[2] / "configs" / "codex.azure-foundry.toml"
    if repo_config.is_file():
        return repo_config
    user_config = Path.home() / ".codex" / "config.toml"
    if user_config.is_file():
        return user_config.resolve()
    pytest.skip("no codex config available for real-codex smoke test")


def _sleep_feed_trajectories(task_id: str, harness_id: str) -> list[Trajectory]:
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": (
                    "The primary agent learned this durable project fact from "
                    "the user: team project code name is Phoenix."
                ),
            },
        }
    ]
    return [
        Trajectory(
            id=f"solve_{ix}",
            kind="solve",
            task_id=task_id,
            harness_id=harness_id,
            instructions="Solve the task.",
            events=events,
            final_message=(
                "I could not answer, but the transcript states: team project "
                "code name is Phoenix."
            ),
            stdout="",
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=0.01,
            sample_index=ix,
        )
        for ix in range(3)
    ]
