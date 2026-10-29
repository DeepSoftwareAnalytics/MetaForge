from __future__ import annotations

from pathlib import Path

import pytest

from rho.agent.fake import FakeAgent, FakeResponse
from rho.datasets.directory import DirectoryTask
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import build_optimize_strategy
from rho.strategies.dynamic_cheatsheet import (
    CHEATSHEET_FILENAME,
    CHEATSHEET_INITIAL_CONTENT,
)


def test_dynamic_cheatsheet_single_stream_accumulates_across_tasks(
    toy_dataset_root,
    tmp_path: Path,
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", harness),
        DirectoryTask(toy_dataset_root / "train" / "task_002", harness),
    ]
    agent = _curator_agent()

    strategy = build_optimize_strategy("dynamic-cheatsheet")
    result = strategy.propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[
            (task, _stub_trajectories(task.id, harness.id)) for task in tasks
        ],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.sample_index == 0
    assert sample.optimize_trajectory.kind == "optimize"
    # Stage records the LAST per-task curator step in this single-stream sample.
    assert sample.optimize_trajectory.stage == "round_optimize:dc_t0001"
    assert sample.candidate is not None

    materialized = tmp_path / "materialized"
    sample.candidate.materialize(materialized)
    contents = (materialized / CHEATSHEET_FILENAME).read_text(encoding="utf-8")
    assert contents != CHEATSHEET_INITIAL_CONTENT
    # Single stream → cumulative across tasks → both tasks appear.
    assert "task_001" in contents
    assert "task_002" in contents

    # M=2 curator runs, all persisted to traj_store.
    persisted_ids = {
        traj.id for task in tasks for traj in traj_store.list_for_task(task.id)
    }
    assert len(persisted_ids) == 2


def test_dynamic_cheatsheet_workspace_layout(toy_dataset_root, tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness_src = tmp_path / "harness_src"
    harness_src.mkdir()
    (harness_src / "notes.md").write_text("base harness\n", encoding="utf-8")
    harness = harness_store.capture(harness_src)
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", harness),
        DirectoryTask(toy_dataset_root / "train" / "task_002", harness),
    ]

    snapshots: list[set[str]] = []

    def capture_workspace(workspace: Path, instructions: str, output_schema):
        del output_schema
        snapshots.append(
            {
                path.relative_to(workspace).as_posix()
                for path in workspace.rglob("*")
                if path.is_file()
                and path.relative_to(workspace).parts[0] != ".rho"
                and path.name != ".sample_index"
            }
        )
        assert "Cheatsheet Curator" in instructions
        assert "<memory_item>" in instructions
        assert "Version" in instructions
        rel = f"harness/{CHEATSHEET_FILENAME}"
        existing = (workspace / rel).read_text(encoding="utf-8")
        new_content = existing + f"- saw {workspace.name}\n"
        return FakeResponse(
            final_message="updated",
            workspace_edits={rel: new_content.encode("utf-8")},
        )

    agent = FakeAgent({"optimize": capture_workspace})

    strategy = build_optimize_strategy("dynamic-cheatsheet")
    result = strategy.propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[
            (task, _stub_trajectories(task.id, harness.id)) for task in tasks
        ],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir_layout",
        n_samples=1,
        round_ix=0,
    )

    assert len(result.samples) == 1
    assert len(snapshots) == 2  # one workspace per task in the single stream

    expected = {
        "harness/notes.md",
        f"harness/{CHEATSHEET_FILENAME}",
        "transcript/prompt.md",
        "transcript/trajectory_0/events.jsonl",
        "transcript/trajectory_0/final_message.txt",
        "transcript/trajectory_1/events.jsonl",
        "transcript/trajectory_1/final_message.txt",
        "transcript/trajectory_2/events.jsonl",
        "transcript/trajectory_2/final_message.txt",
    }
    for task_ix, snap in enumerate(snapshots):
        assert expected.issubset(snap), (
            f"task_{task_ix} workspace missing files: {expected - snap}"
        )

    materialized = tmp_path / "materialized_candidate"
    assert result.samples[0].candidate is not None
    result.samples[0].candidate.materialize(materialized)
    assert (materialized / "notes.md").read_text(encoding="utf-8") == "base harness\n"
    cheatsheet = (materialized / CHEATSHEET_FILENAME).read_text(encoding="utf-8")
    assert cheatsheet.count("- saw ") == 2


def test_dynamic_cheatsheet_rejects_n_samples_other_than_one(
    toy_dataset_root, tmp_path: Path
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)

    strategy = build_optimize_strategy("dynamic-cheatsheet")
    with pytest.raises(ValueError, match="n_samples=1"):
        strategy.propose_candidates(
            agent=FakeAgent({"optimize": lambda *_, **__: FakeResponse()}),
            harness=harness,
            tasks_with_trajectories=[(task, _stub_trajectories(task.id, harness.id))],
            harness_store=harness_store,
            traj_store=traj_store,
            workdir=tmp_path / "workdir_reject",
            n_samples=2,
            round_ix=0,
        )


def test_dynamic_cheatsheet_failed_curator_does_not_advance_chain(
    toy_dataset_root, tmp_path: Path
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", harness),
        DirectoryTask(toy_dataset_root / "train" / "task_002", harness),
    ]
    call_index = {"n": 0}

    def script(workspace: Path, instructions: str, output_schema):
        del instructions, output_schema
        rel = f"harness/{CHEATSHEET_FILENAME}"
        if call_index["n"] == 0:
            call_index["n"] += 1
            existing = (workspace / rel).read_text(encoding="utf-8")
            new_content = existing + "- saw task_001\n"
            return FakeResponse(
                final_message="ok",
                workspace_edits={rel: new_content.encode("utf-8")},
            )
        call_index["n"] += 1
        return FakeResponse(final_message="boom", exit_code=1)

    agent = FakeAgent({"optimize": script})
    strategy = build_optimize_strategy("dynamic-cheatsheet")
    result = strategy.propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[
            (task, _stub_trajectories(task.id, harness.id)) for task in tasks
        ],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir_fail",
        n_samples=1,
        round_ix=0,
    )

    sample = result.samples[0]
    metrics = next(
        e["metrics"]
        for e in sample.optimize_trajectory.events
        if e.get("type") == "dc_cheatsheet.metrics"
    )
    assert metrics["dc_curator_calls"] == 2
    assert metrics["dc_curator_failures"] == 1

    # Candidate carries forward only task_001's edit; task_002's failed run
    # neither captured a new harness nor reverted the chain.
    assert sample.candidate is not None
    materialized = tmp_path / "materialized_fail"
    sample.candidate.materialize(materialized)
    contents = (materialized / CHEATSHEET_FILENAME).read_text(encoding="utf-8")
    assert "- saw task_001" in contents
    assert "task_002" not in contents


def test_dynamic_cheatsheet_emits_metrics_event_on_last_traj(
    toy_dataset_root, tmp_path: Path
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", harness),
        DirectoryTask(toy_dataset_root / "train" / "task_002", harness),
    ]
    agent = _curator_agent()

    strategy = build_optimize_strategy("dynamic-cheatsheet")
    result = strategy.propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[
            (task, _stub_trajectories(task.id, harness.id)) for task in tasks
        ],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir_metrics",
        n_samples=1,
        round_ix=0,
    )

    last_traj = result.samples[0].optimize_trajectory
    metrics_events = [
        event for event in last_traj.events if event.get("type") == "dc_cheatsheet.metrics"
    ]
    assert len(metrics_events) == 1
    metrics = metrics_events[0]["metrics"]
    assert metrics["dc_curator_calls"] == 2
    assert metrics["dc_curator_failures"] == 0
    assert len(metrics["dc_curator_traj_ids"]) == 2
    assert metrics["dc_cheatsheet_word_count"] > 0
    assert metrics["dc_cheatsheet_byte_size"] > 0


def _curator_agent() -> FakeAgent:
    def curator_script(workspace: Path, instructions: str, output_schema):
        del instructions, output_schema
        rel = f"harness/{CHEATSHEET_FILENAME}"
        existing = (workspace / rel).read_text(encoding="utf-8")
        first_final = (
            workspace / "transcript" / "trajectory_0" / "final_message.txt"
        ).read_text(encoding="utf-8")
        task_id = first_final.split("task ")[1].split(" ")[0]
        new_content = existing + f"- saw {task_id}\n"
        return FakeResponse(
            final_message="cheatsheet updated",
            workspace_edits={rel: new_content.encode("utf-8")},
        )

    return FakeAgent({"optimize": curator_script})


def _stub_trajectories(task_id: str, harness_id: str) -> list[Trajectory]:
    return [_solve_trajectory(task_id, harness_id, sample_index) for sample_index in range(3)]


def _solve_trajectory(task_id: str, harness_id: str, sample_index: int) -> Trajectory:
    return Trajectory(
        id=f"solve_{task_id}_{sample_index}",
        kind="solve",
        task_id=task_id,
        harness_id=harness_id,
        instructions="Solve the task.",
        events=[],
        final_message=f"I attempted task {task_id} sample {sample_index}.",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.01,
        sample_index=sample_index,
    )
