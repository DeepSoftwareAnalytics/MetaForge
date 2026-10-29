from __future__ import annotations

import json
from pathlib import Path

import pytest

from rho.agent.fake import FakeAgent, FakeResponse
from rho.datasets.directory import DirectoryTask
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import build_optimize_strategy


def test_strategy_smoke_produces_samples_and_expected_aux_fields(
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

    cases = [
        ("query-only", {}, False),
        ("trajectory", {"trajectories_per_task": 2}, False),
        ("diagnosis", {}, True),
        ("diagnosis-no-consistency", {}, True),
        ("diagnosis-no-validation", {}, True),
    ]
    for name, kwargs, expects_diagnosis in cases:
        strategy = build_optimize_strategy(name, **kwargs)
        result = strategy.propose_candidates(
            agent=_strategy_agent(),
            harness=harness,
            tasks_with_trajectories=[
                (task, _stub_trajectories(task.id, harness.id)) for task in tasks
            ],
            harness_store=harness_store,
            traj_store=traj_store,
            workdir=tmp_path / f"workdir_{name}",
            n_samples=2,
            round_ix=0,
        )

        assert len(result.samples) == 2
        assert any(sample.candidate is not None for sample in result.samples)
        assert all(sample.optimize_trajectory.kind == "optimize" for sample in result.samples)
        assert [sample.sample_index for sample in result.samples] == [0, 1]
        if expects_diagnosis:
            assert result.diagnoses is not None
            assert result.diagnose_trajectories is not None
            assert len(result.diagnoses) == len(tasks)
            assert len(result.diagnose_trajectories) == len(tasks)
        else:
            assert result.diagnoses is None
            assert result.diagnose_trajectories is None


def test_strategy_workspace_layouts(monkeypatch, toy_dataset_root, tmp_path: Path) -> None:
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
    expectations = {
        "query-only": {
            "harness/notes.md",
            "tasks/task_0000/prompt.md",
            "tasks/task_0001/prompt.md",
        },
        "trajectory": {
            "harness/notes.md",
            "tasks/task_0000/prompt.md",
            "tasks/task_0000/trajectory_0/events.jsonl",
            "tasks/task_0000/trajectory_0/final_message.txt",
            "tasks/task_0000/trajectory_1/events.jsonl",
            "tasks/task_0000/trajectory_1/final_message.txt",
            "tasks/task_0000/trajectory_2/events.jsonl",
            "tasks/task_0000/trajectory_2/final_message.txt",
            "tasks/task_0001/prompt.md",
            "tasks/task_0001/trajectory_0/events.jsonl",
            "tasks/task_0001/trajectory_0/final_message.txt",
            "tasks/task_0001/trajectory_1/events.jsonl",
            "tasks/task_0001/trajectory_1/final_message.txt",
            "tasks/task_0001/trajectory_2/events.jsonl",
            "tasks/task_0001/trajectory_2/final_message.txt",
        },
        "diagnosis": {
            "harness/notes.md",
            "diagnoses/task_0000/diagnosis.md",
            "diagnoses/task_0000/prompt.md",
            "diagnoses/task_0001/diagnosis.md",
            "diagnoses/task_0001/prompt.md",
        },
        "diagnosis-no-consistency": {
            "harness/notes.md",
            "diagnoses/task_0000/diagnosis.md",
            "diagnoses/task_0000/prompt.md",
            "diagnoses/task_0001/diagnosis.md",
            "diagnoses/task_0001/prompt.md",
        },
        "diagnosis-no-validation": {
            "harness/notes.md",
            "diagnoses/task_0000/diagnosis.md",
            "diagnoses/task_0000/prompt.md",
            "diagnoses/task_0001/diagnosis.md",
            "diagnoses/task_0001/prompt.md",
        },
    }

    for name in (
        "query-only",
        "trajectory",
        "diagnosis",
        "diagnosis-no-consistency",
        "diagnosis-no-validation",
    ):
        strategy = build_optimize_strategy(name)
        snapshots: list[set[str]] = []
        dumped_diagnoses: list[str] = []

        def intercept(
            agent,
            harness,
            harness_store,
            *,
            workspace_builder,
            instructions,
            workdir,
            stage,
            round_ix,
            sample_index,
        ):
            del agent, harness_store, workdir, stage, round_ix
            ws = tmp_path / f"{name}_workspace_{sample_index}"
            ws.mkdir()
            harness_dir = ws / "harness"
            harness_dir.mkdir()
            harness.materialize(harness_dir)
            (ws / ".sample_index").write_text(str(sample_index), encoding="utf-8")
            workspace_builder(ws)
            if name == "diagnosis-no-validation":
                dumped_diagnoses.extend(
                    (path / "diagnosis.md").read_text(encoding="utf-8")
                    for path in sorted((ws / "diagnoses").iterdir())
                )
            snapshots.append(
                {
                    path.relative_to(ws).as_posix()
                    for path in ws.rglob("*")
                    if path.is_file() and path.name != ".sample_index"
                }
            )
            return _optimize_trajectory(sample_index, instructions=instructions), None

        if name == "query-only":
            monkeypatch.setattr("rho.strategies.query_only.optimize_agent_call", intercept)
        elif name == "trajectory":
            monkeypatch.setattr("rho.strategies.trajectory.optimize_agent_call", intercept)
        else:
            monkeypatch.setattr("rho.strategies.diagnose.optimize_agent_call", intercept)

        result = strategy.propose_candidates(
            agent=_strategy_agent(),
            harness=harness,
            tasks_with_trajectories=[
                (task, _stub_trajectories(task.id, harness.id)) for task in tasks
            ],
            harness_store=harness_store,
            traj_store=traj_store,
            workdir=tmp_path / f"workdir_layout_{name}",
            n_samples=1,
            round_ix=0,
        )

        assert len(result.samples) == 1
        assert snapshots == [expectations[name]]
        if name == "diagnosis-no-consistency":
            assert "inconsistency" not in result.samples[0].optimize_trajectory.instructions.lower()
            assert "consistency" not in result.samples[0].optimize_trajectory.instructions.lower()
        if name == "diagnosis-no-validation":
            instructions = result.samples[0].optimize_trajectory.instructions.lower()
            assert "inconsistency" in instructions
            assert "failure mode" not in instructions
            assert "per-trajectory" not in instructions
            assert dumped_diagnoses
            assert all("Per-trajectory analysis" not in text for text in dumped_diagnoses)
            assert all("Failure mode analysis" not in text for text in dumped_diagnoses)


def _strategy_agent() -> FakeAgent:
    def diagnose_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        prompt = (workspace / "task" / "prompt.md").read_text(encoding="utf-8")
        diagnosis = {
            "task_id": workspace.name,
            "severity": 0.8 if "project code name" in prompt else 0.3,
            "trajectory_analyses": [
                {
                    "trajectory": f"trajectory_{ix}",
                    "successful": 0,
                    "quality_analysis": "Did not answer the prompt completely.",
                    "issues": "Missing facts: project code name",
                }
                for ix in range(3)
            ],
            "failure_mode_analysis": "Missing facts: project code name",
            "inconsistency_analysis": "",
            "harness_improvement_direction": "Add missing facts to the harness.",
        }
        return FakeResponse(final_message=json.dumps(diagnosis))

    def optimize_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        content = "strategy smoke candidate\n"
        return FakeResponse(
            final_message="updated harness",
            workspace_edits={"harness/notes.md": content.encode("utf-8")},
        )

    return FakeAgent({"diagnose": diagnose_script, "optimize": optimize_script})


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
        final_message=f"I don't know {task_id} sample {sample_index}",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.01,
        sample_index=sample_index,
    )


def _optimize_trajectory(sample_index: int, *, instructions: str) -> Trajectory:
    return Trajectory(
        id=f"opt_{sample_index}",
        kind="optimize",
        task_id="*",
        harness_id="harness",
        instructions=instructions,
        events=[],
        final_message="snapshot",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.01,
        sample_index=sample_index,
    )
