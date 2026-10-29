from pathlib import Path

import pytest

from rho.datasets.directory import DirectoryTask
from rho.protocols import Diagnosis, TrajectoryAnalysis
from rho.stores.harness import FilesystemHarnessStore
from rho.strategies._common import optimize_agent_call
from rho.strategies.diagnose import OPTIMIZE_INSTRUCTIONS, _dump_diagnosis

pytestmark = pytest.mark.codex


def test_optimize_real_codex(codex_agent_factory, toy_dataset_root, tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    diagnosis = Diagnosis(
        task_id=task.id,
        trajectory_analyses=[
            TrajectoryAnalysis(
                trajectory="trajectory_0",
                successful=0,
                quality_analysis="The trajectory did not answer the task.",
                issues="Missing facts: project code name",
            ),
            TrajectoryAnalysis(
                trajectory="trajectory_1",
                successful=0,
                quality_analysis="The trajectory did not answer the task.",
                issues="Missing facts: project code name",
            ),
            TrajectoryAnalysis(
                trajectory="trajectory_2",
                successful=0,
                quality_analysis="The trajectory did not answer the task.",
                issues="Missing facts: project code name",
            ),
        ],
        failure_mode_analysis="Missing facts: project code name. Agent could not find the project code name because the harness lacks this information.",
        inconsistency_analysis="All trajectories failed in the same way.",
        harness_improvement_direction="Add the missing project code name to the harness.",
    )
    handle = codex_agent_factory()
    _, new_harness = optimize_agent_call(
        handle.agent,
        harness,
        harness_store,
        workspace_builder=lambda ws: _build_diagnosis_workspace(ws, task, diagnosis),
        instructions=OPTIMIZE_INSTRUCTIONS,
        workdir=tmp_path / "workdir",
        stage="round_optimize",
        round_ix=0,
        sample_index=0,
    )
    assert new_harness is not None
    assert new_harness.id != harness.id


def _build_diagnosis_workspace(ws: Path, task: DirectoryTask, diagnosis: Diagnosis) -> None:
    diagnoses_dir = ws / "diagnoses"
    diagnoses_dir.mkdir()
    _dump_diagnosis(diagnoses_dir / "task_0000", task, diagnosis)
