from pathlib import Path

import pytest

from rho.datasets.directory import DirectoryTask
from rho.orchestrators.solve import solve
from rho.stores.harness import FilesystemHarnessStore

pytestmark = pytest.mark.codex


def test_solve_real_codex(codex_agent_factory, toy_dataset_root, tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    materialized = tmp_path / "src_harness"
    harness.materialize(materialized)
    (materialized / "notes.md").write_text(
        "team project code name is Phoenix\n", encoding="utf-8"
    )
    harness = harness_store.capture(materialized)
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness_store.empty())
    handle = codex_agent_factory()
    trajectory = solve(handle.agent, task, harness, workdir=tmp_path / "workdir")
    assert trajectory.exit_code == 0
    assert task.grade(trajectory).passed is True
