from rho.orchestrators.solve import solve
from rho.stores.harness import FilesystemHarnessStore


def test_solve_primitive(fake_agent_default_scripts, toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask

    store = FilesystemHarnessStore(tmp_path / "harness")
    harness = store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    trajectory = solve(fake_agent_default_scripts, task, harness, workdir=tmp_path / "workdir")
    assert trajectory.task_id == task.id
    assert trajectory.harness_id == harness.id
    assert "I don't know project code name" in trajectory.final_message
