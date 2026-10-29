from pathlib import Path

from rho.datasets.directory import DirectoryDataset, DirectoryTask
from rho.protocols import Dataset, Harness, HarnessStore, Task, TaskSet, TrajectoryStore
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore


def test_runtime_protocols(toy_dataset_root, tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")

    assert isinstance(task, Task)
    assert isinstance(dataset.train, TaskSet)
    assert isinstance(dataset, Dataset)
    assert isinstance(harness_store, HarnessStore)
    assert isinstance(harness, Harness)
    assert isinstance(traj_store, TrajectoryStore)
