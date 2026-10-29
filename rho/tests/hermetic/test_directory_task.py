from pathlib import Path

from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore


def test_directory_task_materialize_and_grade(toy_dataset_root, tmp_path: Path) -> None:
    from rho.datasets.directory import DirectoryTask

    harness_store = FilesystemHarnessStore(tmp_path / "_hs")
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness_store.empty())
    dest = tmp_path / "task"
    task.materialize(dest)

    assert (dest / "prompt.md").exists()
    assert not (dest / "expected.json").exists()

    trajectory = Trajectory(
        id="traj",
        kind="solve",
        task_id=task.id,
        harness_id="h_empty",
        instructions="MODE: solve",
        events=[],
        final_message="team project code name is Phoenix",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )
    grade = task.grade(trajectory)
    assert grade.passed is True
    assert grade.score == 1.0
