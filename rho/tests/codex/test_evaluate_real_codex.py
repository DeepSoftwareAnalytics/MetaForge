import pytest

from rho.datasets.directory import DirectoryTask
from rho.orchestrators.evaluate import evaluate
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore

pytestmark = pytest.mark.codex


def _trajectory(
    final_message: str,
    harness_id: str,
    workspace_diff: dict[str, bytes] | None = None,
) -> Trajectory:
    return Trajectory(
        id=f"traj_{harness_id}",
        kind="solve",
        task_id="task_001",
        harness_id=harness_id,
        instructions="MODE: solve",
        events=[],
        final_message=final_message,
        stdout="",
        stderr="",
        workspace_diff=workspace_diff or {},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )


def test_evaluate_real_codex(codex_agent_factory, toy_dataset_root, tmp_path) -> None:
    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    before = _trajectory("I don't know project code name", "h_before")
    after = _trajectory(
        "team project code name is Phoenix",
        "h_after",
        {"task/harness_evidence.txt": b"team project code name is Phoenix\n"},
    )
    handle = codex_agent_factory()
    eval_traj, score = evaluate(
        handle.agent, task, before, after, workdir=tmp_path / "workdir"
    )
    assert eval_traj.exit_code == 0
    assert score.value > 0
