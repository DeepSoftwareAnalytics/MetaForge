"""Assert LocomoTask satisfies the Task protocol and its materialize/grade."""

from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.locomo import LocomoDataset, LocomoTask
from rho.protocols import Task, Trajectory
from rho.stores.harness import FilesystemHarnessStore

LOCOMO_PATH = Path(__file__).parents[2] / "data" / "locomo10.json"


@pytest.fixture
def dataset(tmp_path: Path) -> LocomoDataset:
    hs = FilesystemHarnessStore(tmp_path / "harness")
    return LocomoDataset(LOCOMO_PATH, harness_store=hs, seed=0, max_per_split=10)


def test_locomo_task_satisfies_task_protocol(dataset: LocomoDataset) -> None:
    for task in dataset.train:
        assert isinstance(task, Task)
        assert isinstance(task, LocomoTask)


def test_all_tasks_share_same_harness(dataset: LocomoDataset) -> None:
    all_tasks = list(dataset.train) + list(dataset.val) + list(dataset.test)
    ids = {t.harness.id for t in all_tasks}
    assert len(ids) == 1


def test_materialize_writes_prompt_md(dataset: LocomoDataset, tmp_path: Path) -> None:
    task = next(iter(dataset.train))
    dest = tmp_path / "workspace"
    task.materialize(dest)
    prompt_path = dest / "prompt.md"
    assert prompt_path.exists()
    content = prompt_path.read_text(encoding="utf-8")
    assert "ANSWER:" in content
    # The prompt should reference the conversation id so the agent knows
    # where to look in its harness.
    assert task._conv_id in content  # type: ignore[attr-defined]
    assert task._question in content  # type: ignore[attr-defined]


def test_task_id_format(dataset: LocomoDataset) -> None:
    for task in dataset.train:
        assert task.id.startswith("conv-")
        assert "/qa_" in task.id
        conv_id, qa_part = task.id.split("/", 1)
        assert qa_part.startswith("qa_")
        assert len(qa_part) == len("qa_0000")


def test_grade_produces_float_score(dataset: LocomoDataset) -> None:
    task = next(iter(dataset.train))
    # Feed the gold answer as the prediction — should score high.
    gold = task._gold  # type: ignore[attr-defined]
    traj = Trajectory(
        id="traj_test",
        kind="solve",
        task_id=task.id,
        harness_id=task.harness.id,
        instructions="MODE: solve",
        events=[],
        final_message=f"reasoning ...\n\nANSWER: {gold}",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )
    grade = task.grade(traj)
    assert isinstance(grade.score, float)
    assert 0.0 <= grade.score <= 1.0
    assert grade.details["prediction"] == gold
