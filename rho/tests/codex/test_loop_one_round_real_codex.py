from pathlib import Path

import pytest

from rho.datasets.directory import DirectoryTask
from rho.loop import run_round
from rho.reporting import grade_on_split
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy

pytestmark = pytest.mark.codex


def test_loop_one_round_real_codex(
    codex_agent_factory, toy_dataset_root, tmp_path: Path
) -> None:
    _task_hs = FilesystemHarnessStore(tmp_path / "_task_hs")
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", _task_hs.empty())

    first_result, first_kinds, first_counts = _run_one_round_with_optional_grade(
        codex_agent_factory, task, tmp_path / "run_a"
    )
    assert (tmp_path / "run_a" / "rounds" / "round_0" / "scores.json").exists() or first_result.accepted is False
    assert {"solve", "diagnose"} <= first_kinds
    first_total = first_counts[0] + first_counts[1]
    assert first_counts[1] > 0
    assert first_total > 0

    second_result, second_kinds, second_counts = _run_one_round_with_optional_grade(
        codex_agent_factory, task, tmp_path / "run_b"
    )
    assert second_result.accepted == first_result.accepted
    assert second_result.mean_score == first_result.mean_score
    assert second_result.candidate.id == first_result.candidate.id
    assert second_kinds == first_kinds
    assert second_counts == (first_total, 0)


def _run_one_round_with_optional_grade(codex_agent_factory, task, root: Path):
    harness_store = FilesystemHarnessStore(root / "harness")
    traj_store = FilesystemTrajectoryStore(root / "trajectories")
    current = harness_store.empty()
    handle = codex_agent_factory(
        cache_mode="on",
        cache_dir=root.parent / "agent-cache",
    )
    assert handle.caching_agent is not None

    result = run_round(
        0,
        current,
        [task],
        handle.agent,
        harness_store,
        traj_store,
        root / "workdir",
        root / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
    )

    if result.accepted:
        grade = grade_on_split(handle.agent, result.candidate, [task], root / "workdir")[0]
        assert grade.grade.passed or (root / "rounds" / "round_0" / "scores.json").exists()

    kinds = {trajectory.kind for trajectory in traj_store._iter_all()}
    return result, kinds, (handle.caching_agent.hit_count, handle.caching_agent.miss_count)
