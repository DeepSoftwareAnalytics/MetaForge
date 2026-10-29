from __future__ import annotations

import json
from pathlib import Path

from rho.meta_harness.history import build_history_dir
from rho.meta_harness.store import CandidateRecord, append_record, write_frontier
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore


def _trajectory(traj_id: str, task_id: str, harness_id: str) -> Trajectory:
    return Trajectory(
        id=traj_id,
        kind="solve",
        task_id=task_id,
        harness_id=harness_id,
        instructions="solve it",
        events=[{"type": "test_event"}],
        final_message="done",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )


def test_build_history_dir_assembles_code_scores_and_traces(tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")

    src = tmp_path / "src_harness"
    src.mkdir()
    (src / "notes.md").write_text("a useful note\n", encoding="utf-8")
    harness = harness_store.capture(src)

    traj = _trajectory("traj_1", "task_a", harness.id)
    traj_store.put(traj)

    run_meta_dir = tmp_path / "meta_harness"
    summary = run_meta_dir / "summary.jsonl"
    frontier = run_meta_dir / "frontier.json"
    reports = run_meta_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "iter_0.md").write_text("seed report\n", encoding="utf-8")

    record = CandidateRecord(
        iteration=0,
        harness_id=harness.id,
        name="seed",
        hypothesis="built-in",
        parent=None,
        per_task={"task_a": 1.0},
        mean_score=1.0,
        pass_rate=1.0,
        solve_traj_ids=["traj_1"],
    )
    append_record(summary, record)
    write_frontier(frontier, [record])

    dest = tmp_path / "history"
    build_history_dir(
        dest=dest,
        summary_path=summary,
        frontier_path=frontier,
        reports_dir=reports,
        harness_store=harness_store,
        traj_store=traj_store,
    )

    assert (dest / "summary.jsonl").read_text(encoding="utf-8").strip() != ""
    assert json.loads((dest / "frontier.json").read_text(encoding="utf-8"))["best"]
    assert (dest / "candidates" / harness.id / "notes.md").read_text(
        encoding="utf-8"
    ) == "a useful note\n"
    assert (dest / "traces" / harness.id / "task_a" / "traj_1" / "events.jsonl").exists()
    assert (dest / "reports" / "iter_0.md").read_text(encoding="utf-8") == "seed report\n"


def test_build_history_dir_empty_state(tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dest = tmp_path / "history"
    build_history_dir(
        dest=dest,
        summary_path=tmp_path / "absent.jsonl",
        frontier_path=tmp_path / "absent.json",
        reports_dir=tmp_path / "absent_reports",
        harness_store=harness_store,
        traj_store=traj_store,
    )
    assert (dest / "summary.jsonl").read_text(encoding="utf-8") == ""
    assert (dest / "frontier.json").read_text(encoding="utf-8") == "{}"
    assert (dest / "candidates").is_dir()
    assert (dest / "traces").is_dir()
