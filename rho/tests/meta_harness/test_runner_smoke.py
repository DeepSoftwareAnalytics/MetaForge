from __future__ import annotations

from pathlib import Path

from rho.datasets.directory import DirectoryTask
from rho.meta_harness.runner import _evaluate_candidate
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from tests.helpers import FACTS, make_meta_harness_fake_agent

ALL_FACTS = "\n".join(FACTS.values()) + "\n"


def test_evaluate_candidate_grades_search_set(toy_dataset_root: Path, tmp_path: Path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")

    src = tmp_path / "good_harness"
    src.mkdir()
    (src / "notes.md").write_text(ALL_FACTS, encoding="utf-8")
    harness = harness_store.capture(src)

    tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", harness),
        DirectoryTask(toy_dataset_root / "train" / "task_002", harness),
    ]
    record = _evaluate_candidate(
        agent=make_meta_harness_fake_agent(),
        harness=harness,
        search_tasks=tasks,
        workdir=tmp_path / "workdir",
        traj_store=traj_store,
        search_trials=1,
        solve_workers=2,
        iteration=0,
        name="seed",
        hypothesis="built-in",
        parent=None,
    )
    assert record.harness_id == harness.id
    assert record.iteration == 0
    assert set(record.per_task) == {"task_001", "task_002"}
    assert record.mean_score == 1.0  # harness contains every fact -> toy tasks pass
    assert record.pass_rate == 1.0
    assert len(record.solve_traj_ids) == 2  # 1 trial x 2 tasks


def test_propose_captures_candidates_from_manifest(tmp_path: Path) -> None:
    from rho.meta_harness.runner import _propose

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    ws = tmp_path / "ws"
    (ws / "proposed").mkdir(parents=True)
    (ws / "history").mkdir()

    traj, captured = _propose(
        agent=make_meta_harness_fake_agent(),
        ws=ws,
        instructions="propose now",
        harness_store=harness_store,
    )
    assert traj.kind == "optimize"
    assert len(captured) == 1
    harness, entry = captured[0]
    assert entry["name"] == "all_facts"
    assert entry["parent"] is None
    materialized = tmp_path / "check"
    harness.materialize(materialized)
    assert (materialized / "notes.md").read_text(encoding="utf-8") == ALL_FACTS


def test_propose_returns_empty_when_manifest_missing(tmp_path: Path) -> None:
    from rho.agent.fake import FakeResponse
    from rho.meta_harness.runner import _propose

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    ws = tmp_path / "ws"
    (ws / "proposed").mkdir(parents=True)

    agent = make_meta_harness_fake_agent()
    agent.scripts["optimize"] = lambda w, i, o: FakeResponse(final_message="nothing")
    traj, captured = _propose(
        agent=agent, ws=ws, instructions="propose", harness_store=harness_store
    )
    assert captured == []


def test_run_meta_harness_end_to_end(toy_dataset_root: Path, tmp_path: Path) -> None:
    from rho.meta_harness import run_meta_harness
    from rho.meta_harness.store import load_records

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    seed = harness_store.empty()

    search_tasks = [
        DirectoryTask(toy_dataset_root / "train" / "task_001", seed),
        DirectoryTask(toy_dataset_root / "train" / "task_002", seed),
    ]
    test_tasks = [DirectoryTask(toy_dataset_root / "val" / "task_v01", seed)]
    run_meta_dir = tmp_path / "run" / "meta_harness"

    result = run_meta_harness(
        agent=make_meta_harness_fake_agent(),
        search_tasks=search_tasks,
        test_tasks=test_tasks,
        seed_harness=seed,
        harness_store=harness_store,
        traj_store=traj_store,
        run_meta_dir=run_meta_dir,
        workdir=tmp_path / "run" / "workdir",
        iterations=2,
        candidates_per_iter=1,
        search_trials=1,
        solve_workers=2,
    )

    # iteration 0 (seed) + 1 candidate per iteration x 2 iterations = 3 records.
    records = load_records(run_meta_dir / "summary.jsonl")
    assert [r.iteration for r in records] == [0, 1, 2]
    assert records[0].name == "seed"
    assert records[0].mean_score == 0.0  # empty seed harness -> toy tasks fail

    # The proposer's harness contains every fact -> best candidate scores 1.0.
    assert result.best.mean_score == 1.0
    assert result.best.harness_id != seed.id

    # Final test eval ran on the best harness.
    assert len(result.test_grades) == 1
    assert result.test_grades[0].grade.passed is True

    # Proposer post-mortem reports persisted across iterations.
    assert (run_meta_dir / "reports" / "iter_0.md").exists()
    # Per-iteration workspaces are cleaned up.
    assert not list((tmp_path / "run" / "workdir").glob("mh_iter*"))
