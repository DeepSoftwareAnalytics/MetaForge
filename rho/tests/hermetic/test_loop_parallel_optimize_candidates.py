import json

from rho.loop import run_round
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.conftest import make_fake_agent


def test_parallel_optimize_dedupes_identical_candidates(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("good")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    tasks = list(dataset.train)

    result = run_round(
        0,
        harness_store.empty(),
        tasks,
        agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
        optimize_samples=3,
    )

    payload = json.loads(
        (tmp_path / "rounds" / "round_0" / "optimize_candidates.json").read_text(encoding="utf-8")
    )

    assert len(payload["samples"]) == 3
    assert len(payload["unique_candidates"]) == 1
    assert payload["winner_candidate_harness_id"] == result.candidate.id
    assert payload["unique_candidates"][0]["sample_indices"] == [0, 1, 2]
    assert len(payload["unique_candidates"][0]["solve_after_traj_ids"]) == len(tasks)
    assert len(payload["unique_candidates"][0]["eval_traj_ids"]) == len(tasks)

    optimize_count = sum(1 for trajectory in traj_store._iter_all() if trajectory.kind == "optimize")
    evaluate_count = sum(1 for trajectory in traj_store._iter_all() if trajectory.kind == "evaluate")
    assert optimize_count == 3
    assert evaluate_count == len(tasks)


def test_parallel_optimize_breaks_ties_by_lowest_sample_index(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("sampled")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)

    result = run_round(
        0,
        harness_store.empty(),
        list(dataset.train),
        agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
        optimize_samples=3,
    )

    payload = json.loads(
        (tmp_path / "rounds" / "round_0" / "optimize_candidates.json").read_text(encoding="utf-8")
    )

    assert result.accepted is True
    assert result.winner_sample_index == 0
    assert payload["winner_sample_index"] == 0
    assert len(payload["unique_candidates"]) == 3
    assert payload["unique_candidates"][0]["mean_score"] == payload["unique_candidates"][1]["mean_score"]
    assert payload["unique_candidates"][0]["accepted"] is True
    assert payload["unique_candidates"][1]["accepted"] is False
