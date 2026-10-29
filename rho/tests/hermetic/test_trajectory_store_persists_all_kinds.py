from rho.loop import run_round
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.conftest import make_fake_agent


def test_trajectory_store_persists_all_kinds(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("good")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    current = harness_store.empty()

    run_round(
        0,
        current,
        list(dataset.train),
        agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
    )

    kinds = {trajectory.kind for trajectory in traj_store._iter_all()}
    assert kinds == {"solve", "evaluate", "optimize", "diagnose"}
    stages = {trajectory.stage for trajectory in traj_store._iter_all()}
    assert {
        "round_solve_before",
        "round_diagnose",
        "round_optimize",
        "round_solve_after",
        "round_evaluate",
    } <= stages
    for path in (tmp_path / "trajectories").iterdir():
        assert (path / "instructions.md").read_text(encoding="utf-8")
        assert (path / "stdout.log").exists()
        assert (path / "stderr.log").exists()
