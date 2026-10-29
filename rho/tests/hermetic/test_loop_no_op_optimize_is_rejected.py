import json

from rho.loop import run_round
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.helpers import FACTS
from tests.conftest import make_fake_agent


def test_loop_no_op_optimize_is_rejected(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("noop")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    current = harness_store.empty()
    tasks = list(dataset.train)

    result = run_round(
        0,
        current,
        tasks,
        agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
    )

    assert result.candidate.id == current.id
    assert result.mean_score == 0
    assert result.accepted is False


def test_loop_passes_low_severity_clean_diagnoses_to_optimize(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("noop")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    current_src = tmp_path / "initial_harness"
    current_src.mkdir()
    (current_src / "notes.md").write_text(
        "\n".join(FACTS.values()) + "\n",
        encoding="utf-8",
    )
    current = harness_store.capture(current_src)

    result = run_round(
        0,
        current,
        list(dataset.train),
        agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        tmp_path / "rounds" / "round_0",
        strategy=DiagnoseStrategy(),
        optimize_samples=3,
    )

    diagnoses = json.loads(
        (tmp_path / "rounds" / "round_0" / "diagnoses.json").read_text(encoding="utf-8")
    )
    optimize_count = sum(1 for trajectory in traj_store._iter_all() if trajectory.kind == "optimize")

    assert diagnoses
    assert all(diagnosis["severity"] == 0.0 for diagnosis in diagnoses)
    assert optimize_count == 3
    assert result.candidate.id == current.id
    assert result.mean_score == 0
    assert result.accepted is False
