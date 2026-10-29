import json

from rho.loop import run_evolution
from rho.reporting import grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.conftest import make_fake_agent


def test_loop_e2e_three_facts(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("good")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    train = dataset.train
    val = dataset.val
    initial = next(iter(train)).harness

    final, rounds = run_evolution(
        train=train,
        n_rounds=3,
        agent=agent,
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        rounds_dir=tmp_path / "rounds",
        initial=initial,
        strategy=DiagnoseStrategy(),
    )

    initial_summary = summarize(grade_on_split(agent, initial, val, tmp_path / "workdir"))
    final_summary = summarize(grade_on_split(agent, final, val, tmp_path / "workdir"))

    assert len(rounds) == 3
    assert initial_summary["n"] == 3
    assert final_summary["n"] == 3
    assert final_summary["mean_score"] > initial_summary["mean_score"]
    assert [round_result.accepted for round_result in rounds] == [True, True, True]

    round0_input = (tmp_path / "rounds" / "round_0" / "input_harness_id").read_text(encoding="utf-8").strip()
    round1_input = (tmp_path / "rounds" / "round_1" / "input_harness_id").read_text(encoding="utf-8").strip()
    round2_input = (tmp_path / "rounds" / "round_2" / "input_harness_id").read_text(encoding="utf-8").strip()
    assert round0_input == initial.id
    assert round1_input == rounds[0].candidate.id
    assert round2_input == rounds[1].candidate.id

    scores = json.loads((tmp_path / "rounds" / "round_0" / "scores.json").read_text(encoding="utf-8"))
    assert scores
