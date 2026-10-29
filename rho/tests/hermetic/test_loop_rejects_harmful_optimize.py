from rho.loop import run_evolution
from rho.reporting import grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.conftest import make_fake_agent


def test_loop_rejects_harmful_optimize(toy_dataset_root, tmp_path) -> None:
    """Drive run_evolution end-to-end with a harmful optimizer and verify
    that the loop driver's accept/reject propagation is correct:
    round 1 must NOT inherit the poisoned candidate from round 0."""
    from rho.datasets.directory import DirectoryDataset

    agent = make_fake_agent("harmful")
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    train = dataset.train
    val = dataset.val
    initial = next(iter(train)).harness

    final, rounds = run_evolution(
        train=train,
        n_rounds=2,
        agent=agent,
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        rounds_dir=tmp_path / "rounds",
        initial=initial,
        strategy=DiagnoseStrategy(),
    )

    # Both rounds must be rejected — harmful optimizer lowers the score.
    assert rounds[0].accepted is False
    assert rounds[0].mean_score < 0
    assert rounds[1].accepted is False

    # The load-bearing assertion: round 1's input harness is still initial,
    # not the poisoned candidate from round 0. This only holds if run_evolution
    # correctly refuses to propagate rejected candidates.
    round1_input = (tmp_path / "rounds" / "round_1" / "input_harness_id").read_text(
        encoding="utf-8"
    ).strip()
    assert round1_input == initial.id

    # And the final harness returned by run_evolution must still be initial.
    assert final.id == initial.id

    # Val grades against the final harness are 0 — loop correctly did nothing
    # rather than actively poisoning the harness.
    final_summary = summarize(grade_on_split(agent, final, val, tmp_path / "workdir"))
    assert final_summary["mean_score"] == 0.0
