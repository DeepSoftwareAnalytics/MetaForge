from rho.agent.cache import AgentResponseCache, CachingAgent
from rho.datasets.directory import DirectoryDataset
from rho.loop import run_evolution
from rho.reporting import grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy
from tests.helpers import make_fake_agent


def test_loop_e2e_cache_replay_hits_every_agent_call(toy_dataset_root, tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")

    first_summary, first_rounds, first_fake, first_agent = _run_loop(
        toy_dataset_root, tmp_path / "run_a", cache
    )
    second_summary, second_rounds, second_fake, second_agent = _run_loop(
        toy_dataset_root, tmp_path / "run_b", cache
    )

    assert first_agent.hit_count + first_agent.miss_count == 72
    assert len(first_fake.calls) == first_agent.miss_count
    assert first_agent.miss_count > 0
    assert len(second_fake.calls) == 0
    assert second_agent.hit_count == 72
    assert second_agent.miss_count == 0
    assert second_summary == first_summary
    assert [
        (round_result.accepted, round_result.mean_score, round_result.candidate.id)
        for round_result in second_rounds
    ] == [
        (round_result.accepted, round_result.mean_score, round_result.candidate.id)
        for round_result in first_rounds
    ]


def _run_loop(dataset_root, run_dir, cache):
    fake = make_fake_agent("good")
    agent = CachingAgent(fake, cache)
    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    dataset = DirectoryDataset(dataset_root, harness_store=harness_store)
    train = dataset.train
    val = dataset.val
    initial = next(iter(train)).harness

    final, rounds = run_evolution(
        train=train,
        n_rounds=2,
        agent=agent,
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=run_dir / "workdir",
        rounds_dir=run_dir / "rounds",
        initial=initial,
        strategy=DiagnoseStrategy(),
    )
    summary = {
        "initial": summarize(grade_on_split(agent, initial, val, run_dir / "workdir")),
        "final": summarize(grade_on_split(agent, final, val, run_dir / "workdir")),
        "rounds": [
            {
                "accepted": round_result.accepted,
                "mean_score": round_result.mean_score,
                "candidate_harness_id": round_result.candidate.id,
            }
            for round_result in rounds
        ],
    }
    return summary, rounds, fake, agent
