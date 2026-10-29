from rho.agent.cache import AgentResponseCache, CachingAgent
from rho.datasets.directory import DirectoryTask
from rho.orchestrators.solve import solve
from rho.stores.harness import FilesystemHarnessStore
from tests.helpers import make_fake_agent


def test_solve_with_cache_replays_second_identical_run(toy_dataset_root, tmp_path) -> None:
    fake = make_fake_agent("good")
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))
    store = FilesystemHarnessStore(tmp_path / "harness")
    harness = store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)

    first = solve(agent, task, harness, workdir=tmp_path / "workdir")
    second = solve(agent, task, harness, workdir=tmp_path / "workdir")

    assert len(fake.calls) == 1
    assert agent.hit_count == 1
    assert second.workspace_diff == first.workspace_diff
    assert second.final_message == first.final_message
