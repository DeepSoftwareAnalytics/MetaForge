from dataclasses import replace

import pytest

from rho.agent.cache import AgentResponseCache, CachingAgent, build_default_agent
from rho.agent.fake import FakeAgent, FakeResponse


def _workspace(root, text: str = "same"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "task").mkdir()
    (root / "harness").mkdir()
    (root / "task" / "prompt.md").write_text(text, encoding="utf-8")
    (root / "harness" / "README.md").write_text("harness\n", encoding="utf-8")
    return root


def _script(final_message: str, edits: dict[str, bytes] | None = None, exit_code: int = 0):
    def run(workspace, instructions, output_schema):
        del workspace, instructions, output_schema
        return FakeResponse(
            final_message=final_message,
            workspace_edits=edits or {"task/answer.txt": final_message.encode("utf-8")},
            events=[{"type": "done", "value": final_message}],
            exit_code=exit_code,
        )

    return run


def _agent(final_message: str = "ok", edits: dict[str, bytes] | None = None, exit_code: int = 0):
    return FakeAgent({"solve": _script(final_message, edits, exit_code)})


def _without_id(traj):
    return replace(traj, id="")


def test_caching_agent_replays_trajectory_and_workspace(tmp_path) -> None:
    fake = _agent("cached")
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))

    first_ws = _workspace(tmp_path / "first")
    first = agent.run(first_ws, "MODE: solve", task_id="t1", harness_id="h1")
    second_ws = _workspace(tmp_path / "second")
    second = agent.run(second_ws, "MODE: solve", task_id="t2", harness_id="h2")

    assert len(fake.calls) == 1
    assert agent.hit_count == 1
    assert agent.miss_count == 1
    assert (second_ws / "task" / "answer.txt").read_text(encoding="utf-8") == "cached"
    assert second.id != first.id
    assert second.task_id == "t2"
    assert second.harness_id == "h2"
    assert _without_id(second) == replace(first, id="", task_id="t2", harness_id="h2")
    assert second.events == first.events


def test_caching_agent_replays_deletions(tmp_path) -> None:
    fake = _agent("deleted", {"task/remove.txt": b""})
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))

    first_ws = _workspace(tmp_path / "first")
    (first_ws / "task" / "remove.txt").write_text("remove me", encoding="utf-8")
    agent.run(first_ws, "MODE: solve")
    second_ws = _workspace(tmp_path / "second")
    (second_ws / "task" / "remove.txt").write_text("remove me", encoding="utf-8")
    agent.run(second_ws, "MODE: solve")

    assert not (second_ws / "task" / "remove.txt").exists()
    assert agent.hit_count == 1


def test_caching_agent_misses_when_workspace_changes(tmp_path) -> None:
    fake = _agent("cached")
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))

    agent.run(_workspace(tmp_path / "first", "same"), "MODE: solve")
    agent.run(_workspace(tmp_path / "second", "different"), "MODE: solve")

    assert len(fake.calls) == 2
    assert agent.hit_count == 0
    assert agent.miss_count == 2


def test_caching_agent_misses_when_env_signature_changes(tmp_path) -> None:
    fake = _agent("cached")
    fake.cache_env_signature = {"agent_class": "FakeAgent", "variant": "a"}
    CachingAgent(fake, AgentResponseCache(tmp_path / "cache")).run(
        _workspace(tmp_path / "first"), "MODE: solve"
    )

    fake.cache_env_signature = {"agent_class": "FakeAgent", "variant": "b"}
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))
    agent.run(_workspace(tmp_path / "second"), "MODE: solve")

    assert len(fake.calls) == 2
    assert agent.hit_count == 0
    assert agent.miss_count == 1


def test_readonly_mode_does_not_store_on_miss_but_can_replay_hits(tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")
    fake = _agent("cached")
    CachingAgent(fake, cache).run(_workspace(tmp_path / "seed"), "MODE: solve")

    readonly_fake = _agent("should not run")
    readonly = CachingAgent(readonly_fake, cache, mode="readonly")
    replayed = readonly.run(_workspace(tmp_path / "hit"), "MODE: solve")

    assert replayed.final_message == "cached"
    assert len(readonly_fake.calls) == 0
    assert readonly.hit_count == 1

    miss_fake = _agent("miss")
    miss_agent = CachingAgent(miss_fake, AgentResponseCache(tmp_path / "miss-cache"), mode="readonly")
    miss_agent.run(_workspace(tmp_path / "miss", "new"), "MODE: solve")
    assert not (tmp_path / "miss-cache").exists()


def test_refresh_mode_overwrites_existing_entry(tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")
    CachingAgent(_agent("old"), cache).run(_workspace(tmp_path / "old"), "MODE: solve")
    CachingAgent(_agent("new"), cache, mode="refresh").run(
        _workspace(tmp_path / "refresh"), "MODE: solve"
    )

    replay = CachingAgent(_agent("unused"), cache)
    traj = replay.run(_workspace(tmp_path / "replay"), "MODE: solve")

    assert traj.final_message == "new"
    assert replay.hit_count == 1


def test_build_default_agent_bypasses_fake_agent_even_when_enabled(tmp_path) -> None:
    fake = _agent("cached")

    assert build_default_agent(fake, mode="on", cache_dir=tmp_path / "cache") is fake
    assert not (tmp_path / "cache").exists()


def test_build_default_agent_off_returns_inner_without_touching_cache(tmp_path) -> None:
    fake = _agent("cached")

    assert build_default_agent(fake) is fake
    assert not (tmp_path / "cache").exists()


def test_build_default_agent_requires_explicit_cache_dir_when_enabled() -> None:
    class NoopAgent:
        def run(self, *args, **kwargs):
            raise AssertionError("should not run")

    with pytest.raises(ValueError, match="cache_dir is required"):
        build_default_agent(NoopAgent(), mode="on")
