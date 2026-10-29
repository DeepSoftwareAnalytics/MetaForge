from pathlib import Path

from rho.agent.cache import AgentResponseCache, CachingAgent
from rho.agent.fake import FakeAgent, FakeResponse
from rho.protocols import Trajectory


def _workspace(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "task").mkdir()
    (root / "task" / "prompt.md").write_text("same", encoding="utf-8")
    return root


def _fake(final_message: str, *, edits: dict[str, bytes] | None = None, exit_code: int = 0):
    def solve(workspace, instructions, output_schema):
        del workspace, instructions, output_schema
        return FakeResponse(
            final_message=final_message,
            workspace_edits=edits or {"task/out.txt": final_message.encode("utf-8")},
            events=[{"type": "answer", "message": final_message}],
            exit_code=exit_code,
        )

    return FakeAgent({"solve": solve})


def _trajectory(final_message: str = "stored") -> Trajectory:
    return Trajectory(
        id="traj_original",
        kind="solve",
        task_id="task",
        harness_id="harness",
        instructions="MODE: solve",
        events=[{"type": "stored"}],
        final_message=final_message,
        stdout="out",
        stderr="err",
        workspace_diff={"task/out.txt": final_message.encode("utf-8")},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.23,
    )


def test_non_zero_exit_is_cached_and_replayed(tmp_path) -> None:
    fake = _fake("failed", edits={"task/fail.txt": b"failed"}, exit_code=2)
    agent = CachingAgent(fake, AgentResponseCache(tmp_path / "cache"))

    first = agent.run(_workspace(tmp_path / "first"), "MODE: solve")
    second = agent.run(_workspace(tmp_path / "second"), "MODE: solve")

    assert len(fake.calls) == 1
    assert first.exit_code == 2
    assert second.exit_code == 2
    assert second.events == first.events


def test_large_file_hash_sentinel_falls_back_without_rewriting_entry(tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")
    fake = _fake("hash", edits={"task/hash.txt": b"HASH:deadbeef"})
    agent = CachingAgent(fake, cache)

    agent.run(_workspace(tmp_path / "first"), "MODE: solve")
    manifest = next((tmp_path / "cache").rglob("manifest.json"))
    before_mtime = manifest.stat().st_mtime_ns
    agent.run(_workspace(tmp_path / "second"), "MODE: solve")

    assert len(fake.calls) == 2
    assert agent.hit_count == 0
    assert agent.miss_count == 2
    assert manifest.stat().st_mtime_ns == before_mtime


def test_lookup_ignores_abandoned_temp_and_stale_siblings(tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")
    key = "a" * 64
    cache.store(key, {"format_version": "1"}, _trajectory())
    shard = tmp_path / "cache" / "v1" / "aa"
    (shard / ".cache_tmp_abandoned").mkdir()
    (shard / ".cache_stale_abandoned").mkdir()

    hit = cache.lookup(key)

    assert hit is not None
    assert hit.final_message == "stored"


def test_store_refreshes_existing_entry_and_cleans_stale(tmp_path) -> None:
    cache = AgentResponseCache(tmp_path / "cache")
    key = "b" * 64
    cache.store(key, {"format_version": "1"}, _trajectory("old"))
    cache.store(key, {"format_version": "1"}, _trajectory("new"))

    hit = cache.lookup(key)
    shard = tmp_path / "cache" / "v1" / "bb"

    assert hit is not None
    assert hit.final_message == "new"
    assert not any(path.name.startswith(".cache_stale_") for path in shard.iterdir())
