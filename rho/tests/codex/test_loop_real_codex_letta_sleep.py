"""Full one-round loop integration with real codex using LettaSleepStrategy.

Mirrors tests/codex/test_loop_one_round_real_codex.py (DiagnoseStrategy) so the
two are directly comparable: same task, same fixture, same cache-determinism
assertion. Adds Letta-specific behavioral assertions to verify the baseline
actually exercises the Letta-faithful mechanism (memory-tool calls, snapshot
delivery, protocol-violation metrics) rather than collapsing to a degenerate
no-op or to free-form file editing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.directory import DirectoryTask
from rho.loop import run_round
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import LettaSleepStrategy

pytestmark = pytest.mark.codex

MEMORY_TOOL_SCRIPTS = (
    "memory_replace.py",
    "memory_insert.py",
    "memory_rethink.py",
    "memory_finish_edits.py",
)


def test_loop_one_round_real_codex_letta_sleep(
    codex_agent_factory, toy_dataset_root, tmp_path: Path
) -> None:
    _task_hs = FilesystemHarnessStore(tmp_path / "_task_hs")
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", _task_hs.empty())

    first_result, first_kinds, first_counts, first_optimize_trajs = _run_one_round(
        codex_agent_factory, task, tmp_path / "run_a"
    )

    # Phase 1 + Phase 2 ran (Phase 3/4 always run regardless of strategy).
    assert {"solve", "optimize"} <= first_kinds

    # LettaSleepStrategy encodes (sample_index, task_ix) into stage; verify it
    # reached the trajectory store via traj_store.put (the strategy persists
    # intermediates itself; loop.py only stores the surfaced sample.optimize_trajectory).
    letta_optimize_trajs = [
        traj
        for traj in first_optimize_trajs
        if traj.stage and traj.stage.startswith("round_optimize:letta_")
    ]
    assert letta_optimize_trajs, (
        "no optimize trajectories with stage prefix 'round_optimize:letta_' — "
        "LettaSleepStrategy is not persisting per-task intermediates correctly."
    )

    # The Letta-faithful mechanism requires that codex actually use the 4
    # narrow memory tools rather than falling back to apply_patch / sed / etc.
    # If none of the 4 scripts is invoked across all letta optimize trajectories,
    # the baseline collapses to "ignore the prompt and edit freely" — which would
    # silently make the comparison against `trajectory` strategy meaningless.
    invoked = _memory_tools_invoked(letta_optimize_trajs)
    assert invoked, (
        "None of the 4 letta memory tools were invoked in any optimize "
        f"trajectory. Expected at least one of {MEMORY_TOOL_SCRIPTS} to appear "
        "in command_execution events. The strategy is not exercising the "
        "Letta-faithful tool surface."
    )

    # Protocol-violation counters get appended as a final letta_sleep.metrics event.
    metrics_events = [
        event
        for traj in letta_optimize_trajs
        for event in traj.events
        if event.get("type") == "letta_sleep.metrics"
    ]
    assert metrics_events, (
        "no letta_sleep.metrics events found in optimize trajectories"
    )
    # Sanity-check the metrics shape: payload is nested under "metrics" with
    # the violation counters described in spec §11.4 (names defined in
    # src/rho/strategies/letta_sleep.py::_letta_metrics).
    payload = metrics_events[0].get("metrics") or {}
    expected_counters = {
        "letta_apply_patch_memory_touch_count",
        "letta_file_write_outside_memory_count",
        "letta_missing_finish_edits_count",
        "letta_finish_violation_count",
        "letta_oversize_block_count",
    }
    missing = expected_counters - payload.keys()
    assert not missing, (
        f"letta_sleep.metrics event missing expected counters {missing}; "
        f"got keys {sorted(payload.keys())}"
    )

    # Cache must have actually run real codex on the first pass.
    first_total = first_counts[0] + first_counts[1]
    assert first_counts[1] > 0, "first run had no cache misses (nothing executed?)"
    assert first_total > 0

    # Second run: full cache hit, deterministic. This confirms env-vars
    # (LETTA_MEMORY_ROOT / LETTA_SCRIPTS) are part of the cache key — without
    # that, materialize-into-fresh-tempdir would change paths and cache would miss.
    second_result, _, second_counts, _ = _run_one_round(
        codex_agent_factory, task, tmp_path / "run_b"
    )
    assert second_result.accepted == first_result.accepted
    assert second_result.mean_score == first_result.mean_score
    assert second_result.candidate.id == first_result.candidate.id
    assert second_counts == (first_total, 0), (
        f"second run was not fully cached: hits={second_counts[0]} "
        f"misses={second_counts[1]} (first run total={first_total})"
    )


def _run_one_round(codex_agent_factory, task, root: Path):
    harness_store = FilesystemHarnessStore(root / "harness")
    traj_store = FilesystemTrajectoryStore(root / "trajectories")
    current = harness_store.empty()
    handle = codex_agent_factory(
        cache_mode="on",
        cache_dir=root.parent / "agent-cache",
    )
    assert handle.caching_agent is not None

    result = run_round(
        0,
        current,
        [task],
        handle.agent,
        harness_store,
        traj_store,
        root / "workdir",
        root / "rounds" / "round_0",
        strategy=LettaSleepStrategy(),
        # n_samples=1 keeps the test cost down (~5 codex calls instead of ~9).
        # The parallel-sample mechanics are covered by the strategy-level smoke
        # in tests/strategies/test_letta_sleep_smoke.py.
        optimize_samples=1,
    )

    kinds = {trajectory.kind for trajectory in traj_store._iter_all()}
    optimize_trajs = [
        traj for traj in traj_store._iter_all() if traj.kind == "optimize"
    ]
    return (
        result,
        kinds,
        (handle.caching_agent.hit_count, handle.caching_agent.miss_count),
        optimize_trajs,
    )


def _memory_tools_invoked(trajectories) -> set[str]:
    """Scan events.jsonl across all trajectories for shell-execution events
    whose command string mentions any of the 4 letta memory scripts."""
    invoked: set[str] = set()
    for traj in trajectories:
        for event in traj.events:
            if event.get("type") not in ("item.started", "item.completed"):
                continue
            item = event.get("item") or {}
            if item.get("type") != "command_execution":
                continue
            command = item.get("command") or ""
            for script in MEMORY_TOOL_SCRIPTS:
                if script in command:
                    invoked.add(script)
    return invoked
