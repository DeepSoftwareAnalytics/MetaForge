from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from rho.agent.cache import build_default_agent
from rho.agent.codex import CodexAgent
from rho.datasets.directory import DirectoryTask
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies.dynamic_cheatsheet import (
    CHEATSHEET_INITIAL_CONTENT,
    CURATOR_INSTRUCTIONS,
    DynamicCheatsheetStrategy,
)

pytestmark = pytest.mark.codex


def test_dynamic_cheatsheet_real_codex_one_task_one_sample(
    toy_dataset_root, tmp_path: Path
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    harness = harness_store.empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_001", harness)
    agent = _real_codex_agent(default_timeout_s=480.0)
    feed_trajectories = _curator_feed_trajectories(task.id, harness.id)

    result = DynamicCheatsheetStrategy().propose_candidates(
        agent=agent,
        harness=harness,
        tasks_with_trajectories=[(task, feed_trajectories)],
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        n_samples=1,
        round_ix=0,
    )

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.optimize_trajectory.kind == "optimize"
    assert sample.candidate is not None

    metrics = next(
        e["metrics"]
        for e in sample.optimize_trajectory.events
        if e.get("type") == "dc_cheatsheet.metrics"
    )
    assert metrics["dc_curator_calls"] == 1
    assert metrics["dc_curator_failures"] == 0
    assert metrics["dc_total_input_tokens"] > 0
    assert metrics["dc_cheatsheet_word_count"] > 0
    assert metrics["dc_cheatsheet_version"] == 1

    materialized = tmp_path / "materialized"
    sample.candidate.materialize(materialized)
    cheatsheet = (materialized / "cheatsheet.md").read_text(encoding="utf-8")
    assert cheatsheet != CHEATSHEET_INITIAL_CONTENT
    stripped = cheatsheet.strip()
    assert stripped.startswith("<cheatsheet>") and stripped.endswith("</cheatsheet>")
    assert cheatsheet.count("<memory_item>") >= 1
    assert cheatsheet.count("<memory_item>") == cheatsheet.count("</memory_item>")
    assert "Count:" in cheatsheet

    persisted = list(traj_store.list_for_task(task.id))
    assert [t.id for t in persisted] == [sample.optimize_trajectory.id]
    assert any(e.get("type") == "dc_cheatsheet.metrics" for e in persisted[0].events)

    _save_snapshot_if_requested(
        cheatsheet=cheatsheet,
        trajectory=sample.optimize_trajectory,
        feed_trajectories=feed_trajectories,
    )


def _real_codex_agent(*, default_timeout_s: float):
    binary = shutil.which("codex")
    if binary is None:
        pytest.skip("codex CLI not found on PATH")
    return build_default_agent(
        CodexAgent(
            codex_config_path=_codex_config_for_tests(),
            binary=binary,
            fallback_sandbox="danger-full-access",
            default_timeout_s=default_timeout_s,
        ),
        mode="off",
    )


def _codex_config_for_tests() -> Path:
    override = os.environ.get("RHO_CODEX_CONFIG")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
    repo_config = Path(__file__).resolve().parents[2] / "configs" / "codex.azure-foundry.toml"
    if repo_config.is_file():
        return repo_config
    user_config = Path.home() / ".codex" / "config.toml"
    if user_config.is_file():
        return user_config.resolve()
    pytest.skip("no codex config available for real-codex smoke test")


def _curator_feed_trajectories(task_id: str, harness_id: str) -> list[Trajectory]:
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": (
                    "Used `grep -ri <keyword> harness/` to scan all harness files "
                    "for relevant context before drafting an answer. This "
                    "grep-then-read pattern reliably surfaces the right reference "
                    "file when present."
                ),
            },
        }
    ]
    return [
        Trajectory(
            id=f"solve_{ix}",
            kind="solve",
            task_id=task_id,
            harness_id=harness_id,
            instructions="Solve the task.",
            events=events,
            final_message=(
                "Pattern: when answering factoid questions, first run a recursive "
                "grep over harness/ for keywords drawn from the question, then "
                "read the matching files. This avoids guessing and surfaces the "
                "canonical source if one exists. Concrete one-liner: grep -ril "
                "'<keyword>' harness/ | head -5"
            ),
            stdout="",
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=0.01,
            sample_index=ix,
        )
        for ix in range(3)
    ]


def _save_snapshot_if_requested(
    *,
    cheatsheet: str,
    trajectory: Trajectory,
    feed_trajectories: list[Trajectory],
) -> None:
    if os.environ.get("RHO_DC_SAVE_SNAPSHOT") != "1":
        return

    snapshot_dir = (
        Path(__file__).resolve().parent / "snapshots" / "dc_real_codex"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    (snapshot_dir / "cheatsheet.md").write_text(cheatsheet, encoding="utf-8")
    (snapshot_dir / "curator_instructions.md").write_text(
        CURATOR_INSTRUCTIONS,
        encoding="utf-8",
    )
    with (snapshot_dir / "curator_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in trajectory.events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (snapshot_dir / "curator_final_message.txt").write_text(
        trajectory.final_message,
        encoding="utf-8",
    )
    (snapshot_dir / "feed_trajectories.json").write_text(
        json.dumps(
            [_trajectory_to_json(traj) for traj in feed_trajectories],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot_dir / "metadata.json").write_text(
        json.dumps(
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "codex_model": trajectory.model,
                "codex_config_path": str(_codex_config_for_tests()),
                "rho_git_sha": _git_sha(),
                "prompt_sha256": hashlib.sha256(
                    CURATOR_INSTRUCTIONS.encode("utf-8")
                ).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trajectory_to_json(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "id": trajectory.id,
        "kind": trajectory.kind,
        "task_id": trajectory.task_id,
        "harness_id": trajectory.harness_id,
        "instructions": trajectory.instructions,
        "events": trajectory.events,
        "final_message": trajectory.final_message,
        "stdout": trajectory.stdout,
        "stderr": trajectory.stderr,
        "workspace_diff": {
            rel: content.decode("utf-8", errors="replace")
            for rel, content in sorted(trajectory.workspace_diff.items())
        },
        "workspace_deletions": sorted(trajectory.workspace_deletions),
        "exit_code": trajectory.exit_code,
        "wall_time_s": trajectory.wall_time_s,
        "timed_out": trajectory.timed_out,
        "stage": trajectory.stage,
        "round_ix": trajectory.round_ix,
        "sample_index": trajectory.sample_index,
        "model": trajectory.model,
        "reasoning_effort": trajectory.reasoning_effort,
        "cache_mode": trajectory.cache_mode,
    }


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"
