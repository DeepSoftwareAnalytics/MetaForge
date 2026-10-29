from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rho.protocols import Trajectory

_RUNTIME_SCRATCH_NAMES = {
    ".rho",
    ".sample_index",
    ".codex",
    ".gaia2",
    ".gaia2_state",
}
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens")


def is_runtime_scratch(rel: str | Path) -> bool:
    path = Path(rel)
    return any(part in _RUNTIME_SCRATCH_NAMES for part in path.parts)


def resolve_agent_metadata(agent: object | None) -> tuple[str | None, str | None, str]:
    cache_mode = "off"
    current = agent
    while current is not None:
        inner = getattr(current, "inner", None)
        mode = getattr(current, "mode", None)
        if isinstance(mode, str):
            cache_mode = mode
        if inner is None:
            break
        current = inner
    model = getattr(current, "model", None)
    reasoning_effort = getattr(current, "reasoning_effort", None)
    return (
        model if isinstance(model, str) or model is None else str(model),
        reasoning_effort
        if isinstance(reasoning_effort, str) or reasoning_effort is None
        else str(reasoning_effort),
        cache_mode,
    )


def annotate_trajectory(
    trajectory: Trajectory,
    *,
    agent: object | None = None,
    stage: str | None = None,
    round_ix: int | None = None,
    sample_index: int | None = None,
) -> Trajectory:
    model = trajectory.model
    reasoning_effort = trajectory.reasoning_effort
    cache_mode = trajectory.cache_mode
    if agent is not None:
        agent_model, agent_reasoning_effort, agent_cache_mode = resolve_agent_metadata(
            agent
        )
        if model is None:
            model = agent_model
        if reasoning_effort is None:
            reasoning_effort = agent_reasoning_effort
        if cache_mode is None:
            cache_mode = agent_cache_mode
    return dataclasses.replace(
        trajectory,
        stage=stage if stage is not None else trajectory.stage,
        round_ix=round_ix if round_ix is not None else trajectory.round_ix,
        sample_index=sample_index if sample_index is not None else trajectory.sample_index,
        model=model,
        reasoning_effort=reasoning_effort,
        cache_mode=cache_mode,
    )


def extract_usage(events: list[dict[str, Any]]) -> dict[str, int] | None:
    totals = Counter()
    saw_usage = False
    for event in events:
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in _USAGE_KEYS:
            value = usage.get(key)
            if value is None:
                continue
            totals[key] += int(value)
            saw_usage = True
    if not saw_usage:
        return None
    return {key: int(totals[key]) for key in _USAGE_KEYS}


def usage_summary(trajectories: list[Trajectory]) -> dict[str, Any]:
    overall = Counter()
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    with_usage = 0
    for trajectory in trajectories:
        usage = extract_usage(trajectory.events)
        stage = trajectory.stage or "(none)"
        kind_bucket = by_kind[trajectory.kind]
        kind_bucket["trajectory_count"] += 1
        stage_bucket = by_stage[stage]
        stage_bucket["trajectory_count"] += 1
        if usage is None:
            continue
        with_usage += 1
        kind_bucket["with_usage_count"] += 1
        stage_bucket["with_usage_count"] += 1
        overall["with_usage_count"] += 1
        for key, value in usage.items():
            overall[key] += value
            kind_bucket[key] += value
            stage_bucket[key] += value
    overall["trajectory_count"] = len(trajectories)
    return {
        "overall": {key: int(value) for key, value in overall.items()},
        "by_kind": {
            kind: {key: int(value) for key, value in counter.items()}
            for kind, counter in sorted(by_kind.items())
        },
        "by_stage": {
            stage: {key: int(value) for key, value in counter.items()}
            for stage, counter in sorted(by_stage.items())
        },
        "with_usage_count": with_usage,
    }
