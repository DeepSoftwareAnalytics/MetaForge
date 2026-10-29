from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rho.protocols import Grade, Harness, Trajectory
from rho.selection import DifficultySelector, TaskJudge
from rho.selection.difficulty_selector import JudgeResult, _parse_judge_output
from rho.selection.llm_client import FakeLLMClient
from tests.hermetic._traj_helper import trajectories_for


@dataclass(frozen=True)
class _QueryTask:
    _id: str
    _query: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        raise NotImplementedError

    def materialize(self, dest: Path) -> None:
        raise NotImplementedError

    def query(self) -> str:
        return self._query

    def grade(self, trajectory: Trajectory, *, artifacts_dir: Path | None = None) -> Grade:
        raise NotImplementedError


def _fingerprint(label: str) -> str:
    return (
        f"{label} failure mode centers on partial propagation across several layers, "
        "where one branch adopts a revised contract but a neighboring branch keeps "
        "older ordering assumptions and only breaks under a narrow boundary case. "
        "The work is moderately hard because invariants are scattered across helper "
        "logic, precedence rules, and state reconciliation paths, so the change "
        "requires contextual tracing rather than a purely local edit. Technical scope "
        "is multi-file bug fixing with contract alignment, propagation auditing, and "
        "global reasoning about how a small update preserves behavior."
    )


def _len_judge(prompt: str, model: str) -> str:
    del model
    score = min(10.0, len(prompt) / 400.0)
    label = f"shape_{len(prompt)}"
    return json.dumps(
        {
            "difficulty": score,
            "abstract_fingerprint": _fingerprint(label),
        }
    )


def test_difficulty_selector_ranks_by_score(tmp_path: Path) -> None:
    pool = [
        _QueryTask(_id="short", _query="x"),
        _QueryTask(_id="medium", _query="x" * 200),
        _QueryTask(_id="long", _query="x" * 1000),
    ]
    judge = TaskJudge(
        client=FakeLLMClient(_len_judge),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([t.id for t in pool]),
    )
    picks = DifficultySelector(judge=judge).select(pool, k=2)
    assert [task.id for task in picks] == ["long", "medium"]


def test_difficulty_selector_k_none_returns_all_ranked(tmp_path: Path) -> None:
    pool = [
        _QueryTask(_id="a", _query="x" * 500),
        _QueryTask(_id="b", _query="x" * 50),
        _QueryTask(_id="c", _query="x" * 1000),
    ]
    judge = TaskJudge(
        client=FakeLLMClient(_len_judge),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([t.id for t in pool]),
    )
    picks = DifficultySelector(judge=judge).select(pool, k=None)
    assert [task.id for task in picks] == ["c", "a", "b"]


def test_task_judge_caches_per_task_and_returns_results(tmp_path: Path) -> None:
    client = FakeLLMClient(_len_judge)
    pool = [_QueryTask(_id=f"t{i}", _query="x" * (i + 1)) for i in range(4)]
    judge = TaskJudge(
        client=client,
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([t.id for t in pool]),
    )
    selector = DifficultySelector(judge=judge)

    selector.select(pool, k=2)
    first = len(client.calls)
    selector.select(pool, k=3)

    assert len(client.calls) == first
    results = selector.results()
    assert set(results) == {task.id for task in pool}
    assert all(isinstance(result, JudgeResult) for result in results.values())


def test_task_judge_persists_call_json(tmp_path: Path) -> None:
    task = _QueryTask(_id="t0", _query="q0" * 100)
    judge = TaskJudge(
        client=FakeLLMClient(_len_judge),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([task.id]),
    )

    result = judge.judge(task)

    record = json.loads((tmp_path / "calls" / "t0.json").read_text(encoding="utf-8"))
    assert result == JudgeResult(
        task_id="t0",
        difficulty=record["parsed_difficulty"],
        fingerprint=record["parsed_fingerprint"],
        digest_token_estimate=result.digest_token_estimate,
    )
    assert record["schema_version"] == 3
    assert record["task_id"] == "t0"
    assert record["model"] == "fake-model"
    assert "prompt" in record
    assert "completion" in record


def test_parse_judge_output_rejects_missing_abstract_fingerprint() -> None:
    with pytest.raises(ValueError, match="missing required key 'abstract_fingerprint'"):
        _parse_judge_output(json.dumps({"difficulty": 5.0}), "task-1")


def test_parse_judge_output_rejects_missing_difficulty() -> None:
    payload = json.dumps(
        {"abstract_fingerprint": " ".join(f"word{i}" for i in range(50))}
    )
    with pytest.raises(ValueError, match="missing required key 'difficulty'"):
        _parse_judge_output(payload, "task-1")


def test_parse_judge_output_rejects_short_fingerprint() -> None:
    payload = json.dumps(
        {
            "difficulty": 5.0,
            "abstract_fingerprint": " ".join(f"word{i}" for i in range(39)),
        }
    )
    with pytest.raises(ValueError, match="too short"):
        _parse_judge_output(payload, "task-1")


def test_task_judge_judge_many_runs_concurrently_and_preserves_results(
    tmp_path: Path,
) -> None:
    import threading
    import time

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def slow_script(prompt: str, model: str) -> str:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            time.sleep(0.05)
            return _len_judge(prompt, model)
        finally:
            with lock:
                in_flight -= 1

    judge = TaskJudge(
        client=FakeLLMClient(slow_script),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([f"t{i}" for i in range(16)]),
        cache_root=None,  # force fresh calls so the instrumented script runs
    )
    pool = [_QueryTask(_id=f"t{i}", _query="x" * (i + 1) * 10) for i in range(16)]
    results = judge.judge_many(pool, max_workers=8)

    assert set(results) == {task.id for task in pool}
    assert all(isinstance(r, JudgeResult) for r in results.values())
    assert max_in_flight >= 2, (
        f"judge_many should overlap calls, observed max {max_in_flight} in flight"
    )
    assert set(judge.results()) == set(results)


def test_difficulty_selector_raises_on_malformed_output(tmp_path: Path) -> None:
    pool = [_QueryTask(_id="t0", _query="q")]
    judge = TaskJudge(
        client=FakeLLMClient(lambda prompt, model: "not valid JSON"),
        model="fake-model",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([t.id for t in pool]),
    )
    with pytest.raises(RuntimeError, match="malformed JSON"):
        DifficultySelector(judge=judge).select(pool, k=1)
