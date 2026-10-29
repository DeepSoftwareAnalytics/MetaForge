from __future__ import annotations

import json
from pathlib import Path

from rho.meta_harness.store import (
    CandidateRecord,
    append_record,
    best_record,
    load_records,
    write_frontier,
)


def _record(iteration: int, harness_id: str, mean: float) -> CandidateRecord:
    return CandidateRecord(
        iteration=iteration,
        harness_id=harness_id,
        name=f"cand_{iteration}",
        hypothesis="test hypothesis",
        parent=None,
        per_task={"task_a": mean},
        mean_score=mean,
        pass_rate=mean,
        solve_traj_ids=["traj_1"],
    )


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    summary = tmp_path / "summary.jsonl"
    append_record(summary, _record(0, "h_seed", 0.2))
    append_record(summary, _record(1, "h_aaa", 0.6))

    records = load_records(summary)
    assert [r.iteration for r in records] == [0, 1]
    assert records[1].harness_id == "h_aaa"
    assert records[1].per_task == {"task_a": 0.6}
    assert isinstance(records[0], CandidateRecord)


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_records(tmp_path / "absent.jsonl") == []


def test_best_record_picks_max_mean_then_earliest() -> None:
    records = [_record(0, "h_seed", 0.5), _record(1, "h_aaa", 0.9), _record(2, "h_bbb", 0.9)]
    best = best_record(records)
    assert best is not None
    assert best.harness_id == "h_aaa"  # ties broken toward earlier iteration
    assert best_record([]) is None


def test_write_frontier(tmp_path: Path) -> None:
    frontier = tmp_path / "frontier.json"
    write_frontier(frontier, [_record(0, "h_seed", 0.5), _record(1, "h_aaa", 0.9)])
    payload = json.loads(frontier.read_text(encoding="utf-8"))
    assert payload["best"]["harness_id"] == "h_aaa"
    assert payload["best"]["mean_score"] == 0.9
