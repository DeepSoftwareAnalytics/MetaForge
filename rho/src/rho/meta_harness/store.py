from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateRecord:
    """One evaluated harness in the Meta-Harness search history.

    mean_score / pass_rate come from ground-truth grading on the fixed search set.
    """

    iteration: int
    harness_id: str
    name: str
    hypothesis: str
    parent: str | None
    per_task: dict[str, float]
    mean_score: float
    pass_rate: float
    solve_traj_ids: list[str]


def append_record(summary_path: Path, record: CandidateRecord) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def load_records(summary_path: Path) -> list[CandidateRecord]:
    if not summary_path.exists():
        return []
    records: list[CandidateRecord] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(CandidateRecord(**json.loads(line)))
    return records


def best_record(records: list[CandidateRecord]) -> CandidateRecord | None:
    """Highest mean_score; ties broken toward the earliest iteration."""
    if not records:
        return None
    return max(records, key=lambda r: (r.mean_score, -r.iteration))


def write_frontier(frontier_path: Path, records: list[CandidateRecord]) -> None:
    best = best_record(records)
    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if best is not None:
        payload = {
            "best": {
                "harness_id": best.harness_id,
                "name": best.name,
                "iteration": best.iteration,
                "mean_score": best.mean_score,
                "pass_rate": best.pass_rate,
            }
        }
    frontier_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
