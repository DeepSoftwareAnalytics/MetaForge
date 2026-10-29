#!/usr/bin/env python3
"""Build the steps-vs-success CSV from the current source-of-truth runs.

For each (dataset, method) pair we pick the canonical run directory, read the
held-out grade (score per task), and count agent steps (number of
`command_execution` events) in each trajectory.

The output is a single CSV consumed by the D3 figure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# (dataset_label, method_label, run_dir) — note rho-gaia2-patched is the rerun.
SPECS = [
    ("SWE-bench Pro", "vanilla", RUNS / "exp-vanilla-swebench"),
    ("SWE-bench Pro", "rho", RUNS / "exp-rho-swebench"),
    ("Terminal-Bench 2", "vanilla", RUNS / "exp-vanilla-tb2"),
    ("Terminal-Bench 2", "rho", RUNS / "20260520-rho-tb2-traj"),
    ("GAIA2", "vanilla", RUNS / "exp-vanilla-gaia2"),
    ("GAIA2", "rho", RUNS / "exp-rho-gaia2-patched"),
]


def load_grades(run_dir: Path) -> list[dict]:
    """Return list of {task_id, trajectory_id, score} from a run.

    Evolve runs persist `reports/final_val_grades.json`; the older
    grade-only vanilla runs (swebench, tb2) only have the JSON dumped at the
    tail of `run.log` after a header line. vanilla-gaia2 has both — prefer
    the report when present.
    """
    grades_path = run_dir / "reports" / "final_val_grades.json"
    if grades_path.exists():
        return json.loads(grades_path.read_text())

    log = (run_dir / "run.log").read_text()
    # strip the leading "[rho cache] ..." line; the rest is a JSON array.
    start = log.index("[", log.index("\n"))
    return json.loads(log[start:])


def count_steps(run_dir: Path, traj_id: str) -> int:
    """Count `command_execution` items in a trajectory's events.jsonl."""
    events_path = run_dir / "trajectories" / traj_id / "events.jsonl"
    if not events_path.exists():
        return 0
    n = 0
    with events_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "item.completed":
                if e.get("item", {}).get("type") == "command_execution":
                    n += 1
    return n


def main() -> None:
    out_path = REPO / "docs" / "paper" / "figures" / "all_datasets_steps_vs_success.csv"
    rows: list[dict] = []
    for dataset, method, run_dir in SPECS:
        grades = load_grades(run_dir)
        n_succ = 0
        for g in grades:
            traj_id = g["trajectory_id"]
            steps = count_steps(run_dir, traj_id)
            score = float(g["score"])
            success = 1.0 if score >= 1.0 else 0.0 if score <= 0.0 else score
            n_succ += success
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "task_id": g["task_id"],
                    "trajectory_id": traj_id,
                    "steps": steps,
                    "success": success,
                }
            )
        n_total = len(grades)
        print(
            f"{dataset:18s} {method:8s} {run_dir.name:30s} "
            f"n={n_total:3d}  pass@1={n_succ / n_total:.3f}"
        )

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "method", "task_id", "trajectory_id", "steps", "success"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
