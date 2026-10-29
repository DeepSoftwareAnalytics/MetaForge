#!/usr/bin/env python3
"""Build per-step category density: for each (dataset, method), how often
each action category appears at step k (averaged over held-out tasks).

For each trajectory in the held-out grade:
  - enumerate command_execution events in chronological order
  - the i-th event sits at step position k=i+1
  - classify with the same rules as `classify_actions.py`

For each category, the per-step count is:
  count[c][k] = (# events at step k of category c, summed over tasks) / N_tasks

Total height of the stacked curves at step k naturally decays as fewer
tasks reach that step (i.e. it equals the "survival" fraction of tasks
still issuing commands at step k).

Output: one JSON consumed by the D3 figure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# Re-use the classifier from the sibling module.
import importlib.util
import sys

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

spec = importlib.util.spec_from_file_location(
    "classify_actions", REPO / "scripts" / "classify_actions.py"
)
classify_actions = importlib.util.module_from_spec(spec)
sys.modules["classify_actions"] = classify_actions
spec.loader.exec_module(classify_actions)
classify = classify_actions.classify
load_grades = classify_actions.load_grades
CATEGORIES = classify_actions.CATEGORIES

SPECS = [
    ("SWE-bench Pro", "vanilla", RUNS / "exp-vanilla-swebench"),
    ("SWE-bench Pro", "rho", RUNS / "exp-rho-swebench"),
    ("Terminal-Bench 2", "vanilla", RUNS / "exp-vanilla-tb2"),
    ("Terminal-Bench 2", "rho", RUNS / "20260520-rho-tb2-traj"),
    ("GAIA2", "vanilla", RUNS / "exp-vanilla-gaia2"),
    ("GAIA2", "rho", RUNS / "exp-rho-gaia2-patched"),
]


def per_trajectory_step_categories(run_dir: Path, traj_id: str) -> list[str]:
    """Return the category at each step position, in order."""
    events_path = run_dir / "trajectories" / traj_id / "events.jsonl"
    cats: list[str] = []
    if not events_path.exists():
        return cats
    with events_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "item.completed":
                continue
            item = e.get("item", {})
            if item.get("type") != "command_execution":
                continue
            cats.append(classify(item.get("command", "")))
    return cats


def main() -> None:
    out: dict = {}
    for dataset, method, run_dir in SPECS:
        grades = load_grades(run_dir)
        per_step: dict[str, list[int]] = {c: [] for c in CATEGORIES}
        max_steps = 0
        n_tasks = len(grades)
        for g in grades:
            cats = per_trajectory_step_categories(run_dir, g["trajectory_id"])
            if len(cats) > max_steps:
                max_steps = len(cats)
            for c in CATEGORIES:
                # extend the per-category histogram to current length
                while len(per_step[c]) < len(cats):
                    per_step[c].append(0)
            for k, c in enumerate(cats):
                per_step[c][k] += 1

        # normalize: mean count per task at step k
        cats_norm = {
            c: [v / n_tasks for v in per_step[c]] for c in CATEGORIES
        }
        out[f"{dataset}|{method}"] = {
            "dataset": dataset,
            "method": method,
            "n_tasks": n_tasks,
            "max_steps": max_steps,
            "categories": cats_norm,
        }
        # quick summary
        total = sum(sum(v) for v in cats_norm.values())
        print(f"{dataset:18s} {method:8s}  n={n_tasks:3d}  max_k={max_steps:3d}  Σmean={total:5.1f}")

    out_path = REPO / "docs" / "paper" / "figures" / "step_category_density.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote → {out_path}")


if __name__ == "__main__":
    main()
