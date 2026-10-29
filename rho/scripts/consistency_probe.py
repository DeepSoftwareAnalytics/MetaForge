#!/usr/bin/env python3
"""Probe consistency on LoCoMo val set: find wrong samples, then re-run each 5x."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rho.agent.cache import build_default_agent
from rho.agent.codex import CodexAgent
from rho.datasets.loader import load_dataset
from rho.orchestrators.solve import solve
from rho.stores.harness import FilesystemHarnessStore

N_REPEATS = 5
RUN_DIR = Path("runs/consistency-probe")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "workdir").mkdir(exist_ok=True)

    harness_store = FilesystemHarnessStore(RUN_DIR / "harness")
    dataset = load_dataset("locomo:data/locomo10.json", harness_store=harness_store)
    val_tasks = list(dataset.val)[: args.max_tasks]
    harness = val_tasks[0].harness
    WORKERS = args.workers

    # --- Phase 1: solve all val tasks (cached), grade them ---
    codex_config_path = Path.home() / ".codex" / "config.toml"
    cached_agent = build_default_agent(
        CodexAgent(codex_config_path=codex_config_path),
        mode="on",
        cache_dir=RUN_DIR / "agent-cache",
    )

    def solve_and_grade(task):
        traj = solve(cached_agent, task, harness, workdir=RUN_DIR / "workdir")
        grade = task.grade(traj)
        return task, traj, grade

    print(f"Phase 1: solving {len(val_tasks)} val tasks with {WORKERS} workers…")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(solve_and_grade, val_tasks))

    passed = [(t, tr, g) for t, tr, g in results if g.passed]
    failed = [(t, tr, g) for t, tr, g in results if not g.passed]
    print(f"  passed={len(passed)}  failed={len(failed)}")

    phase1 = []
    for task, traj, grade in results:
        phase1.append({
            "task_id": task.id,
            "passed": grade.passed,
            "score": grade.score,
            "prediction": grade.details.get("prediction", ""),
            "gold": grade.details.get("gold", ""),
            "question": grade.details.get("question", ""),
            "category": grade.details.get("category"),
        })
    _write_json(RUN_DIR / "phase1_grades.json", phase1)

    if not failed:
        print("No failures — nothing to probe.")
        return 0

    # --- Phase 2: re-run each failed task 5x WITHOUT cache ---
    raw_agent = CodexAgent(codex_config_path=codex_config_path)
    failed_tasks = [t for t, _, _ in failed]

    work_items = [(task, i) for task in failed_tasks for i in range(N_REPEATS)]
    print(f"Phase 2: {len(failed_tasks)} failed tasks × {N_REPEATS} = {len(work_items)} solves…")

    def solve_repeat(item):
        task, repeat_ix = item
        traj = solve(raw_agent, task, harness, workdir=RUN_DIR / "workdir")
        grade = task.grade(traj)
        return {
            "task_id": task.id,
            "repeat": repeat_ix,
            "passed": grade.passed,
            "score": grade.score,
            "prediction": grade.details.get("prediction", ""),
            "final_message": traj.final_message,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        repeat_results = list(pool.map(solve_repeat, work_items))

    by_task: dict[str, list[dict]] = {}
    for r in repeat_results:
        by_task.setdefault(r["task_id"], []).append(r)

    # --- Analysis ---
    print("\n=== Consistency Analysis ===\n")
    phase2 = []
    for task_id, repeats in sorted(by_task.items()):
        orig = next(p for p in phase1 if p["task_id"] == task_id)
        predictions = [r["prediction"] for r in repeats]
        pass_count = sum(1 for r in repeats if r["passed"])
        unique = set(predictions)

        entry = {
            "task_id": task_id,
            "gold": orig["gold"],
            "question": orig["question"],
            "category": orig["category"],
            "original_prediction": orig["prediction"],
            "pass_rate": pass_count / len(repeats),
            "n_unique": len(unique),
            "repeats": repeats,
        }
        phase2.append(entry)

        tag = "CONSISTENT" if len(unique) == 1 else f"DIVERGENT ({len(unique)} unique)"
        print(f"  {task_id}: pass={pass_count}/{len(repeats)}, {tag}")
        print(f"    gold: {orig['gold'][:100]}")
        for i, r in enumerate(repeats):
            mark = "✓" if r["passed"] else "✗"
            print(f"    [{i}] {mark} score={r['score']:.2f} → {r['prediction'][:100]}")
        print()

    _write_json(RUN_DIR / "phase2_consistency.json", phase2)

    n_consistent = sum(1 for e in phase2 if e["n_unique"] == 1)
    n_divergent = len(phase2) - n_consistent
    avg_pass = sum(e["pass_rate"] for e in phase2) / len(phase2) if phase2 else 0

    print("=== Summary ===")
    print(f"  Failed tasks probed:     {len(phase2)}")
    print(f"  Consistent (1 answer):   {n_consistent}")
    print(f"  Divergent (>1 answers):  {n_divergent}")
    print(f"  Avg pass rate on retry:  {avg_pass:.1%}")
    return 0


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
