"""Re-grade SWE-bench Pro tasks that previously failed patch extraction.

Loads the existing solve trajectories for runs whose grades recorded
`patch_extraction_failed`, then re-runs grading with the current code. Used to
verify the patching.py artifact-exclusion fix recovers tasks that were scored 0
only because a generated dependency/build artifact aborted patch extraction.

Default mode prints corrected per-task scores without modifying any files.
With ``--apply``, the script also backfills the corrected scores into the
canonical reports so downstream consumers see one source of truth:

  - ``reports/final_val_grades.json`` is rewritten with the re-graded entries.
  - The original is renamed to ``reports/final_val_grades.uncorrected.json``
    (only on first ``--apply``; subsequent runs are no-ops because no
    ``patch_extraction_failed`` entries remain).
  - ``reports/summary.json`` (when present) has ``final_val.mean_score``
    updated to match the new pass count.

Usage:
  uv run --extra swebench-pro python scripts/regrade_extraction_failures.py
  uv run --extra swebench-pro python scripts/regrade_extraction_failures.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rho.datasets.loader import load_dataset
from rho.protocols import Grade
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore

SOURCE = "swebench-pro:ScaleAI/SWE-bench_Pro"
RUNS = [
    "runs/exp-vanilla-swebench",
    "runs/exp-rho-swebench",
    "runs/exp-letta-swebench",
    "runs/exp-dc-swebench",
]
WORKERS = 6


def load_grades(run: Path) -> list[dict]:
    report = run / "reports" / "final_val_grades.json"
    if report.exists():
        g = json.loads(report.read_text())
        return g if isinstance(g, list) else list(g.values())
    # vanilla grade run: per-task records dumped as a JSON array in run.log
    lines = (run / "run.log").read_text().splitlines()
    i = lines.index("[")
    j = len(lines) - 1 - lines[::-1].index("]")
    return json.loads("\n".join(lines[i : j + 1]))


def _apply_corrections(
    run: Path,
    grades: list[dict],
    regrades: dict[str, Grade],
) -> None:
    """Backfill `regrades` into the canonical reports under ``run/reports``."""

    corrected = []
    for entry in grades:
        new = dict(entry)
        if entry["task_id"] in regrades:
            g = regrades[entry["task_id"]]
            new["score"] = g.score
            new["details"] = g.details
        corrected.append(new)

    reports = run / "reports"
    canonical = reports / "final_val_grades.json"
    backup = reports / "final_val_grades.uncorrected.json"

    if canonical.exists():
        if not backup.exists():
            canonical.rename(backup)
        canonical.write_text(json.dumps(corrected, ensure_ascii=False, indent=2))

    summary_path = reports / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        n = len(corrected)
        new_mean = sum(e["score"] for e in corrected) / n if n else 0.0
        summary.setdefault("final_val", {})
        summary["final_val"]["n"] = n
        summary["final_val"]["mean_score"] = new_mean
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Backfill corrected scores into final_val_grades.json + summary.json.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="regrade_harness_") as tmp:
        dataset = load_dataset(
            SOURCE,
            harness_store=FilesystemHarnessStore(Path(tmp) / "harness"),
            docker_pull="missing",
        )
        tasks_by_id = {t.id: t for t in dataset.val}

        for run_str in RUNS:
            run = Path(run_str)
            grades = load_grades(run)
            traj_store = FilesystemTrajectoryStore(run / "trajectories")
            failed = [
                g
                for g in grades
                if (g.get("details") or {}).get("error") == "patch_extraction_failed"
            ]
            print(f"\n=== {run_str}: {len(failed)} extraction failures to re-grade ===")

            def regrade(g: dict) -> tuple[str, Grade | str]:
                tid = g["task_id"]
                task = tasks_by_id.get(tid)
                if task is None:
                    return tid, "TASK_NOT_IN_VAL"
                traj = traj_store.get(g["trajectory_id"])
                with tempfile.TemporaryDirectory(prefix="regrade_") as art:
                    grade = task.grade(traj, artifacts_dir=Path(art))
                return tid, grade

            results: list[tuple[str, Grade | str]] = []
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for tid, res in pool.map(regrade, failed):
                    results.append((tid, res))
                    shown = res if isinstance(res, str) else res.score
                    print(f"  {shown!s:>8}  {tid}")

            regraded_map = {tid: r for tid, r in results if isinstance(r, Grade)}
            recovered = sum(1 for g in regraded_map.values() if g.score == 1.0)
            still_err = sum(1 for _, r in results if isinstance(r, str))
            base_pass = sum(1 for g in grades if g.get("score"))
            n = len(grades)
            print(
                f"  was: {base_pass}/{n} = {base_pass / n:.2f}  |  "
                f"re-graded {len(results)}: +{recovered} pass, {still_err} still error"
            )
            print(
                f"  corrected: {base_pass + recovered}/{n} = "
                f"{(base_pass + recovered) / n:.2f}"
            )

            if args.apply and regraded_map:
                _apply_corrections(run, grades, regraded_map)
                print(f"  backfilled: {run / 'reports/final_val_grades.json'}")
                print(
                    f"  original kept at: "
                    f"{run / 'reports/final_val_grades.uncorrected.json'}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
