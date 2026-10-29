"""Salvage script: regrade persisted final val trajectories.

Replays task.grade() on the final_val_grade trajectories that were already
persisted to disk, then writes the same final val report that `rho
evolve` would have written. Use when an evolve run completed all codex stages
but crashed in final tempdir cleanup before the reports were serialized.

Usage:
    uv run --extra swebench-pro scripts/regrade_persisted_val.py <run-dir> [grade-workers]

Reads the candidate harness id from <run-dir>/rounds/round_0/. Reads dataset
name and docker_pull from <run-dir>/config.json. Writes
reports/final_val_grades.json.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rho.cli import _serialize_grades, _write_json
from rho.datasets.loader import load_dataset
from rho.reporting import GradedSolve, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    grade_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset_name = config["dataset_spec"]
    docker_pull = config.get("docker_pull", "missing")

    round0 = run_dir / "rounds" / "round_0"
    final_harness_id = (round0 / "candidate_harness_id").read_text(encoding="utf-8").strip()

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")

    print(
        f"[regrade] run_dir={run_dir}\n"
        f"[regrade] dataset={dataset_name} docker_pull={docker_pull}\n"
        f"[regrade] final_harness={final_harness_id}\n"
        f"[regrade] grade_workers={grade_workers}",
        file=sys.stderr,
    )

    dataset = load_dataset(
        dataset_name,
        harness_store=harness_store,
        docker_pull=docker_pull,
    )
    val_tasks = {task.id: task for task in dataset.val}
    print(f"[regrade] val split has {len(val_tasks)} tasks", file=sys.stderr)

    # Persisted val trajectories indexed by task id.
    final_trajs: dict[str, object] = {}
    for traj in traj_store._iter_all():
        if traj.stage == "final_val_grade" and traj.harness_id == final_harness_id:
            final_trajs[traj.task_id] = traj
    print(
        f"[regrade] persisted trajectories: final={len(final_trajs)}",
        file=sys.stderr,
    )

    artifacts_root = run_dir / "workdir" / "grade_artifacts_regrade"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    def grade_stage(stage: str, traj_by_task: dict[str, object]) -> list[GradedSolve]:
        records: list[GradedSolve] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=grade_workers) as pool:
            futures = {}
            for task_id, traj in traj_by_task.items():
                task = val_tasks.get(task_id)
                if task is None:
                    print(f"[regrade] WARNING task {task_id!r} not in val split, skip", file=sys.stderr)
                    continue
                artifacts_dir = artifacts_root / stage / task.id / traj.id
                fut = pool.submit(task.grade, traj, artifacts_dir=artifacts_dir)
                futures[fut] = (task, traj)
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                task, traj = futures[fut]
                done += 1
                try:
                    grade = fut.result()
                except Exception as exc:
                    print(f"[regrade] {stage} {task.id} EXCEPTION {exc!r}", file=sys.stderr)
                    continue
                records.append(GradedSolve(task=task, grade=grade, trajectory=traj, stage=stage))
                print(
                    f"[regrade] {stage} {done}/{total} {task.id} score={grade.score:.2f}",
                    file=sys.stderr,
                )
        elapsed = time.time() - t0
        print(f"[regrade] {stage} done in {elapsed:.1f}s, {len(records)}/{total} graded", file=sys.stderr)
        return records

    final_records = grade_stage("final_val_grade", final_trajs)

    report_dir = run_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    _write_json(report_dir / "final_val_grades.json", _serialize_grades(final_records))

    final_summary = summarize(final_records)
    print(
        f"\n[regrade] FINAL: mean_score={final_summary['mean_score']:.4f} n={final_summary['n']}\n"
        f"[regrade] wrote reports/final_val_grades.json",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
