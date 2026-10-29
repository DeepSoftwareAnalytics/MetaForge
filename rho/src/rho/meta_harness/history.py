from __future__ import annotations

import shutil
from pathlib import Path

from rho.meta_harness.store import load_records
from rho.orchestrators._util import dump_trajectory
from rho.protocols import HarnessStore, TrajectoryStore


def build_history_dir(
    *,
    dest: Path,
    summary_path: Path,
    frontier_path: Path,
    reports_dir: Path,
    harness_store: HarnessStore,
    traj_store: TrajectoryStore,
) -> None:
    """Materialize the proposer's read-only `history/` workspace.

    Writes prior candidate code, ground-truth scores (summary.jsonl), the
    frontier, raw solve traces, and earlier iterations' post-mortem reports.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.jsonl").write_text(
        summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        encoding="utf-8",
    )
    (dest / "frontier.json").write_text(
        frontier_path.read_text(encoding="utf-8") if frontier_path.exists() else "{}",
        encoding="utf-8",
    )
    reports_dest = dest / "reports"
    reports_dest.mkdir(exist_ok=True)
    if reports_dir.exists():
        shutil.copytree(reports_dir, reports_dest, dirs_exist_ok=True)

    candidates_dir = dest / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    traces_dir = dest / "traces"
    traces_dir.mkdir(exist_ok=True)

    materialized: set[str] = set()
    for record in load_records(summary_path):
        if record.harness_id not in materialized:
            materialized.add(record.harness_id)
            harness_store.get(record.harness_id).materialize(
                candidates_dir / record.harness_id
            )
        for traj_id in record.solve_traj_ids:
            traj = traj_store.get(traj_id)
            dump_trajectory(
                traces_dir / record.harness_id / traj.task_id / traj_id,
                None,
                traj,
            )
