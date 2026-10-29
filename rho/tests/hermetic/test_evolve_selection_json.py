from __future__ import annotations

import json
from pathlib import Path

from rho import cli
from tests.helpers import make_fake_agent


def test_evolve_selection_json_preserves_selected_task_order(
    monkeypatch,
    toy_dataset_root: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    selection_json = tmp_path / "selection.json"
    pinned_ids = ["task_003", "task_001", "task_005"]
    selection_json.write_text(
        json.dumps({"selected_task_ids": pinned_ids}, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_build_agent", lambda args, run_dir: make_fake_agent("good"))

    rc = cli.main(
        [
            "evolve",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--rounds",
            "1",
            "--run-dir",
            str(run_dir),
            "--selection-json",
            str(selection_json),
            "--selector",
            "dpp",
            "--theta",
            "0.7",
            "--optimize-strategy",
            "query-only",
            "--optimize-samples",
            "1",
            "--max-grading-tasks",
            "0",
            "--codex-concurrency",
            "1",
        ]
    )

    assert rc == 0

    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["selection_json"] == str(selection_json.resolve())
    assert selection["selected_task_ids"] == pinned_ids
    assert "all_candidate_ids" in selection
    assert set(pinned_ids).issubset(set(selection["all_candidate_ids"]))

    solve_groups = json.loads(
        (run_dir / "rounds" / "round_0" / "solve_before_traj_ids.json").read_text(
            encoding="utf-8"
        )
    )
    first_solve_ids = [group[0] for group in solve_groups]
    solved_task_ids = [
        json.loads((run_dir / "trajectories" / traj_id / "meta.json").read_text(
            encoding="utf-8"
        ))["task_id"]
        for traj_id in first_solve_ids
    ]
    assert solved_task_ids == pinned_ids
