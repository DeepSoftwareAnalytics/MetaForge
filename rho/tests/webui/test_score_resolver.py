from __future__ import annotations

import json

from rho.webui.data import RunRepository

from .helpers import find_trajectory_id, require_run


def test_final_val_grade_trajectory_resolves_to_ctrf_report_score() -> None:
    run_dir = require_run("exp-rho-tb2")
    traj_id = find_trajectory_id(run_dir, stage="final_val_grade")
    score = RunRepository(run_dir.parent).get_trajectory_score(run_dir.name, traj_id)

    assert score["kind"] == "ctrf"
    assert score["score"] in {0.0, 1.0}
    assert score["source"].startswith("reports/")
    assert "ctrf" in score


def test_baseline_grade_trajectory_resolves_from_run_log() -> None:
    run_dir = require_run("exp-vanilla-tb2")
    traj_id = "traj_568853f71d"
    score = RunRepository(run_dir.parent).get_trajectory_score(run_dir.name, traj_id)

    assert score["kind"] == "ctrf"
    assert score["source"] == "run.log"
    assert score["score"] in {0.0, 1.0}
    assert score["ctrf"]["total"] >= score["ctrf"]["passed"]


def test_round_evaluate_trajectory_resolves_to_own_consistency_score() -> None:
    run_dir = require_run("exp-rho-tb2")
    traj_id = find_trajectory_id(run_dir, stage="round_evaluate", kind="evaluate")
    score = RunRepository(run_dir.parent).get_trajectory_score(run_dir.name, traj_id)

    assert score["kind"] == "consistency"
    assert isinstance(score["score"], int)
    assert score["rationale"]
    assert score["source"] == "final_message.txt"

    final_message = json.loads((run_dir / "trajectories" / traj_id / "final_message.txt").read_text(encoding="utf-8"))
    assert score["score"] == final_message["value"]


def test_round_solve_before_trajectory_is_ungraded() -> None:
    run_dir = require_run("exp-rho-tb2")
    traj_id = find_trajectory_id(run_dir, stage="round_solve_before")
    score = RunRepository(run_dir.parent).get_trajectory_score(run_dir.name, traj_id)

    assert score == {
        "kind": "ungraded",
        "score": None,
        "ctrf": None,
        "reward": None,
        "rationale": None,
        "source": "ungraded",
    }
