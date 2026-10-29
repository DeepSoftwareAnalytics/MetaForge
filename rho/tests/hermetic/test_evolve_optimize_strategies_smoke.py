from __future__ import annotations

import json

import pytest

from rho.cli import main
from tests.helpers import make_fake_agent


@pytest.mark.parametrize(
    ("strategy", "extra_args", "expects_diagnosis_files"),
    [
        ("query-only", [], False),
        ("trajectory", ["--optimize-trajectories-per-task", "2"], False),
        ("diagnosis", [], True),
        ("diagnosis-no-consistency", [], True),
        ("diagnosis-no-validation", [], True),
    ],
)
def test_cli_evolve_smoke_for_optimize_strategies(
    monkeypatch,
    toy_dataset_root,
    tmp_path,
    strategy: str,
    extra_args: list[str],
    expects_diagnosis_files: bool,
) -> None:
    monkeypatch.setattr(
        "rho.cli.CodexAgent",
        lambda **kwargs: make_fake_agent("good"),
    )
    run_dir = tmp_path / strategy
    rc = main(
        [
            "evolve",
            "--dataset",
            str(toy_dataset_root),
            "--rounds",
            "1",
            "--run-dir",
            str(run_dir),
            "--max-evolve-tasks",
            "1",
            "--optimize-strategy",
            strategy,
        ]
        + extra_args
    )

    assert rc == 0
    round_dir = run_dir / "rounds" / "round_0"
    candidates = json.loads((round_dir / "optimize_candidates.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "reports" / "summary.json").read_text(encoding="utf-8"))

    assert config["optimize_strategy"] == strategy
    assert summary["optimize_strategy"] == strategy
    assert len(candidates["samples"]) == 3
    assert candidates["winner_candidate_harness_id"] is not None
    assert (round_dir / "optimize_instructions.txt").exists()
    assert (round_dir / "optimize_input_tokens.json").exists()
    assert (round_dir / "candidate_harness_id").read_text(encoding="utf-8").strip() != "(none)"
    assert (round_dir / "diagnoses.json").exists() is expects_diagnosis_files
    assert (round_dir / "diagnose_traj_ids.json").exists() is expects_diagnosis_files
