import json

import pytest

from rho.cli import main
from tests.conftest import make_fake_agent


def test_run_artifact_layout(monkeypatch, toy_dataset_root, tmp_path) -> None:
    monkeypatch.setattr(
        "rho.cli.CodexAgent",
        lambda **kwargs: make_fake_agent("good"),
    )
    run_dir = tmp_path / "run"
    rc = main(
        [
            "evolve",
            "--dataset",
            str(toy_dataset_root),
            "--rounds",
            "1",
            "--run-dir",
            str(run_dir),
            "--codex-concurrency",
            "7",
        ]
    )
    assert rc == 0
    assert (run_dir / "config.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "rounds" / "round_0" / "scores.json").exists()
    assert (run_dir / "rounds" / "round_0" / "optimize_candidates.json").exists()
    assert (run_dir / "rounds" / "round_0" / "optimize_instructions.txt").exists()
    assert (run_dir / "rounds" / "round_0" / "optimize_input_tokens.json").exists()
    assert (run_dir / "rounds" / "round_0" / "diagnose_traj_ids.json").exists()
    assert (run_dir / "rounds" / "round_0" / "diagnoses.json").exists()
    assert (run_dir / "reports" / "summary.json").exists()
    assert (run_dir / "reports" / "final_val_grades.json").exists()
    assert (run_dir / "reports" / "usage_summary.json").exists()
    assert (run_dir / "reports" / "manifest.json").exists()
    assert (run_dir / "reports" / "summary.txt").exists()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["argv"][:2] == ["evolve", "--dataset"]
    assert config["cache_mode"] == "off"
    assert config["cache_dir"] is None
    assert config["optimize_samples"] == 3
    assert config["optimize_strategy"] == "diagnosis"
    assert config["optimize_trajectories_per_task"] == 3
    assert config["codex_concurrency"] == 7
    assert config["model"] == "gpt-5.5"
    assert config["reasoning_effort"] == "high"
    assert config["codex_isolation"]["inherits_user_config"] is False
    assert config["codex_isolation"]["subprocess_env"] == "minimal"
    grades = json.loads((run_dir / "reports" / "final_val_grades.json").read_text(encoding="utf-8"))
    assert grades
    assert "trajectory_id" in grades[0]
    assert "harness_id" in grades[0]
    assert "prediction" in grades[0]
    summary = json.loads((run_dir / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert summary["optimize_strategy"] == "diagnosis"
    assert summary["optimize_trajectories_per_task"] == 3
    assert "initial_val" not in summary
    assert "final_val" in summary
    assert "task_id" in summary["rounds"][0]["scores"][0]

    rc = main(
        [
            "solve",
            "--dataset",
            str(toy_dataset_root),
            "--task",
            "task_001",
            "--harness",
            "h_empty",
            "--run-dir",
            str(run_dir),
            "--codex-concurrency",
            "7",
        ]
    )
    assert rc == 0
    rc = main(
        [
            "grade",
            "--dataset",
            str(toy_dataset_root),
            "--split",
            "val",
            "--harness",
            "h_empty",
            "--run-dir",
            str(run_dir),
            "--codex-concurrency",
            "7",
        ]
    )
    assert rc == 0
    traj_meta = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "trajectories").glob("*/meta.json")
    ]
    manifest = json.loads((run_dir / "reports" / "manifest.json").read_text(encoding="utf-8"))
    usage = json.loads((run_dir / "reports" / "usage_summary.json").read_text(encoding="utf-8"))
    stages = {meta.get("stage") for meta in traj_meta}
    assert "cli_solve" in stages
    assert "cli_val_grade" in stages
    assert manifest["trajectory_count"] == len(traj_meta)
    assert manifest["trajectory_counts_by_stage"]["cli_solve"] >= 1
    assert manifest["trajectory_counts_by_stage"]["cli_val_grade"] >= 1
    assert usage["overall"]["trajectory_count"] == len(traj_meta)
    assert usage["by_stage"]["cli_solve"]["trajectory_count"] >= 1
    assert usage["by_stage"]["cli_val_grade"]["trajectory_count"] >= 1


def test_non_diagnosis_round_artifacts_omit_diagnose_files(monkeypatch, toy_dataset_root, tmp_path) -> None:
    monkeypatch.setattr(
        "rho.cli.CodexAgent",
        lambda **kwargs: make_fake_agent("good"),
    )
    for strategy in ("query-only", "trajectory"):
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
                "--optimize-strategy",
                strategy,
                "--max-evolve-tasks",
                "1",
            ]
            + (
                ["--optimize-trajectories-per-task", "2"]
                if strategy == "trajectory"
                else []
            )
        )
        assert rc == 0
        round_dir = run_dir / "rounds" / "round_0"
        assert (round_dir / "optimize_instructions.txt").exists()
        assert (round_dir / "optimize_input_tokens.json").exists()
        assert not (round_dir / "diagnose_traj_ids.json").exists()
        assert not (round_dir / "diagnoses.json").exists()
        summary = json.loads((run_dir / "reports" / "summary.json").read_text(encoding="utf-8"))
        assert summary["optimize_strategy"] == strategy
        if strategy == "trajectory":
            assert summary["optimize_trajectories_per_task"] == 2


def test_codex_concurrency_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "evolve",
                "--dataset",
                "dummy",
                "--rounds",
                "1",
                "--codex-concurrency",
                "0",
            ]
        )


def test_codex_concurrency_help_on_codex_backed_commands(capsys) -> None:
    # `select` is now codex-backed too (short-solve probe runs the agent
    # before judge+selector — see trajectory-aware-task-selection-design §9.2).
    for command in ("evolve", "solve", "grade", "select"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert "--codex-concurrency" in capsys.readouterr().out
