from __future__ import annotations

import json
from pathlib import Path

import pytest

from rho.cli import main as cli_main


@pytest.fixture
def hermetic_selectors(monkeypatch):
    """Swap real API-backed clients for FakeEmbedder + constant FakeLLMClient,
    so 'difficulty' and 'coverage' selectors run without hitting any API."""
    import rho.selection as selection_mod
    from rho.selection import embedder as embedder_mod
    from rho.selection import llm_client as llm_mod

    fake_embedder_factory = lambda *a, **kw: embedder_mod.FakeEmbedder(dim=16)
    fake_llm_factory = lambda *a, **kw: llm_mod.FakeLLMClient(
            lambda prompt, model: json.dumps(
                {
                    "difficulty": 5.0,
                    "abstract_fingerprint": (
                        "Failure mode is partial propagation across modules where one "
                        "branch adopts a revised contract while another silently keeps "
                        "older ordering assumptions until a boundary case appears. "
                        "Difficulty stays moderate because invariants are distributed "
                        "across helpers, reconciliation, and precedence logic, so the "
                        "change requires contextual tracing, contract alignment, and "
                        "multi-file bug-fix reasoning rather than a purely local edit."
                    ),
                }
            )
        )
    monkeypatch.setattr(embedder_mod, "LiteLLMEmbedder", fake_embedder_factory)
    monkeypatch.setattr(selection_mod, "LiteLLMEmbedder", fake_embedder_factory)
    monkeypatch.setattr(selection_mod, "LocalEmbedder", fake_embedder_factory)
    monkeypatch.setattr(llm_mod, "LiteLLMClient", fake_llm_factory)
    monkeypatch.setattr(selection_mod, "LiteLLMClient", fake_llm_factory)


def test_select_random_writes_selection_json(
    toy_dataset_root: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "2",
            "--seed",
            "7",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0

    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["selector"] == "random"
    assert sel["k"] == 2
    assert sel["seed"] == 7
    assert len(sel["selected_task_ids"]) == 2
    assert set(sel["selected_task_ids"]).issubset(set(sel["all_candidate_ids"]))

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["selector"] == "random"
    assert config["dataset_spec"] == f"directory:{toy_dataset_root}"
    assert config["split"] == "train"
    assert (run_dir / "environment.json").exists()


@pytest.mark.parametrize("selector_name", ["difficulty", "coverage"])
def test_select_with_fakes_writes_selector_artifacts(
    toy_dataset_root: Path,
    tmp_path: Path,
    hermetic_selectors,
    hermetic_short_solve_agent,
    selector_name: str,
) -> None:
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            selector_name,
            "-k",
            "2",
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0

    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["selector"] == selector_name
    assert len(sel["selected_task_ids"]) == 2
    assert "short_solve_trajectory_ids" in sel
    assert "judge_input_token_estimate" in sel

    workdir = run_dir / "selector_calls"
    if selector_name == "difficulty":
        assert "difficulty_scores" in sel
        assert len(list(workdir.glob("*.json"))) == len(sel["all_candidate_ids"])
    else:
        for name in (
            "fingerprints.json",
            "embeddings.npy",
            "similarity.npy",
            "candidate_ids.json",
            "gain_trace.json",
        ):
            assert (workdir / name).exists(), f"missing {name}"


@pytest.mark.parametrize("selector_name", ["difficulty", "coverage"])
def test_select_requires_k_for_non_random(
    toy_dataset_root: Path, tmp_path: Path, hermetic_selectors, selector_name: str
) -> None:
    # regression: Task 1 added the explicit exit-2 guard for non-random selectors.
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            selector_name,
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    assert rc == 2


def test_select_random_without_k_returns_all(
    toy_dataset_root: Path, tmp_path: Path
) -> None:
    # regression: RandomSelector preserves the full pool when k is omitted.
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["k"] is None
    assert sel["selected_task_ids"] == sel["all_candidate_ids"]


def test_select_task_filter_no_match_exits_1(
    toy_dataset_root: Path, tmp_path: Path
) -> None:
    # regression: Task 1 already returns exit 1 for empty post-filter pools.
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "1",
            "--task-filter",
            "this_substring_matches_nothing_xyz",
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    assert rc == 1


def test_select_k_larger_than_pool_returns_all(
    toy_dataset_root: Path, tmp_path: Path, capsys
) -> None:
    # regression: Task 1 warns and returns the full pool when k exceeds pool size.
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "9999",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "exceeds pool size" in captured.err
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert len(sel["selected_task_ids"]) == len(sel["all_candidate_ids"])


def test_select_refuses_existing_selection_json(
    toy_dataset_root: Path, tmp_path: Path
) -> None:
    # regression: Task 1 protects existing selection outputs from overwrite.
    run_dir = tmp_path / "run"
    rc1 = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc1 == 0
    rc2 = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc2 == 1


def test_select_random_writes_report(toy_dataset_root: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "random",
            "-k",
            "2",
            "--seed",
            "7",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0
    text = (run_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "Selection report — random" in text
    assert "## Selected tasks" in text


def test_select_coverage_report_has_gain_trace_and_neighbors(
    toy_dataset_root: Path, tmp_path: Path, hermetic_selectors, hermetic_short_solve_agent
) -> None:
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "coverage",
            "-k",
            "2",
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0
    text = (run_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "## Coverage gain trace" in text
    assert "## Coverage nearest-neighbor spot-check" in text


def test_select_difficulty_report_has_histogram(
    toy_dataset_root: Path, tmp_path: Path, hermetic_selectors, hermetic_short_solve_agent
) -> None:
    run_dir = tmp_path / "run"
    rc = cli_main(
        [
            "select",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--selector",
            "difficulty",
            "-k",
            "2",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert rc == 0
    text = (run_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "## Difficulty score histogram" in text
    assert "Fingerprint:" in text


def test_select_with_fakes_runs_short_solve(
    toy_dataset_root, tmp_path, hermetic_selectors, hermetic_short_solve_agent
):
    run_dir = tmp_path / "run"
    rc = cli_main([
        "select",
        "--dataset", f"directory:{toy_dataset_root}",
        "--selector", "difficulty",
        "-k", "2",
        "--seed", "1",
        "--run-dir", str(run_dir),
    ])
    assert rc == 0
    sel = json.loads((run_dir / "selection.json").read_text())
    assert "short_solve_trajectory_ids" in sel
    assert "judge_input_token_estimate" in sel
    # Random skip case: re-run with random and confirm fields absent.
    rc2 = cli_main([
        "select",
        "--dataset", f"directory:{toy_dataset_root}",
        "--selector", "random",
        "-k", "2",
        "--seed", "1",
        "--run-dir", str(tmp_path / "run2"),
    ])
    assert rc2 == 0
    sel2 = json.loads((tmp_path / "run2" / "selection.json").read_text())
    assert "short_solve_trajectory_ids" not in sel2
