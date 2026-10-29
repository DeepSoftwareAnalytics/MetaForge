from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rho import cli
from tests.codex._az_helper import have_azure_foundry_token
from tests.helpers import make_fake_agent


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rho.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_select_rejects_theta_out_of_range(tmp_path: Path) -> None:
    proc = _run_cli(
        "select",
        "--dataset",
        "locomo",
        "--selector",
        "dpp",
        "-k",
        "2",
        "--theta",
        "1.5",
        "--run-dir",
        str(tmp_path / "run"),
    )
    assert proc.returncode != 0
    assert "theta" in proc.stderr.lower()


def test_evolve_accepts_theta_flag(tmp_path: Path) -> None:
    proc = _run_cli("evolve", "--help")
    assert proc.returncode == 0
    assert "--theta" in proc.stdout


def test_select_accepts_theta_flag() -> None:
    proc = _run_cli("select", "--help")
    assert proc.returncode == 0
    assert "--theta" in proc.stdout


def test_select_dpp_hermetic_with_fake_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    if not have_azure_foundry_token():
        pytest.skip(
            "needs `az login` for the Foundry resource judge; "
            "embedder runs locally via FastEmbed (no API key needed)"
        )

    run_dir = tmp_path / "run"
    proc = _run_cli(
        "select",
        "--dataset",
        "locomo-hard:data/locomo-hard",
        "--selector",
        "dpp",
        "-k",
        "3",
        "--theta",
        "0.7",
        "--max-per-split",
        "8",
        "--run-dir",
        str(run_dir),
    )
    assert proc.returncode == 0, proc.stderr
    assert (run_dir / "selection.json").exists()
    assert (run_dir / "selection_report.md").exists()
    assert (run_dir / "selector_calls" / "dpp_trace.json").exists()
    assert (run_dir / "selector_calls" / "dpp_kernel_eigvals.npy").exists()

    report = (run_dir / "selection_report.md").read_text(encoding="utf-8")
    assert "DPP log-det-gain trace" in report
    assert "score=" in report


def test_evolve_dpp_runs_short_solve_with_fakes(
    toy_dataset_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rho.selection as selection_mod
    from rho.selection import embedder as embedder_mod
    from rho.selection import llm_client as llm_mod

    monkeypatch.setattr(cli, "_build_agent", lambda args, run_dir: make_fake_agent("good"))
    monkeypatch.setattr(selection_mod, "LocalEmbedder", lambda *a, **kw: embedder_mod.FakeEmbedder(dim=16))
    monkeypatch.setattr(
        selection_mod,
        "LiteLLMClient",
        lambda *a, **kw: llm_mod.FakeLLMClient(
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
        ),
    )

    run_dir = tmp_path / "run"
    rc = cli.main(
        [
            "evolve",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--rounds",
            "1",
            "--run-dir",
            str(run_dir),
            "--max-evolve-tasks",
            "2",
            "--max-grading-tasks",
            "0",
            "--selector",
            "dpp",
            "--seed",
            "7",
            "--codex-concurrency",
            "1",
        ]
    )
    assert rc == 0
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert "short_solve_trajectory_ids" in selection
    assert "judge_input_token_estimate" in selection
