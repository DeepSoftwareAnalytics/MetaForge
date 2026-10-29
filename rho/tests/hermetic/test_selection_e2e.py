from __future__ import annotations

import json

import pytest

from rho.cli import main as cli_main


@pytest.mark.parametrize("selector_name", ["random", "coverage", "difficulty"])
def test_evolve_persists_selection_json(
    toy_dataset_root, tmp_path, monkeypatch, selector_name
) -> None:
    """End-to-end: every selector writes selection.json with the right shape.
    Patches the embedder and LLM client with hermetic fakes so no API is hit."""

    from rho import cli as cli_mod
    from rho.selection import embedder as embedder_mod
    from rho.selection import llm_client as llm_mod
    from tests.helpers import make_fake_agent

    monkeypatch.setattr(cli_mod, "_build_agent", lambda args, run_dir: make_fake_agent("good"))

    # Swap the real API-backed clients for hermetic fakes at import time
    # (build_selector uses late-binding imports inside its body).
    monkeypatch.setattr(
        embedder_mod,
        "LiteLLMEmbedder",
        lambda *a, **kw: embedder_mod.FakeEmbedder(dim=16),
    )
    monkeypatch.setattr(
        llm_mod,
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
    argv = [
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
        selector_name,
        "--seed",
        "7",
    ]
    rc = cli_main(argv)
    assert rc == 0

    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["selector"] == selector_name
    assert len(selection["selected_task_ids"]) == 2
    assert "all_candidate_ids" in selection

    if selector_name == "difficulty":
        assert "difficulty_scores" in selection
        # One JSON per task scored.
        scored_files = list((run_dir / "selector_calls").glob("*.json"))
        assert len(scored_files) == len(selection["all_candidate_ids"])
