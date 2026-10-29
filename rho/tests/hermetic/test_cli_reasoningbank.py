from __future__ import annotations

import json
from pathlib import Path

from rho import cli
from tests.helpers import make_fake_agent


def test_reasoningbank_cli_writes_frozen_run_artifacts(
    monkeypatch,
    toy_dataset_root: Path,
    tmp_path: Path,
) -> None:
    from rho.selection import embedder as embedder_mod
    from rho.selection import llm_client as llm_mod

    run_dir = tmp_path / "run"
    monkeypatch.setattr(cli, "_build_agent", lambda args, run_dir: make_fake_agent("good"))
    monkeypatch.setattr(
        embedder_mod,
        "LiteLLMEmbedder",
        lambda *args, **kwargs: embedder_mod.FakeEmbedder(dim=16),
    )
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda *args, **kwargs: embedder_mod.FakeEmbedder(dim=16),
    )
    monkeypatch.setattr(
        llm_mod,
        "LiteLLMClient",
        lambda *args, **kwargs: llm_mod.FakeLLMClient(
            lambda prompt, model: "success"
            if "Did the agent successfully complete the task?" in prompt
            else "remember to inspect the harness"
        ),
    )

    rc = cli.main(
        [
            "reasoningbank",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--run-dir",
            str(run_dir),
            "--max-train-tasks",
            "1",
            "--max-grading-tasks",
            "1",
            "--selector",
            "random",
            "--seed",
            "0",
            "--cache",
            "off",
            "--codex-concurrency",
            "1",
        ]
    )

    assert rc == 0
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["baseline"] == "reasoningbank"
    assert config["eval_variant"] == "frozen"
    assert config["embedding_provider"] == "litellm"
    assert config["embedding_model"] == "local:BAAI/bge-large-en-v1.5"

    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected_task_ids"]

    memory_lines = (run_dir / "reasoningbank" / "memory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(memory_lines) == 1
    assert json.loads(memory_lines[0])["status"] == "success"

    summary = json.loads((run_dir / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_variant"] == "frozen"
    assert summary["embedding_provider"] == "litellm"
    assert summary["embedding_model"] == "local:BAAI/bge-large-en-v1.5"
    assert summary["embedding_dimensions_observed"] == 16
    assert summary["train"]["n"] == 1
    assert summary["eval"]["n"] == 1
    assert (run_dir / "reports" / "manifest.json").exists()


def test_reasoningbank_selection_json_skips_short_solve(
    monkeypatch,
    toy_dataset_root: Path,
    tmp_path: Path,
) -> None:
    from rho.selection import embedder as embedder_mod
    from rho.selection import llm_client as llm_mod

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"selected_task_ids": ["task_001"], "selector": "difficulty"}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(cli, "_build_agent", lambda args, run_dir: make_fake_agent("good"))
    monkeypatch.setattr(
        embedder_mod,
        "LiteLLMEmbedder",
        lambda *args, **kwargs: embedder_mod.FakeEmbedder(dim=16),
    )
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda *args, **kwargs: embedder_mod.FakeEmbedder(dim=16),
    )
    monkeypatch.setattr(
        llm_mod,
        "LiteLLMClient",
        lambda *args, **kwargs: llm_mod.FakeLLMClient(
            lambda prompt, model: "success"
            if "Did the agent successfully complete the task?" in prompt
            else "remember to inspect the harness"
        ),
    )

    rc = cli.main(
        [
            "reasoningbank",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--run-dir",
            str(run_dir),
            "--selection-json",
            str(selection_path),
            "--max-train-tasks",
            "1",
            "--max-grading-tasks",
            "0",
            "--selector",
            "difficulty",
            "--cache",
            "off",
            "--codex-concurrency",
            "1",
        ]
    )

    assert rc == 0
    assert not (run_dir / "selector_calls" / "short_solve").exists()
