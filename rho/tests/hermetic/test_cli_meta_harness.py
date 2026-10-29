from __future__ import annotations

import json
from pathlib import Path

from rho import cli
from tests.helpers import make_meta_harness_fake_agent


def test_meta_harness_cli_smoke(monkeypatch, toy_dataset_root: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        cli, "_build_agent", lambda args, run_dir: make_meta_harness_fake_agent()
    )
    rc = cli.main(
        [
            "meta-harness",
            "--dataset",
            f"directory:{toy_dataset_root}",
            "--run-dir",
            str(run_dir),
            "--iterations",
            "1",
            "--candidates-per-iter",
            "1",
            "--search-trials",
            "1",
            "--max-search-tasks",
            "2",
            "--max-test-tasks",
            "0",
        ]
    )
    assert rc == 0

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"] == "meta-harness"
    # Phase 0 seed + 1 iteration x 1 candidate = 2 evaluated harnesses.
    assert summary["n_candidates"] == 2
    assert summary["test_pass_rate"] is None  # --max-test-tasks 0 skips test eval

    rows = (run_meta := run_dir / "meta_harness" / "summary.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(rows) == 2
    assert (run_dir / "config.json").exists()
    assert (run_dir / "meta_harness" / "frontier.json").exists()
    del run_meta
