from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


RUNS_ROOT = Path("runs")


def require_run(name: str) -> Path:
    run_dir = RUNS_ROOT / name
    if not run_dir.exists():
        pytest.skip(f"fixture run {name!r} is not available")
    return run_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_trajectory_id(run_dir: Path, *, stage: str | None = None, kind: str | None = None) -> str:
    for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
        meta = read_json(meta_path)
        if stage is not None and meta.get("stage") != stage:
            continue
        if kind is not None and meta.get("kind") != kind:
            continue
        return str(meta_path.parent.name)
    pytest.skip(f"no trajectory found in {run_dir.name} for stage={stage!r} kind={kind!r}")
