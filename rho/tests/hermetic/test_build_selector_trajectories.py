from __future__ import annotations

from pathlib import Path

import pytest

from rho.selection import (
    DEFAULT_JUDGE_REASONING, RandomSelector, build_selector,
)


def test_default_judge_reasoning_is_high() -> None:
    assert DEFAULT_JUDGE_REASONING == "high"


def test_build_selector_random_does_not_require_trajectories(tmp_path: Path) -> None:
    sel = build_selector("random", workdir=tmp_path)
    assert isinstance(sel, RandomSelector)


def test_build_selector_judge_consuming_requires_trajectories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trajectories"):
        build_selector("difficulty", workdir=tmp_path)
    with pytest.raises(ValueError, match="trajectories"):
        build_selector("coverage", workdir=tmp_path)
    with pytest.raises(ValueError, match="trajectories"):
        build_selector("dpp", workdir=tmp_path)
