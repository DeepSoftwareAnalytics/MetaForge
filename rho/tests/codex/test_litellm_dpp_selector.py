"""Real-API smoke test for DPPSelector. Skipped without an active az login for the Foundry resource.

Pool: 4 parser-bug tasks with high difficulty + 1 cooking task with
moderate difficulty. At θ=0.7 (difficulty-leaning), the best pick for
k=2 is one parser task (the hardest) plus either the other hardest
parser or the cooking outlier; the θ=0 pure-diversity case should
always include the outlier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.codex._az_helper import have_azure_foundry_token

from rho.selection import DEFAULT_EMBEDDING_MODEL
from rho.selection.difficulty_selector import TaskJudge
from rho.selection.dpp_selector import DPPSelector
from rho.selection.local_embedder import LocalEmbedder
from rho.selection.llm_client import FakeLLMClient
from tests.hermetic._traj_helper import trajectories_for


pytestmark = [
    pytest.mark.codex,
    pytest.mark.skipif(
        not have_azure_foundry_token(),
        reason="needs `az login` for the Foundry resource (cognitiveservices audience)",
    ),
]


@dataclass(frozen=True)
class _QueryTask:
    _id: str
    _query: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self):
        raise NotImplementedError

    def materialize(self, dest: Path) -> None:
        raise NotImplementedError

    def query(self) -> str:
        return self._query

    def grade(self, trajectory, *, artifacts_dir=None):
        raise NotImplementedError


_SCORES = {
    "parser_a": 9.0,
    "parser_b": 8.5,
    "parser_c": 8.0,
    "parser_d": 7.5,
    "cooking": 6.0,
}


def _fingerprint_for(tid: str) -> str:
    return (
        f"Task {tid} fails under a narrow boundary case when one call site "
        "adopts a revised contract but a neighboring branch keeps older "
        "ordering assumptions, causing partial propagation that only "
        "surfaces on specific inputs. Difficulty is driven by scattered "
        "invariants and precedence rules requiring contextual reasoning "
        "across multiple layers rather than a localized edit. Scope is "
        "multi-file bug fixing with global contract tracing."
    )


def _scripted_scorer(prompt: str, model: str) -> str:
    del model
    for tid, score in _SCORES.items():
        if f"[[ID={tid}]]" in prompt:
            return json.dumps(
                {
                    "difficulty": score,
                    "abstract_fingerprint": _fingerprint_for(tid),
                }
            )
    raise AssertionError(f"unscripted prompt: {prompt[:120]!r}")


def _make_pool() -> list[_QueryTask]:
    return [
        _QueryTask("parser_a", "[[ID=parser_a]] Fix JSON parser crash on trailing commas."),
        _QueryTask(
            "parser_b",
            "[[ID=parser_b]] CSV parser silently truncates embedded-newline rows.",
        ),
        _QueryTask("parser_c", "[[ID=parser_c]] YAML loader raises on anchors."),
        _QueryTask(
            "parser_d",
            "[[ID=parser_d]] Segfault in TOML parser when key starts with digit.",
        ),
        _QueryTask(
            "cooking",
            "[[ID=cooking]] Recipe for sourdough bread, hydration and timing notes.",
        ),
    ]


def _build_selector(tmp_path: Path, theta: float) -> DPPSelector:
    judge = TaskJudge(
        client=FakeLLMClient(_scripted_scorer),
        model="fake-judge",
        workdir=tmp_path / "scores",
        trajectories=trajectories_for([t.id for t in _make_pool()]),
        cache_root=None,
    )
    return DPPSelector(
        judge=judge,
        embedder=LocalEmbedder(
            model=DEFAULT_EMBEDDING_MODEL.removeprefix("local:"),
            cache_root=tmp_path / "cache",
        ),
        theta=theta,
        workdir=tmp_path / "dpp",
    )


def test_dpp_theta_zero_covers_outlier(tmp_path: Path) -> None:
    selector = _build_selector(tmp_path / "t0", theta=0.0)
    picks = selector.select(_make_pool(), k=2)
    picked_ids = {t.id for t in picks}
    assert "cooking" in picked_ids, f"expected outlier at θ=0, got {picked_ids}"


def test_dpp_high_theta_prefers_hardest(tmp_path: Path) -> None:
    selector = _build_selector(tmp_path / "thigh", theta=0.9)
    picks = selector.select(_make_pool(), k=2)
    picked_ids = [t.id for t in picks]
    assert picked_ids[0] == "parser_a"


def test_dpp_emits_artifacts(tmp_path: Path) -> None:
    selector = _build_selector(tmp_path / "art", theta=0.7)
    selector.select(_make_pool(), k=3)
    assert all(r.digest_token_estimate <= 10_000 for r in selector.results().values())
    artdir = tmp_path / "art" / "dpp"
    for name in (
        "embeddings.npy",
        "similarity.npy",
        "dpp_kernel_eigvals.npy",
        "candidate_ids.json",
        "dpp_trace.json",
    ):
        assert (artdir / name).exists(), f"missing {name}"
    trace = json.loads((artdir / "dpp_trace.json").read_text(encoding="utf-8"))
    assert len(trace) == 3
    assert trace[0]["step"] == 0
    assert "log_det_gain" in trace[0]
