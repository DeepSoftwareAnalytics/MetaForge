from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from rho.protocols import Grade, Harness, Trajectory
from rho.selection import DPPSelector, TaskJudge
from rho.selection.embedder import FakeEmbedder
from rho.selection.llm_client import FakeLLMClient
from tests.hermetic._traj_helper import trajectories_for


@dataclass(frozen=True)
class _QueryTask:
    _id: str
    _query: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        raise NotImplementedError

    def materialize(self, dest: Path) -> None:
        raise NotImplementedError

    def query(self) -> str:
        return self._query

    def grade(
        self, trajectory: Trajectory, *, artifacts_dir: Path | None = None
    ) -> Grade:
        raise NotImplementedError


_SCORE_TABLE = {
    "hard_a": 9.0,
    "hard_b": 8.5,
    "mid_a": 5.0,
    "mid_b": 4.5,
}


def _fingerprint_for(task_id: str) -> str:
    # Min 40 words per TaskJudge parser. Per-task unique text keeps
    # FakeEmbedder (hash-seeded) vectors distinct, so the kernel isn't
    # accidentally rank-1.
    return (
        f"Task {task_id} exhibits a failure mode of partial propagation "
        "across several layers where one branch adopts a revised contract "
        "but a neighboring branch silently keeps older ordering assumptions "
        "and only breaks under a narrow boundary case. Difficulty comes from "
        "scattered invariants and precedence rules that require contextual "
        "tracing rather than a local edit. Technical scope is multi-file "
        "bug fixing with global reasoning about how a small update preserves "
        f"behavior in shape {task_id}."
    )


def _script(prompt: str, model: str) -> str:
    del model
    for task_id, score in _SCORE_TABLE.items():
        if f"__{task_id}__" in prompt:
            return json.dumps(
                {
                    "difficulty": score,
                    "abstract_fingerprint": _fingerprint_for(task_id),
                }
            )
    raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")


def _make_pool() -> list[_QueryTask]:
    return [
        _QueryTask(_id=tid, _query=f"__{tid}__ task text body")
        for tid in ("hard_a", "hard_b", "mid_a", "mid_b")
    ]


def _build_judge(workdir: Path, *, script=_script) -> TaskJudge:
    tasks = _make_pool()
    return TaskJudge(
        client=FakeLLMClient(script),
        model="fake-model",
        workdir=workdir,
        trajectories=trajectories_for([t.id for t in tasks]),
        cache_root=None,
    )


def test_first_pick_is_highest_score(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    sel = DPPSelector(
        judge=judge,
        embedder=FakeEmbedder(dim=32),
        theta=0.7,
        workdir=tmp_path / "dpp",
    )
    picks = sel.select(_make_pool(), k=1)
    assert [t.id for t in picks] == ["hard_a"]


def test_theta_zero_uses_kernel_not_scores(tmp_path: Path) -> None:
    pool = _make_pool()
    embedder = FakeEmbedder(dim=32)

    def _alt_judge(mapping: dict[str, float], *, suffix: str) -> TaskJudge:
        def script(prompt: str, model: str) -> str:
            del model
            for tid, score in mapping.items():
                if f"__{tid}__" in prompt:
                    return json.dumps(
                        {
                            "difficulty": score,
                            "abstract_fingerprint": _fingerprint_for(tid),
                        }
                    )
            raise AssertionError(prompt)

        return TaskJudge(
            client=FakeLLMClient(script),
            model="fake-model",
            workdir=tmp_path / f"calls-{suffix}",
            trajectories=trajectories_for([t.id for t in pool]),
            cache_root=None,
        )

    scores_a = {"hard_a": 9.0, "hard_b": 8.5, "mid_a": 5.0, "mid_b": 4.5}
    scores_b = {"hard_a": 4.5, "hard_b": 5.0, "mid_a": 8.5, "mid_b": 9.0}
    sel_a = DPPSelector(
        judge=_alt_judge(scores_a, suffix="a"),
        embedder=embedder,
        theta=0.0,
        workdir=tmp_path / "dpp-a",
    )
    sel_b = DPPSelector(
        judge=_alt_judge(scores_b, suffix="b"),
        embedder=embedder,
        theta=0.0,
        workdir=tmp_path / "dpp-b",
    )
    picks_a = [t.id for t in sel_a.select(pool, k=3)]
    picks_b = [t.id for t in sel_b.select(pool, k=3)]
    assert picks_a == picks_b, (
        f"θ=0 picks must be invariant under score permutation: {picks_a} vs {picks_b}"
    )


def test_theta_one_degenerates_to_pure_difficulty(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    sel = DPPSelector(
        judge=judge,
        embedder=FakeEmbedder(dim=32),
        theta=1.0,
        workdir=tmp_path / "dpp",
    )
    picks = sel.select(_make_pool(), k=2)
    assert [t.id for t in picks] == ["hard_a", "hard_b"]
    trace = json.loads((tmp_path / "dpp" / "dpp_trace.json").read_text())
    assert len(trace) == 2
    for entry in trace:
        assert isinstance(entry["log_det_gain"], float)
        assert math.isfinite(entry["log_det_gain"])


def test_persists_artifacts(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    workdir = tmp_path / "dpp"
    sel = DPPSelector(
        judge=judge,
        embedder=FakeEmbedder(dim=32),
        theta=0.7,
        workdir=workdir,
    )
    sel.select(_make_pool(), k=3)
    for name in (
        "embeddings.npy",
        "similarity.npy",
        "dpp_kernel_eigvals.npy",
        "candidate_ids.json",
        "fingerprints.json",
        "dpp_trace.json",
    ):
        assert (workdir / name).exists(), f"missing {name}"

    trace = json.loads((workdir / "dpp_trace.json").read_text(encoding="utf-8"))
    assert len(trace) == 3
    for step, entry in enumerate(trace):
        assert entry["step"] == step
        assert "picked_id" in entry
        assert "log_det_gain" in entry
        assert "score" in entry

    fingerprints = json.loads(
        (workdir / "fingerprints.json").read_text(encoding="utf-8")
    )
    assert set(fingerprints.keys()) == {"hard_a", "hard_b", "mid_a", "mid_b"}


def test_results_snapshot(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    sel = DPPSelector(
        judge=judge,
        embedder=FakeEmbedder(dim=32),
        theta=0.7,
        workdir=tmp_path / "dpp",
    )
    sel.select(_make_pool(), k=2)
    snap = sel.results()
    assert snap["hard_a"].difficulty == pytest.approx(9.0)
    assert snap["hard_b"].difficulty == pytest.approx(8.5)
    assert snap["hard_a"].fingerprint  # non-empty


def test_k_none_returns_all(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    sel = DPPSelector(
        judge=judge,
        embedder=FakeEmbedder(dim=32),
        theta=0.7,
        workdir=tmp_path / "dpp",
    )
    pool = _make_pool()
    picks = sel.select(pool, k=None)
    assert {t.id for t in picks} == {t.id for t in pool}


def test_rejects_invalid_theta(tmp_path: Path) -> None:
    judge = _build_judge(tmp_path / "calls")
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            DPPSelector(
                judge=judge,
                embedder=FakeEmbedder(dim=32),
                theta=bad,
                workdir=tmp_path / "dpp",
            )


def test_build_selector_dpp(tmp_path: Path) -> None:
    from rho.selection import SELECTOR_CHOICES, build_selector

    assert "dpp" in SELECTOR_CHOICES
    sel = build_selector(
        "dpp",
        workdir=tmp_path / "calls",
        cache_root=None,
        trajectories=trajectories_for([t.id for t in _make_pool()]),
    )
    assert isinstance(sel, DPPSelector)
