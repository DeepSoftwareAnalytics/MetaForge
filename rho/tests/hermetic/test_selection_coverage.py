from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rho.protocols import Grade, Harness, Trajectory
from rho.selection import CoverageSelector, JudgeResult


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

    def grade(self, trajectory: Trajectory, *, artifacts_dir: Path | None = None) -> Grade:
        raise NotImplementedError


class _FakeJudge:
    def __init__(self, fingerprints: dict[str, str]) -> None:
        self._fingerprints = fingerprints

    def judge(self, task: _QueryTask) -> JudgeResult:
        return JudgeResult(
            task_id=task.id,
            difficulty=5.0,
            fingerprint=self._fingerprints[task.id],
        )

    def judge_many(
        self, tasks: list[_QueryTask], *, max_workers: int = 16
    ) -> dict[str, JudgeResult]:
        del max_workers
        return {task.id: self.judge(task) for task in tasks}


class _FakeEmbedder:
    def __init__(self, vecs_by_text: dict[str, np.ndarray]) -> None:
        self._vecs_by_text = vecs_by_text
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([self._vecs_by_text[text] for text in texts]).astype(np.float32)


def _pool() -> list[_QueryTask]:
    return [
        _QueryTask(_id="a", _query="query a"),
        _QueryTask(_id="b", _query="query b"),
        _QueryTask(_id="c", _query="query c"),
        _QueryTask(_id="d", _query="query d"),
    ]


def test_coverage_selector_uses_fingerprints_for_greedy_order_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    fingerprints = {
        "a": "fingerprint a",
        "b": "fingerprint b",
        "c": "fingerprint c",
        "d": "fingerprint d",
    }
    vecs = {
        "fingerprint a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "fingerprint b": np.array([0.9938837, 0.11043153, 0.0, 0.0], dtype=np.float32),
        "fingerprint c": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "fingerprint d": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    embedder = _FakeEmbedder(vecs)
    selector = CoverageSelector(
        judge=_FakeJudge(fingerprints),
        embedder=embedder,
        workdir=tmp_path / "coverage",
    )

    picks = selector.select(_pool(), k=3, seed=0)

    assert [task.id for task in picks] == ["b", "d", "c"]
    assert embedder.calls == [[
        "fingerprint a",
        "fingerprint b",
        "fingerprint c",
        "fingerprint d",
    ]]

    workdir = tmp_path / "coverage"
    assert json.loads((workdir / "fingerprints.json").read_text(encoding="utf-8")) == fingerprints
    for name in ("embeddings.npy", "similarity.npy", "candidate_ids.json", "gain_trace.json"):
        assert (workdir / name).exists(), f"missing {name}"


def test_coverage_selector_k_none_returns_all_tasks(tmp_path: Path) -> None:
    fingerprints = {task.id: f"fingerprint {task.id}" for task in _pool()}
    vecs = {
        text: np.eye(4, dtype=np.float32)[i]
        for i, text in enumerate(fingerprints.values())
    }
    selector = CoverageSelector(
        judge=_FakeJudge(fingerprints),
        embedder=_FakeEmbedder(vecs),
        workdir=tmp_path / "coverage",
    )

    picks = selector.select(_pool(), k=None, seed=0)

    assert {task.id for task in picks} == {"a", "b", "c", "d"}


def test_coverage_selector_k_larger_than_pool_returns_all_tasks(tmp_path: Path) -> None:
    pool = _pool()
    fingerprints = {task.id: f"fingerprint {task.id}" for task in pool}
    vecs = {
        text: np.eye(4, dtype=np.float32)[i]
        for i, text in enumerate(fingerprints.values())
    }
    selector = CoverageSelector(
        judge=_FakeJudge(fingerprints),
        embedder=_FakeEmbedder(vecs),
        workdir=tmp_path / "coverage",
    )

    picks = selector.select(pool, k=10, seed=0)

    assert {task.id for task in picks} == {"a", "b", "c", "d"}
