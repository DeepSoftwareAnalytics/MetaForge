from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rho.protocols import Task
from rho.selection.difficulty_selector import JudgeResult, TaskJudge
from rho.selection.embedder import TaskEmbedder


def fast_greedy_map(
    L: np.ndarray,
    k: int,
    *,
    eps: float = 1e-10,
) -> tuple[list[int], list[float]]:
    """Chen 2018 Algorithm 1: greedy MAP for a DPP with kernel L.

    Returns (picks, log_det_gains):
      - picks[t] is the index picked at step t (0-indexed)
      - log_det_gains[t] = log det(L_R_{t+1}) - log det(L_R_t)

    Stops early if all remaining d^2 <= eps (no candidate adds meaningful
    volume), which happens when k exceeds the effective rank of L.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")
    M = L.shape[0]
    if k <= 0:
        return [], []
    k = min(k, M)

    d2 = np.array(np.diag(L), dtype=np.float64)
    c: list[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(M)]
    picked_mask = np.zeros(M, dtype=bool)

    first = int(np.argmax(d2))
    if d2[first] <= eps:
        return [], []
    picks: list[int] = [first]
    gains: list[float] = [math.log(d2[first])]
    picked_mask[first] = True

    while len(picks) < k:
        j = picks[-1]
        dj = math.sqrt(max(d2[j], 0.0))
        if dj <= 0.0:
            break
        cj = c[j]
        unpicked_ix = np.flatnonzero(~picked_mask)
        if cj.size > 0:
            C = np.stack([c[i] for i in unpicked_ix], axis=0)
            e_vec = (L[j, unpicked_ix] - C @ cj) / dj
        else:
            e_vec = L[j, unpicked_ix] / dj

        for local_i, i in enumerate(unpicked_ix):
            c[i] = np.append(c[i], e_vec[local_i])
            d2[i] = max(d2[i] - e_vec[local_i] ** 2, 0.0)

        d2_masked = np.where(picked_mask, -np.inf, d2)
        nxt = int(np.argmax(d2_masked))
        if d2_masked[nxt] <= eps:
            break
        picks.append(nxt)
        gains.append(math.log(d2_masked[nxt]))
        picked_mask[nxt] = True

    return picks, gains


class DPPSelector:
    """DPP-based task selector (Chen 2018 fast greedy MAP).

    Composes a TaskJudge (per-task difficulty + abstract fingerprint)
    and a TaskEmbedder. Kernel L = Diag(r) S Diag(r) with S = X X^T
    cosine similarity over **fingerprint** embeddings (not raw queries —
    raw queries were shown to cluster by repo identity on swebench-pro;
    see docs/superpowers/specs/2026-04-18-rationale-coverage-selector-design.md).
    θ ∈ [0, 1] controls the difficulty/diversity tradeoff via
    α = θ / (2 · max(1 - theta, EPS)); r_i ← max(score/10, floor) ** α.

    The max(1 - theta, EPS) clamp (EPS = 1e-6) keeps α finite at θ → 1, so
    the kernel degenerates smoothly to lexicographic-by-score without a
    code-path branch — this preserves continuous dpp_trace semantics
    across the spec §5 θ-sweep. θ = 0 → α = 0 → all r_i = 1 → pure
    diversity (plain facility-of-volume on S).

    When `fast_greedy_map` returns fewer than k picks (the kernel's
    effective rank collapsed — every remaining candidate's Schur
    complement is ≤ eps, i.e. it adds no volume), `select` returns
    exactly what greedy produced. We deliberately do NOT pad with
    highest-score fallbacks: that would create a silent divergence
    between selection.json and dpp_trace.json and mask the rank-collapse
    signal that callers actually want to see in a research setting.
    """

    def __init__(
        self,
        *,
        judge: TaskJudge,
        embedder: TaskEmbedder,
        theta: float = 0.7,
        score_floor: float = 0.1,
        workdir: Path | None = None,
    ) -> None:
        if not (0.0 <= theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        if not (0.0 < score_floor <= 1.0):
            raise ValueError(f"score_floor must be in (0, 1], got {score_floor}")
        self._judge = judge
        self._embedder = embedder
        self._theta = theta
        self._score_floor = score_floor
        self._workdir = workdir

    @property
    def theta(self) -> float:
        return self._theta

    def results(self) -> dict[str, JudgeResult]:
        return self._judge.results()

    def select(
        self,
        candidates: list[Task],
        k: int | None,
        *,
        seed: int | None = None,
    ) -> list[Task]:
        del seed
        if k is None or k >= len(candidates):
            target_k = len(candidates)
        else:
            target_k = k

        judged_map = self._judge.judge_many(candidates)
        judged = [judged_map[t.id] for t in candidates]
        raw_scores = [r.difficulty for r in judged]
        fingerprints = {t.id: r.fingerprint for t, r in zip(candidates, judged)}
        texts = [r.fingerprint for r in judged]
        vecs = self._embedder.embed(texts)

        floored = np.array(
            [max(s / 10.0, self._score_floor) for s in raw_scores],
            dtype=np.float64,
        )
        S = (vecs @ vecs.T).astype(np.float64)
        if self._theta == 1.0:
            # α = 5e5 at the clamp boundary; r_raw ** α underflows to exact
            # zero for every non-max-score item and the kernel rank collapses
            # to 1, so greedy returns a single pick. Pure-difficulty is the
            # semantic we want here anyway, so substitute a diagonal L to get
            # top-k by score directly. Spec §5 ablation uses θ ≤ 0.9, where
            # the continuous formula below runs without underflow.
            r = floored / float(floored.max())
            L = np.diag(r**2)
        else:
            alpha = self._theta / (2.0 * max(1.0 - self._theta, 1e-6))
            r_raw = floored / float(floored.max())
            r = r_raw**alpha
            L = (r[:, None] * S) * r[None, :]

        picks_ix, gains = fast_greedy_map(L, target_k)

        trace = [
            {
                "step": step,
                "picked_ix": int(i),
                "picked_id": candidates[i].id,
                "log_det_gain": float(gain),
                "score": float(raw_scores[i]),
            }
            for step, (i, gain) in enumerate(zip(picks_ix, gains))
        ]

        if self._workdir is not None:
            eigvals = np.linalg.eigvalsh(L)
            self._persist(candidates, fingerprints, vecs, S, eigvals, trace)

        return [candidates[i] for i in picks_ix]

    def _persist(
        self,
        candidates: list[Task],
        fingerprints: dict[str, str],
        vecs: np.ndarray,
        sim: np.ndarray,
        eigvals: np.ndarray,
        trace: list[dict],
    ) -> None:
        import json

        assert self._workdir is not None
        self._workdir.mkdir(parents=True, exist_ok=True)
        np.save(self._workdir / "embeddings.npy", vecs)
        np.save(self._workdir / "similarity.npy", sim)
        np.save(self._workdir / "dpp_kernel_eigvals.npy", eigvals)
        (self._workdir / "candidate_ids.json").write_text(
            json.dumps([t.id for t in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._workdir / "fingerprints.json").write_text(
            json.dumps(fingerprints, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._workdir / "dpp_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
