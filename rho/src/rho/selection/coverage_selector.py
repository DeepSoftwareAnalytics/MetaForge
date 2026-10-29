from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from rho.protocols import Task
from rho.selection.difficulty_selector import JudgeResult, TaskJudge
from rho.selection.embedder import TaskEmbedder


class CoverageSelector:
    """Greedy facility location over task fingerprint embeddings.

    Maximizes sum_{t in pool} max_{s in S} sim(t, s). Equivalent to
    Nemhauser-Wolsey greedy on a submodular monotone objective;
    gives (1 - 1/e) approximation. Unlike DPP, rewards representative
    coverage over outlier novelty.

    If `workdir` is provided, persists the computed fingerprints,
    embeddings, similarity matrix, and per-step gain trace there so a
    coverage pick can be audited after the fact.
    """

    def __init__(
        self,
        *,
        judge: TaskJudge,
        embedder: TaskEmbedder,
        workdir: Path | None = None,
    ) -> None:
        self._judge = judge
        self._embedder = embedder
        self._workdir = workdir

    def results(self) -> dict[str, JudgeResult]:
        return self._judge.results()

    def select(
        self,
        candidates: list[Task],
        k: int | None,
        *,
        seed: int | None = None,
    ) -> list[Task]:
        if not candidates:
            return []

        effective_k = len(candidates) if k is None else min(k, len(candidates))
        judged_map = self._judge.judge_many(candidates)
        judged = [judged_map[task.id] for task in candidates]
        fingerprints = {task.id: result.fingerprint for task, result in zip(candidates, judged)}
        texts = [result.fingerprint for result in judged]
        vecs = self._embedder.embed(texts)
        sim = vecs @ vecs.T
        n = len(candidates)

        rng = random.Random(seed)

        row_sums = sim.sum(axis=1)
        max_sum = row_sums.max()
        top = [i for i in range(n) if abs(row_sums[i] - max_sum) < 1e-9]
        first = rng.choice(top) if len(top) > 1 else top[0]
        chosen = [first]
        coverage = sim[first].copy()
        gain_trace: list[dict[str, object]] = [
            {
                "step": 0,
                "picked_ix": first,
                "picked_id": candidates[first].id,
                "gain": float(row_sums[first]),
            }
        ]

        while len(chosen) < effective_k:
            gains = np.maximum(sim - coverage[np.newaxis, :], 0.0).sum(axis=1)
            for j in chosen:
                gains[j] = -np.inf
            best_gain = float(gains.max())
            candidates_ix = [
                i
                for i in range(n)
                if i not in chosen and abs(gains[i] - best_gain) < 1e-9
            ]
            pick = rng.choice(candidates_ix) if len(candidates_ix) > 1 else candidates_ix[0]
            gain_trace.append(
                {
                    "step": len(chosen),
                    "picked_ix": pick,
                    "picked_id": candidates[pick].id,
                    "gain": best_gain,
                }
            )
            chosen.append(pick)
            coverage = np.maximum(coverage, sim[pick])

        if self._workdir is not None:
            self._persist(candidates, fingerprints, vecs, sim, gain_trace)

        return [candidates[i] for i in chosen]

    def _persist(
        self,
        candidates: list[Task],
        fingerprints: dict[str, str],
        vecs: np.ndarray,
        sim: np.ndarray,
        gain_trace: list[dict[str, object]],
    ) -> None:
        assert self._workdir is not None
        self._workdir.mkdir(parents=True, exist_ok=True)
        np.save(self._workdir / "embeddings.npy", vecs)
        np.save(self._workdir / "similarity.npy", sim)
        (self._workdir / "candidate_ids.json").write_text(
            json.dumps([task.id for task in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._workdir / "fingerprints.json").write_text(
            json.dumps(fingerprints, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._workdir / "gain_trace.json").write_text(
            json.dumps(gain_trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
