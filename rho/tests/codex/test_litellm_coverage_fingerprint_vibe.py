"""End-to-end real-API vibe check: coverage fingerprints capture *problem
shape* (refactor vs doc fix), not surface domain (web auth vs ETL).

Three tasks:
- `async_refactor_auth`: multi-site async migration in a web auth codebase.
- `async_refactor_etl`:  multi-site async migration in a data pipeline.
- `doc_typo`:            one-line README typo fix.

The two refactor tasks share abstract shape (scoped multi-site async
migration preserving semantics) in completely different domains. The doc
fix is a different shape entirely.

Assertions:
1. Both refactor tasks score difficulty >= 6; the doc fix scores <= 3.
2. Cosine similarity of the two refactor fingerprint embeddings is
   meaningfully greater than either refactor's similarity to the doc fix
   — i.e. the kernel clusters by problem shape, not by domain text.
3. Every fingerprint is >= 40 words and contains no banned domain tokens
   copied verbatim from the query.

Hits the Azure OpenAI Foundry (Entra-auth) with the default
judge + text-embedding-3-large. Skipped without an active az login for the Foundry resource.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tests.codex._az_helper import have_azure_foundry_token

from rho.selection import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING,
    TaskJudge,
)
from rho.selection.local_embedder import LocalEmbedder
from rho.selection.llm_client import LiteLLMClient

pytestmark = [
    pytest.mark.codex,
    pytest.mark.skipif(
        not have_azure_foundry_token(),
        reason="needs `az login` for the Foundry resource (cognitiveservices audience)",
    ),
]

_BANNED_DOMAIN_TOKENS = {
    # Proper nouns & framework/product names: zero-tolerance — these are
    # never legitimate abstract vocabulary.
    "Flask",
    "Django",
    "FastAPI",
    # File formats and domain-specific acronyms from the query surface.
    "CSV",
    "ETL",
    # Auth-specific surface nouns. ("session" alone is allowed — the
    # judge legitimately uses "shared session state" as generic
    # concurrency vocabulary. Compound cookies/auth/middleware stays
    # banned because those are codebase-identifying.)
    "cookie",
    "middleware",
    # Doc-fix surface tokens.
    "README",
    "recieve",
}


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


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def test_coverage_fingerprint_clusters_by_problem_shape(tmp_path: Path) -> None:
    auth_refactor = _QueryTask(
        "async_refactor_auth",
        "Migrate the synchronous authentication middleware to async/await "
        "across every request handler in this web service. The existing sync "
        "checker is called from 30+ route sites; every caller must be "
        "updated together and session-cookie semantics must be preserved so "
        "existing clients are not broken. All current integration tests must "
        "keep passing, with no behavioral regression under concurrent "
        "requests.",
    )
    etl_refactor = _QueryTask(
        "async_refactor_etl",
        "Convert the row-by-row CSV ingestion pipeline into a batched async "
        "streaming pipeline. The pipeline has 20+ consumer sites each pulling "
        "one record at a time; every consumer must be updated to accept a "
        "batch iterator while preserving exactly-once delivery semantics for "
        "downstream sinks. Throughput must improve without introducing "
        "duplicates or reordering under failure-retry, and existing "
        "end-to-end tests must continue passing.",
    )
    doc_fix = _QueryTask(
        "doc_typo",
        "Fix a typo in the README file: 'recieve' should be 'receive'. No "
        "code changes.",
    )
    pool = [auth_refactor, etl_refactor, doc_fix]

    workdir = tmp_path / "selector_calls"
    judge = TaskJudge(
        client=LiteLLMClient(),
        model=DEFAULT_JUDGE_MODEL,
        workdir=workdir,
        reasoning_effort=DEFAULT_JUDGE_REASONING,
        cache_root=tmp_path / "judge_cache",
    )

    results = {task.id: judge.judge(task) for task in pool}
    for task_id, result in results.items():
        assert _word_count(result.fingerprint) >= 40, (
            f"Fingerprint for {task_id!r} too short "
            f"({_word_count(result.fingerprint)} words): {result.fingerprint!r}"
        )
        lower = result.fingerprint.lower()
        hits = {t for t in _BANNED_DOMAIN_TOKENS if t.lower() in lower}
        assert not hits, (
            f"Fingerprint for {task_id!r} leaked surface-domain tokens "
            f"{sorted(hits)}. Fingerprint: {result.fingerprint!r}"
        )

    easy = results["doc_typo"].difficulty
    hard_a = results["async_refactor_auth"].difficulty
    hard_b = results["async_refactor_etl"].difficulty
    assert easy <= 3.0, f"doc_typo scored {easy}, expected <= 3"
    assert hard_a >= 6.0, f"async_refactor_auth scored {hard_a}, expected >= 6"
    assert hard_b >= 6.0, f"async_refactor_etl scored {hard_b}, expected >= 6"

    embedder = LocalEmbedder(
        model=DEFAULT_EMBEDDING_MODEL.removeprefix("local:"),
        cache_root=tmp_path / "embed_cache",
    )
    vecs = embedder.embed([results[t.id].fingerprint for t in pool])
    sim = vecs @ vecs.T
    idx = {t.id: i for i, t in enumerate(pool)}
    sim_refactors = float(sim[idx["async_refactor_auth"], idx["async_refactor_etl"]])
    sim_auth_doc = float(sim[idx["async_refactor_auth"], idx["doc_typo"]])
    sim_etl_doc = float(sim[idx["async_refactor_etl"], idx["doc_typo"]])

    # The two refactors should cluster against the doc fix. Require a
    # clear margin (0.05) so a near-tie doesn't falsely pass: if the
    # kernel is still dominated by surface domain, the two refactors will
    # not be meaningfully closer to each other than to the doc fix.
    assert sim_refactors > sim_auth_doc + 0.05, (
        f"Refactor-refactor similarity ({sim_refactors:.3f}) should exceed "
        f"auth↔doc similarity ({sim_auth_doc:.3f}) by >= 0.05"
    )
    assert sim_refactors > sim_etl_doc + 0.05, (
        f"Refactor-refactor similarity ({sim_refactors:.3f}) should exceed "
        f"etl↔doc similarity ({sim_etl_doc:.3f}) by >= 0.05"
    )

    # Persist a small human-readable snapshot so post-hoc inspection is easy.
    snapshot = tmp_path / "vibe_snapshot.txt"
    lines = [
        "Coverage fingerprint vibe check",
        "================================",
        "",
        f"sim(refactor_auth, refactor_etl) = {sim_refactors:.3f}",
        f"sim(refactor_auth, doc_typo)     = {sim_auth_doc:.3f}",
        f"sim(refactor_etl, doc_typo)      = {sim_etl_doc:.3f}",
        "",
        "Difficulty scores:",
    ]
    for task_id, result in results.items():
        lines.append(f"  {task_id:24s}  difficulty={result.difficulty:.1f}")
    lines.append("")
    lines.append("Fingerprints:")
    for task_id, result in results.items():
        lines.append(f"\n--- {task_id} ---")
        lines.append(result.fingerprint)
    snapshot.write_text("\n".join(lines), encoding="utf-8")
    print(snapshot.read_text(encoding="utf-8"))
    assert np.isfinite([sim_refactors, sim_auth_doc, sim_etl_doc]).all()
