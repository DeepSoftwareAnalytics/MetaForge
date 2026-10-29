"""End-to-end real-API smoke test: CoverageSelector runs over TaskJudge
fingerprints + LiteLLMEmbedder and writes well-formed artifacts.

Hits the Azure OpenAI Foundry resource directly via Entra Bearer with the
default judge model + openai/text-embedding-3-large. Skipped without an
active az login for the Foundry resource.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.codex._az_helper import have_azure_foundry_token

from rho.selection import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING,
    CoverageSelector,
    TaskJudge,
)
from rho.selection.local_embedder import LocalEmbedder
from rho.selection.llm_client import LiteLLMClient
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


def test_coverage_selector_end_to_end_produces_well_formed_artifacts(
    tmp_path: Path,
) -> None:
    # Three tasks spanning distinct problem shapes. We don't assert which
    # ones get picked — that depends on how the judge abstracts the queries,
    # which is exactly the axis we validate end-to-end in runs/20260418-*.
    # Here we only check that the TaskJudge + LiteLLMEmbedder + CoverageSelector
    # pipeline runs without error and produces the documented artifacts.
    pool = [
        _QueryTask(
            "parser_a",
            "Fix a bug in the JSON parser that crashes on trailing commas.",
        ),
        _QueryTask(
            "migration_b",
            "Migrate the synchronous authentication middleware to async, "
            "preserving session-cookie semantics across multiple routers.",
        ),
        _QueryTask(
            "cooking_c",
            "Write a recipe for sourdough bread with hydration and timing notes.",
        ),
    ]

    workdir = tmp_path / "selector_calls"
    judge = TaskJudge(
        client=LiteLLMClient(),
        model=DEFAULT_JUDGE_MODEL,
        workdir=workdir,
        trajectories=trajectories_for([t.id for t in pool]),
        reasoning_effort=DEFAULT_JUDGE_REASONING,
        cache_root=tmp_path / "judge_cache",
    )
    selector = CoverageSelector(
        judge=judge,
        embedder=LocalEmbedder(
            model=DEFAULT_EMBEDDING_MODEL.removeprefix("local:"),
            cache_root=tmp_path / "embed_cache",
        ),
        workdir=workdir,
    )
    picks = selector.select(pool, k=2, seed=0)
    assert all(r.digest_token_estimate <= 10_000 for r in selector.results().values())
    picked_ids = [t.id for t in picks]
    assert len(picked_ids) == 2
    assert set(picked_ids).issubset({t.id for t in pool})

    for filename in (
        "embeddings.npy",
        "similarity.npy",
        "candidate_ids.json",
        "fingerprints.json",
        "gain_trace.json",
    ):
        assert (workdir / filename).exists(), f"missing {filename}"

    candidate_ids = json.loads(
        (workdir / "candidate_ids.json").read_text(encoding="utf-8")
    )
    assert candidate_ids == [t.id for t in pool]

    fingerprints = json.loads(
        (workdir / "fingerprints.json").read_text(encoding="utf-8")
    )
    assert set(fingerprints) == {t.id for t in pool}
    for task_id, fp in fingerprints.items():
        assert isinstance(fp, str) and len(fp.split()) >= 40, (
            f"fingerprint for {task_id!r} too short: {fp!r}"
        )

    gain_trace = json.loads(
        (workdir / "gain_trace.json").read_text(encoding="utf-8")
    )
    assert len(gain_trace) == 2
    assert {entry["picked_id"] for entry in gain_trace} == set(picked_ids)
