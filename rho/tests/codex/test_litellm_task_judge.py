"""End-to-end real-API smoke test: TaskJudge ranks hard tasks above easy
ones and produces abstract fingerprints of the requested length.

Hits the Azure OpenAI Foundry resource directly via Entra Bearer with
the default judge model (openai/gpt-5.5 + medium reasoning). Skipped
without an active az login for the Foundry resource.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.codex._az_helper import have_azure_foundry_token

from rho.selection import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING,
    DifficultySelector,
    TaskJudge,
)
from rho.selection.llm_client import LiteLLMClient
from tests.hermetic._traj_helper import trajectories_for

_FILE_PATH_RE = re.compile(r"\b[\w./-]+\.(py|js|ts|go|rb|c|h|cpp|java|md|txt)\b")
_IMPORT_RE = re.compile(r"\b(?:from \w+ import|import \w+)\b")

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


def test_task_judge_ranks_hard_above_easy_and_produces_abstract_fingerprints(
    tmp_path: Path,
) -> None:
    easy = _QueryTask(
        _id="easy_typo",
        _query="Fix a typo in the README file: 'recieve' should be 'receive'.",
    )
    hard = _QueryTask(
        _id="hard_migration",
        _query=(
            "Migrate the synchronous authentication middleware to async/await "
            "throughout, while preserving session-cookie semantics across the "
            "Flask, Django, and FastAPI routers. The session store must switch "
            "from SQLite to Redis without breaking existing API contracts, and "
            "all existing integration tests must keep passing."
        ),
    )

    judge = TaskJudge(
        client=LiteLLMClient(),
        model=DEFAULT_JUDGE_MODEL,
        workdir=tmp_path / "selector_calls",
        trajectories=trajectories_for(["easy_typo", "hard_migration"]),
        reasoning_effort=DEFAULT_JUDGE_REASONING,
        # Fresh tmp cache so a warm repo-level data/cache can't let this
        # test pass without actually calling the API.
        cache_root=tmp_path / "cache",
    )
    selector = DifficultySelector(judge=judge)
    ranked = selector.select([easy, hard], k=2)
    results = selector.results()

    assert set(results) == {"easy_typo", "hard_migration"}
    easy_result = results["easy_typo"]
    hard_result = results["hard_migration"]

    # A one-line typo fix must score strictly below a cross-framework async
    # + storage migration. If the gap is <1.0 the judge probably isn't
    # discriminating, so we require a real separation.
    assert hard_result.difficulty > easy_result.difficulty + 1.0, (
        f"Hard task ({hard_result.difficulty}) should substantially outrank "
        f"easy task ({easy_result.difficulty})"
    )
    assert easy_result.difficulty <= 3.0, (
        f"Easy task scored {easy_result.difficulty}, expected <= 3"
    )
    assert hard_result.difficulty >= 6.0, (
        f"Hard task scored {hard_result.difficulty}, expected >= 6"
    )

    # Ranked output reflects the scores.
    assert [t.id for t in ranked] == ["hard_migration", "easy_typo"]

    # Fingerprints are present, long enough, and abstract (no task-specific
    # tokens leaked from the query).
    banned_tokens = {
        "Flask",
        "Django",
        "FastAPI",
        "SQLite",
        "Redis",
        "README",
        "recieve",
        "receive",
    }
    for result in (easy_result, hard_result):
        word_count = len(re.findall(r"\b\w+\b", result.fingerprint))
        assert word_count >= 40, (
            f"Fingerprint for {result.task_id!r} is {word_count} words, "
            f"expected >= 40"
        )
        lower = result.fingerprint.lower()
        hits = {t for t in banned_tokens if t.lower() in lower}
        assert not hits, (
            f"Fingerprint for {result.task_id!r} leaked codebase-specific "
            f"tokens: {sorted(hits)}. Fingerprint was: {result.fingerprint!r}"
        )
        assert not _FILE_PATH_RE.search(result.fingerprint), (
            f"Fingerprint contains file-path tokens: {result.fingerprint!r}"
        )
        assert not _IMPORT_RE.search(result.fingerprint), (
            f"Fingerprint contains import-shaped tokens: {result.fingerprint!r}"
        )
        assert result.digest_token_estimate <= 10_000

    # Each call was persisted with the v3 schema.
    for task_id in ("easy_typo", "hard_migration"):
        record = json.loads(
            (tmp_path / "selector_calls" / f"{task_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["task_id"] == task_id
        assert record["model"] == DEFAULT_JUDGE_MODEL
        assert record["schema_version"] == 3
        assert 0.0 <= record["parsed_difficulty"] <= 10.0
        assert len(re.findall(r"\b\w+\b", record["parsed_fingerprint"])) >= 40
        assert record["completion"].strip()
        assert record["digest_token_estimate"] <= 10_000
