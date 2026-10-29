from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rho.selection.embedder import LiteLLMEmbedder
from tests.codex._az_helper import have_remote_embedder_credential


pytestmark = [
    pytest.mark.codex,
    pytest.mark.skipif(
        not have_remote_embedder_credential(),
        reason="LiteLLMEmbedder real-API test needs an embedder credential — set "
               "OPENAI_API_KEY or OPENROUTER_API_KEY (East US 2 Foundry has no embedding deployment)",
    ),
]


def test_litellm_embedder_shape_and_norm(tmp_path: Path) -> None:
    # Fresh tmp cache so a warm repo-level data/cache can't short-circuit
    # this real-API test.
    emb = LiteLLMEmbedder(
        model="openai/text-embedding-3-large",
        cache_root=tmp_path / "cache",
    )
    vecs = emb.embed([
        "fix a parser bug",
        "add a CLI flag",
        "refactor tests",
    ])
    assert vecs.shape[0] == 3
    assert vecs.shape[1] > 0
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


def test_litellm_embedder_semantic_similarity(tmp_path: Path) -> None:
    emb = LiteLLMEmbedder(
        model="openai/text-embedding-3-large",
        cache_root=tmp_path / "cache",
    )
    vecs = emb.embed([
        "fix a parser bug",
        "parser has a bug, please fix it",
        "cooking recipes",
    ])
    sim01 = float(vecs[0] @ vecs[1])
    sim02 = float(vecs[0] @ vecs[2])
    assert sim01 > sim02
