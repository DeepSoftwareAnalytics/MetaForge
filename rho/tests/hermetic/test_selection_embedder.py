from __future__ import annotations

import numpy as np

from rho.selection.embedder import FakeEmbedder, LiteLLMEmbedder


def test_fake_embedder_returns_matrix_of_correct_shape() -> None:
    vecs = FakeEmbedder(dim=8).embed(["hello world", "another task", "third one"])
    assert vecs.shape == (3, 8)


def test_fake_embedder_deterministic_same_text_same_vector() -> None:
    emb = FakeEmbedder(dim=16)
    np.testing.assert_allclose(emb.embed(["foo", "bar"]), emb.embed(["foo", "bar"]))


def test_fake_embedder_distinct_text_distinct_vector() -> None:
    vecs = FakeEmbedder(dim=16).embed(["alpha", "beta"])
    assert not np.allclose(vecs[0], vecs[1])


def test_fake_embedder_vectors_are_unit_normalized() -> None:
    vecs = FakeEmbedder(dim=32).embed(["one", "two", "three"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)


def test_litellm_embedder_chunks_large_inputs(monkeypatch) -> None:
    """A 200-input request must be split into multiple API calls so we don't
    blow past Azure's TPM cap on a single embedding request."""
    calls: list[int] = []

    def fake_embedding(*, model, input, **kwargs):
        calls.append(len(input))
        # Mimic the OpenAI/Azure response shape used by LiteLLMEmbedder.
        return {
            "data": [
                {"embedding": [float(i + 1)] + [0.0] * 7} for i in range(len(input))
            ]
        }

    import litellm

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    emb = LiteLLMEmbedder(model="openai/text-embedding-3-large", cache_root=None)
    n = 200
    vecs = emb.embed([f"task-{i}" for i in range(n)])
    assert vecs.shape == (n, 8)
    # 200 inputs at chunk size 64 → 4 calls of [64, 64, 64, 8].
    assert calls == [64, 64, 64, 8], calls
