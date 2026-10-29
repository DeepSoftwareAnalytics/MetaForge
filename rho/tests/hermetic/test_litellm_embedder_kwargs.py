from __future__ import annotations

from unittest.mock import patch

from rho.selection import embedder as emb_mod


def _fake_response(n: int, dim: int = 8):
    return {"data": [{"embedding": [float(i + j) for j in range(dim)]} for i in range(n)]}


def test_litellm_embedder_provider_agnostic_default() -> None:
    captured: dict = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return _fake_response(len(kwargs["input"]), dim=8)

    with patch("litellm.embedding", side_effect=fake_embedding):
        e = emb_mod.LiteLLMEmbedder(model="openai/text-embedding-3-large", cache_root=None)
        vecs = e.embed(["a", "b"])

    assert captured["model"] == "openai/text-embedding-3-large"
    assert "api_base" not in captured       # default is provider-agnostic
    assert "api_key" not in captured
    assert vecs.shape == (2, 8)


def test_litellm_embedder_azure_foundry_opt_in() -> None:
    captured: dict = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return _fake_response(len(kwargs["input"]), dim=4)

    with patch.object(
        emb_mod, "azure_foundry_kwargs", return_value={"api_base": "X", "api_key": "JWT"}
    ), patch("litellm.embedding", side_effect=fake_embedding):
        e = emb_mod.LiteLLMEmbedder(
            model="openai/text-embedding-3-large",
            cache_root=None,
            use_azure_foundry=True,
        )
        e.embed(["hello"])

    assert captured["api_base"] == "X"
    assert captured["api_key"] == "JWT"
