from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from rho.selection.azure_auth import azure_foundry_kwargs
from rho.selection.cache import DEFAULT_CACHE_ROOT, EmbeddingCache


@runtime_checkable
class TaskEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of unit-normalized embeddings."""


class FakeEmbedder:
    """Deterministic hash-seeded embedder for hermetic tests.

    Each text seeds a numpy RNG via SHA-256; identical texts give identical
    vectors; distinct texts give (with overwhelming probability) distinct
    vectors. Output is L2-normalized so cosine similarity equals dot product.
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            vecs[i] = v
        return vecs


class LiteLLMEmbedder:
    """Real embedder backed by ``litellm.embedding()``. L2-normalized output.

    By default this is provider-agnostic: litellm picks the backend from the
    model prefix (e.g. ``openai/…``, ``openrouter/…``) and reads the API key
    from the corresponding env var. Pass ``use_azure_foundry=True`` to route
    through the Azure OpenAI Foundry resource using an Entra Bearer instead.

    Note: as of 2026-05, the Azure OpenAI Foundry resource has no
    embedding deployment. Use ``LocalEmbedder`` (the new default) unless you
    have a remote API account.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-large",
        *,
        cache_root: Path | None = DEFAULT_CACHE_ROOT,
        use_azure_foundry: bool = False,
    ) -> None:
        self._model = model
        self._use_azure_foundry = use_azure_foundry
        self._cache = (
            EmbeddingCache(cache_root, model) if cache_root is not None else None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        results: list[np.ndarray | None] = [None] * n
        miss_ix: list[int] = []
        miss_texts: list[str] = []

        if self._cache is not None:
            for i, text in enumerate(texts):
                hit = self._cache.get(text)
                if hit is not None:
                    results[i] = hit
                else:
                    miss_ix.append(i)
                    miss_texts.append(text)
        else:
            miss_ix = list(range(n))
            miss_texts = list(texts)

        if miss_texts:
            fresh = self._embed_batch(miss_texts)
            for local_i, global_i in enumerate(miss_ix):
                vec = fresh[local_i]
                results[global_i] = vec
                if self._cache is not None:
                    self._cache.put(texts[global_i], vec)

        vecs = np.stack(
            [r if r is not None else np.zeros(0, dtype=np.float32) for r in results]
        ).astype(np.float32)
        # Cached vectors are already unit-norm (we normalize before caching);
        # re-normalize defensively so identity is preserved regardless of
        # cache provenance.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / (norms + 1e-12)).astype(np.float32)

    # Chunk size for the embedding API. Azure S0 tier text-embedding-3-large
    # in Sweden Central rate-limits around 350k TPM; a batch of 64 long-form
    # SWE-bench Pro queries (~25-30k tokens) stays comfortably below per-request
    # caps and lets litellm's retry handle the occasional 429. Larger
    # all-in-one batches (e.g. all 585 train tasks at once) trip TPM.
    _BATCH_SIZE = 64

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import litellm

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            chunk = list(texts[start : start + self._BATCH_SIZE])
            kwargs: dict = {
                "model": self._model,
                "input": chunk,
                "num_retries": 5,  # exp-backoff on transient 429 / 5xx
            }
            if self._use_azure_foundry:
                kwargs.update(azure_foundry_kwargs())

            response = litellm.embedding(**kwargs)
            data = response["data"] if isinstance(response, dict) else response.data
            embeddings = [
                (item["embedding"] if isinstance(item, dict) else item.embedding)
                for item in data
            ]
            all_vecs.append(np.array(embeddings, dtype=np.float32))

        vecs = np.concatenate(all_vecs, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / (norms + 1e-12)).astype(np.float32)
