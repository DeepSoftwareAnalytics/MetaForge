"""Local ONNX-runtime embedder via FastEmbed (Qdrant).

Default model BAAI/bge-large-en-v1.5 (1024-dim) is a strong general-purpose
encoder; first call downloads ~640MB ONNX weights to ~/.cache/fastembed.
Subsequent embeds are CPU-fast (single-core ~50 short texts/s on a recent
x86) and never touch the network. Output is L2-normalized (FastEmbed
returns unit vectors by default; we renormalize defensively).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rho.selection.cache import DEFAULT_CACHE_ROOT, EmbeddingCache


class LocalEmbedder:
    """Implements the ``TaskEmbedder`` Protocol (defined in embedder.py).

    Model is loaded lazily on first ``embed()`` call so that just
    constructing the object doesn't pay the ONNX import cost (matters
    for CLI startup and test collection).
    """

    _BATCH_SIZE = 32  # FastEmbed default; CPU-friendly

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        *,
        cache_root: Path | None = DEFAULT_CACHE_ROOT,
    ) -> None:
        self._model_name = model
        # Namespace cache by the local: prefix so it never collides with
        # remote-API caches keyed by openai/... or azure/...
        self._cache = (
            EmbeddingCache(cache_root, f"local:{model}") if cache_root is not None else None
        )
        self._model = None  # lazy

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

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
            model = self._get_model()
            fresh = np.stack(list(model.embed(miss_texts, batch_size=self._BATCH_SIZE)))
            fresh = fresh.astype(np.float32)
            norms = np.linalg.norm(fresh, axis=1, keepdims=True)
            fresh = fresh / (norms + 1e-12)
            for local_i, global_i in enumerate(miss_ix):
                vec = fresh[local_i]
                results[global_i] = vec
                if self._cache is not None:
                    self._cache.put(texts[global_i], vec)

        return np.stack(results).astype(np.float32)


__all__ = ["LocalEmbedder"]
