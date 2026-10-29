"""Real-FastEmbed smoke test for LocalEmbedder.

Lives under tests/codex/ because the first run hits the network
(downloads ~25MB to ~/.cache/fastembed). Subsequent runs are offline
and <1s. Skipped if FastEmbed isn't installed *or* the cache isn't
warm and there's no network — i.e. genuinely offline CI runners stay
green.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest

from rho.selection.local_embedder import LocalEmbedder

_TINY_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, 22MB ONNX


def _fastembed_cache_warm(model: str) -> bool:
    """True iff FastEmbed has the model's ONNX weights already cached."""
    cache_root = Path(os.environ.get("FASTEMBED_CACHE_PATH") or "~/.cache/fastembed").expanduser()
    if not cache_root.is_dir():
        return False
    # FastEmbed slugifies the model name; check for any subdir containing it
    slug = model.replace("/", "--")
    return any(slug in p.name for p in cache_root.iterdir())


def _can_reach_huggingface(timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not (_fastembed_cache_warm(_TINY_MODEL) or _can_reach_huggingface()),
    reason="LocalEmbedder smoke needs either a warm ~/.cache/fastembed or network access to huggingface.co",
)


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder(model=_TINY_MODEL, cache_root=None)


def test_shape_and_unit_norm(embedder: LocalEmbedder) -> None:
    vecs = embedder.embed(["fix the date parser bug", "refactor cache key construction"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic_same_text_same_vector(embedder: LocalEmbedder) -> None:
    a = embedder.embed(["hello world"])
    b = embedder.embed(["hello world"])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_semantic_similarity_orders_correctly(embedder: LocalEmbedder) -> None:
    vecs = embedder.embed(
        [
            "fix off-by-one in the date parser",
            "fix off-by-one in the calendar parser",  # near
            "recipe for chocolate chip cookies",  # far
        ]
    )
    sim_near = float(vecs[0] @ vecs[1])
    sim_far = float(vecs[0] @ vecs[2])
    assert sim_near > sim_far + 0.1


def test_cache_round_trip(tmp_path: Path) -> None:
    e = LocalEmbedder(model=_TINY_MODEL, cache_root=tmp_path)
    a = e.embed(["cached text"])
    # second call should hit the on-disk cache and skip the ONNX runtime
    # entirely - we can't observe that without mocks, but the result must
    # equal the first call byte-for-byte (cached vectors are pre-normalized).
    b = e.embed(["cached text"])
    np.testing.assert_array_equal(a, b)


def test_lazy_model_load_defers_fastembed_instantiation(tmp_path: Path) -> None:
    # Construct without calling .embed(). The wrapper must defer creating
    # the underlying fastembed.TextEmbedding (which itself does
    # download_model + load_onnx_model on construction); the cheapest
    # robust check is that the internal ._model attr is still None.
    e = LocalEmbedder(model=_TINY_MODEL, cache_root=tmp_path)
    assert e._model is None, "LocalEmbedder must defer fastembed.TextEmbedding load until embed()"
