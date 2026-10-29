"""Plain content-addressed cache for selector judge calls and embeddings.

Layout:

    data/cache/
        difficulty/<model-slug>__<reasoning>/<content-hash>.json
        embedding/<model-slug>/<content-hash>.npy

Each unique prompt/text gets its own file keyed by sha256 of that exact
content. Changing model or reasoning_effort goes in a different subdir
so stale entries don't collide. No locking needed — each cache
lookup/write touches its own path.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_CACHE_ROOT = Path("data/cache")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _atomic_write(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    try:
        writer(Path(tmp_path))
        os.replace(tmp_path, target)
    except Exception:
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass
        raise


class DifficultyCache:
    def __init__(self, root: Path, model: str, reasoning_effort: str | None) -> None:
        namespace = f"{_slug(model)}__{_slug(reasoning_effort or 'none')}"
        self._dir = root / "difficulty" / namespace

    def get(self, query: str) -> dict | None:
        path = self._dir / f"{_key(query)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, query: str, record: dict) -> None:
        path = self._dir / f"{_key(query)}.json"
        _atomic_write(
            path,
            lambda p: p.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
        )


class EmbeddingCache:
    def __init__(self, root: Path, model: str) -> None:
        self._dir = root / "embedding" / _slug(model)

    def get(self, query: str) -> np.ndarray | None:
        path = self._dir / f"{_key(query)}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None

    def put(self, query: str, vec: np.ndarray) -> None:
        path = self._dir / f"{_key(query)}.npy"

        # np.save auto-appends ".npy" to path-like args that don't already
        # end in it, which breaks our atomic rename. Use a file handle to
        # keep the write on the exact path we hand it.
        def _writer(p: Path) -> None:
            with open(p, "wb") as f:
                np.save(f, vec)

        _atomic_write(path, _writer)
