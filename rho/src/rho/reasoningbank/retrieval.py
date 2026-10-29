from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from rho.reasoningbank.store import ReasoningMemoryEntry

logger = logging.getLogger(__name__)

REASONINGBANK_RETRIEVAL_TASK = (
    "Given the prior software engineering queries, your task is to analyze a "
    "current query's intent and select relevant prior queries that could help "
    "resolve it."
)


@runtime_checkable
class ReasoningBankEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of embeddings."""


@dataclass(frozen=True)
class CachedEmbeddingRecord:
    id: str
    text: str
    embedding: np.ndarray


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


class CachedEmbeddingStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[CachedEmbeddingRecord]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            return []

        records: list[CachedEmbeddingRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            vec = _l2_normalize_1d(np.array(payload["embedding"], dtype=np.float32))
            records.append(
                CachedEmbeddingRecord(
                    id=str(payload["id"]),
                    text=str(payload.get("text", "")),
                    embedding=vec,
                )
            )
        return records

    def append(self, task_id: str, text: str, embedding: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        record = {
            "id": task_id,
            "text": text,
            "embedding": vec.tolist(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class GeminiReasoningBankEmbedder:
    """Official ReasoningBank retrieval embedder.

    The upstream SWE-bench code uses vertexai `gemini-embedding-001`,
    `RETRIEVAL_DOCUMENT`, and 3072 output dimensions.
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        *,
        output_dimensionality: int = 3072,
    ) -> None:
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.output_dimensionality), dtype=np.float32)
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

        if self._model is None:
            self._model = TextEmbeddingModel.from_pretrained(self.model_name)
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in texts]
        response = self._model.get_embeddings(
            inputs,
            output_dimensionality=self.output_dimensionality,
        )
        vecs = np.array([item.values for item in response], dtype=np.float32)
        return _l2_normalize_2d(vecs)


class ReasoningBankRetriever:
    def __init__(
        self,
        *,
        embedder: ReasoningBankEmbedder,
        cache: CachedEmbeddingStore,
        trace_dir: Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.cache = cache
        self.trace_dir = trace_dir

    def select_memory(
        self,
        memory_entries: list[ReasoningMemoryEntry],
        *,
        cur_query: str,
        task_id: str,
        n: int = 1,
        append_current: bool = True,
    ) -> list[ReasoningMemoryEntry]:
        if n <= 0:
            raise ValueError("n must be positive")
        if n > 10:
            logger.warning("ReasoningBank retrieval requested n=%s; upstream caps at 10", n)

        cached = self.cache.load()
        if append_current:
            current_embedding = self.embedder.embed([cur_query])[0]
            self.cache.append(task_id, cur_query, current_embedding)

        if not cached:
            logger.warning("No cached ReasoningBank embeddings found at %s", self.cache.path)
            self._write_trace(
                task_id=task_id,
                cur_query=cur_query,
                append_current=append_current,
                ranked=[],
                selected_task_ids=[],
            )
            return []

        instruct_query = get_detailed_instruct(REASONINGBANK_RETRIEVAL_TASK, cur_query)
        query_vec = _l2_normalize_1d(self.embedder.embed([instruct_query])[0])
        matrix = _l2_normalize_2d(np.stack([record.embedding for record in cached]))
        scores = (matrix @ query_vec) * 100.0
        ranked = sorted(
            zip(cached, scores.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )

        entry_by_task_id: dict[str, ReasoningMemoryEntry] = {}
        for entry in memory_entries:
            entry_by_task_id.setdefault(entry.task_id, entry)

        selected: list[ReasoningMemoryEntry] = []
        for record, _score in ranked[:n]:
            entry = entry_by_task_id.get(record.id)
            if entry is not None:
                selected.append(entry)
        self._write_trace(
            task_id=task_id,
            cur_query=cur_query,
            append_current=append_current,
            ranked=[
                {"task_id": record.id, "score": float(score)}
                for record, score in ranked
            ],
            selected_task_ids=[entry.task_id for entry in selected],
        )
        return selected

    def _write_trace(
        self,
        *,
        task_id: str,
        cur_query: str,
        append_current: bool,
        ranked: list[dict[str, object]],
        selected_task_ids: list[str],
    ) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "cur_query": cur_query,
            "append_current": append_current,
            "ranked": ranked,
            "selected_task_ids": selected_task_ids,
        }
        path = self.trace_dir / f"{task_id.replace('/', '__')}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _l2_normalize_1d(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return (arr / (norm + 1e-12)).astype(np.float32)


def _l2_normalize_2d(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / (norms + 1e-12)).astype(np.float32)
