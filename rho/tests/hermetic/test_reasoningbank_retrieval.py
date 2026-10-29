from __future__ import annotations

import json

import numpy as np

from rho.reasoningbank.retrieval import (
    CachedEmbeddingStore,
    REASONINGBANK_RETRIEVAL_TASK,
    ReasoningBankRetriever,
    get_detailed_instruct,
)
from rho.reasoningbank.store import ReasoningMemoryEntry, ReasoningStatus


class StaticEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            if text == "train-a":
                rows.append([1.0, 0.0])
            elif text == "train-b":
                rows.append([0.0, 1.0])
            elif text == get_detailed_instruct(REASONINGBANK_RETRIEVAL_TASK, "eval-b"):
                rows.append([0.0, 1.0])
            elif text == "eval-b":
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0, 0.0])
        return np.array(rows, dtype=np.float32)


def test_retriever_ranks_cached_memory_without_appending_in_frozen_mode(tmp_path) -> None:
    cache = CachedEmbeddingStore(tmp_path / "embeddings.jsonl")
    cache.append("task_a", "train-a", np.array([1.0, 0.0], dtype=np.float32))
    cache.append("task_b", "train-b", np.array([0.0, 1.0], dtype=np.float32))
    retriever = ReasoningBankRetriever(embedder=StaticEmbedder(), cache=cache)
    memory = [
        ReasoningMemoryEntry("task_a", "train-a", ["memory a"], ReasoningStatus.SUCCESS),
        ReasoningMemoryEntry("task_b", "train-b", ["memory b"], ReasoningStatus.SUCCESS),
    ]

    selected = retriever.select_memory(
        memory,
        cur_query="eval-b",
        task_id="eval_task",
        n=1,
        append_current=False,
    )

    assert [entry.task_id for entry in selected] == ["task_b"]
    lines = (tmp_path / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["task_a", "task_b"]


def test_retriever_appends_current_query_for_online_stream(tmp_path) -> None:
    cache = CachedEmbeddingStore(tmp_path / "embeddings.jsonl")
    retriever = ReasoningBankRetriever(embedder=StaticEmbedder(), cache=cache)

    selected = retriever.select_memory(
        [],
        cur_query="eval-b",
        task_id="eval_task",
        n=1,
        append_current=True,
    )

    assert selected == []
    lines = (tmp_path / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["eval_task"]
