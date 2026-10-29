from __future__ import annotations

import json

from rho.reasoningbank.store import (
    ReasoningMemoryEntry,
    ReasoningMemoryStore,
    ReasoningStatus,
    split_memory_items,
)


def test_split_memory_items_matches_official_double_newline_split() -> None:
    assert split_memory_items("# One\n\n# Two\n\n") == ["# One", "# Two", ""]


def test_memory_store_writes_official_jsonl_shape(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = ReasoningMemoryStore(path)
    entry = ReasoningMemoryEntry(
        task_id="task_a",
        query="What happened?",
        memory_items=["item 1", "item 2"],
        status=ReasoningStatus.SUCCESS,
    )

    store.append(entry)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {
        "task_id": "task_a",
        "query": "What happened?",
        "memory_items": ["item 1", "item 2"],
        "status": "success",
    }
    assert store.load() == [entry]


def test_memory_store_missing_file_loads_empty(tmp_path) -> None:
    assert ReasoningMemoryStore(tmp_path / "missing.jsonl").load() == []
