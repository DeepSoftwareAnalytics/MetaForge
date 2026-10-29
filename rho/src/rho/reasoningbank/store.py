from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ReasoningStatus(str, Enum):
    SUCCESS = "success"
    FAIL = "fail"


@dataclass(frozen=True)
class ReasoningMemoryEntry:
    task_id: str
    query: str
    memory_items: list[str]
    status: ReasoningStatus

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ReasoningMemoryEntry":
        return cls(
            task_id=str(payload["task_id"]),
            query=str(payload["query"]),
            memory_items=[str(item) for item in payload.get("memory_items", [])],
            status=ReasoningStatus(str(payload["status"])),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "memory_items": list(self.memory_items),
            "status": self.status.value,
        }


class ReasoningMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[ReasoningMemoryEntry]:
        if not self.path.exists():
            return []
        entries: list[ReasoningMemoryEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(ReasoningMemoryEntry.from_json(json.loads(line)))
        return entries

    def append(self, entry: ReasoningMemoryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


def split_memory_items(text: str) -> list[str]:
    return text.split("\n\n")
