from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rho.reasoningbank.prompts import CODING_FAILED_SI, CODING_SUCCESSFUL_SI
from rho.reasoningbank.store import split_memory_items
from rho.selection.llm_client import LLMClient

JUDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant that judges whether the agent successfully "
    "completed the task."
)


@dataclass
class ReasoningBankLLM:
    client: LLMClient
    model: str
    judge_reasoning_effort: str | None = None
    extraction_reasoning_effort: str | None = None
    max_tokens: int = 65536
    snapshot_dir: Path | None = None

    def judge_success(self, query: str, trajectory: str, *, task_id: str) -> bool:
        prompt = (
            f"Task: {query}\n\n"
            f"Trajectory:\n{trajectory}\n\n"
            "Did the agent successfully complete the task? "
            "Answer with 'success' or 'fail' only."
        )
        response = self.client.complete(
            prompt,
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            reasoning_effort=self.judge_reasoning_effort,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )
        success = "success" in response.strip().lower()
        verdict = "success" if success else "fail"
        if self.snapshot_dir is not None:
            _write_snapshot(
                self.snapshot_dir,
                task_id,
                "judge.json",
                {
                    "model": self.model,
                    "system_prompt": JUDGE_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "completion": response,
                    "verdict": verdict,
                },
            )
        return success

    def extract_memory_items(
        self,
        *,
        query: str,
        trajectory: str,
        success: bool,
        task_id: str,
    ) -> list[str]:
        prompt = f"**Query:** {query}\n\n**Trajectory:**\n{trajectory}"
        system_prompt = CODING_SUCCESSFUL_SI if success else CODING_FAILED_SI
        response = self.client.complete(
            prompt,
            model=self.model,
            temperature=1.0,
            max_tokens=self.max_tokens,
            reasoning_effort=self.extraction_reasoning_effort,
            system_prompt=system_prompt.strip(),
        )
        memory_items = split_memory_items(response)
        if self.snapshot_dir is not None:
            _write_snapshot(
                self.snapshot_dir,
                task_id,
                "extract.json",
                {
                    "model": self.model,
                    "system_prompt": system_prompt.strip(),
                    "prompt": prompt,
                    "completion": response,
                    "success": success,
                    "memory_items": memory_items,
                },
            )
        return memory_items


def _write_snapshot(
    snapshot_dir: Path,
    task_id: str,
    filename: str,
    payload: dict[str, object],
) -> None:
    task_dir = snapshot_dir / task_id.replace("/", "__")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
