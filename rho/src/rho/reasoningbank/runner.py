from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from rho.agent.base import Agent
from rho.observability import annotate_trajectory
from rho.orchestrators.solve import SOLVE_INSTRUCTIONS, solve_workspace
from rho.protocols import Grade, Harness, Task, Trajectory, TrajectoryStore
from rho.reasoningbank.llm import ReasoningBankLLM
from rho.reasoningbank.prompts import MEMORY_INJECTION_PREAMBLE
from rho.reasoningbank.store import (
    ReasoningMemoryEntry,
    ReasoningMemoryStore,
    ReasoningStatus,
)

EvalVariant = Literal["frozen", "online"]
_MAX_TRAJECTORY_TEXT_CHARS = 64_000
_MAX_EVENT_TEXT_CHARS = 4_000
_MAX_COMMAND_OUTPUT_CHARS = 2_000
# Low-level per-line streaming events. Codex emits thousands of these; the
# command output they carry is already captured (bounded) by each
# command_execution item's aggregated_output, so rendering them verbatim
# only floods the judge prompt and crowds out the real trajectory.
_RAW_STREAM_EVENT_TYPES = frozenset({"raw_stderr", "raw_stdout"})


class MemoryRetriever(Protocol):
    def select_memory(
        self,
        memory_entries: list[ReasoningMemoryEntry],
        *,
        cur_query: str,
        task_id: str,
        n: int,
        append_current: bool,
    ) -> list[ReasoningMemoryEntry]: ...


@dataclass(frozen=True)
class ReasoningBankTaskRecord:
    task: Task
    trajectory: Trajectory
    grade: Grade
    stage: str
    selected_memory_task_ids: list[str]
    memory_status: ReasoningStatus | None = None
    memory_item_count: int | None = None


@dataclass(frozen=True)
class ReasoningBankRunResult:
    train_records: list[ReasoningBankTaskRecord]
    eval_records: list[ReasoningBankTaskRecord]
    train_summary: dict[str, float | int]
    eval_summary: dict[str, float | int]


class ReasoningBankRunner:
    def __init__(
        self,
        *,
        agent: Agent,
        memory_llm: ReasoningBankLLM,
        retriever: MemoryRetriever,
        memory_store: ReasoningMemoryStore,
        traj_store: TrajectoryStore,
        workdir: Path,
        harness: Harness,
        memory_n: int = 1,
        eval_variant: EvalVariant = "frozen",
        grade_workers: int = 1,
        solve_workers: int = 1,
        artifacts_root: Path | None = None,
    ) -> None:
        if memory_n <= 0:
            raise ValueError("memory_n must be positive")
        if eval_variant not in ("frozen", "online"):
            raise ValueError("eval_variant must be 'frozen' or 'online'")
        if grade_workers <= 0:
            raise ValueError("grade_workers must be positive")
        if solve_workers <= 0:
            raise ValueError("solve_workers must be positive")
        self.agent = agent
        self.memory_llm = memory_llm
        self.retriever = retriever
        self.memory_store = memory_store
        self.traj_store = traj_store
        self.workdir = workdir
        self.harness = harness
        self.memory_n = memory_n
        self.eval_variant: EvalVariant = eval_variant
        self.grade_workers = grade_workers
        self.solve_workers = solve_workers
        self.artifacts_root = artifacts_root
        self._memory_lock = threading.Lock()
        self._grade_gate = threading.Semaphore(grade_workers)

    def run(
        self,
        *,
        train_tasks: list[Task],
        eval_tasks: list[Task],
    ) -> ReasoningBankRunResult:
        train_records = [
            self._solve_select_judge_extract(
                task,
                stage="reasoningbank_train",
                update_memory=True,
                append_current_embedding=True,
            )
            for task in train_tasks
        ]

        if self.eval_variant == "online":
            eval_records = [
                self._solve_select_judge_extract(
                    task,
                    stage="reasoningbank_online_eval",
                    update_memory=True,
                    append_current_embedding=True,
                )
                for task in eval_tasks
            ]
        else:
            eval_records = self._run_frozen_eval(eval_tasks)

        return ReasoningBankRunResult(
            train_records=train_records,
            eval_records=eval_records,
            train_summary=_summarize(train_records),
            eval_summary=_summarize(eval_records),
        )

    def _run_frozen_eval(self, eval_tasks: list[Task]) -> list[ReasoningBankTaskRecord]:
        if not eval_tasks:
            return []
        if self.solve_workers == 1:
            return [
                self._solve_select_judge_extract(
                    task,
                    stage="reasoningbank_frozen_eval",
                    update_memory=False,
                    append_current_embedding=False,
                )
                for task in eval_tasks
            ]

        workers = min(self.solve_workers, len(eval_tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda task: self._solve_select_judge_extract(
                        task,
                        stage="reasoningbank_frozen_eval",
                        update_memory=False,
                        append_current_embedding=False,
                    ),
                    eval_tasks,
                )
            )

    def _solve_select_judge_extract(
        self,
        task: Task,
        *,
        stage: str,
        update_memory: bool,
        append_current_embedding: bool,
    ) -> ReasoningBankTaskRecord:
        query = task.query()
        with self._memory_lock:
            memory_entries = self.memory_store.load()
            selected = self.retriever.select_memory(
                memory_entries,
                cur_query=query,
                task_id=task.id,
                n=self.memory_n,
                append_current=append_current_embedding,
            )
        selected_memory = _render_selected_memory(selected)
        instructions = _solve_instructions(selected_memory)

        with solve_workspace(task, self.harness, self.workdir) as ws:
            trajectory = self.agent.run(
                ws,
                instructions,
                task_id=task.id,
                harness_id=self.harness.id,
                kind="solve",
                timeout_s=task.agent_timeout_s,
            )
            trajectory = annotate_trajectory(trajectory, agent=self.agent, stage=stage)
            # grade() must run while the workspace is still open: datasets like
            # GAIA-2 validate against the live runtime session that
            # solve_workspace enters, and that session is torn down once this
            # block exits (grading afterwards returns a `no_runtime` error).
            artifacts_dir = _grade_artifacts_dir(
                self.artifacts_root, stage, task, trajectory
            )
            with self._grade_gate:
                grade = task.grade(trajectory, artifacts_dir=artifacts_dir)
        self.traj_store.put(trajectory)

        memory_status: ReasoningStatus | None = None
        memory_item_count: int | None = None
        if update_memory:
            rendered_trajectory = render_trajectory_text(trajectory)
            success = self.memory_llm.judge_success(
                query,
                rendered_trajectory,
                task_id=task.id,
            )
            memory_status = ReasoningStatus.SUCCESS if success else ReasoningStatus.FAIL
            memory_items = self.memory_llm.extract_memory_items(
                query=query,
                trajectory=rendered_trajectory,
                success=success,
                task_id=task.id,
            )
            memory_item_count = len(memory_items)
            with self._memory_lock:
                self.memory_store.append(
                    ReasoningMemoryEntry(
                        task_id=task.id,
                        query=query,
                        memory_items=memory_items,
                        status=memory_status,
                    )
                )

        return ReasoningBankTaskRecord(
            task=task,
            trajectory=trajectory,
            grade=grade,
            stage=stage,
            selected_memory_task_ids=[entry.task_id for entry in selected],
            memory_status=memory_status,
            memory_item_count=memory_item_count,
        )


def _solve_instructions(selected_memory: str) -> str:
    if not selected_memory:
        return SOLVE_INSTRUCTIONS
    return f"{MEMORY_INJECTION_PREAMBLE}\n{selected_memory}\n\n{SOLVE_INSTRUCTIONS}"


def _render_selected_memory(entries: list[ReasoningMemoryEntry]) -> str:
    items = [item for entry in entries for item in entry.memory_items]
    return "\n\n".join(items)


def render_trajectory_text(trajectory: Trajectory) -> str:
    parts: list[str] = []
    for event in trajectory.events:
        if event.get("role") == "system":
            continue
        if event.get("type") in _RAW_STREAM_EVENT_TYPES:
            continue
        text = _event_text(event)
        if text:
            parts.append(text)
    if trajectory.final_message:
        parts.append(f"Final message:\n{trajectory.final_message}")
    return _truncate_middle("\n\n".join(parts), _MAX_TRAJECTORY_TEXT_CHARS)


def _event_text(event: dict) -> str:
    item = event.get("item")
    if isinstance(item, dict):
        rendered = _item_text(item)
        if rendered:
            return rendered
    for key in ("content", "message", "text", "output"):
        value = event.get(key)
        if isinstance(value, str):
            return _truncate_middle(value, _MAX_EVENT_TEXT_CHARS)
    return (
        _truncate_middle(json.dumps(event, ensure_ascii=False, sort_keys=True), _MAX_EVENT_TEXT_CHARS)
        if event
        else ""
    )


def _item_text(item: dict) -> str:
    item_type = item.get("type")
    if item_type == "agent_message":
        text = item.get("text")
        return (
            f"Assistant:\n{_truncate_middle(text, _MAX_EVENT_TEXT_CHARS)}"
            if isinstance(text, str)
            else ""
        )
    if item_type == "command_execution":
        lines: list[str] = []
        command = item.get("command")
        if isinstance(command, str) and command:
            lines.append(f"$ {command}")
        exit_code = item.get("exit_code")
        if exit_code is not None:
            lines.append(f"exit_code={exit_code}")
        output = item.get("aggregated_output")
        if not isinstance(output, str):
            output = item.get("output")
        if isinstance(output, str) and output:
            lines.append(
                "Output:\n" + _truncate_middle(output, _MAX_COMMAND_OUTPUT_CHARS)
            )
        return "\n".join(lines)
    text = item.get("text")
    if isinstance(text, str):
        return _truncate_middle(text, _MAX_EVENT_TEXT_CHARS)
    return _truncate_middle(json.dumps(item, ensure_ascii=False, sort_keys=True), _MAX_EVENT_TEXT_CHARS)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 64:
        return text[:limit]
    omitted = len(text) - limit
    marker = f"\n[truncated {omitted} chars]\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _grade_artifacts_dir(
    artifacts_root: Path | None,
    stage: str,
    task: Task,
    trajectory: Trajectory,
) -> Path | None:
    if artifacts_root is None:
        return None
    safe_task_id = task.id.replace("/", "__")
    return artifacts_root / stage / safe_task_id / trajectory.id


def _summarize(records: list[ReasoningBankTaskRecord]) -> dict[str, float | int]:
    total = len(records)
    mean_score = sum(record.grade.score for record in records) / total if total else 0.0
    return {"n": total, "mean_score": mean_score}
