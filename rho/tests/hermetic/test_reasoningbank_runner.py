from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rho.agent.fake import FakeResponse
from rho.datasets.directory import DirectoryTask
from rho.protocols import Trajectory, TrajectoryKind
from rho.reasoningbank.llm import ReasoningBankLLM
from rho.reasoningbank.retrieval import CachedEmbeddingStore, ReasoningBankRetriever
from rho.reasoningbank.runner import ReasoningBankRunner, render_trajectory_text
from rho.reasoningbank.store import ReasoningMemoryEntry, ReasoningMemoryStore
from rho.selection.llm_client import FakeLLMClient
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore

MEMORY_ITEM = "team project code name is Phoenix"


class MemoryAwareAgent:
    model = "fake-solver"
    reasoning_effort = "low"

    def __init__(self) -> None:
        self.instructions_by_task: dict[str, str] = {}

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        del output_schema, timeout_s, env
        self.instructions_by_task[task_id] = instructions
        if task_id == "task_001" or MEMORY_ITEM in instructions:
            final = MEMORY_ITEM
        else:
            final = "I don't know project code name"
        response = FakeResponse(final_message=final)
        return Trajectory(
            id=f"traj_{task_id}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=response.events,
            final_message=response.final_message,
            stdout="",
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=0.01,
        )


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def select_memory(
        self,
        memory_entries: list[ReasoningMemoryEntry],
        *,
        cur_query: str,
        task_id: str,
        n: int,
        append_current: bool,
    ) -> list[ReasoningMemoryEntry]:
        del cur_query, n
        self.calls.append((task_id, append_current))
        return list(memory_entries[:1])


class PromptEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rows.append([0.0, 1.0] if "oncall" in text.lower() else [1.0, 0.0])
        return np.array(rows, dtype=np.float32)


def _task(root: Path, harness, split: str, task_id: str) -> DirectoryTask:
    return DirectoryTask(root / split / task_id, harness)


def test_frozen_eval_uses_train_memory_without_val_updates(toy_dataset_root, tmp_path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    train = [_task(toy_dataset_root, harness, "train", "task_001")]
    val = [_task(toy_dataset_root, harness, "val", "task_v01")]
    agent = MemoryAwareAgent()
    retriever = RecordingRetriever()
    client = FakeLLMClient(
        lambda prompt, model: "success"
        if "Did the agent successfully complete the task?" in prompt
        else MEMORY_ITEM
    )
    memory_store = ReasoningMemoryStore(tmp_path / "memory.jsonl")
    runner = ReasoningBankRunner(
        agent=agent,
        memory_llm=ReasoningBankLLM(client=client, model="memory-model"),
        retriever=retriever,
        memory_store=memory_store,
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        harness=harness,
        memory_n=1,
        eval_variant="frozen",
    )

    result = runner.run(train_tasks=train, eval_tasks=val)

    assert result.train_summary == {"n": 1, "mean_score": 1.0}
    assert result.eval_summary == {"n": 1, "mean_score": 1.0}
    assert [entry.task_id for entry in memory_store.load()] == ["task_001"]
    assert retriever.calls == [("task_001", True), ("task_v01", False)]
    assert len(client.calls) == 2
    assert "Below are some memory items" in agent.instructions_by_task["task_v01"]
    assert MEMORY_ITEM in agent.instructions_by_task["task_v01"]
    assert [record.stage for record in result.eval_records] == ["reasoningbank_frozen_eval"]


def test_frozen_eval_writes_train_memory_snapshots_and_retrieval_traces(
    toy_dataset_root,
    tmp_path,
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    train = [
        _task(toy_dataset_root, harness, "train", "task_001"),
        _task(toy_dataset_root, harness, "train", "task_002"),
    ]
    val = [_task(toy_dataset_root, harness, "val", "task_v01")]
    snapshot_dir = tmp_path / "memory_llm"
    trace_dir = tmp_path / "retrieval"
    client = FakeLLMClient(
        lambda prompt, model: "success"
        if "Did the agent successfully complete the task?" in prompt
        else MEMORY_ITEM
    )
    runner = ReasoningBankRunner(
        agent=MemoryAwareAgent(),
        memory_llm=ReasoningBankLLM(
            client=client,
            model="memory-model",
            snapshot_dir=snapshot_dir,
        ),
        retriever=ReasoningBankRetriever(
            embedder=PromptEmbedder(),
            cache=CachedEmbeddingStore(tmp_path / "embeddings.jsonl"),
            trace_dir=trace_dir,
        ),
        memory_store=ReasoningMemoryStore(tmp_path / "memory.jsonl"),
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        harness=harness,
        memory_n=1,
        eval_variant="frozen",
    )

    runner.run(train_tasks=train, eval_tasks=val)

    for task_id in ("task_001", "task_002"):
        judge_snapshot = json.loads(
            (snapshot_dir / task_id / "judge.json").read_text(encoding="utf-8")
        )
        assert set(judge_snapshot) == {
            "model",
            "system_prompt",
            "prompt",
            "completion",
            "verdict",
        }
        assert judge_snapshot["model"] == "memory-model"
        assert judge_snapshot["completion"] == "success"
        assert judge_snapshot["verdict"] == "success"

        extract_snapshot = json.loads(
            (snapshot_dir / task_id / "extract.json").read_text(encoding="utf-8")
        )
        assert set(extract_snapshot) == {
            "model",
            "system_prompt",
            "prompt",
            "completion",
            "success",
            "memory_items",
        }
        assert extract_snapshot["success"] is True
        assert extract_snapshot["memory_items"] == [MEMORY_ITEM]

    assert not (snapshot_dir / "task_v01").exists()

    first_train_trace = json.loads(
        (trace_dir / "task_001.json").read_text(encoding="utf-8")
    )
    assert first_train_trace["append_current"] is True
    assert first_train_trace["ranked"] == []
    assert first_train_trace["selected_task_ids"] == []

    second_train_trace = json.loads(
        (trace_dir / "task_002.json").read_text(encoding="utf-8")
    )
    assert second_train_trace["append_current"] is True
    assert [item["task_id"] for item in second_train_trace["ranked"]] == ["task_001"]
    assert second_train_trace["selected_task_ids"] == ["task_001"]

    frozen_eval_trace = json.loads(
        (trace_dir / "task_v01.json").read_text(encoding="utf-8")
    )
    assert frozen_eval_trace["append_current"] is False
    assert [item["task_id"] for item in frozen_eval_trace["ranked"]] == [
        "task_001",
        "task_002",
    ]
    assert frozen_eval_trace["selected_task_ids"] == ["task_001"]


def test_online_eval_updates_memory_on_eval_tasks(toy_dataset_root, tmp_path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    harness = harness_store.empty()
    train = [_task(toy_dataset_root, harness, "train", "task_001")]
    val = [_task(toy_dataset_root, harness, "val", "task_v01")]
    retriever = RecordingRetriever()
    client = FakeLLMClient(
        lambda prompt, model: "success"
        if "Did the agent successfully complete the task?" in prompt
        else MEMORY_ITEM
    )
    runner = ReasoningBankRunner(
        agent=MemoryAwareAgent(),
        memory_llm=ReasoningBankLLM(client=client, model="memory-model"),
        retriever=retriever,
        memory_store=ReasoningMemoryStore(tmp_path / "memory.jsonl"),
        traj_store=FilesystemTrajectoryStore(tmp_path / "trajectories"),
        workdir=tmp_path / "workdir",
        harness=harness,
        memory_n=1,
        eval_variant="online",
    )

    runner.run(train_tasks=train, eval_tasks=val)

    assert retriever.calls == [("task_001", True), ("task_v01", True)]
    assert len(runner.memory_store.load()) == 2
    assert len(client.calls) == 4


def test_render_trajectory_text_keeps_codex_event_summary_bounded() -> None:
    large_output = "line\n" * 20_000
    trajectory = Trajectory(
        id="traj_large",
        kind="solve",
        task_id="task",
        harness_id="harness",
        instructions="solve",
        events=[
            {"role": "system", "content": "hidden system content"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "I will inspect the failing test.",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "pytest tests/unit/test_urlutils.py -q",
                    "aggregated_output": large_output,
                    "exit_code": 1,
                },
            },
        ],
        final_message="Changed urlutils.py and compiled it.",
        stdout="stdout noise\n" * 20_000,
        stderr="stderr noise\n" * 20_000,
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )

    text = render_trajectory_text(trajectory)

    assert len(text) < 70_000
    assert "hidden system content" not in text
    assert "I will inspect the failing test." in text
    assert "$ pytest tests/unit/test_urlutils.py -q" in text
    assert "exit_code=1" in text


def test_render_trajectory_text_drops_raw_stream_events() -> None:
    noise_line = "RAW-STDERR-NOISE codex_home AbsolutePathBuf temp dir"
    trajectory = Trajectory(
        id="traj_raw",
        kind="solve",
        task_id="task",
        harness_id="harness",
        instructions="solve",
        events=(
            [{"type": "raw_stderr", "line": noise_line}] * 500
            + [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "The real fix lives in urlutils.",
                    },
                },
                {"type": "raw_stdout", "line": "RAW-STDOUT-NOISE chatter"},
            ]
        ),
        final_message="done",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )

    text = render_trajectory_text(trajectory)

    # Raw streaming events are low-level noise: command output is already
    # carried by command_execution.aggregated_output. They must not flood
    # the judge prompt or crowd out the real trajectory.
    assert noise_line not in text
    assert "RAW-STDOUT-NOISE" not in text
    assert "The real fix lives in urlutils." in text
    assert "Final message:\ndone" in text
