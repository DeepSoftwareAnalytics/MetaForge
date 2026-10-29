"""LocomoDataset / LocomoTaskSet / LocomoTask.

On construction, reads ``locomo10.json``, materializes raw sessions into
a temporary directory, captures that directory into the harness store as
a single shared ``Harness`` (via ``HarnessStore.capture``), and computes
a deterministic stratified split. All tasks in the dataset share the
same ``Harness`` reference.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rho.datasets.locomo.ingest import (
    LocomoConversation,
    load_locomo,
    write_harness_tree,
)
from rho.datasets.locomo.scoring import extract_answer, score_qa
from rho.datasets.locomo.splits import (
    QARef,
    apply_max_per_split,
    stratified_split,
)
from rho.protocols import Grade, Harness, HarnessStore, Task, TaskSet, Trajectory

_PROMPT_TEMPLATE = """\
# Question

{question}

You are answering a question about {conv_id}. The raw conversation
sessions are available in your harness under `{conv_id}/`. Use them
to answer the question as concisely as possible.

Write your final answer on a single line at the end of your response,
after a line containing exactly `ANSWER:`.
"""


@dataclass(frozen=True)
class LocomoTask:
    _conv_id: str
    _qa_index: int
    _question: str
    _gold: str
    _category: int
    _harness: Harness

    @property
    def id(self) -> str:
        return f"{self._conv_id}/qa_{self._qa_index:04d}"

    @property
    def harness(self) -> Harness:
        return self._harness

    @property
    def agent_timeout_s(self) -> float | None:
        return None

    def materialize(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        prompt = _PROMPT_TEMPLATE.format(question=self._question, conv_id=self._conv_id)
        (dest / "prompt.md").write_text(prompt, encoding="utf-8")

    def query(self) -> str:
        return self._question

    def grade(
        self,
        trajectory: Trajectory,
        *,
        artifacts_dir: Path | None = None,
    ) -> Grade:
        prediction = extract_answer(trajectory.final_message)
        score = score_qa(prediction, self._gold, self._category)
        return Grade(
            passed=score > 0.5,
            score=float(score),
            details={
                "category": self._category,
                "question": self._question,
                "gold": self._gold,
                "prediction": prediction,
            },
        )


@dataclass(frozen=True)
class LocomoTaskSet:
    _split: str
    _tasks: tuple[LocomoTask, ...]

    @property
    def split(self) -> str:
        return self._split

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class LocomoDataset:
    def __init__(
        self,
        path: Path,
        *,
        harness_store: HarnessStore,
        seed: int = 0,
        max_per_split: int | None = None,
    ) -> None:
        self._path = path
        self._seed = seed
        self._max_per_split = max_per_split

        conversations = load_locomo(path)
        self._harness = self._build_harness(conversations, harness_store)

        qa_refs: list[QARef] = []
        qa_by_ref: dict[tuple[str, int], dict] = {}
        for conv in conversations:
            for idx, qa in enumerate(conv.qa):
                qa_refs.append(
                    QARef(conv_id=conv.sample_id, qa_index=idx, category=qa["category"])
                )
                qa_by_ref[(conv.sample_id, idx)] = qa

        split_map = stratified_split(qa_refs, seed=seed)
        if max_per_split is not None:
            split_map = {k: apply_max_per_split(v, max_per_split=max_per_split) for k, v in split_map.items()}

        self._splits: dict[str, LocomoTaskSet] = {}
        for split_name, refs in split_map.items():
            tasks = tuple(
                self._build_task(ref, qa_by_ref[(ref.conv_id, ref.qa_index)])
                for ref in refs
            )
            self._splits[split_name] = LocomoTaskSet(_split=split_name, _tasks=tasks)

    def _build_harness(
        self,
        conversations: tuple[LocomoConversation, ...],
        harness_store: HarnessStore,
    ) -> Harness:
        with tempfile.TemporaryDirectory(prefix="locomo_harness_") as tmp:
            tree_root = Path(tmp)
            write_harness_tree(conversations, tree_root)
            return harness_store.capture(tree_root)

    def _build_task(self, ref: QARef, qa: dict) -> LocomoTask:
        gold = str(qa.get("answer", ""))
        return LocomoTask(
            _conv_id=ref.conv_id,
            _qa_index=ref.qa_index,
            _question=qa["question"],
            _gold=gold,
            _category=ref.category,
            _harness=self._harness,
        )

    @property
    def train(self) -> TaskSet:
        return self._splits["train"]

    @property
    def val(self) -> TaskSet:
        return self._splits["val"]

    @property
    def test(self) -> TaskSet:
        return self._splits["test"]
