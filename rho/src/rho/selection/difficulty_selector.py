from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rho.protocols import Task, Trajectory
from rho.selection.cache import DEFAULT_CACHE_ROOT, DifficultyCache
from rho.selection.llm_client import LLMClient
from rho.selection.trajectory_digest import render_digest

_JUDGE_INSTRUCTIONS = """\
Rate the difficulty of the following software engineering task and write
an abstract structural fingerprint of it. You may also use the observed
agent run below to inform your judgment.

Output a JSON object (no markdown fences, no extra text):
{
  "difficulty": <float in [0.0, 10.0]>,
  "abstract_fingerprint": "<see guide below>"
}

Difficulty scale:
- 0-2: trivial (obvious one-line fix, cosmetic change).
- 3-5: moderate (localized change, well-defined spec).
- 6-8: hard (multi-file, non-obvious design, subtle bugs).
- 9-10: very hard (cross-cutting refactor, deep reasoning required).

Abstract fingerprint guide — 3-5 sentences (~60-120 words) describing
the *shape* of the problem in vocabulary that would apply equally to
any software project. Cover:
- Failure mode: what typically goes wrong (partial propagation,
  missed boundary case, silent precedence regression, broken invariant
  under a new branch, etc.).
- Source of difficulty: what makes it hard or easy (scattered
  invariants, ambiguous spec, tight coupling, purely localized
  arithmetic, etc.).
- Technical complexity: scope (single-function / multi-file /
  cross-module / architectural), reasoning depth (local / contextual /
  global invariant tracking), type of change (bug fix / feature /
  refactor / rollback).

Do NOT mention: repository, product, company, framework, or library
names; file paths; function, class, config, or variable names;
domain-specific nouns tied to a particular codebase. Use only abstract
structural programming vocabulary (invariant, precedence, boundary,
state reconciliation, propagation, ordering, contract, etc.).

The observed agent run contains concrete file paths, library names, and
tool output. Abstract these out the same way — fingerprints describe
shapes, not the specific codebase the agent happened to run in.

The observed run is a single noisy sample, not ground truth. Do not
lower difficulty just because the agent appeared to succeed in one
attempt, and do not raise difficulty just because one attempt thrashed
on bootstrap issues. Treat the trajectory as evidence that adjusts your
prior on task difficulty and failure mode; weight it relative to what
the task description itself implies.

Example of a well-abstracted fingerprint:
  "A multi-file refactor whose difficulty comes from keeping a single
  shared invariant consistent across several independently-evolving
  modules; the typical failure mode is partial propagation, where one
  call site adopts the new contract while another silently keeps the
  old, producing a latent bug that only surfaces under a specific
  input ordering. Spec ambiguity is low but reasoning must be global —
  the change is mechanically small per site but requires tracing a
  contract through the call graph."

Task:
---
{query}
---

Observed agent run (under the current harness):
---
{trajectory_digest}
---
"""

_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)
_WORD_RE = re.compile(r"\b\w+\b")
_MIN_FINGERPRINT_WORDS = 40
_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class JudgeResult:
    task_id: str
    difficulty: float
    fingerprint: str
    digest_token_estimate: int = 0


def _extract_json(text: str) -> str:
    m = _FENCED_JSON.search(text)
    return m.group(1).strip() if m else text.strip()


def _build_prompt(query: str, trajectory_digest: str) -> str:
    return (_JUDGE_INSTRUCTIONS
            .replace("{query}", query)
            .replace("{trajectory_digest}", trajectory_digest))


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _parse_judge_output(text: str, task_id: str) -> tuple[float, str]:
    try:
        parsed = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"TaskJudge: malformed JSON from judge for task {task_id!r}: {exc}. "
            f"Raw: {text[:200]!r}"
        ) from exc

    for required in ("difficulty", "abstract_fingerprint"):
        if required not in parsed:
            raise ValueError(
                f"TaskJudge: missing required key {required!r} in judge "
                f"output for task {task_id!r}. Raw: {text[:200]!r}"
            )

    try:
        difficulty = float(parsed["difficulty"])
        fingerprint = str(parsed["abstract_fingerprint"]).strip()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TaskJudge: malformed field types in judge output for task "
            f"{task_id!r}: {exc}. Raw: {text[:200]!r}"
        ) from exc

    if not (0.0 <= difficulty <= 10.0):
        raise ValueError(
            f"TaskJudge: difficulty {difficulty} out of [0,10] for task {task_id!r}"
        )
    if not fingerprint:
        raise ValueError(
            f"TaskJudge: empty abstract_fingerprint for task {task_id!r}"
        )
    n_words = _word_count(fingerprint)
    if n_words < _MIN_FINGERPRINT_WORDS:
        raise ValueError(
            "TaskJudge: abstract_fingerprint too short for task "
            f"{task_id!r}: {n_words} words < {_MIN_FINGERPRINT_WORDS}"
        )
    return difficulty, fingerprint


def _result_from_cache_record(record: dict, task_id: str) -> JudgeResult:
    schema_version = record.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            "TaskJudge: cached record has incompatible schema_version for task "
            f"{task_id!r}: expected {_SCHEMA_VERSION}, got {schema_version!r}"
        )
    try:
        difficulty = float(record["parsed_difficulty"])
        fingerprint = str(record["parsed_fingerprint"]).strip()
        token_estimate = int(record.get("digest_token_estimate", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"TaskJudge: malformed cached record for task {task_id!r}: {exc}"
        ) from exc
    if not (0.0 <= difficulty <= 10.0):
        raise ValueError(
            "TaskJudge: cached parsed_difficulty out of [0,10] for task "
            f"{task_id!r}: {difficulty}"
        )
    if _word_count(fingerprint) < _MIN_FINGERPRINT_WORDS:
        raise ValueError(
            "TaskJudge: cached parsed_fingerprint too short for task "
            f"{task_id!r}"
        )
    return JudgeResult(
        task_id=task_id, difficulty=difficulty, fingerprint=fingerprint,
        digest_token_estimate=token_estimate,
    )


class TaskJudge:
    """Judge a task's intrinsic difficulty and abstract fingerprint via one LLM call.

    Caches by task id in memory and by rendered prompt on disk. Persists every
    call as <workdir>/<task_id>.json.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        workdir: Path,
        trajectories: dict[str, Trajectory],
        reasoning_effort: str | None = None,
        max_tokens: int = 16384,
        cache_root: Path | None = DEFAULT_CACHE_ROOT,
    ) -> None:
        self._client = client
        self._model = model
        self._workdir = workdir
        self._trajectories = trajectories
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._cache: dict[str, JudgeResult] = {}
        self._disk_cache = (
            DifficultyCache(cache_root, model, reasoning_effort)
            if cache_root is not None
            else None
        )

    def judge(self, task: Task) -> JudgeResult:
        if task.id in self._cache:
            return self._cache[task.id]
        if task.id not in self._trajectories:
            raise KeyError(task.id)
        trajectory = self._trajectories[task.id]
        digest_text, digest_tokens = render_digest(trajectory)

        # Persist the digest sidecar for human inspection.
        safe_id = task.id.replace("/", "__")
        digest_dir = self._workdir / "short_solve"
        digest_dir.mkdir(parents=True, exist_ok=True)
        (digest_dir / f"{safe_id}.txt").write_text(digest_text, encoding="utf-8")

        query = task.query()
        prompt = _build_prompt(query, digest_text)

        if self._disk_cache is not None:
            hit = self._disk_cache.get(prompt)
            if hit is not None:
                result_no_tokens = _result_from_cache_record(hit, task.id)
                result = JudgeResult(
                    task_id=task.id, difficulty=result_no_tokens.difficulty,
                    fingerprint=result_no_tokens.fingerprint,
                    digest_token_estimate=digest_tokens,
                )
                self._persist(
                    task.id,
                    prompt,
                    str(hit.get("completion", "")),
                    result,
                    wall_s=0.0,
                    cache_hit=True,
                )
                self._cache[task.id] = result
                return result

        t0 = time.monotonic()
        completion = self._client.complete(
            prompt,
            model=self._model,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
        )
        wall_s = time.monotonic() - t0
        difficulty, fingerprint = _parse_judge_output(completion, task.id)
        result = JudgeResult(
            task_id=task.id, difficulty=difficulty, fingerprint=fingerprint,
            digest_token_estimate=digest_tokens,
        )

        if self._disk_cache is not None:
            self._disk_cache.put(
                prompt,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "model": self._model,
                    "reasoning_effort": self._reasoning_effort,
                    "prompt": prompt,
                    "completion": completion,
                    "parsed_difficulty": result.difficulty,
                    "parsed_fingerprint": result.fingerprint,
                    "digest_token_estimate": result.digest_token_estimate,
                    "wall_time_s": wall_s,
                },
            )

        self._persist(task.id, prompt, completion, result, wall_s, cache_hit=False)
        self._cache[task.id] = result
        return result

    def judge_many(
        self, tasks: list[Task], *, max_workers: int = 64
    ) -> dict[str, JudgeResult]:
        """Judge a batch of tasks in parallel. Returns id -> JudgeResult.

        Each `judge()` call is independent network I/O; ThreadPoolExecutor
        overlaps them to cut wall time from serial-per-task to
        roughly serial/max_workers. In-memory cache writes are atomic
        under the GIL (single dict assignment); disk writes go to
        separate files per task/hash so there is no contention.
        """
        if not tasks:
            return {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [(t, pool.submit(self.judge, t)) for t in tasks]
            results_list: list[JudgeResult] = []
            failures: list[tuple[str, BaseException]] = []
            for task, f in futures:
                try:
                    results_list.append(f.result())
                except BaseException as exc:  # noqa: BLE001
                    failures.append((task.id, exc))
        if failures:
            lines = [
                f"TaskJudge.judge_many: {len(failures)}/{len(tasks)} tasks failed:"
            ]
            for tid, exc in failures:
                lines.append(f"  - {tid}: {type(exc).__name__}: {exc}")
            raise RuntimeError("\n".join(lines))
        return {r.task_id: r for r in results_list}

    def results(self) -> dict[str, JudgeResult]:
        return dict(self._cache)

    def _persist(
        self,
        task_id: str,
        prompt: str,
        completion: str,
        result: JudgeResult,
        wall_s: float,
        *,
        cache_hit: bool,
    ) -> None:
        self._workdir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "task_id": task_id,
            "model": self._model,
            "reasoning_effort": self._reasoning_effort,
            "prompt": prompt,
            "completion": completion,
            "parsed_difficulty": result.difficulty,
            "parsed_fingerprint": result.fingerprint,
            "wall_time_s": wall_s,
            "timestamp_ms": int(time.time() * 1000),
            "cache_hit": cache_hit,
        }
        safe_id = task_id.replace("/", "__")
        (self._workdir / f"{safe_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class DifficultySelector:
    """Top-k by judge-rated difficulty. Ties broken by stable task-id sort."""

    def __init__(self, *, judge: TaskJudge) -> None:
        self._judge = judge

    def results(self) -> dict[str, JudgeResult]:
        return self._judge.results()

    def select(
        self,
        candidates: list[Task],
        k: int | None,
        *,
        seed: int | None = None,
    ) -> list[Task]:
        del seed
        judged = self._judge.judge_many(candidates)
        scored = [
            (judged[task.id].difficulty, task.id, task) for task in candidates
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        ordered = [row[2] for row in scored]
        if k is None:
            return ordered
        return ordered[:k]
