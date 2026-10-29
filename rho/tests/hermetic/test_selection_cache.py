from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from rho.protocols import Grade, Harness, Trajectory
from rho.selection.cache import DifficultyCache, EmbeddingCache
from rho.selection.difficulty_selector import TaskJudge, _build_prompt
from rho.selection.trajectory_digest import render_digest
from rho.selection.embedder import LiteLLMEmbedder
from rho.selection.llm_client import FakeLLMClient
from tests.hermetic._traj_helper import fake_trajectory, trajectories_for


@dataclass(frozen=True)
class _QT:
    _id: str
    _q: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        raise NotImplementedError

    def materialize(self, dest: Path) -> None:
        raise NotImplementedError

    def query(self) -> str:
        return self._q

    def grade(self, trajectory: Trajectory, *, artifacts_dir=None) -> Grade:
        raise NotImplementedError


def _fingerprint(label: str) -> str:
    return (
        f"{label} failure mode centers on partial propagation across several layers, "
        "where one branch adopts a revised contract but a neighboring branch keeps "
        "older ordering assumptions and only breaks under a narrow boundary case. "
        "The work is moderately hard because invariants are scattered across helper "
        "logic, precedence rules, and state reconciliation paths, so the change "
        "requires contextual tracing rather than a purely local edit. Technical scope "
        "is multi-file bug fixing with contract alignment, propagation auditing, and "
        "global reasoning about how a small update preserves behavior."
    )


def test_difficulty_cache_roundtrip(tmp_path: Path) -> None:
    cache = DifficultyCache(tmp_path, "openai/gpt-5.4", "medium")
    assert cache.get("hello") is None
    cache.put(
        "hello",
        {
            "schema_version": 2,
            "parsed_difficulty": 7.5,
            "parsed_fingerprint": _fingerprint("hello"),
            "completion": "...",
        },
    )
    hit = cache.get("hello")
    assert hit is not None
    assert hit["schema_version"] == 2
    assert hit["parsed_difficulty"] == 7.5
    assert hit["parsed_fingerprint"] == _fingerprint("hello")


def test_difficulty_cache_namespaces_by_model_and_reasoning(tmp_path: Path) -> None:
    a = DifficultyCache(tmp_path, "openai/gpt-5.4", "medium")
    b = DifficultyCache(tmp_path, "openai/gpt-5.4", "low")
    c = DifficultyCache(tmp_path, "openai/gpt-5-mini", "medium")
    a.put(
        "same text",
        {
            "schema_version": 2,
            "parsed_difficulty": 1.0,
            "parsed_fingerprint": _fingerprint("same"),
        },
    )
    assert b.get("same text") is None
    assert c.get("same text") is None
    assert a.get("same text")["parsed_difficulty"] == 1.0


def test_task_judge_uses_disk_cache_on_second_call(tmp_path: Path) -> None:
    calls = []

    def script(prompt, model):
        calls.append(prompt)
        return json.dumps(
            {"difficulty": 6.5, "abstract_fingerprint": _fingerprint("cached")}
        )

    task = _QT(_id="t0", _q="identical query")
    trajectories = trajectories_for([task.id])

    first = TaskJudge(
        client=FakeLLMClient(script),
        model="openai/gpt-5.4",
        workdir=tmp_path / "calls_1",
        trajectories=trajectories,
        cache_root=tmp_path / "cache",
    )
    assert first.judge(task).difficulty == 6.5
    assert len(calls) == 1

    second = TaskJudge(
        client=FakeLLMClient(script),
        model="openai/gpt-5.4",
        workdir=tmp_path / "calls_2",
        trajectories=trajectories,
        cache_root=tmp_path / "cache",
    )
    result = second.judge(task)
    assert result.difficulty == 6.5
    assert result.fingerprint == _fingerprint("cached")
    assert len(calls) == 1

    persisted = json.loads(
        (tmp_path / "calls_2" / "t0.json").read_text(encoding="utf-8")
    )
    assert persisted["cache_hit"] is True
    assert persisted["schema_version"] == 3


def test_task_judge_cache_invalidates_when_prompt_template_changes(
    tmp_path: Path, monkeypatch
) -> None:
    import rho.selection.difficulty_selector as ds_mod

    calls = []

    def script(prompt, model):
        calls.append(prompt)
        return json.dumps(
            {"difficulty": 6.5, "abstract_fingerprint": _fingerprint("fresh")}
        )

    task = _QT(_id="t0", _q="identical query")
    trajectories = trajectories_for([task.id])

    first = TaskJudge(
        client=FakeLLMClient(script),
        model="m",
        workdir=tmp_path / "calls_1",
        trajectories=trajectories,
        cache_root=tmp_path / "cache",
    )
    assert first.judge(task).difficulty == 6.5
    assert len(calls) == 1

    original_tpl = ds_mod._JUDGE_INSTRUCTIONS
    monkeypatch.setattr(
        ds_mod,
        "_JUDGE_INSTRUCTIONS",
        original_tpl + "\nNEW LINE THAT CHANGES THE PROMPT.",
    )

    second = TaskJudge(
        client=FakeLLMClient(script),
        model="m",
        workdir=tmp_path / "calls_2",
        trajectories=trajectories,
        cache_root=tmp_path / "cache",
    )
    assert second.judge(task).difficulty == 6.5
    assert len(calls) == 2


@pytest.mark.parametrize("schema_version", [None, 1, 2])
def test_task_judge_rejects_incompatible_cache_schema(
    tmp_path: Path, schema_version: int | None
) -> None:
    task = _QT(_id="t0", _q="identical query")
    digest, _ = render_digest(fake_trajectory(task.id))
    prompt = _build_prompt(task.query(), digest)
    cache = DifficultyCache(tmp_path / "cache", "m", "none")
    record = {
        "parsed_difficulty": 5.0,
        "parsed_fingerprint": _fingerprint("stale"),
        "completion": "{}",
    }
    if schema_version is not None:
        record["schema_version"] = schema_version
    cache.put(prompt, record)

    judge = TaskJudge(
        client=FakeLLMClient(
            lambda prompt, model: json.dumps(
                {"difficulty": 1.0, "abstract_fingerprint": _fingerprint("fresh")}
            )
        ),
        model="m",
        workdir=tmp_path / "calls",
        trajectories=trajectories_for([task.id]),
        reasoning_effort=None,
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(ValueError, match="schema_version"):
        judge.judge(task)


def test_task_judge_cache_disabled_passthrough(tmp_path: Path) -> None:
    calls = []

    def script(prompt, model):
        calls.append(prompt)
        return json.dumps(
            {"difficulty": 4.2, "abstract_fingerprint": _fingerprint("uncached")}
        )

    task = _QT(_id="t0", _q="x")
    trajectories = trajectories_for([task.id])

    a = TaskJudge(
        client=FakeLLMClient(script),
        model="m",
        workdir=tmp_path / "a",
        trajectories=trajectories,
        cache_root=None,
    )
    b = TaskJudge(
        client=FakeLLMClient(script),
        model="m",
        workdir=tmp_path / "b",
        trajectories=trajectories,
        cache_root=None,
    )
    assert a.judge(task).difficulty == 4.2
    assert b.judge(task).difficulty == 4.2
    assert len(calls) == 2


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, "openai/text-embedding-3-large")
    assert cache.get("hello") is None
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.put("hello", vec)
    hit = cache.get("hello")
    assert hit is not None
    np.testing.assert_allclose(hit, vec)


def _fake_litellm_embedding(texts):
    vecs = np.array(
        [[float(ord(t[0]) % 7), 1.0, 2.0, 3.0] for t in texts], dtype=np.float32
    )
    return vecs


def test_embedder_uses_disk_cache_and_only_embeds_misses(tmp_path, monkeypatch) -> None:
    batch_sizes_seen: list[int] = []

    class _FakeResp:
        def __init__(self, texts):
            self.data = [{"embedding": v} for v in _fake_litellm_embedding(texts)]

    def fake_embedding(*, model, input, **kw):
        batch_sizes_seen.append(len(input))
        return _FakeResp(input)

    import rho.selection.embedder as embedder_mod

    import types

    fake_mod = types.SimpleNamespace(embedding=fake_embedding)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_mod)

    emb = LiteLLMEmbedder(
        model="openai/text-embedding-3-large", cache_root=tmp_path / "cache"
    )
    out1 = emb.embed(["alpha", "beta", "gamma"])
    assert out1.shape[0] == 3
    assert batch_sizes_seen == [3]

    out2 = emb.embed(["alpha", "beta", "delta"])
    assert out2.shape[0] == 3
    assert batch_sizes_seen == [3, 1]

    _ = emb.embed(["alpha", "beta", "delta", "gamma"])
    assert batch_sizes_seen == [3, 1]
