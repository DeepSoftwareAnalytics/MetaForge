from __future__ import annotations

from rho.selection.llm_client import FakeLLMClient


def test_fake_llm_client_returns_canned_response() -> None:
    client = FakeLLMClient(lambda prompt, model: f"response to {prompt[:20]!r} via {model}")
    out = client.complete("What is 2+2?", model="dummy-model")
    assert "What is 2+2?" in out
    assert "dummy-model" in out


def test_fake_llm_client_records_calls() -> None:
    client = FakeLLMClient(lambda prompt, model: "ok")
    client.complete("a", model="m")
    client.complete("b", model="m")
    assert [c.prompt for c in client.calls] == ["a", "b"]


def test_fake_llm_client_records_system_prompt() -> None:
    client = FakeLLMClient(lambda prompt, model: "ok")
    client.complete("a", model="m", system_prompt="system")
    assert client.calls[0].system_prompt == "system"
