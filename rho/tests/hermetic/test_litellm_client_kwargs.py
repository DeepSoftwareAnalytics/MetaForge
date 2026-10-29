from __future__ import annotations

from unittest.mock import patch

from rho.selection import llm_client


class _FakeResp:
    def __init__(self, content: str = "ok") -> None:
        class _Msg:
            def __init__(self, c: str) -> None:
                self.content = c
        class _Choice:
            def __init__(self, c: str) -> None:
                self.message = _Msg(c)
        self.choices = [_Choice(content)]


def test_azure_foundry_path_injects_base_and_bearer() -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch.object(
        llm_client,
        "azure_foundry_kwargs",
        return_value={"api_base": "X", "api_key": "JWT"},
    ), patch("litellm.completion", side_effect=fake_completion):
        client = llm_client.LiteLLMClient(use_azure_foundry=True)
        client.complete(
            "hi", model="openai/gpt-5.5", max_tokens=512, reasoning_effort="medium"
        )

    assert captured["model"] == "openai/gpt-5.5"
    assert captured["api_base"] == "X"
    assert captured["api_key"] == "JWT"
    assert captured["max_completion_tokens"] == 512
    assert "max_tokens" not in captured
    assert captured["reasoning_effort"] == "medium"


def test_provider_agnostic_path_omits_api_base() -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch("litellm.completion", side_effect=fake_completion):
        client = llm_client.LiteLLMClient(use_azure_foundry=False)
        client.complete("hi", model="openrouter/openai/gpt-5", max_tokens=200)

    assert captured["model"] == "openrouter/openai/gpt-5"
    assert "api_base" not in captured       # litellm uses its own provider routing
    assert "api_key" not in captured        # litellm reads OPENROUTER_API_KEY from env


def test_non_reasoning_model_keeps_max_tokens_and_temperature() -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch("litellm.completion", side_effect=fake_completion):
        client = llm_client.LiteLLMClient(use_azure_foundry=False)
        client.complete("hi", model="openai/gpt-4o-mini", max_tokens=100, temperature=0.2)

    assert captured["max_tokens"] == 100
    assert captured["temperature"] == 0.2
    assert "max_completion_tokens" not in captured


def test_o4_is_detected_as_reasoning() -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch("litellm.completion", side_effect=fake_completion):
        client = llm_client.LiteLLMClient(use_azure_foundry=False)
        client.complete("hi", model="openai/o4-mini", max_tokens=100)

    assert captured["max_completion_tokens"] == 100
    assert "max_tokens" not in captured
