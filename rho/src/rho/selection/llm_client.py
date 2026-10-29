from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from rho.selection.azure_auth import azure_foundry_kwargs


def _is_reasoning_model(model: str) -> bool:
    name = model.split("/")[-1].lower()
    return (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Return the completion's final text."""


@dataclass
class _Call:
    prompt: str
    model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str | None
    system_prompt: str | None


class FakeLLMClient:
    def __init__(self, script: Callable[[str, str], str]) -> None:
        self._script = script
        self.calls: list[_Call] = []

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        self.calls.append(
            _Call(prompt, model, temperature, max_tokens, reasoning_effort, system_prompt)
        )
        return self._script(prompt, model)


class LiteLLMClient:
    """Real LLM client backed by litellm.

    When ``use_azure_foundry=True`` (default), injects the Azure OpenAI
    Foundry ``api_base`` and an Entra Bearer ``api_key``. Otherwise
    passes the call straight through to ``litellm.completion`` and lets
    litellm pick provider + credentials from the model prefix and env
    (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, …).

    gpt-5 / o-series reasoning models require ``max_completion_tokens``
    instead of ``max_tokens`` and refuse non-default ``temperature``;
    we detect them by model-name suffix and translate kwargs.
    """

    def __init__(self, *, use_azure_foundry: bool = True) -> None:
        self._use_azure_foundry = use_azure_foundry

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        import litellm

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "num_retries": 5,  # exp-backoff on transient 429 / 5xx under concurrency
        }
        if self._use_azure_foundry:
            kwargs.update(azure_foundry_kwargs())
        if _is_reasoning_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
            # ARE pins litellm==1.71.1, whose model registry predates gpt-5.5
            # and rejects reasoning_effort as unsupported. Force passthrough.
            kwargs["allowed_openai_params"] = ["reasoning_effort"]

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        return content or ""
