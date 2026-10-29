import os
from pathlib import Path

from rho.protocols import TaskSelector
from rho.selection.cache import DEFAULT_CACHE_ROOT
from rho.selection.coverage_selector import CoverageSelector
from rho.selection.difficulty_selector import (
    DifficultySelector,
    JudgeResult,
    TaskJudge,
)
from rho.selection.dpp_selector import DPPSelector
from rho.selection.embedder import LiteLLMEmbedder, TaskEmbedder
from rho.selection.llm_client import LiteLLMClient
from rho.selection.local_embedder import LocalEmbedder
from rho.selection.random_selector import RandomSelector

SELECTOR_CHOICES = ("random", "difficulty", "coverage", "dpp")
DEFAULT_DPP_THETA = 0.7

# Default judge: gpt-5.5 on Azure OpenAI Foundry via direct Entra Bearer.
# Override via `RHO_LLM_BACKEND=off` to let litellm pick provider
# + creds from the model prefix and env (OPENROUTER_API_KEY etc.).
DEFAULT_JUDGE_MODEL = "openai/gpt-5.5"
DEFAULT_JUDGE_REASONING = "high"

# Default embedder: local FastEmbed ONNX (no network, no API key, no
# Azure dependency). Prefix `local:` selects the LocalEmbedder; bare
# litellm-style prefixes (`openai/…`, `openrouter/…`) route through
# LiteLLMEmbedder. The new Foundry resource has no embedding deployment,
# so an API embedder is opt-in and requires a remote provider key.
DEFAULT_EMBEDDING_MODEL = "local:BAAI/bge-large-en-v1.5"


def _use_azure_foundry_for_llm() -> bool:
    return os.environ.get("RHO_LLM_BACKEND", "azure-foundry") == "azure-foundry"


def _use_azure_foundry_for_embedder() -> bool:
    return os.environ.get("RHO_EMBEDDER_BACKEND", "off") == "azure-foundry"


def build_embedder(model: str, cache_root: Path | None) -> TaskEmbedder:
    """Map an embedding-model string to an embedder.

    A ``local:`` prefix selects the on-machine FastEmbed ONNX encoder (no
    network, no API key); any other prefix (`openai/…`, `azure/…`) routes
    through litellm. Shared by the task selector and the ReasoningBank
    retriever so both pick the backend the same way.
    """
    if model.startswith("local:"):
        return LocalEmbedder(model=model[len("local:") :], cache_root=cache_root)
    return LiteLLMEmbedder(
        model=model,
        cache_root=cache_root,
        use_azure_foundry=_use_azure_foundry_for_embedder(),
    )


def build_selector(
    name: str,
    *,
    workdir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_reasoning: str | None = DEFAULT_JUDGE_REASONING,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
    theta: float = DEFAULT_DPP_THETA,
    trajectories: dict | None = None,
) -> TaskSelector:
    if name == "random":
        return RandomSelector()
    if name not in {"difficulty", "coverage", "dpp"}:
        raise ValueError(f"unknown selector: {name}")
    if trajectories is None:
        raise ValueError(
            f"trajectories required for selector={name!r}; "
            "run short_solve_all before build_selector"
        )

    judge = TaskJudge(
        client=LiteLLMClient(use_azure_foundry=_use_azure_foundry_for_llm()),
        model=judge_model,
        workdir=workdir,
        trajectories=trajectories,
        reasoning_effort=judge_reasoning,
        cache_root=cache_root,
    )
    if name == "difficulty":
        return DifficultySelector(judge=judge)

    embedder = build_embedder(embedding_model, cache_root)
    if name == "coverage":
        return CoverageSelector(judge=judge, embedder=embedder, workdir=workdir)
    return DPPSelector(judge=judge, embedder=embedder, theta=theta, workdir=workdir)


__all__ = [
    "RandomSelector",
    "CoverageSelector",
    "DifficultySelector",
    "DPPSelector",
    "JudgeResult",
    "TaskJudge",
    "DEFAULT_CACHE_ROOT",
    "SELECTOR_CHOICES",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_REASONING",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_DPP_THETA",
    "build_selector",
    "build_embedder",
]
