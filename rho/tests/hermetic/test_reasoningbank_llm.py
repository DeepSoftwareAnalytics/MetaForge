from __future__ import annotations

from rho.reasoningbank.llm import JUDGE_SYSTEM_PROMPT, ReasoningBankLLM
from rho.reasoningbank.prompts import CODING_SUCCESSFUL_SI
from rho.selection.llm_client import FakeLLMClient


def test_judge_uses_system_prompt_role() -> None:
    client = FakeLLMClient(lambda prompt, model: "success")
    llm = ReasoningBankLLM(client=client, model="memory-model")

    assert llm.judge_success("query", "trajectory", task_id="task_001") is True

    assert client.calls[0].system_prompt == JUDGE_SYSTEM_PROMPT
    assert "Answer with 'success' or 'fail' only" in client.calls[0].prompt


def test_extract_memory_items_uses_official_split_and_system_prompt() -> None:
    client = FakeLLMClient(lambda prompt, model: "# Memory Item 1\n\n# Memory Item 2\n\n")
    llm = ReasoningBankLLM(client=client, model="memory-model")

    items = llm.extract_memory_items(
        query="query",
        trajectory="trajectory",
        success=True,
        task_id="task_001",
    )

    assert items == ["# Memory Item 1", "# Memory Item 2", ""]
    assert client.calls[0].system_prompt == CODING_SUCCESSFUL_SI.strip()
    assert client.calls[0].prompt.startswith("**Query:** query")
