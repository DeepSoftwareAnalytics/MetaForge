from __future__ import annotations

from rho.datasets.terminal_bench_2.prompts import render_prompt


def test_prompt_substitutes_all_placeholders() -> None:
    out = render_prompt(
        task_id="break-filter",
        difficulty="medium",
        category="software-engineering",
        container_name="tbench2-break-filter-abc123",
        agent_timeout_sec=900.0,
        verifier_timeout_sec=900.0,
        instruction_md="# Instruction\n\nFix the bug.",
    )
    assert "tbench2-break-filter-abc123" in out
    assert "/host-ws" in out
    assert "# Instruction" in out
    assert "Fix the bug." in out
    assert "medium" in out
    assert "900" in out
    assert "docker exec -i tbench2-break-filter-abc123 bash /host-ws/" in out


def test_prompt_has_no_stray_placeholders() -> None:
    out = render_prompt(
        task_id="t",
        difficulty="d",
        category="c",
        container_name="n",
        agent_timeout_sec=1.0,
        verifier_timeout_sec=1.0,
        instruction_md="x",
    )
    assert "{" not in out or "}" not in out
