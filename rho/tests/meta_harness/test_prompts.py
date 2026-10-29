from __future__ import annotations

from rho.meta_harness.prompts import render_proposer_instructions


def test_render_substitutes_candidate_count() -> None:
    text = render_proposer_instructions(3)
    assert "propose 3 new candidate harnesses" in text or "exactly 3 candidates" in text
    assert "{candidates_per_iter}" not in text
    assert "{harness_description}" not in text


def test_render_covers_required_sections() -> None:
    text = render_proposer_instructions(1)
    # Workspace layout the runner actually creates.
    assert "history/" in text
    assert "candidates/" in text
    assert "summary.jsonl" in text
    assert "traces/" in text
    assert "proposed/" in text
    assert "manifest.json" in text
    # New post-mortems must be written under proposed/ (history/ is read-only).
    assert "proposed/reports/" in text
    # Faithfulness-critical guidance.
    assert "ground-truth" in text.lower()
    assert "anti-overfitting" in text.lower() or "general-purpose" in text.lower()
    assert "dataset names" in text.lower()
