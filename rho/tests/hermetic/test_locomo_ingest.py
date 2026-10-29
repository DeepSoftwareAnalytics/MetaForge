"""Assert LOCOMO ingestion matches the vendored data and spec §6/§7."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from rho.datasets.locomo.ingest import (
    LocomoConversation,
    load_locomo,
    write_harness_tree,
)

LOCOMO_PATH = Path(__file__).parents[2] / "data" / "locomo10.json"


@pytest.fixture(scope="module")
def conversations() -> tuple[LocomoConversation, ...]:
    return load_locomo(LOCOMO_PATH)


def test_ten_conversations(conversations: tuple[LocomoConversation, ...]) -> None:
    assert len(conversations) == 10


def test_total_qa_count(conversations: tuple[LocomoConversation, ...]) -> None:
    total = sum(len(c.qa) for c in conversations)
    assert total == 1986


def test_total_session_count(conversations: tuple[LocomoConversation, ...]) -> None:
    total = sum(len(c.sessions) for c in conversations)
    assert total == 272


def test_category_distribution(conversations: tuple[LocomoConversation, ...]) -> None:
    cats = Counter(q["category"] for c in conversations for q in c.qa)
    assert cats[1] == 282
    assert cats[2] == 321
    assert cats[3] == 96
    assert cats[4] == 841
    assert cats[5] == 446


def test_usable_qa_after_cat5_filter(conversations: tuple[LocomoConversation, ...]) -> None:
    usable = sum(1 for c in conversations for q in c.qa if q["category"] != 5)
    assert usable == 1540


def test_sessions_are_numerically_ordered(
    conversations: tuple[LocomoConversation, ...],
) -> None:
    for conv in conversations:
        indexes = [s.index for s in conv.sessions]
        assert indexes == sorted(indexes)
        assert len(indexes) == len(set(indexes))


def test_write_harness_tree_shape(
    conversations: tuple[LocomoConversation, ...], tmp_path: Path
) -> None:
    out = tmp_path / "harness"
    write_harness_tree(conversations, out)

    for conv in conversations:
        conv_dir = out / conv.sample_id
        assert conv_dir.is_dir()
        assert (conv_dir / "INDEX.md").is_file()
        for session in conv.sessions:
            session_file = conv_dir / f"session_{session.index:02d}.md"
            assert session_file.is_file()
            content = session_file.read_text(encoding="utf-8")
            assert content.startswith(f"# session_{session.index:02d} —")
            assert session.date_time in content
            # One non-blank non-image line per turn (approximately).
            dialog_lines = [
                line for line in content.splitlines()[2:]
                if line and not line.startswith("  [image:")
            ]
            assert len(dialog_lines) >= len(session.turns) * 0.9


def test_no_leaked_fields_in_harness_tree(
    conversations: tuple[LocomoConversation, ...], tmp_path: Path
) -> None:
    out = tmp_path / "harness"
    write_harness_tree(conversations, out)
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # observation/session_summary/event_summary leak information
        assert "observation" not in text.lower().split()
        # dia_id should not appear in rendered harness
        assert not re.search(r"\bD\d+:\d+\b", text)


def test_harness_index_lists_speakers(
    conversations: tuple[LocomoConversation, ...], tmp_path: Path
) -> None:
    out = tmp_path / "harness"
    write_harness_tree(conversations, out)
    for conv in conversations:
        index = (out / conv.sample_id / "INDEX.md").read_text(encoding="utf-8")
        assert conv.speakers[0] in index
        assert conv.speakers[1] in index
        assert f"{len(conv.sessions)} files" in index
