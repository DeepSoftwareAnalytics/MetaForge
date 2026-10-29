from __future__ import annotations

import json
from pathlib import Path

from rho.strategies.letta_sleep import (
    LETTA_SLEEP_INPUT_TEMPLATE,
    LETTA_SLEEP_SYSTEM_PROMPT,
    MEMORY_TOOL_FAITHFULNESS,
)
from rho.strategies.letta_memory_tools.vendored_constants import (
    CORE_MEMORY_LINE_NUMBER_WARNING,
)


SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "letta_bb52a89"


def test_letta_sleep_prompt_is_pinned_verbatim() -> None:
    assert LETTA_SLEEP_SYSTEM_PROMPT == (
        SNAPSHOT_DIR / "sleeptime_v2_prompt.txt"
    ).read_text(encoding="utf-8")


def test_letta_sleep_user_template_is_pinned_verbatim() -> None:
    assert LETTA_SLEEP_INPUT_TEMPLATE == (
        SNAPSHOT_DIR / "sleeptime_input_template.txt"
    ).read_text(encoding="utf-8")


def test_memory_tool_error_messages_and_validation_patterns_are_pinned() -> None:
    expected = json.loads(
        (SNAPSHOT_DIR / "base_memory_tool_faithfulness.json").read_text(
            encoding="utf-8"
        )
    )
    assert MEMORY_TOOL_FAITHFULNESS == expected
    assert CORE_MEMORY_LINE_NUMBER_WARNING == expected["core_memory_line_number_warning"]
