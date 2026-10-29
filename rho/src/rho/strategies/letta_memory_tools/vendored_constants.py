from __future__ import annotations

CORE_MEMORY_LINE_NUMBER_WARNING = "# NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. Do NOT include line number prefixes in your memory edit tool calls."
LINE_NUMBER_PREFIX_REGEX = r"\nLine \d+: "
BASE_SLEEPTIME_TOOLS = [
    "memory_replace",
    "memory_insert",
    "memory_rethink",
    "memory_finish_edits",
]

LABEL_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]*"
READ_ONLY_FILE = ".read_only"
