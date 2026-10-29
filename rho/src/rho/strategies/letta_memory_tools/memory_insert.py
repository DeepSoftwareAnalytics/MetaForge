from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

CORE_MEMORY_LINE_NUMBER_WARNING = "# NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. Do NOT include line number prefixes in your memory edit tool calls."
LINE_NUMBER_PREFIX_REGEX = r"\nLine \d+: "
LABEL_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]*"
SNIPPET_LINES: int = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--new_string", required=True)
    parser.add_argument("--insert_line", type=int, default=-1)
    args = parser.parse_args()
    try:
        result = memory_insert(args.label, args.new_string, args.insert_line)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(result)
    return 0


def memory_insert(label: str, new_string: str, insert_line: int = -1) -> str:
    if bool(re.search(LINE_NUMBER_PREFIX_REGEX, new_string)):
        raise ValueError(
            "new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in new_string:
        raise ValueError(
            "new_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )

    block_path = _block_path(label, must_exist=True)
    current_value = block_path.read_text(encoding="utf-8").expandtabs()
    new_string = str(new_string).expandtabs()
    current_value_lines = current_value.split("\n")
    n_lines = len(current_value_lines)

    if insert_line == -1:
        insert_line = n_lines
    elif insert_line < 0 or insert_line > n_lines:
        raise ValueError(
            f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the memory block: {[0, n_lines]}, or -1 to append to the end of the memory block."
        )

    new_string_lines = new_string.split("\n")
    new_value_lines = current_value_lines[:insert_line] + new_string_lines + current_value_lines[insert_line:]
    (
        current_value_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
        + new_string_lines
        + current_value_lines[insert_line : insert_line + SNIPPET_LINES]
    )

    new_value = "\n".join(new_value_lines)
    block_path.write_text(new_value, encoding="utf-8")
    return new_value


def _block_path(label: str, *, must_exist: bool) -> Path:
    if not re.fullmatch(LABEL_PATTERN, label):
        raise ValueError(f"invalid memory block label `{label}`.")
    root_raw = os.environ.get("LETTA_MEMORY_ROOT")
    if not root_raw:
        raise ValueError("LETTA_MEMORY_ROOT is required.")
    root = Path(root_raw).resolve()
    if label in _read_only(root):
        raise ValueError(f"Cannot edit read-only memory block with label `{label}`.")
    path = root / f"{label}.md"
    if must_exist and not path.exists():
        raise ValueError(f"Memory block with label `{label}` does not exist.")
    return path


def _read_only(root: Path) -> set[str]:
    path = root / ".read_only"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
