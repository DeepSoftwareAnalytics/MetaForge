from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

CORE_MEMORY_LINE_NUMBER_WARNING = "# NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. Do NOT include line number prefixes in your memory edit tool calls."
LINE_NUMBER_PREFIX_REGEX = r"\nLine \d+: "
LABEL_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]*"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--old_string", required=True)
    parser.add_argument("--new_string", required=True)
    args = parser.parse_args()
    try:
        result = memory_replace(args.label, args.old_string, args.new_string)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(result)
    return 0


def memory_replace(label: str, old_string: str, new_string: str) -> str:
    if bool(re.search(LINE_NUMBER_PREFIX_REGEX, old_string)):
        raise ValueError(
            "old_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in old_string:
        raise ValueError(
            "old_string contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )
    if bool(re.search(LINE_NUMBER_PREFIX_REGEX, new_string)):
        raise ValueError(
            "new_string contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )

    old_string = str(old_string).expandtabs()
    new_string = str(new_string).expandtabs()
    block_path = _block_path(label, must_exist=True)
    current_value = block_path.read_text(encoding="utf-8").expandtabs()

    occurences = current_value.count(old_string)
    if occurences == 0:
        raise ValueError(
            f"No replacement was performed, old_string `{old_string}` did not appear verbatim in memory block with label `{label}`."
        )
    elif occurences > 1:
        content_value_lines = current_value.split("\n")
        lines = [idx + 1 for idx, line in enumerate(content_value_lines) if old_string in line]
        raise ValueError(
            f"No replacement was performed. Multiple occurrences of old_string `{old_string}` in lines {lines}. Please ensure it is unique."
        )

    new_value = current_value.replace(str(old_string), str(new_string))
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
