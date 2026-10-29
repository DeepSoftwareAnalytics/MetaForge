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
    parser.add_argument("--new_memory", required=True)
    args = parser.parse_args()
    try:
        result = memory_rethink(args.label, args.new_memory)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(result)
    return 0


def memory_rethink(label: str, new_memory: str) -> str:
    if bool(re.search(LINE_NUMBER_PREFIX_REGEX, new_memory)):
        raise ValueError(
            "new_memory contains a line number prefix, which is not allowed. Do not include line numbers when calling memory tools (line numbers are for display purposes only)."
        )
    if CORE_MEMORY_LINE_NUMBER_WARNING in new_memory:
        raise ValueError(
            "new_memory contains a line number warning, which is not allowed. Do not include line number information when calling memory tools (line numbers are for display purposes only)."
        )

    block_path = _block_path(label)
    block_path.write_text(str(new_memory), encoding="utf-8")
    return str(new_memory)


def _block_path(label: str) -> Path:
    if not re.fullmatch(LABEL_PATTERN, label):
        raise ValueError(f"invalid memory block label `{label}`.")
    root_raw = os.environ.get("LETTA_MEMORY_ROOT")
    if not root_raw:
        raise ValueError("LETTA_MEMORY_ROOT is required.")
    root = Path(root_raw).resolve()
    if label in _read_only(root):
        raise ValueError(f"Cannot edit read-only memory block with label `{label}`.")
    return root / f"{label}.md"


def _read_only(root: Path) -> set[str]:
    path = root / ".read_only"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
