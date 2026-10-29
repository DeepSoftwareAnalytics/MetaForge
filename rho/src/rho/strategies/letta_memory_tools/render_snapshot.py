from __future__ import annotations

from pathlib import Path

from rho.strategies.letta_memory_tools.vendored_constants import (
    CORE_MEMORY_LINE_NUMBER_WARNING,
    READ_ONLY_FILE,
)


def render_memory_snapshot(letta_memory_dir: Path) -> str:
    read_only = _read_only_labels(letta_memory_dir)
    blocks = [
        path
        for path in sorted(letta_memory_dir.glob("*.md"))
        if path.is_file() and path.parent == letta_memory_dir
    ]
    lines: list[str] = [
        "<letta>",
        "You have persistent memory blocks shown below. Modify them via the memory tools.",
        "<memory_blocks>",
        CORE_MEMORY_LINE_NUMBER_WARNING,
        "",
    ]
    for path in blocks:
        label = path.stem
        value = path.read_text(encoding="utf-8", errors="replace")
        value_lines = value.split("\n")
        lines.append(
            f'<block label="{label}" read_only="{str(label in read_only).lower()}" line_count="{len(value_lines)}">'
        )
        for ix, line in enumerate(value_lines, start=1):
            lines.append(f"{ix}→ {line}")
        lines.append("</block>")
        lines.append("")
    lines.extend(["</memory_blocks>", "</letta>"])
    return "\n".join(lines)


def _read_only_labels(letta_memory_dir: Path) -> set[str]:
    path = letta_memory_dir / READ_ONLY_FILE
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    }
