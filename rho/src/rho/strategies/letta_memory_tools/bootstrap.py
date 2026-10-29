from __future__ import annotations

import shutil
from pathlib import Path

PERSONA_STUB = "# persona\n\n(no persona policies in this deployment)\n"
_TOOL_NAMES = (
    "memory_replace.py",
    "memory_insert.py",
    "memory_rethink.py",
    "memory_finish_edits.py",
)


def ensure_letta_memory_initialized(harness_dir: Path) -> None:
    memory_dir = harness_dir / "letta_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    notes = memory_dir / "notes.md"
    if not notes.exists():
        notes.write_text("", encoding="utf-8")
    persona = memory_dir / "persona.md"
    if not persona.exists():
        persona.write_text(PERSONA_STUB, encoding="utf-8")
    read_only = memory_dir / ".read_only"
    if not read_only.exists():
        read_only.write_text("persona\n", encoding="utf-8")


def install_memory_tools(scripts_dir: Path) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    for name in _TOOL_NAMES:
        shutil.copy2(source_dir / name, scripts_dir / name)
