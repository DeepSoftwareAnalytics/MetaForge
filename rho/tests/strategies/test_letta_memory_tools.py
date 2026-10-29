from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rho.strategies.letta_memory_tools.bootstrap import (
    ensure_letta_memory_initialized,
    install_memory_tools,
)
from rho.strategies.letta_memory_tools.vendored_constants import (
    CORE_MEMORY_LINE_NUMBER_WARNING,
)


def test_memory_tools_match_letta_edit_semantics_and_errors(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    notes = ws / "harness" / "letta_memory" / "notes.md"
    notes.write_text("alpha\nbeta\n", encoding="utf-8")

    result = _run_tool(
        ws,
        "memory_replace.py",
        "--label",
        "notes",
        "--old_string",
        "beta",
        "--new_string",
        "gamma",
    )
    assert result.returncode == 0
    assert notes.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert result.stdout == "alpha\ngamma\n"

    result = _run_tool(
        ws,
        "memory_replace.py",
        "--label",
        "notes",
        "--old_string",
        "missing",
        "--new_string",
        "x",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "No replacement was performed, old_string `missing` did not appear "
        "verbatim in memory block with label `notes`.\n"
    )

    notes.write_text("dupe\ndupe\n", encoding="utf-8")
    result = _run_tool(
        ws,
        "memory_replace.py",
        "--label",
        "notes",
        "--old_string",
        "dupe",
        "--new_string",
        "once",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "No replacement was performed. Multiple occurrences of old_string "
        "`dupe` in lines [1, 2]. Please ensure it is unique.\n"
    )

    result = _run_tool(
        ws,
        "memory_replace.py",
        "--label",
        "notes",
        "--old_string",
        "\nLine 7: dupe",
        "--new_string",
        "once",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "old_string contains a line number prefix, which is not allowed. "
        "Do not include line numbers when calling memory tools (line numbers "
        "are for display purposes only).\n"
    )

    result = _run_tool(
        ws,
        "memory_replace.py",
        "--label",
        "notes",
        "--old_string",
        CORE_MEMORY_LINE_NUMBER_WARNING,
        "--new_string",
        "once",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "old_string contains a line number warning, which is not allowed. "
        "Do not include line number information when calling memory tools "
        "(line numbers are for display purposes only).\n"
    )


def test_memory_insert_rethink_finish_and_read_only_cases(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    memory_root = ws / "harness" / "letta_memory"
    notes = memory_root / "notes.md"
    persona = memory_root / "persona.md"
    notes.write_text("first\nthird", encoding="utf-8")

    result = _run_tool(
        ws,
        "memory_insert.py",
        "--label",
        "notes",
        "--new_string",
        "second",
        "--insert_line",
        "1",
    )
    assert result.returncode == 0
    assert notes.read_text(encoding="utf-8") == "first\nsecond\nthird"

    result = _run_tool(
        ws,
        "memory_insert.py",
        "--label",
        "notes",
        "--new_string",
        "bad",
        "--insert_line",
        "99",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "Invalid `insert_line` parameter: 99. It should be within the range "
        "of lines of the memory block: [0, 3], or -1 to append to the end "
        "of the memory block.\n"
    )

    result = _run_tool(
        ws,
        "memory_insert.py",
        "--label",
        "notes",
        "--new_string",
        f"prefix\nLine 3: {CORE_MEMORY_LINE_NUMBER_WARNING}",
    )
    assert result.returncode != 0
    assert result.stderr == (
        "new_string contains a line number prefix, which is not allowed. "
        "Do not include line numbers when calling memory tools (line numbers "
        "are for display purposes only).\n"
    )

    result = _run_tool(
        ws,
        "memory_rethink.py",
        "--label",
        "research",
        "--new_memory",
        "consolidated\nnotes",
    )
    assert result.returncode == 0
    assert (memory_root / "research.md").read_text(encoding="utf-8") == "consolidated\nnotes"

    result = _run_tool(
        ws,
        "memory_rethink.py",
        "--label",
        "research",
        "--new_memory",
        CORE_MEMORY_LINE_NUMBER_WARNING,
    )
    assert result.returncode != 0
    assert result.stderr == (
        "new_memory contains a line number warning, which is not allowed. "
        "Do not include line number information when calling memory tools "
        "(line numbers are for display purposes only).\n"
    )

    result = _run_tool(
        ws,
        "memory_rethink.py",
        "--label",
        "persona",
        "--new_memory",
        "changed",
    )
    assert result.returncode != 0
    assert "read-only memory block" in result.stderr
    assert persona.read_text(encoding="utf-8") == "# persona\n\n(no persona policies in this deployment)\n"

    result = _run_tool(ws, "memory_finish_edits.py")
    assert result.returncode == 0
    assert (ws / ".rho" / "letta_finish_edits").read_text(encoding="utf-8") == "finished\n"


def test_memory_tools_reject_unsafe_labels_and_memory_root_cli_override(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)

    result = _run_tool(
        ws,
        "memory_rethink.py",
        "--label",
        "../outside",
        "--new_memory",
        "bad",
    )
    assert result.returncode != 0
    assert "invalid memory block label" in result.stderr

    result = _run_tool(
        ws,
        "memory_rethink.py",
        "--label",
        "notes",
        "--new_memory",
        "bad",
        "--memory-root",
        str(tmp_path),
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --memory-root" in result.stderr


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "harness").mkdir(parents=True)
    (ws / ".rho").mkdir()
    ensure_letta_memory_initialized(ws / "harness")
    install_memory_tools(ws / "scripts")
    return ws


def _run_tool(ws: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LETTA_MEMORY_ROOT"] = str((ws / "harness" / "letta_memory").resolve())
    env["LETTA_SCRIPTS"] = str((ws / "scripts").resolve())
    return subprocess.run(
        [sys.executable, str(ws / "scripts" / script), *args],
        cwd=ws / "harness",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
