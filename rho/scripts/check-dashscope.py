#!/usr/bin/env python3
"""Opt-in live smoke test (one Codex request; incurs API usage)."""
from __future__ import annotations

import json
from pathlib import Path
import random
import runpy
import tempfile

from rho.agent.codex import CodexAgent

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    launcher = runpy.run_path(str(ROOT / "scripts/rho-ds.py"))
    launcher["configure_environment"]()
    (ROOT / "runs").mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="setup-smoke-", dir=ROOT / "runs"))
    workspace = run / "workspace"
    workspace.mkdir()
    a, b = random.randint(100, 999), random.randint(100, 999)
    (workspace / "input.json").write_text(json.dumps({"a": a, "b": b}))
    agent = CodexAgent(
        ROOT / "configs/codex.dashscope.toml",
        model=launcher["MODEL"],
        binary=str(ROOT.parent / ".local/bin/codex"),
        sandbox="danger-full-access",
        codex_auth_home=run / "empty-auth",
    )
    trajectory = agent.run(
        workspace,
        "Use shell tools to read input.json, add a and b, and write the decimal sum "
        "to result.txt. Report sum and file_written. Stay inside this workspace "
        "and do not access the network.",
        output_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}, "file_written": {"type": "boolean"}},
            "required": ["sum", "file_written"],
            "additionalProperties": False,
        },
        timeout_s=120,
    )
    (run / "events.jsonl").write_text(trajectory.stdout)
    (run / "stderr.log").write_text(trajectory.stderr)
    (run / "final.txt").write_text(trajectory.final_message)
    try:
        parsed = json.loads(trajectory.final_message)
    except json.JSONDecodeError:
        parsed = None
    file = workspace / "result.txt"
    tools = sum(e.get("type") == "item.completed" and
                e.get("item", {}).get("type") == "command_execution"
                for e in trajectory.events)
    ok = (trajectory.exit_code == 0 and not trajectory.timed_out and tools > 0
          and parsed == {"sum": a + b, "file_written": True}
          and file.is_file() and file.read_text().strip() == str(a + b))
    summary = {"ok": ok, "exit_code": trajectory.exit_code, "timed_out": trajectory.timed_out,
               "tool_calls": tools, "final_json": parsed, "seconds": trajectory.wall_time_s,
               "model": launcher["MODEL"], "run_dir": str(run)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
