#!/usr/bin/env python3
"""Local RHO entry point using DashScope and the existing Codex CLI."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from rho.local_settings import codex_binary, codex_config, configure_network

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-v4-flash-0731"


def configure_environment() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        key_file = os.environ.get("RHO_DASHSCOPE_KEY_FILE")
        if not key_file:
            raise RuntimeError('Set DASHSCOPE_API_KEY or RHO_DASHSCOPE_KEY_FILE outside the repository')
        path = Path(key_file).expanduser()
        keys = [line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.strip().isdigit() and not line.lstrip().startswith("#")]
        index = int(os.environ.get("RHO_DASHSCOPE_KEY_INDEX", "0"))
        if index < 0 or index >= len(keys):
            raise ValueError("RHO_DASHSCOPE_KEY_INDEX does not identify an available credential")
        os.environ["DASHSCOPE_API_KEY"] = keys[index]
    # The optional difficulty/DPP selector uses LiteLLM, separate from Codex.
    os.environ["RHO_LLM_BACKEND"] = "off"
    os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""),
                                                 "dashscope.aliyuncs.com,localhost,127.0.0.1"]))
    configure_network()


def command_arguments(argv: list[str]) -> list[str]:
    if not argv:
        return ["--help"]
    agent_commands = {"evolve", "solve", "grade", "select"}
    if argv[0] not in agent_commands:
        return argv
    def supplied(flag: str) -> bool:
        return any(arg == flag or arg.startswith(flag + "=") for arg in argv[1:])
    defaults = {
        "--codex-config": str(codex_config(ROOT)),
        "--model": MODEL,
        "--codex-binary": str(codex_binary()),
        "--codex-sandbox": "workspace-write",
        "--codex-concurrency": "8",
    }
    if argv[0] in {"evolve", "select"}:
        defaults["--judge-model"] = "openai/" + MODEL
    result = list(argv)
    for flag, value in defaults.items():
        if not supplied(flag):
            result.extend([flag, value])
    return result


def main() -> int:
    argv = sys.argv[1:]
    if argv and not any(arg in {"-h", "--help"} for arg in argv) and argv[0] in {
        "evolve", "solve", "grade", "select"
    }:
        configure_environment()
    from rho.cli import main as rho_main
    return rho_main(command_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
