from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from rho.agent.base import Agent
from rho.agent.cache import CachingAgent, build_default_agent
from rho.agent.codex import CodexAgent


@dataclass(frozen=True)
class CodexAgentHandle:
    agent: Agent
    caching_agent: CachingAgent | None


@pytest.fixture(scope="session")
def codex_binary() -> str:
    binary = shutil.which("codex")
    if binary is None:
        pytest.skip(
            "codex CLI not found on PATH; install from https://github.com/openai/codex to run real-codex smoke tests"
        )

    if not _codex_exec_is_usable(binary):
        pytest.skip(
            "codex CLI is present but `codex exec` cannot use local command execution in this environment; skipping real-codex smoke tests"
        )
    return binary


def _oss_extra_flags() -> tuple[str, ...]:
    """Build extra CLI flags when RHO_OSS_PROVIDER is set."""
    provider = os.environ.get("RHO_OSS_PROVIDER", "")
    if not provider:
        return ()
    return ("--oss", "--local-provider", provider)


@pytest.fixture(scope="session")
def codex_agent_factory(codex_binary: str):
    """Factory for CodexAgent configured for real-codex tests.

    Set ``RHO_MODEL`` and ``RHO_OSS_PROVIDER`` env vars
    to run tests against a local model (e.g. vLLM behind LM Studio
    compatible endpoint).

    Relies on :func:`rho.agent.codex.default_sandbox` autodetect:
    - Hosts with working bubblewrap (default elsewhere): sandbox defaults
      to ``workspace-write``.
    - Hosts where /proc/sys/kernel/apparmor_restrict_unprivileged_userns
      is ``1`` (Ubuntu 23.10+ default): sandbox defaults to
      ``danger-full-access`` with a one-shot runtime warning so the test
      suite still runs end-to-end without wasting time on a
      guaranteed-to-fail bwrap retry.
    """
    env_model = os.environ.get("RHO_MODEL")
    oss_flags = _oss_extra_flags()
    default_codex_config = _default_codex_config_for_tests()

    def _make(**kwargs) -> CodexAgentHandle:
        cache_mode = kwargs.pop("cache_mode", "off")
        cache_dir = kwargs.pop("cache_dir", None)
        kwargs.setdefault("binary", codex_binary)
        kwargs.setdefault("fallback_sandbox", "danger-full-access")
        kwargs.setdefault("codex_config_path", default_codex_config)
        if env_model:
            kwargs.setdefault("model", env_model)
        if oss_flags:
            existing = kwargs.get("extra_flags", ())
            kwargs["extra_flags"] = tuple(existing) + oss_flags
        agent = build_default_agent(
            CodexAgent(**kwargs),
            mode=cache_mode,
            cache_dir=cache_dir,
        )
        caching_agent = agent if isinstance(agent, CachingAgent) else None
        return CodexAgentHandle(agent=agent, caching_agent=caching_agent)

    return _make


def _default_codex_config_for_tests() -> Path:
    """Pick a codex config for real-codex smoke tests.

    Prefer ``$RHO_CODEX_CONFIG`` if set, then the in-repo
    ``configs/codex.azure-foundry.toml`` (Codex 0.130 direct Entra-auth via
    az account get-access-token), and finally fall back to the user's own
    ``~/.codex/config.toml``.
    """
    override = os.environ.get("RHO_CODEX_CONFIG")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
    repo_config = (
        Path(__file__).resolve().parents[2] / "configs" / "codex.azure-foundry.toml"
    )
    if repo_config.is_file():
        return repo_config
    return (Path.home() / ".codex" / "config.toml").resolve()


def _codex_exec_is_usable(binary: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "probe.txt").write_text("OK\n", encoding="utf-8")
        output_path = root / "last.txt"
        proc = subprocess.run(
            [
                binary,
                "exec",
                "--cd",
                str(root),
                "--json",
                "--sandbox",
                "danger-full-access",
                "--skip-git-repo-check",
                "--output-last-message",
                str(output_path),
                "Read probe.txt and reply with exactly OK.",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        if proc.returncode != 0:
            return False
        final_message = output_path.read_text(encoding="utf-8", errors="replace").strip()
        combined = f"{proc.stdout}\n{proc.stderr}\n{final_message}"
        if "Operation not permitted" in combined or "bwrap:" in combined:
            return False
        return final_message == "OK"
