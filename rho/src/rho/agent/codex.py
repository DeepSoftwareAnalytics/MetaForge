from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
import uuid
import warnings
from pathlib import Path
from typing import Any

from rho.agent.codex_pool import global_codex_pool
from rho.observability import is_runtime_scratch
from rho.protocols import Trajectory, TrajectoryKind

DEFAULT_TIMEOUT_S = 900.0
EVAL_TIMEOUT_S = 300.0
MAX_SNAPSHOT_FILE_BYTES = 1 * 1024 * 1024
# Keep this value in sync with rho.agent.cache.MAX_SNAPSHOT_FILE_BYTES.
# Changing either value independently is a cache format-breaking change.
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh")
_CODEX_AUTH_FILES = ("auth.json",)
_ISOLATED_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    # AZURE_CONFIG_DIR lets `az account get-access-token` (invoked by
    # Codex's [model_providers.*.auth] block) find the user's tenant +
    # refresh-token cache. Without it `az` looks at $HOME/.azure, but
    # we override HOME to the isolated Codex home below — so we must
    # forward an explicit override. _build_subprocess_env synthesizes a
    # default of <real-HOME>/.azure when this env var isn't already set.
    "AZURE_CONFIG_DIR",
)


# ═══════════════════════════════════════════════════════════════════════
# Sandbox autodetect
#
# The codex CLI wraps its tool calls in bubblewrap when --sandbox is
# workspace-write or read-only. On Ubuntu 24.04+, the sysctl
# `kernel.apparmor_restrict_unprivileged_userns=1` (on by default) puts
# any unconfined process entering a user namespace into a catch-all
# apparmor profile at /etc/apparmor.d/unprivileged_userns that contains
# `audit deny capability`, which denies CAP_NET_ADMIN. bubblewrap then
# cannot bring up its loopback interface and dies with
# `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
# The outer `codex exec` exits 0 with the error in stdout; the inner
# tool calls silently fail.
#
# We detect this condition at first-use time via a cheap /proc read and,
# if triggered, default to `danger-full-access` with a runtime warning.
# Users who have a working bubblewrap apparmor profile (e.g. the local
# override described in the spec §15 assumption 3) can override by
# passing sandbox="workspace-write" explicitly to CodexAgent(...).
# ═══════════════════════════════════════════════════════════════════════

_APPARMOR_USERNS_SYSCTL = Path(
    "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
)

_detected_sandbox: str | None = None
_sandbox_warning_emitted = False
_sandbox_lock = threading.Lock()


def _detect_default_sandbox() -> tuple[str, str | None]:
    """Return (sandbox, warning_message_or_None) based on host heuristics.

    Pure stdlib, no codex subprocess, no I/O beyond one /proc read.
    """
    try:
        if _APPARMOR_USERNS_SYSCTL.read_text().strip() == "1":
            return (
                "danger-full-access",
                (
                    "CodexAgent: bubblewrap sandbox cannot configure its "
                    "network namespace on this host — detected "
                    "kernel.apparmor_restrict_unprivileged_userns=1 "
                    "(typical on Ubuntu 23.10+). Falling back to "
                    "'danger-full-access': codex tool calls will run "
                    "WITHOUT bwrap-level filesystem/network isolation. "
                    "To restore the sandbox, see "
                    "docs/superpowers/specs/"
                    "2026-04-11-rho-basic-impl-design.md §15 "
                    "assumption 3, then pass sandbox='workspace-write' "
                    "to CodexAgent explicitly."
                ),
            )
    except (OSError, ValueError):
        pass  # not Linux, /proc not readable, etc.
    return "workspace-write", None


def default_sandbox() -> str:
    """Cached per-process default sandbox. Emits a one-shot warning on
    degraded environments so it shows up in pytest output and dev runs
    without being printed per-call."""
    global _detected_sandbox, _sandbox_warning_emitted
    if _detected_sandbox is not None:
        return _detected_sandbox
    with _sandbox_lock:
        if _detected_sandbox is not None:
            return _detected_sandbox
        sandbox, warning = _detect_default_sandbox()
        _detected_sandbox = sandbox
        if warning and not _sandbox_warning_emitted:
            warnings.warn(warning, RuntimeWarning, stacklevel=3)
            _sandbox_warning_emitted = True
    return _detected_sandbox


class CodexAgent:
    def __init__(
        self,
        codex_config_path: Path,
        model: str | None = DEFAULT_CODEX_MODEL,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        sandbox: str | None = None,
        # fallback_sandbox is a LAST-RESORT retry when the primary sandbox
        # emits a bubblewrap failure at runtime (exit code 0 + `bwrap:` /
        # `Operation not permitted` in stdout/stderr). It defaults to None
        # — no silent escalation to a weaker sandbox. In most environments
        # autodetect (sandbox=None) already picks the right default so the
        # fallback path is not exercised.
        fallback_sandbox: str | None = None,
        network_access: bool = True,
        binary: str = "codex",
        extra_flags: tuple[str, ...] = (),
        default_timeout_s: float | None = DEFAULT_TIMEOUT_S,
        isolate_codex_home: bool = True,
        codex_auth_home: Path | None = None,
        ephemeral: bool = True,
        stream_logs: bool = False,
    ) -> None:
        if (
            reasoning_effort is not None
            and reasoning_effort not in REASONING_EFFORT_CHOICES
        ):
            allowed = ", ".join(REASONING_EFFORT_CHOICES)
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        self.codex_config_path = Path(codex_config_path).expanduser().resolve()
        if not self.codex_config_path.is_file():
            raise FileNotFoundError(
                f"codex config not found: {self.codex_config_path}"
            )
        config = tomllib.loads(self.codex_config_path.read_text(encoding="utf-8"))
        provider = config.get("model_providers", {}).get(config.get("model_provider"), {})
        self._provider_env_key = provider.get("env_key")
        self.model = model
        self.reasoning_effort = reasoning_effort
        # sandbox=None → autodetect (may emit a warning on degraded hosts).
        # Passing an explicit string disables autodetect and the warning.
        self.sandbox = sandbox if sandbox is not None else default_sandbox()
        self.fallback_sandbox = fallback_sandbox
        self.network_access = network_access
        self.binary = binary
        self.extra_flags = tuple(extra_flags)
        self.default_timeout_s = default_timeout_s
        self.isolate_codex_home = isolate_codex_home
        self.codex_auth_home = codex_auth_home or default_codex_auth_home()
        self.ephemeral = ephemeral
        self.stream_logs = stream_logs
        self._isolated_home_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._isolated_codex_home: Path | None = None
        self._isolated_config_sha256 = ""
        if self.isolate_codex_home:
            self._prepare_isolated_codex_home()
        self._binary_version_cache: str | None = None
        self._version_lock = threading.Lock()

    @property
    def cache_env_signature(self) -> dict[str, str]:
        return {
            "agent_class": "CodexAgent",
            "model": self.model or "",
            "model_reasoning_effort": self.reasoning_effort or "",
            "binary": self.binary,
            "binary_version": self._binary_version(),
            "sandbox": self.sandbox,
            "fallback_sandbox": self.fallback_sandbox or "",
            "network_access": str(self.network_access).lower(),
            "extra_flags": "\x1f".join(self.extra_flags),
            "codex_home_mode": "isolated" if self.isolate_codex_home else "ambient",
            "codex_config_sha256": self._isolated_config_sha256,
            "ephemeral": str(self.ephemeral).lower(),
        }

    @property
    def isolation_metadata(self) -> dict[str, Any]:
        return {
            "codex_home_mode": "isolated" if self.isolate_codex_home else "ambient",
            "inherits_user_config": not self.isolate_codex_home,
            "auth_source": str(self.codex_auth_home) if self.isolate_codex_home else None,
            "subprocess_env": "minimal" if self.isolate_codex_home else "inherited",
            "ephemeral": self.ephemeral,
            "config_sha256": self._isolated_config_sha256 or None,
            "config_source": str(self.codex_config_path),
        }

    def _prepare_isolated_codex_home(self) -> None:
        self._isolated_home_tmp = tempfile.TemporaryDirectory(
            prefix="rho_codex_home_"
        )
        codex_home = Path(self._isolated_home_tmp.name)
        codex_home.chmod(0o700)
        self._isolated_codex_home = codex_home

        self.codex_auth_home = self.codex_auth_home.expanduser().resolve()
        for name in _CODEX_AUTH_FILES:
            src = self.codex_auth_home / name
            if src.exists():
                dst = codex_home / name
                shutil.copy2(src, dst)
                dst.chmod(0o600)

        config_bytes = self.codex_config_path.read_bytes()
        dst_config = codex_home / "config.toml"
        dst_config.write_bytes(config_bytes)
        dst_config.chmod(0o600)
        self._isolated_config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    def _subprocess_env(self) -> dict[str, str] | None:
        return self._build_subprocess_env(None)

    def _build_subprocess_env(self, extra_env: dict[str, str] | None) -> dict[str, str] | None:
        if not self.isolate_codex_home:
            if extra_env is None:
                return None
            env = os.environ.copy()
            env.update({key: str(value) for key, value in extra_env.items()})
            return env
        if self._isolated_codex_home is None:
            raise RuntimeError("isolated Codex home was not initialized")

        env = {
            key: value
            for key in _ISOLATED_ENV_ALLOWLIST
            if (value := os.environ.get(key)) is not None
        }
        env.setdefault("PATH", os.defpath)
        # Forward only the credential requested by the selected provider. Do not
        # inherit unrelated API keys or the ambient Codex configuration.
        if self._provider_env_key and self._provider_env_key in os.environ:
            env[self._provider_env_key] = os.environ[self._provider_env_key]
        # If the user hasn't set AZURE_CONFIG_DIR but has a real ~/.azure,
        # forward that so Codex's [model_providers.*.auth] az invocation
        # can locate the tenant + refresh-token cache. We capture the
        # outer HOME *before* overriding it to the isolated codex home.
        if "AZURE_CONFIG_DIR" not in env:
            outer_home = os.environ.get("HOME")
            if outer_home:
                candidate = Path(outer_home) / ".azure"
                if candidate.is_dir():
                    env["AZURE_CONFIG_DIR"] = str(candidate)
        env["CODEX_HOME"] = str(self._isolated_codex_home)
        env["HOME"] = str(self._isolated_codex_home)
        if extra_env is not None:
            env.update({key: str(value) for key, value in extra_env.items()})
        return env

    def _binary_version(self) -> str:
        if self._binary_version_cache is not None:
            return self._binary_version_cache
        with self._version_lock:
            if self._binary_version_cache is not None:
                return self._binary_version_cache
            try:
                proc = subprocess.run(
                    [self.binary, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self._binary_version_cache = (
                    (proc.stdout or proc.stderr).strip() or "unknown"
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                self._binary_version_cache = "unknown"
        return self._binary_version_cache

    def run(
        self,
        workspace: Path,
        instructions: str,
        *,
        output_schema: dict[str, Any] | None = None,
        task_id: str = "",
        harness_id: str = "",
        kind: TrajectoryKind = "solve",
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Trajectory:
        meta = workspace / ".rho"
        meta.mkdir(exist_ok=True)
        (meta / "instructions.md").write_text(instructions, encoding="utf-8")
        last_msg_path = meta / "last_message.txt"
        last_msg_path.write_text("", encoding="utf-8")

        before = _snapshot(workspace, exclude={".rho"})
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        retry_note: dict[str, Any] | None = None

        with global_codex_pool().acquire():
            t0 = time.monotonic()
            stdout_raw, stderr_raw, exit_code, timed_out = self._run_once(
                workspace=workspace,
                instructions=instructions,
                output_schema=output_schema,
                last_msg_path=last_msg_path,
                sandbox=self.sandbox,
                timeout_s=effective_timeout,
                env=env,
            )
            # Trigger fallback on any sandbox failure, regardless of outer exit code.
            # On codex-cli 0.120+ the outer `codex exec` process can exit 0 even
            # when its inner tool calls are blocked by bubblewrap (e.g. RTM_NEWADDR
            # failing in a namespace-restricted host). In that case the first run's
            # exit_code is 0 but stdout/stderr carry `bwrap:` / `Operation not
            # permitted`, and we still need to retry with the fallback sandbox.
            if (
                not timed_out
                and self.fallback_sandbox
                and self.fallback_sandbox != self.sandbox
                and _looks_like_sandbox_failure(stdout_raw, stderr_raw)
            ):
                retry_note = {
                    "type": "sandbox_fallback",
                    "from": self.sandbox,
                    "to": self.fallback_sandbox,
                }
                stdout_raw, stderr_raw, exit_code, timed_out = self._run_once(
                    workspace=workspace,
                    instructions=instructions,
                    output_schema=output_schema,
                    last_msg_path=last_msg_path,
                    sandbox=self.fallback_sandbox,
                    timeout_s=effective_timeout,
                    env=env,
                )
            wall = time.monotonic() - t0
        after = _snapshot(workspace, exclude={".rho"})

        events: list[dict[str, Any]] = []
        if retry_note is not None:
            events.append(retry_note)
        for line in stdout_raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                events.append({"type": "raw_stdout", "line": stripped})
        # stderr stays in stderr.log / Trajectory.stderr — replaying it as events floods the stream when codex dumps its model-refresh JSON response.

        try:
            final = last_msg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            final = ""

        diff, deletions = _diff(before, after)
        return Trajectory(
            id=f"traj_{uuid.uuid4().hex[:10]}",
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=events,
            final_message=final,
            stdout=stdout_raw,
            stderr=stderr_raw,
            workspace_diff=diff,
            workspace_deletions=frozenset(deletions),
            exit_code=exit_code,
            wall_time_s=wall,
            timed_out=timed_out,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def _run_once(
        self,
        *,
        workspace: Path,
        instructions: str,
        output_schema: dict[str, Any] | None,
        last_msg_path: Path,
        sandbox: str,
        timeout_s: float | None,
        env: dict[str, str] | None,
    ) -> tuple[str, str, int, bool]:
        cmd: list[str] = [
            self.binary,
            "exec",
            "--cd",
            str(workspace),
            "--json",
            "--sandbox",
            sandbox,
            "--skip-git-repo-check",
            "--output-last-message",
            str(last_msg_path),
        ]
        if self.ephemeral:
            cmd += ["--ephemeral"]
        if self.model:
            cmd += ["-m", self.model]
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        if self.network_access:
            cmd += ["-c", "sandbox_workspace_write.network_access=true"]
        if output_schema is not None:
            schema_path = workspace / ".rho" / "output_schema.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cmd += ["--output-schema", str(schema_path)]
            # Some Responses-compatible providers treat output schemas as a
            # hint. Also state the contract in the prompt; keep the raw reply
            # intact so callers can detect malformed output.
            instructions += (
                "\n\nYour final reply must be exactly one JSON value matching "
                "the following JSON Schema, with no Markdown fences or extra text:\n"
                + json.dumps(output_schema, ensure_ascii=False)
            )
        cmd += list(self.extra_flags)
        cmd += [instructions]

        if self.stream_logs:
            stdout_path = workspace / ".rho" / "live.stdout.jsonl"
            stderr_path = workspace / ".rho" / "live.stderr.log"
            with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
                try:
                    proc = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, stdout=stdout_file, stderr=stderr_file,
                        text=True, check=False, timeout=timeout_s,
                        env=self._build_subprocess_env(env),
                    )
                    code, timed_out = proc.returncode, False
                except subprocess.TimeoutExpired:
                    code, timed_out = -1, True
            return (stdout_path.read_text(errors="replace"),
                    stderr_path.read_text(errors="replace"), code, timed_out)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=self._build_subprocess_env(env),
            )
            return proc.stdout or "", proc.stderr or "", proc.returncode, False
        except subprocess.TimeoutExpired as exc:
            return _coerce_timeout_stream(exc.stdout), _coerce_timeout_stream(exc.stderr), -1, True


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def default_codex_auth_home() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".codex"
    except (ImportError, KeyError, OSError):
        return Path.home() / ".codex"


def _looks_like_sandbox_failure(stdout: str, stderr: str) -> bool:
    haystack = f"{stdout}\n{stderr}"
    return "bwrap:" in haystack or "Operation not permitted" in haystack


def _snapshot(root: Path, exclude: set[str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in exclude:
            continue
        if is_runtime_scratch(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SNAPSHOT_FILE_BYTES:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            out[str(rel)] = b"HASH:" + digest.hexdigest().encode("ascii")
        else:
            out[str(rel)] = path.read_bytes()
    return out


def _diff(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> tuple[dict[str, bytes], set[str]]:
    changes: dict[str, bytes] = {}
    for key, value in after.items():
        if before.get(key) != value:
            changes[key] = value
    deletions = {key for key in before if key not in after}
    return changes, deletions
