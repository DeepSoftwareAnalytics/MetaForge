from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from rho.agent.cache import build_default_agent
from rho.agent.codex import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CodexAgent,
)
from rho.agent.codex_pool import (
    _reset_global_codex_pool_for_tests,
    configure_global_codex_pool,
    global_codex_pool,
)


@pytest.fixture(autouse=True)
def _reset_pool():
    _reset_global_codex_pool_for_tests()
    yield
    _reset_global_codex_pool_for_tests()


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _workspace(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.txt").write_text("same\n", encoding="utf-8")
    return root


def test_codex_agent_with_true_binary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.4"\n')
    agent = CodexAgent(codex_config_path=config, binary="/bin/true")
    trajectory = agent.run(workspace, "MODE: solve\n\nsay hi")
    assert trajectory.exit_code == 0
    assert trajectory.final_message == ""
    assert trajectory.events == []


def test_codex_config_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CodexAgent(codex_config_path=tmp_path / "does-not-exist.toml")


def test_codex_config_copied_verbatim_into_isolated_home(
    tmp_path: Path, monkeypatch
) -> None:
    external_codex_home = tmp_path / "external_codex_home"
    external_codex_home.mkdir()
    (external_codex_home / "config.toml").write_text(
        'model = "bad-model"\nmodel_reasoning_effort = "xhigh"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(external_codex_home))
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")

    config_text = (
        'model_provider = "azure-foundry"\n'
        'model = "gpt-5.4"\n'
        'model_reasoning_effort = "high"\n'
        "\n"
        "[model_providers.azure-foundry]\n"
        'name = "Azure OpenAI Sweden (via local proxy)"\n'
        'base_url = "http://127.0.0.1:4000/openai/v1"\n'
        'wire_api = "responses"\n'
    )
    config_path = _write_config(tmp_path / "codex_config.toml", config_text)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    binary = tmp_path / "fake_codex.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
if "--output-last-message" in argv:
    output_path = Path(argv[argv.index("--output-last-message") + 1])
    output_path.write_text("ok", encoding="utf-8")
codex_home = Path(os.environ["CODEX_HOME"])
print(json.dumps({
    "type": "argv",
    "argv": argv,
    "codex_home": str(codex_home),
    "home": os.environ.get("HOME"),
    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
    "config": (codex_home / "config.toml").read_text(encoding="utf-8"),
}))
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)

    agent = CodexAgent(
        codex_config_path=config_path,
        binary=str(binary),
        sandbox="danger-full-access",
    )
    trajectory = agent.run(workspace, "MODE: solve\n\nsay hi")

    event = trajectory.events[0]
    argv = event["argv"]
    assert "--ephemeral" in argv
    assert argv[argv.index("-m") + 1] == DEFAULT_CODEX_MODEL
    assert argv[argv.index("-c") + 1] == (
        f'model_reasoning_effort="{DEFAULT_REASONING_EFFORT}"'
    )
    assert event["codex_home"] != str(external_codex_home)
    assert event["home"] == event["codex_home"]
    assert event["openai_api_key"] is None
    # Config is a byte-for-byte copy of the file we pointed at.
    assert event["config"] == config_text
    assert "bad-model" not in event["config"]
    assert trajectory.model == DEFAULT_CODEX_MODEL
    assert trajectory.reasoning_effort == DEFAULT_REASONING_EFFORT


def test_codex_agent_obeys_global_pool_limit(tmp_path: Path) -> None:
    configure_global_codex_pool(2)
    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.4"\n')
    state_path = tmp_path / "state.json"
    binary = tmp_path / "fake_codex.py"
    binary.write_text(
        """#!/usr/bin/env python3
import fcntl
import json
import os
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
if argv == ["--version"]:
    print("fake-codex 1.0")
    raise SystemExit(0)

state_path = Path(os.environ["RHO_POOL_STATE"])
state_path.touch()
with state_path.open("r+") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    raw = handle.read()
    state = json.loads(raw) if raw else {"current": 0, "max_current": 0, "invocations": 0}
    state["current"] += 1
    state["invocations"] += 1
    state["max_current"] = max(state["max_current"], state["current"])
    handle.seek(0)
    handle.truncate()
    json.dump(state, handle)
    handle.flush()
    fcntl.flock(handle, fcntl.LOCK_UN)

time.sleep(0.1)

if "--output-last-message" in argv:
    output_path = Path(argv[argv.index("--output-last-message") + 1])
    output_path.write_text("ok", encoding="utf-8")

with state_path.open("r+") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    state = json.loads(handle.read())
    state["current"] -= 1
    handle.seek(0)
    handle.truncate()
    json.dump(state, handle)
    handle.flush()
    fcntl.flock(handle, fcntl.LOCK_UN)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    agent = CodexAgent(
        codex_config_path=config,
        binary=str(binary),
        sandbox="danger-full-access",
    )

    def run_one(ix: int):
        return agent.run(
            _workspace(tmp_path / f"ws_{ix}"),
            "MODE: solve\n\nsay hi",
            env={"RHO_POOL_STATE": str(state_path)},
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        trajectories = list(executor.map(run_one, range(5)))

    state = json.loads(state_path.read_text(encoding="utf-8"))
    snapshot = global_codex_pool().snapshot()
    assert all(traj.exit_code == 0 for traj in trajectories)
    assert state["invocations"] == 5
    assert state["max_current"] == 2
    assert snapshot["submitted"] == 5
    assert snapshot["completed"] == 5


def test_fallback_retry_uses_one_pool_permit(tmp_path: Path) -> None:
    configure_global_codex_pool(1)
    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.4"\n')
    state_path = tmp_path / "fallback_state.json"
    binary = tmp_path / "fake_fallback_codex.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
if argv == ["--version"]:
    print("fake-codex 1.0")
    raise SystemExit(0)

state_path = Path(os.environ["RHO_POOL_STATE"])
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {"invocations": 0}
state["invocations"] += 1
state_path.write_text(json.dumps(state), encoding="utf-8")

if state["invocations"] == 1:
    print("bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted")
else:
    if "--output-last-message" in argv:
        output_path = Path(argv[argv.index("--output-last-message") + 1])
        output_path.write_text("ok after fallback", encoding="utf-8")
    print(json.dumps({"type": "done"}))
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    agent = CodexAgent(
        codex_config_path=config,
        binary=str(binary),
        sandbox="workspace-write",
        fallback_sandbox="danger-full-access",
    )

    trajectory = agent.run(
        _workspace(tmp_path / "ws"),
        "MODE: solve\n\nsay hi",
        env={"RHO_POOL_STATE": str(state_path)},
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    snapshot = global_codex_pool().snapshot()
    assert trajectory.final_message == "ok after fallback"
    assert trajectory.events[0] == {
        "type": "sandbox_fallback",
        "from": "workspace-write",
        "to": "danger-full-access",
    }
    assert state["invocations"] == 2
    assert snapshot["submitted"] == 1
    assert snapshot["completed"] == 1


def test_cache_hit_bypasses_global_pool(tmp_path: Path) -> None:
    configure_global_codex_pool(1)
    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.4"\n')
    state_path = tmp_path / "cache_state.json"
    binary = tmp_path / "fake_cache_codex.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
if argv == ["--version"]:
    print("fake-codex 1.0")
    raise SystemExit(0)

state_path = Path(os.environ["RHO_POOL_STATE"])
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {"invocations": 0}
state["invocations"] += 1
state_path.write_text(json.dumps(state), encoding="utf-8")

workspace = Path(argv[argv.index("--cd") + 1])
(workspace / "answer.txt").write_text("cached answer\\n", encoding="utf-8")
if "--output-last-message" in argv:
    output_path = Path(argv[argv.index("--output-last-message") + 1])
    output_path.write_text("cached answer", encoding="utf-8")
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    agent = build_default_agent(
        CodexAgent(
            codex_config_path=config,
            binary=str(binary),
            sandbox="danger-full-access",
        ),
        mode="on",
        cache_dir=tmp_path / "agent-cache",
    )

    kwargs = {"env": {"RHO_POOL_STATE": str(state_path)}}
    first = agent.run(_workspace(tmp_path / "first"), "MODE: solve\n\nsay hi", **kwargs)
    second = agent.run(_workspace(tmp_path / "second"), "MODE: solve\n\nsay hi", **kwargs)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    snapshot = global_codex_pool().snapshot()
    assert first.final_message == "cached answer"
    assert second.final_message == "cached answer"
    assert (tmp_path / "second" / "answer.txt").read_text(encoding="utf-8") == "cached answer\n"
    assert state["invocations"] == 1
    assert snapshot["submitted"] == 1
    assert snapshot["completed"] == 1


def test_isolated_subprocess_env_forwards_azure_config_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: az account get-access-token (invoked by Codex's
    [model_providers.*.auth] block) needs AZURE_CONFIG_DIR to find the
    user's tenant cache, because we override HOME to the isolated codex
    home. If the user didn't set AZURE_CONFIG_DIR, default to the real
    HOME/.azure when it exists.
    """
    real_home = tmp_path / "real_home"
    (real_home / ".azure").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.delenv("AZURE_CONFIG_DIR", raising=False)

    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.5"\n')
    agent = CodexAgent(codex_config_path=config, binary="/bin/true")
    env = agent._build_subprocess_env(None)

    assert env is not None
    # HOME got overridden, but AZURE_CONFIG_DIR captured the real one.
    assert env["HOME"] != str(real_home)
    assert env["AZURE_CONFIG_DIR"] == str(real_home / ".azure")


def test_isolated_subprocess_env_respects_explicit_azure_config_dir(
    tmp_path: Path, monkeypatch
) -> None:
    explicit = tmp_path / "custom_az"
    explicit.mkdir()
    monkeypatch.setenv("AZURE_CONFIG_DIR", str(explicit))

    config = _write_config(tmp_path / "config.toml", 'model = "gpt-5.5"\n')
    agent = CodexAgent(codex_config_path=config, binary="/bin/true")
    env = agent._build_subprocess_env(None)

    assert env is not None
    assert env["AZURE_CONFIG_DIR"] == str(explicit)
