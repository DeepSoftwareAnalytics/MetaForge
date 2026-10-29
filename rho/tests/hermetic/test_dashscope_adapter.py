import importlib.util
import json
from pathlib import Path

import pytest

from rho.agent.codex import CodexAgent
from rho.cli import _build_agent, _build_parser


def test_only_selected_provider_credential_reaches_codex(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-selected-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("OTHER_PROVIDER_KEY", "other-key")
    config = tmp_path / "config.toml"
    config.write_text('model_provider = "ds"\n[model_providers.ds]\nenv_key = "DASHSCOPE_API_KEY"\n'
                      '[model_providers.other]\nenv_key = "OTHER_PROVIDER_KEY"\n')
    binary = tmp_path / "codex"
    binary.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv
Path(args[args.index("--output-last-message") + 1]).write_text("ok")
print(json.dumps({"type": "environment", "selected": os.getenv("DASHSCOPE_API_KEY"),
                  "unrelated": os.getenv("OPENAI_API_KEY"), "other": os.getenv("OTHER_PROVIDER_KEY")}))
''')
    binary.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = CodexAgent(config, binary=str(binary), sandbox="workspace-write",
                       codex_auth_home=tmp_path / "empty-auth")
    trajectory = agent.run(workspace, "test")
    assert trajectory.exit_code == 0
    assert trajectory.events == [{"type": "environment", "selected": "test-selected-key",
                                   "unrelated": None, "other": None}]
    assert "test-selected-key" not in json.dumps(agent.isolation_metadata)


def test_cli_passes_explicit_binary_and_sandbox(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model = "test"\n')
    args = _build_parser().parse_args([
        "solve", "--dataset", "directory:tests/fixtures/toy_dataset", "--task", "task_001",
        "--harness", str(tmp_path), "--run-dir", str(tmp_path / "run"),
        "--codex-config", str(config), "--codex-binary", "/bin/true",
        "--codex-sandbox", "read-only", "--cache", "off",
    ])
    agent = _build_agent(args, run_dir=tmp_path / "run")
    assert agent.binary == "/bin/true"
    assert agent.sandbox == "read-only"


def test_local_launcher_preserves_explicit_options():
    script = Path(__file__).resolve().parents[2] / "scripts/rho-ds.py"
    spec = importlib.util.spec_from_file_location("rho_ds_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = module.command_arguments(["evolve", "--model=custom", "--codex-concurrency", "2"])
    assert "--model" not in argv
    assert argv.count("--codex-concurrency") == 1
    assert argv[argv.index("--codex-concurrency") + 1] == "2"
    assert argv[argv.index("--judge-model") + 1] == "openai/deepseek-v4-flash-0731"


def test_schema_is_also_explained_to_compatible_providers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config = tmp_path / "config.toml"
    config.write_text('model = "test"\n')
    captured = []

    def run(cmd, **kwargs):
        captured.append(cmd)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("rho.agent.codex.subprocess.run", run)
    agent = CodexAgent(config, sandbox="read-only", codex_auth_home=tmp_path / "auth")
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    agent.run(tmp_path, "Original task.", output_schema=schema)
    cmd = next(cmd for cmd in captured if "exec" in cmd)
    assert "--output-schema" in cmd
    assert cmd[-1].startswith("Original task.")
    assert "no Markdown fences or extra text" in cmd[-1]
    assert json.dumps(schema) in cmd[-1]


@pytest.mark.parametrize("validation,consistency", [(True, True), (True, False), (False, True)])
def test_diagnosis_schema_preserves_ablation_inputs(tmp_path, validation, consistency):
    from rho.agent.fake import FakeAgent, FakeResponse
    from rho.datasets.directory import DirectoryTask
    from rho.orchestrators.diagnose import diagnose
    from rho.stores.harness import FilesystemHarnessStore

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "prompt.md").write_text("Compute 1 + 1.")
    harness = FilesystemHarnessStore(tmp_path / "harness").empty()
    task = DirectoryTask(task_dir, harness)

    def respond(workspace, instructions, schema):
        fields = schema["properties"]
        assert ("trajectory_analyses" in fields) == validation
        assert ("failure_mode_analysis" in fields) == validation
        assert ("inconsistency_analysis" in fields) == consistency
        result = {"task_id": task.id, "severity": 0.0, "harness_improvement_direction": ""}
        if validation:
            result["trajectory_analyses"] = [
                {"trajectory": f"trajectory_{i}", "successful": 1,
                 "quality_analysis": "Correct", "issues": ""} for i in range(3)
            ]
            result["failure_mode_analysis"] = ""
        if consistency:
            result["inconsistency_analysis"] = ""
        return FakeResponse(final_message=json.dumps(result))

    agent = FakeAgent({"diagnose": respond})
    _, result = diagnose(agent, task, [], harness, workdir=tmp_path / "work",
                         include_validation=validation, include_consistency=consistency)
    assert result.severity == 0.0


@pytest.mark.parametrize("reply", [
    '{"value": 3, "rationale": "ok"}',
    '```json\n{"value": 3, "rationale": "ok"}\n```',
    'Evidence inspected.\n{"value": 3, "rationale": "ok"}',
])
def test_structured_reply_accepts_provider_preamble(reply):
    from rho.orchestrators._util import extract_json_object
    assert json.loads(extract_json_object(reply)) == {"value": 3, "rationale": "ok"}


@pytest.mark.parametrize("reply", [
    'Evidence.\n{"trajectory_2", "successful": 1}',
    'Evidence.\n{"value": -3}\n{"value": 3}',
    'Evidence.\n{"broken": {"value": 3}',
])
def test_structured_reply_does_not_repair_or_choose_between_objects(reply):
    from rho.orchestrators._util import extract_json_object
    with pytest.raises(json.JSONDecodeError):
        json.loads(extract_json_object(reply))
