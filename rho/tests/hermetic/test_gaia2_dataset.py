from __future__ import annotations

import json

from rho.datasets.loader import load_dataset
from rho.stores.harness import EMPTY_HARNESS_ID, FilesystemHarnessStore


def test_loads_local_gaia2_jsonl_and_uses_empty_harness(tmp_path) -> None:
    dataset_path = tmp_path / "mini.jsonl"
    rows = [
        {
            "id": "row-1",
            "scenario_id": "scenario-a",
            "data": {"metadata": {"definition": {"hints": ["Send the note."]}}},
        },
        {
            "id": "row-2",
            "scenario_id": "scenario-b",
            "data": {"metadata": {"definition": {"hints": ["Find the file."]}}},
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    harness_store = FilesystemHarnessStore(tmp_path / "harness_store")

    dataset = load_dataset(f"gaia2:{dataset_path}#config=mini", harness_store=harness_store)
    tasks = list(dataset.train) + list(dataset.val) + list(dataset.test)

    assert sorted(task.id for task in tasks) == ["mini/scenario-a", "mini/scenario-b"]
    assert {task.harness.id for task in tasks} == {EMPTY_HARNESS_ID}


def test_gaia2_task_materialize_writes_prompt_and_runtime_dir(tmp_path) -> None:
    dataset_path = tmp_path / "mini.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "scenario_id": "scenario-a",
                "data": {"metadata": {"definition": {"hints": ["Send the note."]}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    harness_store = FilesystemHarnessStore(tmp_path / "harness_store")
    dataset = load_dataset(f"gaia2:{dataset_path}", harness_store=harness_store)
    task = next(iter(list(dataset.train) + list(dataset.val) + list(dataset.test)))

    dest = tmp_path / "task"
    task.materialize(dest)

    prompt = (dest / "prompt.md").read_text(encoding="utf-8")
    assert "Send the note." in prompt
    assert "tools/are.py" in prompt
    assert "read the user's request" in prompt
    assert (
        "python task/tools/are.py call AgentUserInterface get_all_messages --json '{}'"
        in prompt
    )
    assert (
        "python task/tools/are.py call AgentUserInterface send_message_to_user --json"
        in prompt
    )
    assert "Codex final message is only a run summary" in prompt
    assert "Do not read `task/.gaia2/scenario.json`" in prompt
    assert "Use `task/tools/catalog.json` for tool names" in prompt
    assert (dest / ".gaia2").is_dir()
    assert hasattr(task, "runtime_session")


def test_gaia2_task_query_is_scenario_text_without_runtime_protocol(tmp_path) -> None:
    dataset_path = tmp_path / "mini.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "scenario_id": "scenario-a",
                "data": {"metadata": {"definition": {"hints": ["Send the note."]}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    harness_store = FilesystemHarnessStore(tmp_path / "harness_store")
    dataset = load_dataset(f"gaia2:{dataset_path}", harness_store=harness_store)
    task = next(iter(list(dataset.train) + list(dataset.val) + list(dataset.test)))

    query = task.query()

    assert "Send the note." in query
    assert "Operating Protocol" not in query
    assert "tools/are.py" not in query
