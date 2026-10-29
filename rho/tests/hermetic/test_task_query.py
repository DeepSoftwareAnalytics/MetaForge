from __future__ import annotations

from pathlib import Path

import pytest

from rho.datasets.directory import DirectoryDataset
from rho.stores.harness import FilesystemHarnessStore


def test_directory_task_query_returns_prompt_text(toy_dataset_root, tmp_path) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    dataset = DirectoryDataset(toy_dataset_root, harness_store=harness_store)
    task = next(iter(dataset.train))
    query = task.query()
    assert isinstance(query, str)
    assert len(query) > 0


def test_swebench_pro_task_query_omits_metadata(tmp_path) -> None:
    from rho.datasets.swebench_pro.dataset import SWEbenchProDataset
    from rho.datasets.swebench_pro.repo_cache import RepoCache
    from rho.stores.harness import FilesystemHarnessStore

    row = {
        "instance_id": "demo-1",
        "repo": "django/django",
        "repo_language": "Python",
        "base_commit": "abc123",
        "problem_statement": "The parser fails on nested comprehensions.",
        "requirements": "Fix the parser; tests in tests/test_parser.py must pass.",
        "interface": "No interface changes required.",
    }
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    # Route the RepoCache to tmp_path so the hermetic test doesn't touch
    # the user's real ~/.cache/swebench-pro.
    dataset = SWEbenchProDataset.from_records(
        [row],
        harness_store=harness_store,
        docker_pull="never",
        repo_cache=RepoCache(tmp_path / "repo_cache"),
    )
    task = next(iter(dataset.train))
    query = task.query()

    # Must include the semantic content
    assert "parser fails on nested comprehensions" in query
    assert "tests in tests/test_parser.py" in query
    # Must NOT include metadata headers that would cause topic/repo clustering
    assert "django/django" not in query
    assert "abc123" not in query
    assert "Instance ID" not in query
    assert "Python" not in query
