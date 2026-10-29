from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path

import pytest

from rho.datasets.loader import load_dataset
from rho.datasets.swebench_pro import SWEbenchProDataset
from rho.datasets.swebench_pro.dataset import _split_rows, load_rows
from rho.datasets.swebench_pro.evaluator import SWEbenchProDockerEvaluator
from rho.datasets.swebench_pro.patching import extract_prediction_patch
from rho.datasets.swebench_pro.repo_cache import RepoCache
from rho.protocols import Trajectory
from rho.stores.harness import FilesystemHarnessStore


def test_local_fixture_loader_splits_stably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RHO_CACHE_DIR", str(tmp_path / "cache"))
    repo, commit = _make_repo(tmp_path)
    rows = [_row(repo, commit, f"instance_fixture-{ix}") for ix in range(5)]
    fixture = tmp_path / "swebench_pro.json"
    fixture.write_text(json.dumps(rows), encoding="utf-8")

    hs = FilesystemHarnessStore(tmp_path / "harness")
    dataset = load_dataset(
        f"swebench-pro:{fixture}",
        harness_store=hs,
        max_per_split=1,
    )

    assert len(dataset.train) == 1
    assert len(dataset.val) == 1
    assert len(dataset.test) == 1
    again = load_dataset(
        f"swebench-pro:{fixture}",
        harness_store=FilesystemHarnessStore(tmp_path / "harness2"),
        max_per_split=1,
    )
    assert [task.id for task in dataset.train] == [task.id for task in again.train]
    assert [task.id for task in dataset.val] == [task.id for task in again.val]
    assert [task.id for task in dataset.test] == [task.id for task in again.test]


def test_split_on_731_public_set_is_100_500_131() -> None:
    rows = [{"instance_id": str(ix)} for ix in range(731)]

    splits = _split_rows(rows, seed=0)

    assert {name: len(split) for name, split in splits.items()} == {
        "train": 100,
        "val": 500,
        "test": 131,
    }
    split_ids = [
        row["instance_id"]
        for split in (splits["train"], splits["val"], splits["test"])
        for row in split
    ]
    assert sorted(split_ids, key=int) == [str(ix) for ix in range(731)]


def test_split_rows_is_deterministic_by_seed() -> None:
    rows = [{"instance_id": str(ix)} for ix in range(731)]

    seed_0_once = _split_rows(rows, seed=0)
    seed_0_again = _split_rows(rows, seed=0)
    seed_1 = _split_rows(rows, seed=1)

    assert seed_0_once == seed_0_again
    assert {name: len(split) for name, split in seed_1.items()} == {
        "train": 100,
        "val": 500,
        "test": 131,
    }
    assert seed_0_once["train"] != seed_1["train"]


def test_materialize_writes_repo_and_hides_gold_fields(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    dataset = SWEbenchProDataset.from_records(
        [_row(repo, commit)],
        harness_store=FilesystemHarnessStore(tmp_path / "harness"),
        repo_cache=RepoCache(tmp_path / "repo-cache"),
    )
    task = next(iter(dataset.train))

    dest = tmp_path / "task"
    task.materialize(dest)

    prompt = (dest / "prompt.md").read_text(encoding="utf-8")
    assert (dest / "repo" / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    assert "fix the fixture bug" in prompt
    assert "GOLD_PATCH_SECRET" not in prompt
    assert "TEST_PATCH_SECRET" not in prompt
    assert "SECRET_DOCKER_TAG" not in prompt
    assert "test_fix" not in prompt


def test_patch_extraction_from_workspace_diff(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    cache = RepoCache(tmp_path / "repo-cache")
    row = _row(repo, commit)
    trajectory = _trajectory(
        workspace_diff={"task/repo/hello.txt": b"hello\nchanged\n"},
        final_message="done",
    )

    patch = extract_prediction_patch(
        trajectory,
        materialize_repo=lambda dest: cache.materialize(row, dest),
        artifacts_dir=tmp_path / "artifacts",
    )

    assert "diff --git a/hello.txt b/hello.txt" in patch
    assert "+changed" in patch


def test_hf_loader_missing_datasets_extra_has_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="swebench-pro"):
        load_rows("ScaleAI/SWE-bench_Pro")


def test_docker_evaluator_command_and_pass_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts_root = _make_eval_assets(tmp_path)
    row = _row(tmp_path, "abc123")
    commands: list[list[str]] = []

    import rho.datasets.swebench_pro.evaluator as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "ensure_official_assets",
        lambda root: scripts_root,
    )

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "[]", "")
        if cmd[:2] == ["docker", "run"]:
            mount = cmd[cmd.index("-v") + 1]
            workspace = Path(mount.split(":", 1)[0])
            (workspace / "stdout.log").write_text("", encoding="utf-8")
            (workspace / "stderr.log").write_text("", encoding="utf-8")
            (workspace / "output.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "test_fix", "status": "PASSED"},
                            {"name": "test_existing", "status": "PASSED"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, "container stdout", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = SWEbenchProDockerEvaluator(
        docker_pull="missing",
        assets_root=tmp_path / "unused",
    ).evaluate(
        row=row,
        patch="diff --git a/hello.txt b/hello.txt\n",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert result.passed is True
    assert any(cmd[:2] == ["docker", "run"] for cmd in commands)
    run_cmd = next(cmd for cmd in commands if cmd[:2] == ["docker", "run"])
    assert "jefzda/sweap-images:SECRET_DOCKER_TAG" in run_cmd
    entryscript = (tmp_path / "artifacts" / "entryscript.sh").read_text(
        encoding="utf-8"
    )
    assert "bash /workspace/run_script.sh test_fix,test_existing" in entryscript


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "hello.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    return repo, commit.strip()


def _row(
    repo: Path,
    commit: str,
    instance_id: str = "instance_fixture-canary",
) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "fixture/repo",
        "repo_language": "python",
        "base_commit": commit,
        "problem_statement": "fix the fixture bug",
        "requirements": "make hello.txt mention the change",
        "interface": "No new interfaces.",
        "patch": "GOLD_PATCH_SECRET",
        "test_patch": "TEST_PATCH_SECRET",
        "fail_to_pass": "['test_fix']",
        "pass_to_pass": "['test_existing']",
        "before_repo_set_cmd": "git checkout abc123 -- tests/test_fixture.py",
        "selected_test_files_to_run": "['test_fix', 'test_existing']",
        "dockerhub_tag": "SECRET_DOCKER_TAG",
        "repo_path": str(repo),
    }


def _trajectory(
    *,
    workspace_diff: dict[str, bytes],
    final_message: str,
) -> Trajectory:
    return Trajectory(
        id="traj_fixture",
        kind="solve",
        task_id="instance_fixture-canary",
        harness_id="h_empty",
        instructions="solve",
        events=[],
        final_message=final_message,
        stdout="",
        stderr="",
        workspace_diff=workspace_diff,
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=0.1,
    )


def _make_eval_assets(tmp_path: Path) -> Path:
    root = tmp_path / "eval_assets"
    uid = "instance_fixture-canary"
    run_dir = root / "run_scripts" / uid
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("", encoding="utf-8")
    for kind in ("base_dockerfile", "instance_dockerfile"):
        dockerfile_dir = root / "dockerfiles" / kind / uid
        dockerfile_dir.mkdir(parents=True)
        (dockerfile_dir / "Dockerfile").write_text(
            "FROM scratch\nENV FIXTURE_ENV=1\n",
            encoding="utf-8",
        )
    return root


def test_local_dataset_image_mapping_is_preserved():
    from rho.datasets.swebench_pro.evaluator import docker_image
    image = "jefzda/sweap-images:" + "x" * 128
    row = {"instance_id": "instance_" + "x" * 150, "repo": "owner/repo", "image_name": image}
    assert docker_image(row) == image
    row["dockerhub_tag"] = "explicit-tag"
    assert docker_image(row) == "jefzda/sweap-images:explicit-tag"
