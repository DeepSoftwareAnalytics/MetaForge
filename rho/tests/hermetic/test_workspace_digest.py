from rho.agent.cache import MAX_SNAPSHOT_FILE_BYTES, workspace_digest


def test_workspace_digest_empty_dir_is_fixed(tmp_path) -> None:
    assert workspace_digest(tmp_path) == workspace_digest(tmp_path)
    assert len(workspace_digest(tmp_path)) == 64


def test_workspace_digest_ignores_creation_order(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_text("a", encoding="utf-8")
    (left / "b.txt").write_text("b", encoding="utf-8")
    (right / "b.txt").write_text("b", encoding="utf-8")
    (right / "a.txt").write_text("a", encoding="utf-8")

    assert workspace_digest(left) == workspace_digest(right)


def test_workspace_digest_changes_when_file_content_changes(tmp_path) -> None:
    target = tmp_path / "task" / "prompt.md"
    target.parent.mkdir()
    target.write_text("before", encoding="utf-8")
    before = workspace_digest(tmp_path)
    target.write_text("after", encoding="utf-8")

    assert workspace_digest(tmp_path) != before


def test_workspace_digest_excludes_rho_dir(tmp_path) -> None:
    (tmp_path / "task.txt").write_text("stable", encoding="utf-8")
    before = workspace_digest(tmp_path)
    meta = tmp_path / ".rho"
    meta.mkdir()
    (meta / "instructions.md").write_text("changed", encoding="utf-8")

    assert workspace_digest(tmp_path) == before


def test_workspace_digest_excludes_nested_git_and_rho_dirs(tmp_path) -> None:
    task_repo = tmp_path / "task" / "repo"
    task_repo.mkdir(parents=True)
    (task_repo / "answer.py").write_text("print('stable')\n", encoding="utf-8")
    before = workspace_digest(tmp_path)

    nested_git = task_repo / ".git"
    nested_git.mkdir()
    (nested_git / "index").write_text("changed", encoding="utf-8")
    nested_meta = task_repo / ".rho"
    nested_meta.mkdir()
    (nested_meta / "events.jsonl").write_text("changed", encoding="utf-8")

    assert workspace_digest(tmp_path) == before


def test_workspace_digest_large_file_is_stable(tmp_path) -> None:
    (tmp_path / "large.bin").write_bytes(b"x" * (MAX_SNAPSHOT_FILE_BYTES + 1))

    assert workspace_digest(tmp_path) == workspace_digest(tmp_path)
