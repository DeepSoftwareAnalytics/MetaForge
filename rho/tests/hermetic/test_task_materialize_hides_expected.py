def test_task_materialize_hides_expected(toy_dataset_root, tmp_path) -> None:
    from rho.datasets.directory import DirectoryTask
    from rho.stores.harness import FilesystemHarnessStore

    harness = FilesystemHarnessStore(tmp_path / "_hs").empty()
    task = DirectoryTask(toy_dataset_root / "train" / "task_002", harness)
    task.materialize(tmp_path / "task")
    assert not (tmp_path / "task" / "expected.json").exists()
