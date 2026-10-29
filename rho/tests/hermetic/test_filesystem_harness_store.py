from pathlib import Path

from rho.stores.harness import EMPTY_HARNESS_ID, FilesystemHarnessStore


def test_filesystem_harness_store_round_trip(tmp_path: Path) -> None:
    store = FilesystemHarnessStore(tmp_path / "harness")
    empty = store.empty()
    assert empty.id == EMPTY_HARNESS_ID
    empty.materialize(tmp_path / "materialized")
    assert (tmp_path / "materialized" / "README.md").exists()

    captured_same = store.capture(tmp_path / "materialized")
    assert captured_same.id == empty.id

    (tmp_path / "materialized" / "notes.md").write_text("version is 3.12\n", encoding="utf-8")
    captured_new = store.capture(tmp_path / "materialized")
    assert captured_new.id != empty.id
    round_trip = store.get(captured_new.id)
    assert round_trip.id == captured_new.id
