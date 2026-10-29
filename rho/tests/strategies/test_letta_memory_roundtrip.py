from __future__ import annotations

from pathlib import Path

from rho.stores.harness import FilesystemHarnessStore


def test_letta_memory_roundtrip_preserves_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    memory = src / "letta_memory"
    memory.mkdir(parents=True)
    (memory / "notes.md").write_bytes(
        b"first line  \nblank follows\n\nwindows\r\nunicode: \xe2\x86\x92\ntrailing"
    )
    (memory / "persona.md").write_bytes(
        "# persona\r\n\r\n(no persona policies in this deployment)\r\n".encode("utf-8")
    )
    (memory / ".read_only").write_bytes(b"persona\n")

    store = FilesystemHarnessStore(tmp_path / "store")
    harness = store.capture(src)
    dest = tmp_path / "dest"
    harness.materialize(dest)

    source_bytes = {
        path.relative_to(memory).as_posix(): path.read_bytes()
        for path in sorted(memory.iterdir())
        if path.is_file()
    }
    dest_memory = dest / "letta_memory"
    roundtrip_bytes = {
        path.relative_to(dest_memory).as_posix(): path.read_bytes()
        for path in sorted(dest_memory.iterdir())
        if path.is_file()
    }
    assert roundtrip_bytes == source_bytes
