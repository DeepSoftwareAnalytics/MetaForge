from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rho.selection.report import write_selection_report


def _section(text: str, heading: str) -> str:
    """Return the body of a markdown section (## heading) up to the next ## or EOF."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    assert m, f"section {heading!r} not found in:\n{text}"
    return m.group(1)


def test_report_random_has_only_selected_section(tmp_path: Path) -> None:
    selection = {
        "selector": "random",
        "k": 2,
        "seed": 7,
        "all_candidate_ids": ["a", "b", "c"],
        "selected_task_ids": ["a", "c"],
    }
    out = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries={"a": "task A query", "b": "task B query", "c": "task C query"},
        dataset_spec="directory:toy",
        split="train",
    )
    text = out.read_text(encoding="utf-8")
    sel_body = _section(text, "Selected tasks")
    assert "task A query" in sel_body and "task C query" in sel_body
    assert "task B query" not in text
    for heading in (
        "Difficulty score histogram",
        "Coverage gain trace",
        "Coverage nearest-neighbor spot-check",
    ):
        assert heading not in text


def test_report_difficulty_histogram_bucket_counts(tmp_path: Path) -> None:
    scores = {"a": 8.5, "b": 2.0, "c": 5.0, "d": 9.5, "e": 10.0}
    selection = {
        "selector": "difficulty",
        "k": 2,
        "seed": None,
        "all_candidate_ids": list(scores),
        "selected_task_ids": ["d", "a"],
        "difficulty_scores": scores,
    }
    out = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries={task_id: f"query {task_id}" for task_id in scores},
        dataset_spec="directory:toy",
        split="train",
        scores=scores,
    )
    text = out.read_text(encoding="utf-8")
    sel_body = _section(text, "Selected tasks")
    assert "score=9.50" in sel_body and "score=8.50" in sel_body
    hist = _section(text, "Difficulty score histogram")
    assert "[8.0, 10.0]" in hist
    assert "[2.0, 4.0)" in hist
    top_line = next(line for line in hist.splitlines() if "[8.0, 10.0]" in line)
    m = re.search(r"\]:\s*(\d+)", top_line)
    assert m and int(m.group(1)) == 3
    assert "median=9.00" in hist


def test_report_includes_truncated_fingerprint_preview(tmp_path: Path) -> None:
    selection = {
        "selector": "coverage",
        "k": 1,
        "seed": 0,
        "all_candidate_ids": ["a"],
        "selected_task_ids": ["a"],
    }
    fingerprint = " ".join(f"word{i:03d}" for i in range(80))
    out = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries={"a": "query a"},
        dataset_spec="directory:toy",
        split="train",
        fingerprints={"a": fingerprint},
    )
    text = out.read_text(encoding="utf-8")
    sel_body = _section(text, "Selected tasks")
    assert "Fingerprint:" in sel_body
    assert "word079" not in sel_body
    assert "..." in sel_body


def test_report_coverage_neighbors_exclude_self_and_match_similarity(
    tmp_path: Path,
) -> None:
    ids = ["a", "b", "c", "d"]
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.95, 0.0, 0.0, 0.31],
        ],
        dtype=np.float32,
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T
    gain_trace = [
        {"step": 0, "picked_ix": 0, "picked_id": "a", "gain": 4.0},
        {"step": 1, "picked_ix": 1, "picked_id": "b", "gain": 1.5},
        {"step": 2, "picked_ix": 2, "picked_id": "c", "gain": 0.5},
    ]
    selection = {
        "selector": "coverage",
        "k": 3,
        "seed": 0,
        "all_candidate_ids": ids,
        "selected_task_ids": ["a", "b", "c"],
    }
    out = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries={task_id: f"query {task_id}" for task_id in ids},
        dataset_spec="directory:toy",
        split="train",
        gain_trace=gain_trace,
        similarity=sim,
        candidate_ids=ids,
    )
    text = out.read_text(encoding="utf-8")
    sel_body = _section(text, "Selected tasks")
    assert "gain=4.00" in sel_body and "gain=1.50" in sel_body and "gain=0.50" in sel_body
    trace = _section(text, "Coverage gain trace")
    for task_id in ("a", "b", "c"):
        assert task_id in trace
    nbrs = _section(text, "Coverage nearest-neighbor spot-check")
    a_line = next(line for line in nbrs.splitlines() if line.startswith("- a:"))
    nbr_ids = re.findall(r"([a-z])\s*\(", a_line)
    assert "a" not in nbr_ids
    assert nbr_ids[0] == "d"


def test_report_coverage_handles_missing_artifacts_gracefully(
    tmp_path: Path,
) -> None:
    selection = {
        "selector": "coverage",
        "k": 2,
        "seed": 0,
        "all_candidate_ids": ["a", "b"],
        "selected_task_ids": ["a", "b"],
    }
    out = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries={"a": "qa", "b": "qb"},
        dataset_spec="directory:toy",
        split="train",
    )
    text = out.read_text(encoding="utf-8")
    assert _section(text, "Selected tasks").strip()
    assert "Coverage gain trace" not in text
    assert "Coverage nearest-neighbor spot-check" not in text
