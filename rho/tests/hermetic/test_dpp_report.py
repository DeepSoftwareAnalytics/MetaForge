from __future__ import annotations

from pathlib import Path

import numpy as np

from rho.selection.report import write_selection_report


def test_dpp_report_includes_all_sections(tmp_path: Path) -> None:
    candidate_ids = ["hard_a", "hard_b", "mid_a", "mid_b"]
    selection = {
        "selector": "dpp",
        "k": 3,
        "seed": None,
        "all_candidate_ids": candidate_ids,
        "selected_task_ids": ["hard_a", "mid_a", "hard_b"],
        "difficulty_scores": {
            "hard_a": 9.0,
            "hard_b": 8.5,
            "mid_a": 5.0,
            "mid_b": 4.5,
        },
    }
    queries = {tid: f"{tid} text" for tid in candidate_ids}
    gain_trace = [
        {
            "step": 0,
            "picked_ix": 0,
            "picked_id": "hard_a",
            "log_det_gain": 0.5,
            "score": 9.0,
        },
        {
            "step": 1,
            "picked_ix": 2,
            "picked_id": "mid_a",
            "log_det_gain": 0.3,
            "score": 5.0,
        },
        {
            "step": 2,
            "picked_ix": 1,
            "picked_id": "hard_b",
            "log_det_gain": 0.1,
            "score": 8.5,
        },
    ]
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((4, 4)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T

    path = write_selection_report(
        run_dir=tmp_path,
        selection=selection,
        queries=queries,
        dataset_spec="fake-dataset",
        split="train",
        scores=selection["difficulty_scores"],
        gain_trace=gain_trace,
        similarity=sim,
        candidate_ids=candidate_ids,
    )
    text = path.read_text(encoding="utf-8")
    assert "# Selection report — dpp on fake-dataset" in text
    assert "score=9.00" in text
    assert "log_gain=0.50" in text
    assert "Difficulty score histogram" in text
    assert "DPP log-det-gain trace" in text
    assert "Nearest-neighbor spot-check" in text
