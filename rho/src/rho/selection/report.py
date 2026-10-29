from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_selection_report(
    *,
    run_dir: Path,
    selection: dict[str, Any],
    queries: dict[str, str],
    dataset_spec: str,
    split: str,
    scores: dict[str, float] | None = None,
    fingerprints: dict[str, str] | None = None,
    gain_trace: list[dict[str, Any]] | None = None,
    similarity: np.ndarray | None = None,
    candidate_ids: list[str] | None = None,
) -> Path:
    """Write `selection_report.md` summarizing a completed selection."""
    selector = selection["selector"]
    selected = selection["selected_task_ids"]
    pool_ids = selection["all_candidate_ids"]

    lines: list[str] = [
        f"# Selection report — {selector} on {dataset_spec}",
        "",
        f"- Run dir: {run_dir}",
        f"- Selector: {selector}",
        f"- k: {selection['k']}  (pool size: {len(pool_ids)})",
        f"- Seed: {selection['seed']}",
        f"- Dataset: {dataset_spec}",
        f"- Split: {split}",
        "",
    ]

    score_map = scores if scores is not None else (selection.get("difficulty_scores") or {})
    gain_by_id = {
        entry["picked_id"]: entry["gain"]
        for entry in (gain_trace or [])
        if "gain" in entry
    }

    lines += ["## Selected tasks", ""]
    log_gain_by_id = {
        entry["picked_id"]: entry.get("log_det_gain")
        for entry in (gain_trace or [])
        if "log_det_gain" in entry
    }
    for rank, task_id in enumerate(selected):
        annotations: list[str] = []
        if selector in ("difficulty", "dpp") and task_id in score_map:
            annotations.append(f"score={score_map[task_id]:.2f}")
        if selector == "coverage" and task_id in gain_by_id:
            annotations.append(f"gain={gain_by_id[task_id]:.2f}")
        if selector == "dpp" and task_id in log_gain_by_id:
            annotations.append(f"log_gain={log_gain_by_id[task_id]:.2f}")
        annot = "  [" + " | ".join(annotations) + "]" if annotations else ""
        lines.append(f"### {rank + 1}. {task_id}{annot}  (pick order: {rank})")
        preview = (queries.get(task_id, "") or "").replace("\n", " ").strip()[:240]
        lines.append(f"> {preview}")
        if fingerprints is not None and task_id in fingerprints:
            lines.append(f"Fingerprint: {_truncate_preview(fingerprints[task_id], limit=300)}")
        lines.append("")

    if selector == "difficulty" and score_map:
        _append_difficulty_sections(lines, score_map, selected)
    if selector == "coverage" and gain_trace is not None:
        _append_gain_trace(lines, gain_trace, heading="Coverage gain trace", gain_key="gain")
    if selector == "coverage" and similarity is not None and candidate_ids is not None:
        _append_neighbors(
            lines,
            similarity,
            candidate_ids,
            selected,
            heading="Coverage nearest-neighbor spot-check",
        )
    if selector == "dpp":
        if score_map:
            _append_difficulty_sections(lines, score_map, selected)
        if gain_trace is not None:
            _append_gain_trace(
                lines,
                gain_trace,
                heading="DPP log-det-gain trace",
                gain_key="log_det_gain",
            )
        if similarity is not None and candidate_ids is not None:
            _append_neighbors(
                lines,
                similarity,
                candidate_ids,
                selected,
                heading="Nearest-neighbor spot-check",
            )

    out = run_dir / "selection_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _append_difficulty_sections(
    lines: list[str], scores: dict[str, float], selected: list[str]
) -> None:
    lines += ["## Difficulty score histogram", ""]
    edges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    counts = [0] * len(edges)
    for score in scores.values():
        for i, (lo, hi) in enumerate(edges):
            is_last = i == len(edges) - 1
            if (lo <= score < hi) or (is_last and score == hi):
                counts[i] += 1
                break
    for i, ((lo, hi), count) in enumerate(zip(edges, counts)):
        if i == len(edges) - 1:
            label = f"[{lo:.1f}, {hi:.1f}]"
        else:
            label = f"[{lo:.1f}, {hi:.1f})"
        lines.append(f"- {label}:  {count}  {'█' * count}")
    lines.append("")

    selected_scores = [scores[task_id] for task_id in selected if task_id in scores]
    if selected_scores:
        arr = np.array(selected_scores, dtype=float)
        lines.append(
            "Selected score stats: "
            f"min={arr.min():.2f}  max={arr.max():.2f}  "
            f"mean={arr.mean():.2f}  median={float(np.median(arr)):.2f}"
        )
        lines.append("")


def _truncate_preview(text: str, *, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 3)].rstrip() + "..."


def _append_gain_trace(
    lines: list[str],
    gain_trace: list[dict[str, Any]],
    *,
    heading: str = "Coverage gain trace",
    gain_key: str = "gain",
) -> None:
    label = "marginal gain" if gain_key == "gain" else "marginal log-det gain"
    lines += [
        f"## {heading}",
        "",
        f"| step | picked id | {label} |",
        "|------|-----------|----------------|",
    ]
    for entry in gain_trace:
        lines.append(
            f"| {entry['step']} | {entry['picked_id']} | {entry[gain_key]:.2f} |"
        )
    lines.append("")


def _append_neighbors(
    lines: list[str],
    similarity: np.ndarray,
    candidate_ids: list[str],
    selected: list[str],
    *,
    heading: str = "Coverage nearest-neighbor spot-check",
) -> None:
    lines += [
        f"## {heading}",
        "",
        "For each selected task, top-5 candidates by cosine similarity (excluding self):",
        "",
    ]
    ix_of = {task_id: i for i, task_id in enumerate(candidate_ids)}
    top_n = min(5, max(0, len(candidate_ids) - 1))
    for task_id in selected:
        i = ix_of.get(task_id)
        if i is None:
            continue
        row = similarity[i].copy()
        row[i] = -np.inf
        order = np.argsort(-row)[:top_n]
        neighbors = ", ".join(
            f"{candidate_ids[j]} ({similarity[i, j]:.2f})" for j in order
        )
        lines.append(f"- {task_id}: {neighbors}")
    lines.append("")
