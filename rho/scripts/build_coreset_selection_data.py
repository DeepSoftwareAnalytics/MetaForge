#!/usr/bin/env python3
"""Build data for fig-coreset-selection: t-SNE of the 100-task SWE-Bench Pro
train pool + each selector's k=10 picks + held-out val Pass% per selector.

Reads:
  - ablation worktree exp-rho-swebench/selection.json (DPP, has fingerprints)
  - ablation worktree exp-abl-sel-{random,difficulty,coverage}-swebench/selection.json
  - val Pass% from each run's reports/summary.txt
  - empty-harness baseline from 20260509-h-empty-val100-baseline/reports/cli_val_grade_summary.json

Writes:
  - docs/paper/figures/data/fig-coreset-selection.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / ".claude" / "worktrees" / "ablation-campaign" / "runs"
OUT = REPO / "docs" / "paper" / "figures" / "data" / "fig-coreset-selection.json"

SELECTORS = [
    # (label, run_dir, color_key)
    ("random",     RUNS / "exp-abl-sel-random-swebench",     "random"),
    ("difficulty", RUNS / "exp-abl-sel-difficulty-swebench", "difficulty"),
    ("coverage",   RUNS / "exp-abl-sel-coverage-swebench",   "coverage"),
    ("dpp",        RUNS / "exp-rho-swebench",                 "dpp"),
]

# Source of fingerprints (canonical t-SNE input — pool is identical across runs)
DPP_RUN = RUNS / "exp-rho-swebench"

# Vanilla / empty-harness baseline (val n=100)
EMPTY_BASELINE = REPO / ".claude" / "worktrees" / "ablation-campaign" / "runs" / "20260509-h-empty-val100-baseline"


def read_val_score(run_dir: Path) -> float:
    txt = (run_dir / "reports" / "summary.txt").read_text()
    m = re.search(r"final:\s*mean_score=([0-9.]+)", txt)
    return float(m.group(1))


def read_empty_score(run_dir: Path) -> float:
    data = json.loads((run_dir / "reports" / "cli_val_grade_summary.json").read_text())
    return float(data["mean_score"])


def main() -> None:
    dpp = json.loads((DPP_RUN / "selection.json").read_text())
    pool_ids: list[str] = dpp["all_candidate_ids"]
    difficulty_by_id: dict[str, float] = {
        tid: float(score) for tid, score in dpp["difficulty_scores"].items()
    }
    difficulty_scores = np.asarray(
        [difficulty_by_id[i] for i in pool_ids], dtype=float
    )

    # the DPP selector persisted the 1024-d fingerprint embeddings + their
    # ordering during the run; reuse them so the t-SNE matches what the
    # selector actually saw.
    selcalls = DPP_RUN / "selector_calls"
    embed_ids: list[str] = json.loads((selcalls / "candidate_ids.json").read_text())
    embeddings = np.load(selcalls / "embeddings.npy").astype(float)
    assert embed_ids == pool_ids, "candidate_ids.json order differs from selection.json"
    assert embeddings.shape == (100, 1024), f"unexpected embeddings shape {embeddings.shape}"

    # t-SNE (deterministic)
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        random_state=0,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    )
    coords = tsne.fit_transform(embeddings)
    # normalize to [0, 1] for stable layout in the figure
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    coords01 = (coords - cmin) / (cmax - cmin + 1e-9)

    # collect picks per selector + val Pass%
    picks: dict[str, list[str]] = {}
    val_scores: dict[str, float] = {}
    round_means: dict[str, float] = {}
    accepted: dict[str, bool] = {}
    for label, run_dir, _color in SELECTORS:
        sel = json.loads((run_dir / "selection.json").read_text())
        picks[label] = list(sel["selected_task_ids"])
        val_scores[label] = read_val_score(run_dir)
        # also parse round_mean and accepted from summary.txt for caption use
        txt = (run_dir / "reports" / "summary.txt").read_text()
        m = re.search(r"mean_score=([\-0-9.]+)\s+accepted=(true|false)", txt)
        round_means[label] = float(m.group(1))
        accepted[label] = m.group(2) == "true"

    vanilla = read_empty_score(EMPTY_BASELINE)

    # validate that every pick is present in the canonical pool
    pool_set = set(pool_ids)
    for label, ids in picks.items():
        unknown = [i for i in ids if i not in pool_set]
        if unknown:
            raise RuntimeError(
                f"selector {label!r} has {len(unknown)} picks outside the canonical pool"
            )

    payload = {
        "pool": [
            {
                "id": pool_ids[i],
                "x": float(coords01[i, 0]),
                "y": float(coords01[i, 1]),
                "difficulty": float(difficulty_scores[i]),
            }
            for i in range(len(pool_ids))
        ],
        "picks": picks,
        "val_pass": val_scores,
        "round_mean": round_means,
        "accepted": accepted,
        "vanilla_pass": vanilla,
        "n_val": 100,
        "k": 10,
        "pool_size": len(pool_ids),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"  pool_size={len(pool_ids)} k={len(picks['dpp'])}")
    print(f"  val_pass: {val_scores}")
    print(f"  vanilla_pass: {vanilla}")


if __name__ == "__main__":
    main()
