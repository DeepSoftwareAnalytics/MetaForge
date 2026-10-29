"""t-SNE projection of swebench-pro query embeddings, colored by repo."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

REPO_RE = re.compile(r"^instance_([^_]+(?:-[^_]+)*)__([A-Za-z0-9._-]+?)-[0-9a-f]{8,}")


def parse_repo(task_id: str) -> str:
    m = REPO_RE.match(task_id)
    return f"{m.group(1)}/{m.group(2)}" if m else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/20260418-select-coverage-smoke")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    emb = np.load(run / "selector_calls" / "embeddings.npy")
    ids = json.loads((run / "selector_calls" / "candidate_ids.json").read_text())
    picks = json.loads((run / "selection.json").read_text())["selected_task_ids"]
    repos = [parse_repo(i) for i in ids]
    pick_idx = [ids.index(p) for p in picks]
    pick_order = {p: i for i, p in enumerate(picks)}

    print(f"[data] embeddings={emb.shape}, n_repos={len(set(repos))}")

    cache = out / f"tsne_p{int(args.perplexity)}_s{args.seed}.npy"
    if cache.exists():
        xy = np.load(cache)
        print(f"[cache] loaded {cache}")
    else:
        print(f"[tsne] running (perplexity={args.perplexity}, seed={args.seed})...")
        xy = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
            metric="cosine",
        ).fit_transform(emb)
        np.save(cache, xy)
        print(f"[write] {cache}")

    repo_count = Counter(repos)
    repo_order = [r for r, _ in sorted(repo_count.items(), key=lambda kv: -kv[1])]
    cmap = plt.get_cmap("tab20")
    color_for = {r: cmap(i % 20) for i, r in enumerate(repo_order)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for r in repo_order:
        m = np.array([rr == r for rr in repos])
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=22,
            color=color_for[r],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            label=f"{r} (n={repo_count[r]})",
        )
    ax.scatter(
        xy[pick_idx, 0],
        xy[pick_idx, 1],
        s=320,
        marker="*",
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
        zorder=10,
        label="picked (coverage greedy)",
    )
    for i, p in enumerate(picks):
        idx = ids.index(p)
        ax.annotate(
            str(i + 1),
            (xy[idx, 0], xy[idx, 1]),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            path_effects=None,
            zorder=11,
        )
        ax.annotate(
            str(i + 1),
            (xy[idx, 0], xy[idx, 1]),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=12,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"t-SNE of swebench-pro query embeddings (n={len(ids)}, perplexity={args.perplexity})\n"
        f"colors = repo, ★ = coverage-greedy pick (numbered by pick order)"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tsne_by_repo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out / 'tsne_by_repo.png'}")

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, (i, p) in zip(axes.flat, enumerate(picks)):
        repo = parse_repo(p)
        m_other = np.array([rr != repo for rr in repos])
        m_same = np.array([rr == repo for rr in repos])
        ax.scatter(xy[m_other, 0], xy[m_other, 1], s=8, color="lightgrey", alpha=0.4)
        ax.scatter(
            xy[m_same, 0],
            xy[m_same, 1],
            s=18,
            color=color_for[repo],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        idx = ids.index(p)
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=240,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i + 1}  {repo}\n(n={repo_count[repo]})", fontsize=9)
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    fig.suptitle(
        "Each pick (red ★) within its own repo cluster (colored) vs all other tasks (grey)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "tsne_picks_per_repo.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'tsne_picks_per_repo.png'}")


if __name__ == "__main__":
    main()
