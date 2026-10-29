"""t-SNE of DPP selection run.

Two panels:
  1. All pool items, colored by difficulty (viridis gradient). DPP picks
     marked with black stars numbered by pick order.
  2. Same layout, colored by source repo (tab20). Lets you eyeball whether
     picks span different repos / failure-mode clusters.

Also prints pair-wise cosine similarity among picks so you can quantify
how spread out they are in fingerprint space.
"""
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
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    emb = np.load(run / "selector_calls" / "embeddings.npy")
    ids = json.loads((run / "selector_calls" / "candidate_ids.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    picks = selection["selected_task_ids"]
    diffs_map = selection.get("difficulty_scores", {})
    diffs = np.array([diffs_map.get(i, float("nan")) for i in ids], dtype=float)
    repos = [parse_repo(i) for i in ids]
    id_to_ix = {i: k for k, i in enumerate(ids)}
    pick_idx = [id_to_ix[p] for p in picks]

    print(f"[data] embeddings={emb.shape}, picks={len(picks)}, repos={len(set(repos))}")
    print(
        f"[data] difficulty: min={np.nanmin(diffs):.2f} max={np.nanmax(diffs):.2f} "
        f"mean={np.nanmean(diffs):.2f} median={np.nanmedian(diffs):.2f}"
    )

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

    # Panel 1: color by difficulty
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=diffs,
        cmap="viridis",
        s=18,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.scatter(
        xy[pick_idx, 0],
        xy[pick_idx, 1],
        s=420,
        marker="*",
        facecolor="none",
        edgecolor="black",
        linewidth=2.2,
        zorder=10,
        label=f"DPP picks (k={len(picks)})",
    )
    for i, p in enumerate(picks):
        idx = id_to_ix[p]
        for color, z in (("white", 11), ("black", 12)):
            ax.annotate(
                str(i + 1),
                (xy[idx, 0], xy[idx, 1]),
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                color=color,
                zorder=z,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"t-SNE of fingerprint embeddings (n={len(ids)}, perplexity={args.perplexity})\n"
        f"color = LLM difficulty score, ★ = DPP pick (numbered by order)"
    )
    fig.colorbar(sc, ax=ax, label="difficulty (0-10)", shrink=0.7)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out / "tsne_dpp_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out / 'tsne_dpp_by_difficulty.png'}")

    # Panel 2: color by repo
    repo_count = Counter(repos)
    repo_order = [r for r, _ in sorted(repo_count.items(), key=lambda kv: -kv[1])]
    cmap = plt.get_cmap("tab20")
    color_for = {r: cmap(i % 20) for i, r in enumerate(repo_order)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for r in repo_order:
        m = np.array([rr == r for rr in repos])
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=18,
            color=color_for[r],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            label=f"{r} (n={repo_count[r]})",
        )
    ax.scatter(
        xy[pick_idx, 0],
        xy[pick_idx, 1],
        s=420,
        marker="*",
        facecolor="none",
        edgecolor="black",
        linewidth=2.2,
        zorder=10,
    )
    for i, p in enumerate(picks):
        idx = id_to_ix[p]
        for color, z in (("white", 11), ("black", 12)):
            ax.annotate(
                str(i + 1),
                (xy[idx, 0], xy[idx, 1]),
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                color=color,
                zorder=z,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"t-SNE of fingerprint embeddings (same layout)\n"
        f"color = source repo, ★ = DPP pick"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tsne_dpp_by_repo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out / 'tsne_dpp_by_repo.png'}")

    # Pairwise cosine similarity among picks
    if len(pick_idx) >= 2:
        pick_vecs = emb[pick_idx]
        sim = pick_vecs @ pick_vecs.T
        iu = np.triu_indices(len(pick_idx), k=1)
        pair_sims = sim[iu]
        print(
            f"[picks] pairwise cosine similarity: "
            f"min={pair_sims.min():.3f} max={pair_sims.max():.3f} "
            f"mean={pair_sims.mean():.3f} median={np.median(pair_sims):.3f}"
        )
        print(
            f"[picks] difficulty: "
            f"{[float(diffs[i]) for i in pick_idx]}"
        )
        print(
            f"[picks] repos covered: "
            f"{sorted(set(parse_repo(p) for p in picks))}"
        )


if __name__ == "__main__":
    main()
