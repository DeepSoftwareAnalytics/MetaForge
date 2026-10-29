"""Diagnostic: does the fingerprint embedding space have any structure?

t-SNE showed no repo clustering (good), but could mean either (a) structure
is organized by a different axis (difficulty / problem shape) or (b) no
discriminative signal at all (boilerplate collapsed the geometry).

Checks:
1. Pairwise cosine distribution — narrow = low discrimination.
2. t-SNE colored by difficulty score — does difficulty form clusters?
3. Nearest-neighbor difficulty coherence — if task X has difficulty d, do
   its top-5 neighbors have difficulty near d? Quantifies "kernel matches
   difficulty".
4. Compare to query-based baseline for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _topk_neighbors(sim: np.ndarray, k: int) -> np.ndarray:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, kth=k, axis=1)[:, :k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    sim = np.load(run / "selector_calls" / "similarity.npy")
    emb = np.load(run / "selector_calls" / "embeddings.npy")
    ids: list[str] = json.loads(
        (run / "selector_calls" / "candidate_ids.json").read_text()
    )

    # Pull difficulties from per-task audit records.
    diffs: dict[str, float] = {}
    for f in (run / "selector_calls").glob("instance_*.json"):
        rec = json.loads(f.read_text())
        diffs[rec["task_id"]] = float(rec["parsed_difficulty"])
    difficulty = np.array([diffs[i] for i in ids])

    print(f"[data] n={len(ids)}, embeddings={emb.shape}")
    print(
        f"[difficulty] min={difficulty.min():.2f} max={difficulty.max():.2f} "
        f"mean={difficulty.mean():.2f} std={difficulty.std():.2f}"
    )

    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    flat = off[~np.isnan(off)]
    print(
        f"[cosine] range=[{flat.min():.3f}, {flat.max():.3f}]  "
        f"mean={flat.mean():.3f}  std={flat.std():.3f}  "
        f"p5={np.percentile(flat, 5):.3f}  p95={np.percentile(flat, 95):.3f}"
    )

    # t-SNE coordinates — reuse cache if present.
    cache = out / f"tsne_p{int(args.perplexity)}_s{args.seed}.npy"
    if cache.exists():
        xy = np.load(cache)
    else:
        xy = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
            metric="cosine",
        ).fit_transform(emb)
        np.save(cache, xy)

    # ---- Cosine distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=60, color="#4C72B0", edgecolor="white")
    ax.axvline(flat.mean(), color="#C44E52", lw=2, linestyle="--", label=f"mean={flat.mean():.3f}")
    ax.set_xlabel("pairwise off-diagonal cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(
        f"Pairwise cosine distribution (n={len(ids)}, "
        f"{len(ids) * (len(ids) - 1) // 2} pairs)\n"
        f"Narrow = low discriminative signal; wide = meaningful geometry"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cosine_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'cosine_distribution.png'}")

    # ---- t-SNE colored by difficulty
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=difficulty,
        cmap="viridis",
        s=22,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"t-SNE colored by difficulty (perplexity={args.perplexity})\n"
        f"If fingerprints encode difficulty, bright and dark regions should separate"
    )
    fig.colorbar(sc, ax=ax, label="judge difficulty (0-10)")
    fig.tight_layout()
    fig.savefig(out / "tsne_by_difficulty.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'tsne_by_difficulty.png'}")

    # ---- Neighbor-difficulty coherence: for each task, mean |Δdifficulty| to top-5 neighbors
    k_values = [5, 10, 20]
    fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 4))
    for ax, k in zip(axes, k_values):
        nbrs = _topk_neighbors(sim, k)
        nbr_diff = difficulty[nbrs]  # (n, k)
        self_diff = difficulty[:, None]
        abs_delta = np.abs(nbr_diff - self_diff).mean(axis=1)

        # Baseline: random k neighbors (shuffled indices)
        rng = np.random.RandomState(0)
        rand_nbrs = np.array(
            [rng.choice(len(ids), size=k, replace=False) for _ in range(len(ids))]
        )
        rand_delta = np.abs(difficulty[rand_nbrs] - self_diff).mean(axis=1)

        ax.hist(
            rand_delta,
            bins=30,
            alpha=0.5,
            color="#888888",
            label=f"random k={k} (mean={rand_delta.mean():.2f})",
        )
        ax.hist(
            abs_delta,
            bins=30,
            alpha=0.7,
            color="#4C72B0",
            label=f"top-{k} cosine (mean={abs_delta.mean():.2f})",
        )
        ax.set_xlabel("mean |Δdifficulty| to k neighbors")
        ax.set_ylabel("count")
        ax.set_title(f"k={k}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Neighbor-difficulty coherence: if the kernel tracks difficulty,\n"
        "top-k cosine neighbors should have lower |Δdifficulty| than random neighbors.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "neighbor_difficulty_coherence.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'neighbor_difficulty_coherence.png'}")

    # Print the key numbers
    print("\n[neighbor-difficulty coherence]")
    for k in k_values:
        nbrs = _topk_neighbors(sim, k)
        self_diff = difficulty[:, None]
        top_delta = np.abs(difficulty[nbrs] - self_diff).mean()
        rng = np.random.RandomState(0)
        rand_nbrs = np.array(
            [rng.choice(len(ids), size=k, replace=False) for _ in range(len(ids))]
        )
        rand_delta = np.abs(difficulty[rand_nbrs] - self_diff).mean()
        print(
            f"  k={k:3d}: top-cos mean|Δd|={top_delta:.3f}  "
            f"random mean|Δd|={rand_delta:.3f}  "
            f"ratio={top_delta / rand_delta:.3f} (closer to 0 = more coherent)"
        )


if __name__ == "__main__":
    main()
