"""Quantify intra-repo nearest-neighbor concentration in the coverage selector.

Loads similarity.npy + candidate_ids.json from a `rho select --selector coverage`
run, parses the GitHub repo from each task id, and reports / plots the fraction of
top-K nearest neighbors that come from the same repo. Hypothesis: embeddings cluster
by repo, so coverage greedy is largely picking one task per repo.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_RE = re.compile(r"^instance_([^_]+(?:-[^_]+)*)__([A-Za-z0-9._-]+?)-[0-9a-f]{8,}")


def parse_repo(task_id: str) -> str:
    m = REPO_RE.match(task_id)
    if not m:
        return "UNKNOWN"
    return f"{m.group(1)}/{m.group(2)}"


def topk_neighbors(sim: np.ndarray, k: int) -> np.ndarray:
    n = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, kth=k, axis=1)[:, :k]


def intra_repo_fraction(neighbors: np.ndarray, repos: list[str]) -> np.ndarray:
    out = np.empty(len(repos))
    for i, row in enumerate(neighbors):
        same = sum(1 for j in row if repos[j] == repos[i])
        out[i] = same / len(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/20260418-select-coverage-smoke")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", default=None, help="Default: <run-dir>/analysis/")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    sim = np.load(run / "selector_calls" / "similarity.npy")
    ids: list[str] = json.loads((run / "selector_calls" / "candidate_ids.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    picks: list[str] = selection["selected_task_ids"]

    repos = [parse_repo(i) for i in ids]
    repo_count = Counter(repos)
    n = len(ids)
    print(f"[data] n_tasks={n}, n_repos={len(repo_count)}, k={args.k}")
    print(f"[data] picks={len(picks)}, unknown_repo={repo_count['UNKNOWN']}")

    nbrs = topk_neighbors(sim, args.k)
    frac = intra_repo_fraction(nbrs, repos)

    pick_idx = [ids.index(p) for p in picks]
    pick_frac = frac[pick_idx]

    print(f"[overall] mean intra-repo top-{args.k} fraction: {frac.mean():.3f}")
    print(f"[overall] median: {np.median(frac):.3f}")
    print(
        f"[overall] frac of tasks with all top-{args.k} same-repo: "
        f"{(frac == 1.0).mean():.3f}"
    )
    print(f"[picks]   mean intra-repo top-{args.k} fraction: {pick_frac.mean():.3f}")
    print(f"[picks]   per-pick same-repo counts (out of {args.k}):")
    for p, f in zip(picks, pick_frac):
        print(f"    {int(round(f * args.k))}/{args.k}  {parse_repo(p):40s}  {p}")

    sizes = np.array([repo_count[r] for r in repos])
    sized_repos = sorted(repo_count.items(), key=lambda kv: -kv[1])
    top_repos = sized_repos[:20]
    print("[repos] top-20 by task count:")
    for r, c in top_repos:
        print(f"    {c:4d}  {r}")

    summary = {
        "k": args.k,
        "n_tasks": n,
        "n_repos": len(repo_count),
        "overall_mean_intra_repo_frac": float(frac.mean()),
        "overall_median": float(np.median(frac)),
        "frac_all_same_repo": float((frac == 1.0).mean()),
        "picks_mean_intra_repo_frac": float(pick_frac.mean()),
        "per_pick": [
            {
                "task_id": p,
                "repo": parse_repo(p),
                "same_repo_count": int(round(f * args.k)),
                "intra_repo_fraction": float(f),
            }
            for p, f in zip(picks, pick_frac)
        ],
        "top_repos_by_size": [{"repo": r, "n_tasks": c} for r, c in top_repos],
    }
    (out / "intra_repo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[write] {out / 'intra_repo_summary.json'}")

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bin_w = 1.0 / args.k
    bins = np.arange(-bin_w / 2, 1.0 + bin_w, bin_w)
    ax.hist(frac, bins=bins, color="#4C72B0", edgecolor="white", label="all 585 tasks")
    for f in pick_frac:
        ax.axvline(f, color="#C44E52", alpha=0.45, lw=1.5)
    ax.axvline(
        pick_frac.mean(),
        color="#C44E52",
        lw=2.5,
        linestyle="--",
        label=f"picks mean = {pick_frac.mean():.2f}",
    )
    ax.axvline(
        frac.mean(),
        color="#2F2F2F",
        lw=2,
        linestyle=":",
        label=f"overall mean = {frac.mean():.2f}",
    )
    ax.set_xlabel(f"fraction of top-{args.k} neighbors from the same repo")
    ax.set_ylabel("number of tasks")
    ax.set_title(
        f"Intra-repo concentration of top-{args.k} cosine neighbors\n"
        f"({n} swebench-pro train tasks, {len(repo_count)} repos)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "intra_repo_histogram.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'intra_repo_histogram.png'}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    jitter = (np.random.RandomState(0).rand(n) - 0.5) * 0.6
    ax.scatter(
        sizes + jitter,
        frac + (np.random.RandomState(1).rand(n) - 0.5) * 0.04,
        s=10,
        alpha=0.35,
        color="#4C72B0",
        label="task",
    )
    ax.scatter(
        sizes[pick_idx],
        pick_frac,
        s=110,
        marker="*",
        color="#C44E52",
        edgecolor="black",
        linewidth=0.5,
        label="picked",
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("repo size (number of tasks in pool, log scale)")
    ax.set_ylabel(f"intra-repo top-{args.k} fraction")
    ax.set_title("Intra-repo concentration vs repo size")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "intra_repo_vs_size.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'intra_repo_vs_size.png'}")

    repo_order = [r for r, _ in sorted(repo_count.items(), key=lambda kv: -kv[1])]
    perm = np.argsort([repo_order.index(r) for r in repos])
    sim_ord = sim[perm][:, perm]
    repos_ord = [repos[p] for p in perm]
    boundaries = []
    cur = repos_ord[0]
    start = 0
    for i, r in enumerate(repos_ord + [None]):
        if r != cur:
            boundaries.append((cur, start, i))
            cur = r
            start = i
    big = [b for b in boundaries if b[2] - b[1] >= 8]

    off = sim_ord.copy()
    np.fill_diagonal(off, np.nan)
    vmin = float(np.nanpercentile(off, 2))
    vmax = float(np.nanpercentile(off, 98))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim_ord, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    for _, s, _e in big:
        ax.axhline(s - 0.5, color="cyan", lw=0.5, alpha=0.8)
        ax.axvline(s - 0.5, color="cyan", lw=0.5, alpha=0.8)
    centers = [(s + e) / 2 for _, s, e in big]
    labels = [r for r, _, _ in big]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        f"Cosine similarity, tasks sorted by repo (largest first)\n"
        f"Bright on-diagonal blocks = intra-repo similarity. "
        f"Color range = {vmin:.2f}..{vmax:.2f} (2nd–98th pct, off-diagonal)"
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(out / "similarity_block_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'similarity_block_heatmap.png'}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [int(round(f * args.k)) for f in pick_frac]
    pick_repos = [parse_repo(p) for p in picks]
    pick_repo_sizes = [repo_count[r] for r in pick_repos]
    labels = [
        f"#{i + 1} {r}\n(repo size={s})"
        for i, (r, s) in enumerate(zip(pick_repos, pick_repo_sizes))
    ]
    colors = ["#C44E52" if c == args.k else "#DD8452" if c >= args.k - 1 else "#4C72B0" for c in counts]
    bars = ax.barh(range(len(picks)), counts, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, args.k + 0.2)
    ax.set_xlabel(f"# of top-{args.k} neighbors from the same repo")
    ax.set_title(f"Per-pick intra-repo neighbor count (max={args.k})")
    ax.axvline(args.k, color="grey", linestyle=":", alpha=0.5)
    for bar, c in zip(bars, counts):
        ax.text(c + 0.05, bar.get_y() + bar.get_height() / 2, f"{c}/{args.k}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_pick_intra_repo.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'per_pick_intra_repo.png'}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_repo: dict[str, list[float]] = {r: [] for r in repo_count}
    for r, f in zip(repos, frac):
        by_repo[r].append(float(f))
    repo_order_box = [r for r, _ in sized_repos]
    data = [by_repo[r] for r in repo_order_box]
    means = [float(np.mean(by_repo[r])) for r in repo_order_box]
    ax.boxplot(data, positions=range(len(repo_order_box)), widths=0.6, showfliers=False)
    ax.scatter(range(len(repo_order_box)), means, color="#C44E52", s=40, zorder=5, label="mean")
    ax.set_xticks(range(len(repo_order_box)))
    ax.set_xticklabels(
        [f"{r}\n(n={repo_count[r]})" for r in repo_order_box],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel(f"intra-repo top-{args.k} fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Per-repo intra-repo concentration distribution")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_repo_concentration_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'per_repo_concentration_boxplot.png'}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sizes_sorted = [c for _, c in sized_repos]
    repo_labels = [r if c >= 8 else "" for r, c in sized_repos]
    colors2 = ["#C44E52" if r in pick_repos else "#4C72B0" for r, _ in sized_repos]
    ax.bar(range(len(sized_repos)), sizes_sorted, color=colors2, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sized_repos)))
    ax.set_xticklabels(repo_labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("# tasks (log)")
    dups = [r for r, c in Counter(pick_repos).items() if c > 1]
    ax.set_title(
        f"Per-repo task count (red = source of a pick)\n"
        f"{len(set(pick_repos))}/{len(repo_count)} repos covered, "
        f"{len(pick_repos)} picks, duplicates from: {', '.join(dups)}"
    )
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "repo_size_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[write] {out / 'repo_size_distribution.png'}")


if __name__ == "__main__":
    main()
