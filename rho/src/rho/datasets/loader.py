from __future__ import annotations

from pathlib import Path

from rho.datasets.directory import DirectoryDataset
from rho.datasets.locomo import LocomoDataset, LocomoSubsetDataset
from rho.datasets.swebench_pro import SWEbenchProDataset
from rho.protocols import Dataset, HarnessStore


def load_dataset(
    spec: str,
    *,
    harness_store: HarnessStore,
    max_per_split: int | None = None,
    docker_pull: str = "missing",
    difficulty_filter: tuple[str, ...] | None = None,
) -> Dataset:
    if ":" in spec:
        scheme, _, payload = spec.partition(":")
    else:
        scheme, payload = "directory", spec

    if scheme == "directory":
        path = Path(payload).resolve()
        if not path.exists():
            raise ValueError(
                f"Dataset path {payload!r} does not exist. "
                f"If this is a HuggingFace dataset, use a scheme prefix, "
                f"e.g. 'swebench-pro:{payload}'."
            )
        return DirectoryDataset(path, harness_store=harness_store)
    if scheme == "locomo":
        path = Path(payload).resolve()
        return LocomoDataset(path, harness_store=harness_store, max_per_split=max_per_split)
    if scheme == "locomo-hard":
        path = Path(payload).resolve()
        return LocomoSubsetDataset(
            path,
            harness_store=harness_store,
            max_per_split=max_per_split,
        )
    if scheme == "swebench-pro":
        return SWEbenchProDataset(
            payload,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
        )
    if scheme == "terminal-bench-2":
        from rho.datasets.terminal_bench_2 import TerminalBench2Dataset

        path = Path(payload).resolve()
        return TerminalBench2Dataset(
            path,
            harness_store=harness_store,
            max_per_split=max_per_split,
            docker_pull=docker_pull,
            difficulty_filter=difficulty_filter,
        )
    if scheme == "gaia2":
        import os

        from rho.datasets.gaia2 import Gaia2Dataset

        raw = os.getenv("RHO_GAIA2_ENABLE_JUDGE", "").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "GAIA-2 requires RHO_GAIA2_ENABLE_JUDGE=1 in the env. "
                "Without it the LLM judge stays off and every task silently "
                "scores 0. Canonical invocation:\n"
                "  RHO_GAIA2_ENABLE_JUDGE=1 uv run --extra gaia2 rho "
                "<cmd> --dataset gaia2:...\n"
                "See scripts/run-table1.sh for the full env-prefix pattern."
            )

        return Gaia2Dataset(
            payload,
            harness_store=harness_store,
            max_per_split=max_per_split,
        )
    raise ValueError(f"Unknown dataset scheme: {scheme!r}")
