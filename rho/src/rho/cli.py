from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rho.agent.base import Agent
from rho.agent.cache import build_default_agent
from rho.agent.codex import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_CHOICES,
    CodexAgent,
    default_codex_auth_home,
)
from rho.agent.codex_pool import (
    DEFAULT_CODEX_CONCURRENCY,
    configure_global_codex_pool,
)
from rho.datasets.loader import load_dataset
from rho.loop import run_evolution
from rho.observability import usage_summary
from rho.orchestrators.solve import solve
from rho.protocols import Dataset, Harness, Task, TaskSet, TrajectoryStore
from rho.reporting import grade_on_split, summarize
from rho.selection import (
    DEFAULT_DPP_THETA,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING,
    SELECTOR_CHOICES as _SELECTOR_CHOICES,
    build_embedder,
)
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import (
    DEFAULT_TRAJECTORIES_PER_TASK,
    OPTIMIZE_STRATEGY_CHOICES,
    build_optimize_strategy,
)

DEFAULT_CACHE_MODE = "off"
DOCKER_PULL_CHOICES = ("missing", "always", "never")
DEFAULT_REASONINGBANK_EMBEDDING_PROVIDER = "litellm"
# Mirror the task selector's default: a `local:` prefix routes to the
# on-machine FastEmbed ONNX encoder (no API key, no network). The previous
# `openrouter/...` default failed — litellm has no embedding route for the
# openrouter provider.
DEFAULT_REASONINGBANK_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_CONFIG_PATH = _REPO_ROOT / "configs" / "codex.azure-foundry.toml"
_CODEX_CONFIG_HELP = (
    "Path to a codex config.toml. Copied verbatim into the isolated "
    "CODEX_HOME for every agent run. Default: configs/codex.azure-foundry.toml "
    "(hits Azure OpenAI Foundry directly with an Entra Bearer refreshed by "
    "`az account get-access-token`). See configs/ for alternatives."
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    from rho.datasets.terminal_bench_2 import cleanup as _tb2_cleanup

    try:
        _tb2_cleanup.startup_sweep()
    except Exception:
        pass
    args = parser.parse_args(raw_argv)
    args._raw_argv = raw_argv
    return args.func(args)


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return n


def _nonnegative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a non-negative integer")
    return n


def _theta_value(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"theta must be a float, got {raw!r}") from exc
    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(f"theta must be in [0, 1], got {value}")
    return value


def _difficulty_filter(args: argparse.Namespace) -> tuple[str, ...] | None:
    if getattr(args, "difficulty", None) is None:
        return None
    return tuple(part.strip() for part in args.difficulty.split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rho")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evolve = subparsers.add_parser("evolve")

    evolve.add_argument("--dataset", required=True)
    evolve.add_argument("--rounds", type=int, required=True)
    evolve.add_argument(
        "--run-dir",
        default=None,
        help="Output directory. Default: runs/<timestamp>-<dataset>/",
    )
    evolve.add_argument(
        "--max-evolve-tasks",
        type=_positive_int,
        default=None,
        help="Max train tasks per evolution round (solve/optimize/evaluate). Default: all.",
    )
    evolve.add_argument(
        "--max-grading-tasks",
        type=int,
        default=None,
        help="Max val tasks for post-evolution grading. 0 to skip val grading. Default: all.",
    )
    evolve.add_argument(
        "--optimize-samples",
        type=_positive_int,
        default=3,
        help="How many parallel optimize samples to run per round. Default: 3.",
    )
    evolve.add_argument(
        "--optimize-strategy",
        choices=OPTIMIZE_STRATEGY_CHOICES,
        default="diagnosis",
        help=(
            "Optimize strategy. Default: diagnosis. "
            f"Choices: {list(OPTIMIZE_STRATEGY_CHOICES)}."
        ),
    )
    evolve.add_argument(
        "--optimize-trajectories-per-task",
        type=_positive_int,
        default=DEFAULT_TRAJECTORIES_PER_TASK,
        help=(
            "For --optimize-strategy=trajectory: how many solve trajectories "
            "per task to show the optimize agent (1..3). Default: 3."
        ),
    )
    evolve.add_argument(
        "--initial-harness",
        type=str,
        default=None,
        help="Harness directory path or ID in the run's store to start from. Default: dataset built-in harness.",
    )
    evolve.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help="Only include train tasks whose ID contains this substring.",
    )
    evolve.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for train task sampling order. Does not affect model or dataset split randomness. Default: no shuffle.",
    )
    evolve.add_argument(
        "--max-per-split",
        type=_positive_int,
        default=None,
        help="Cap tasks per dataset split (train/val/test). Default: all.",
    )
    evolve.add_argument(
        "--grade-workers",
        type=_positive_int,
        default=1,
        help="Max concurrent dataset grade() calls. Codex solve submission is limited by --codex-concurrency. Default: 1.",
    )
    _add_codex_concurrency_argument(evolve)
    evolve.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
        help="Docker image pull policy for datasets that grade in Docker. Default: missing.",
    )
    evolve.add_argument(
        "--difficulty",
        default=None,
        help="Comma-separated difficulty filter (easy,medium,hard,extreme). Only honored by TB2 dataset.",
    )
    evolve.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_CODEX_MODEL}.",
    )
    evolve.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Codex model reasoning effort via model_reasoning_effort. "
            f"Default: {DEFAULT_REASONING_EFFORT}."
        ),
    )
    evolve.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    evolve.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory when cache is enabled. Default: <run-dir>/agent-cache.",
    )
    evolve.add_argument(
        "--selector",
        choices=_SELECTOR_CHOICES,
        default="random",
        help=f"Task selection strategy. Default: random. Choices: {list(_SELECTOR_CHOICES)}.",
    )
    evolve.add_argument(
        "--selection-json",
        default=None,
        help="Reuse selected_task_ids from an existing selection.json instead of running a selector.",
    )
    evolve.add_argument(
        "--theta",
        type=_theta_value,
        default=DEFAULT_DPP_THETA,
        help=(
            "DPP tradeoff parameter in [0, 1]. 0 = pure diversity, "
            "1 = pure difficulty. Only used with --selector dpp. "
            f"Default: {DEFAULT_DPP_THETA}."
        ),
    )
    evolve.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    evolve.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Selector judge model. Default: {DEFAULT_JUDGE_MODEL}.",
    )
    evolve.add_argument(
        "--selector-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_JUDGE_REASONING,
        help=f"Selector judge reasoning effort. Default: {DEFAULT_JUDGE_REASONING}.",
    )
    evolve.set_defaults(func=_cmd_evolve)

    solve_cmd = subparsers.add_parser("solve")
    solve_cmd.add_argument("--dataset", required=True)
    solve_cmd.add_argument("--task", required=True)
    solve_cmd.add_argument("--harness", required=True)
    solve_cmd.add_argument("--run-dir", required=True)
    solve_cmd.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_CODEX_MODEL}.",
    )
    solve_cmd.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Codex model reasoning effort via model_reasoning_effort. "
            f"Default: {DEFAULT_REASONING_EFFORT}."
        ),
    )
    solve_cmd.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    solve_cmd.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory when cache is enabled. Default: <run-dir>/agent-cache.",
    )
    _add_codex_concurrency_argument(solve_cmd)
    solve_cmd.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
        help="Docker image pull policy for datasets that grade in Docker. Default: missing.",
    )
    solve_cmd.add_argument(
        "--difficulty",
        default=None,
        help="Comma-separated difficulty filter (easy,medium,hard,extreme). Only honored by TB2 dataset.",
    )
    solve_cmd.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    solve_cmd.set_defaults(func=_cmd_solve)

    grade = subparsers.add_parser("grade")
    grade.add_argument("--dataset", required=True)
    grade.add_argument("--split", required=True, choices=["train", "val", "test"])
    grade.add_argument("--harness", required=True)
    grade.add_argument("--run-dir", required=True)
    grade.add_argument(
        "--max-grading-tasks",
        type=_positive_int,
        default=None,
        help="Max tasks to grade. Default: all.",
    )
    grade.add_argument(
        "--grade-workers",
        type=_positive_int,
        default=1,
        help="Max concurrent dataset grade() calls. Codex solve submission is limited by --codex-concurrency. Default: 1.",
    )
    _add_codex_concurrency_argument(grade)
    grade.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
        help="Docker image pull policy for datasets that grade in Docker. Default: missing.",
    )
    grade.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_CODEX_MODEL}.",
    )
    grade.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Codex model reasoning effort via model_reasoning_effort. "
            f"Default: {DEFAULT_REASONING_EFFORT}."
        ),
    )
    grade.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    grade.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory when cache is enabled. Default: <run-dir>/agent-cache.",
    )
    grade.add_argument(
        "--difficulty",
        default=None,
        help="Comma-separated difficulty filter (easy,medium,hard,extreme). Only honored by TB2 dataset.",
    )
    grade.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    grade.set_defaults(func=_cmd_grade)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--run-dir", required=True)
    inspect.add_argument("--round", type=int, required=True)
    inspect.set_defaults(func=_cmd_inspect)

    select = subparsers.add_parser("select")
    select.add_argument("--dataset", required=True)
    select.add_argument(
        "--selector",
        choices=_SELECTOR_CHOICES,
        required=True,
        help=f"Task selection strategy. Choices: {list(_SELECTOR_CHOICES)}.",
    )
    select.add_argument(
        "-k",
        type=_positive_int,
        default=None,
        help="Number of tasks to pick. Required for difficulty/coverage; default 'all' for random.",
    )
    select.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to select from. Default: train.",
    )
    select.add_argument("--seed", type=int, default=None)
    select.add_argument("--task-filter", type=str, default=None)
    select.add_argument(
        "--max-per-split",
        type=_positive_int,
        default=None,
        help="Cap tasks loaded per split. Default: all.",
    )
    select.add_argument("--run-dir", default=None)
    select.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
    )
    select.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    select.add_argument(
        "--selector-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_JUDGE_REASONING,
        help=f"Selector judge reasoning effort. Default: {DEFAULT_JUDGE_REASONING}.",
    )
    select.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex solver model for short-solve probe. Default: {DEFAULT_CODEX_MODEL}.",
    )
    select.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=f"Codex solver reasoning effort for short-solve. Default: {DEFAULT_REASONING_EFFORT}.",
    )
    select.add_argument(
        "--initial-harness",
        default=None,
        help="Path or ID of the harness used for short-solve probe. "
             "Default: first task's dataset-built-in harness.",
    )
    select.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    _add_codex_concurrency_argument(select)
    select.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default="off",
        help="Agent response cache mode. Default: off.",
    )
    select.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory. Default: <run-dir>/agent-cache.",
    )
    select.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    select.add_argument(
        "--theta",
        type=_theta_value,
        default=DEFAULT_DPP_THETA,
        help=(
            "DPP tradeoff parameter in [0, 1]. 0 = pure diversity, "
            "1 = pure difficulty. Only used with --selector dpp. "
            f"Default: {DEFAULT_DPP_THETA}."
        ),
    )
    select.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Bypass the on-disk selector cache (always call the API). "
            "Note: this is the selection cache (data/cache/), not the agent "
            "cache used by --cache in evolve/solve/grade."
        ),
    )
    select.set_defaults(func=_cmd_select)

    reasoningbank = subparsers.add_parser(
        "reasoningbank",
        help="Run the ReasoningBank baseline on train, then evaluate frozen or online.",
    )
    reasoningbank.add_argument("--dataset", required=True)
    reasoningbank.add_argument(
        "--run-dir",
        default=None,
        help="Output directory. Default: runs/<timestamp>-reasoningbank-<dataset>/.",
    )
    reasoningbank.add_argument(
        "--max-train-tasks",
        type=_positive_int,
        default=None,
        help="Max selected train tasks for the memory stream. Default: all selected tasks.",
    )
    reasoningbank.add_argument(
        "--max-grading-tasks",
        type=_nonnegative_int,
        default=None,
        help="Max val tasks to evaluate. 0 skips val evaluation. Default: all.",
    )
    reasoningbank.add_argument(
        "--selector",
        choices=_SELECTOR_CHOICES,
        default="random",
        help=f"Train task selection strategy. Default: random. Choices: {list(_SELECTOR_CHOICES)}.",
    )
    reasoningbank.add_argument(
        "--selection-json",
        default=None,
        help="Reuse selected_task_ids from an existing selection.json instead of running a selector.",
    )
    reasoningbank.add_argument("--seed", type=int, default=None)
    reasoningbank.add_argument("--task-filter", type=str, default=None)
    reasoningbank.add_argument(
        "--max-per-split",
        type=_positive_int,
        default=None,
        help="Cap tasks per dataset split before selection/eval. Default: all.",
    )
    reasoningbank.add_argument(
        "--theta",
        type=_theta_value,
        default=DEFAULT_DPP_THETA,
        help=f"DPP theta in [0, 1]. Default: {DEFAULT_DPP_THETA}.",
    )
    reasoningbank.add_argument(
        "--eval-variant",
        choices=["frozen", "online"],
        default="frozen",
        help="Frozen keeps train memory fixed during val; online updates through val. Default: frozen.",
    )
    reasoningbank.add_argument(
        "--memory-n",
        type=_positive_int,
        default=1,
        help="Number of retrieved ReasoningBank entries per task. Default: 1.",
    )
    reasoningbank.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex solver model to use. Default: {DEFAULT_CODEX_MODEL}.",
    )
    reasoningbank.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=f"Codex solver reasoning effort. Default: {DEFAULT_REASONING_EFFORT}.",
    )
    reasoningbank.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Selector judge model. Default: {DEFAULT_JUDGE_MODEL}.",
    )
    reasoningbank.add_argument(
        "--selector-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_JUDGE_REASONING,
        help=f"Selector judge reasoning effort. Default: {DEFAULT_JUDGE_REASONING}.",
    )
    reasoningbank.add_argument(
        "--initial-harness",
        default=None,
        help="Path or ID of the harness used for short-solve probe. "
             "Default: first task's dataset-built-in harness.",
    )
    reasoningbank.add_argument(
        "--memory-model",
        default=None,
        help="ReasoningBank judge/extraction model. Default: openai/ form of --model.",
    )
    reasoningbank.add_argument(
        "--memory-reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_JUDGE_REASONING,
        help=f"ReasoningBank judge/extraction reasoning effort. Default: {DEFAULT_JUDGE_REASONING}.",
    )
    reasoningbank.add_argument(
        "--embedding-provider",
        choices=["official-gemini", "litellm"],
        default=DEFAULT_REASONINGBANK_EMBEDDING_PROVIDER,
        help=f"Retrieval embedding provider. Default: {DEFAULT_REASONINGBANK_EMBEDDING_PROVIDER}.",
    )
    reasoningbank.add_argument(
        "--embedding-model",
        default=DEFAULT_REASONINGBANK_EMBEDDING_MODEL,
        help=(
            "Embedding model when --embedding-provider litellm. A 'local:' "
            "prefix uses the on-machine FastEmbed ONNX encoder; other prefixes "
            f"route through litellm. Default: {DEFAULT_REASONINGBANK_EMBEDDING_MODEL}."
        ),
    )
    reasoningbank.add_argument(
        "--grade-workers",
        type=_positive_int,
        default=1,
        help="Max concurrent dataset grade() calls. Default: 1.",
    )
    _add_codex_concurrency_argument(reasoningbank)
    reasoningbank.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
        help="Docker image pull policy for Docker-backed datasets. Default: missing.",
    )
    reasoningbank.add_argument(
        "--difficulty",
        default=None,
        help="Comma-separated difficulty filter. Only honored by TB2 dataset.",
    )
    reasoningbank.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    reasoningbank.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory when cache is enabled. Default: <run-dir>/agent-cache.",
    )
    reasoningbank.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    reasoningbank.set_defaults(func=_cmd_reasoningbank)

    meta_harness = subparsers.add_parser(
        "meta-harness",
        help="Run the Meta-Harness baseline: filesystem-history harness search with ground-truth scoring.",
    )
    meta_harness.add_argument("--dataset", required=True)
    meta_harness.add_argument(
        "--run-dir",
        default=None,
        help="Output directory. Default: runs/<timestamp>-meta-harness-<dataset>/.",
    )
    meta_harness.add_argument(
        "--iterations",
        type=_positive_int,
        default=20,
        help="Number of Meta-Harness search iterations. Default: 20.",
    )
    meta_harness.add_argument(
        "--candidates-per-iter",
        type=_positive_int,
        default=3,
        help="Candidate harnesses the proposer produces per iteration. Default: 3.",
    )
    meta_harness.add_argument(
        "--search-trials",
        type=_positive_int,
        default=2,
        help="Solve attempts per task when scoring a candidate on the search set. Default: 2.",
    )
    meta_harness.add_argument(
        "--max-search-tasks",
        type=_positive_int,
        default=None,
        help="Cap the fixed search set drawn from the train split. Default: all.",
    )
    meta_harness.add_argument(
        "--max-test-tasks",
        type=_nonnegative_int,
        default=None,
        help="Max test tasks for the final evaluation. 0 skips it. Default: all.",
    )
    meta_harness.add_argument(
        "--selection-json",
        default=None,
        help="Reuse selected_task_ids from an existing selection.json as the fixed search set.",
    )
    meta_harness.add_argument(
        "--final-split",
        choices=("val", "test"),
        default="test",
        help="Dataset split used for the final held-out evaluation. Default: test.",
    )
    meta_harness.add_argument("--seed", type=int, default=None)
    meta_harness.add_argument("--task-filter", type=str, default=None)
    meta_harness.add_argument(
        "--max-per-split",
        type=_positive_int,
        default=None,
        help="Cap tasks loaded per dataset split. Default: all.",
    )
    meta_harness.add_argument(
        "--initial-harness",
        default=None,
        help="Seed harness directory path or store ID. Default: dataset built-in harness.",
    )
    meta_harness.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model for the proposer and solver. Default: {DEFAULT_CODEX_MODEL}.",
    )
    meta_harness.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=f"Codex reasoning effort. Default: {DEFAULT_REASONING_EFFORT}.",
    )
    meta_harness.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    meta_harness.add_argument(
        "--cache-dir",
        default=None,
        help="Agent response cache directory when cache is enabled. Default: <run-dir>/agent-cache.",
    )
    _add_codex_concurrency_argument(meta_harness)
    meta_harness.add_argument(
        "--docker-pull",
        choices=DOCKER_PULL_CHOICES,
        default="missing",
        help="Docker image pull policy for datasets that grade in Docker. Default: missing.",
    )
    meta_harness.add_argument(
        "--difficulty",
        default=None,
        help="Comma-separated difficulty filter (easy,medium,hard,extreme). Only honored by TB2.",
    )
    meta_harness.add_argument("--codex-config", default=None, help=_CODEX_CONFIG_HELP)
    meta_harness.set_defaults(func=_cmd_meta_harness)

    ui = subparsers.add_parser("ui")
    ui.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing run folders. Default: runs/",
    )
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.set_defaults(func=_cmd_ui)

    tb2_cleanup = subparsers.add_parser("tb2-cleanup", help="Remove orphaned TB2 containers")
    tb2_cleanup.add_argument(
        "--all",
        action="store_true",
        help="Remove every tbench2-* container, live or not.",
    )
    tb2_cleanup.set_defaults(func=_cmd_tb2_cleanup)

    return parser


def _add_codex_concurrency_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--codex-binary", default="codex", help="Codex CLI executable or absolute path.",
    )
    parser.add_argument(
        "--codex-sandbox", choices=("read-only", "workspace-write", "danger-full-access"),
        default=None, help="Explicit Codex sandbox mode; defaults to host autodetection.",
    )
    parser.add_argument(
        "--codex-concurrency",
        type=_positive_int,
        default=DEFAULT_CODEX_CONCURRENCY,
        help=(
            "Max concurrent codex exec subprocesses in this Python process. "
            f"Default: {DEFAULT_CODEX_CONCURRENCY}."
        ),
    )


def _cmd_evolve(args: argparse.Namespace) -> int:
    if args.optimize_strategy == "dynamic-cheatsheet" and args.optimize_samples != 1:
        print(
            "error: --optimize-strategy dynamic-cheatsheet is single-stream and "
            "requires --optimize-samples 1 "
            f"(got --optimize-samples {args.optimize_samples}).",
            file=sys.stderr,
        )
        return 2
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{timestamp}-{_dataset_slug(args.dataset)}"
    else:
        run_dir = Path(args.run_dir)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "rounds").mkdir(exist_ok=True)
    (run_dir / "workdir").mkdir(exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    strategy = build_optimize_strategy(
        args.optimize_strategy,
        trajectories_per_task=args.optimize_trajectories_per_task,
    )
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        max_per_split=args.max_per_split,
        docker_pull=args.docker_pull,
        difficulty_filter=_difficulty_filter(args),
    )

    agent_cache_dir = _agent_cache_dir(args, run_dir)
    codex_config_path = _resolved_codex_config_path(args)
    config = {
        "argv": args._raw_argv,
        "dataset_spec": args.dataset,
        "dataset_digest": _digest_dataset(args.dataset),
        "n_rounds": args.rounds,
        "selector": args.selector,
        "selection_json": str(Path(args.selection_json).expanduser().resolve())
        if args.selection_json is not None
        else None,
        "max_per_split": args.max_per_split,
        "max_evolve_tasks": args.max_evolve_tasks,
        "max_grading_tasks": args.max_grading_tasks,
        "optimize_samples": args.optimize_samples,
        "optimize_strategy": args.optimize_strategy,
        "optimize_trajectories_per_task": args.optimize_trajectories_per_task,
        "grade_workers": args.grade_workers,
        "codex_concurrency": args.codex_concurrency,
        "docker_pull": args.docker_pull,
        "initial_harness": args.initial_harness,
        "seed": args.seed,
        "accept_rule": "mean>0",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "judge_model": args.judge_model,
        "selector_reasoning_effort": args.selector_reasoning_effort,
        "cache_mode": args.cache,
        "cache_dir": str(agent_cache_dir) if agent_cache_dir is not None else None,
        "codex_isolation": _codex_isolation_config(),
        **_audit_codex_config(run_dir, codex_config_path),
        "start_timestamp": _now_iso(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _capture_environment())
    strategy_log = f"[evolve] optimize_strategy={args.optimize_strategy}"
    if args.optimize_strategy == "trajectory":
        strategy_log += f" trajectories_per_task={args.optimize_trajectories_per_task}"
    print(strategy_log)

    agent = _build_agent(args, run_dir=run_dir)

    train_tasks = dataset.train
    if args.task_filter is not None:
        train_tasks = [t for t in train_tasks if args.task_filter in t.id]
        if not train_tasks:
            print(f"No train tasks match filter {args.task_filter!r}")
            return 1
        print(f"Task filter matched {len(train_tasks)} train task(s)")

    if args.initial_harness is not None:
        p = Path(args.initial_harness)
        initial = harness_store.capture(p) if p.is_dir() else harness_store.get(args.initial_harness)
    else:
        first_task = train_tasks[0] if isinstance(train_tasks, list) else next(iter(train_tasks))
        initial = first_task.harness

    all_candidates = list(train_tasks)

    if args.selection_json is not None:
        # Reusing a stored selection — skip short-solve, judge, selector
        # entirely. Matches the §9.3 --selection-json bypass in reasoningbank.
        selector_args = ("--selector", "--theta")
        if any(
            raw == option or raw.startswith(f"{option}=")
            for raw in args._raw_argv
            for option in selector_args
        ):
            print(
                "warning: --selection-json provided; ignoring --selector/--theta.",
                file=sys.stderr,
            )
        selection_path = Path(args.selection_json).expanduser().resolve()
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_ids = list(payload["selected_task_ids"])
        if args.max_evolve_tasks is not None:
            selected_ids = selected_ids[: args.max_evolve_tasks]
        by_id = {task.id: task for task in all_candidates}
        missing = [task_id for task_id in selected_ids if task_id not in by_id]
        if missing:
            raise KeyError(
                f"selection_json references task ids absent from train split: {missing[:5]}"
            )
        selected_tasks = [by_id[task_id] for task_id in selected_ids]
        selection_record = {
            "selection_json": str(selection_path),
            "selected_task_ids": selected_ids,
            "all_candidate_ids": [t.id for t in all_candidates],
        }
    else:
        from rho.selection import CoverageSelector as _CS
        from rho.selection import DifficultySelector as _DS
        from rho.selection import DPPSelector as _DPP
        from rho.selection import build_selector
        from rho.selection.short_solve import short_solve_all

        trajectories: dict | None = None
        if args.selector != "random":
            print(
                f"[evolve] short-solve probe over {len(all_candidates)} candidate(s) "
                f"on initial harness {initial.id} (selector={args.selector})"
            )
            trajectories = short_solve_all(
                all_candidates,
                agent=agent,
                harness=initial,
                traj_store=traj_store,
                workdir=run_dir / "workdir" / "short_solve",
                max_workers=args.codex_concurrency,
            )

        selector = build_selector(
            args.selector,
            workdir=run_dir / "selector_calls",
            judge_model=args.judge_model,
            judge_reasoning=args.selector_reasoning_effort,
            theta=args.theta,
            trajectories=trajectories,
        )

        selected_tasks = selector.select(
            all_candidates, k=args.max_evolve_tasks, seed=args.seed
        )
        selection_record = {
            "selector": args.selector,
            "k": args.max_evolve_tasks,
            "seed": args.seed,
            "all_candidate_ids": [t.id for t in all_candidates],
            "selected_task_ids": [t.id for t in selected_tasks],
        }
        if isinstance(selector, (_DS, _DPP)):
            selection_record["difficulty_scores"] = {
                task_id: result.difficulty
                for task_id, result in selector.results().items()
            }
        if isinstance(selector, _DPP):
            selection_record["theta"] = selector.theta
        if isinstance(selector, (_CS, _DPP)):
            fingerprints_path = run_dir / "selector_calls" / "fingerprints.json"
            if fingerprints_path.exists():
                selection_record["fingerprints"] = json.loads(
                    fingerprints_path.read_text(encoding="utf-8")
                )
        if trajectories is not None:
            selection_record["short_solve_trajectory_ids"] = {
                tid: traj.id for tid, traj in trajectories.items()
            }
            selection_record["judge_input_token_estimate"] = {
                task_id: result.digest_token_estimate
                for task_id, result in selector.results().items()
            }
    _write_json(run_dir / "selection.json", selection_record)

    final, rounds = run_evolution(
        train=selected_tasks,
        n_rounds=args.rounds,
        agent=agent,
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=run_dir / "workdir",
        rounds_dir=run_dir / "rounds",
        initial=initial,
        strategy=strategy,
        optimize_samples=args.optimize_samples,
        solve_workers=args.codex_concurrency,
    )

    skip_val = args.max_grading_tasks == 0
    if skip_val:
        final_grades = []
        final_summary = {"mean_score": None, "n": 0}
    else:
        final_grades = grade_on_split(
            agent,
            final,
            dataset.val,
            run_dir / "workdir",
            max_tasks=args.max_grading_tasks,
            traj_store=traj_store,
            stage="final_val_grade",
            artifacts_root=run_dir / "workdir" / "grade_artifacts",
            max_workers=args.grade_workers,
            solve_workers=args.codex_concurrency,
        )
        final_summary = summarize(final_grades)

    report_dir = run_dir / "reports"
    _write_json(report_dir / "final_val_grades.json", _serialize_grades(final_grades))

    summary = {
        "initial_harness_id": initial.id,
        "final_harness_id": final.id,
        "optimize_strategy": args.optimize_strategy,
        "optimize_trajectories_per_task": args.optimize_trajectories_per_task,
        "final_val": final_summary,
        "rounds": [
            {
                "round_ix": round_result.round_ix,
                "optimize_samples": args.optimize_samples,
                "unique_candidate_count": len(round_result.candidate_pool),
                "candidate_harness_id": round_result.candidate.id,
                "accepted": round_result.accepted,
                "mean_score": round_result.mean_score,
                "winner_sample_index": round_result.winner_sample_index,
                "candidates": [
                    {
                        "candidate_harness_id": candidate_result.candidate.id,
                        "sample_indices": candidate_result.sample_indices,
                        "mean_score": candidate_result.mean_score,
                        "accepted": round_result.accepted
                        and candidate_result.candidate.id == round_result.candidate.id,
                    }
                    for candidate_result in round_result.candidate_pool
                ],
                "scores": [
                    {"task_id": task_id, "value": score.value, "rationale": score.rationale}
                    for task_id, score in zip(round_result.score_task_ids, round_result.scores)
                ],
            }
            for round_result in rounds
        ],
        "end_timestamp": _now_iso(),
    }
    _write_json(report_dir / "summary.json", summary)
    all_trajs = list(traj_store._iter_all())
    _write_json(report_dir / "usage_summary.json", usage_summary(all_trajs))
    _write_json(report_dir / "manifest.json", _build_manifest(run_dir, all_trajs, rounds))
    summary_text = _format_summary(summary)
    (report_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    codex_config_path = _resolved_codex_config_path(args)
    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        docker_pull=args.docker_pull,
        difficulty_filter=_difficulty_filter(args),
    )
    task = _find_task(dataset, args.task)
    harness = harness_store.get(args.harness)
    _audit_codex_config(run_dir, codex_config_path)
    agent = _build_agent(args, run_dir=run_dir)
    traj = solve(agent, task, harness, workdir=run_dir / "workdir", stage="cli_solve")
    traj_store.put(traj)
    _refresh_run_reports(run_dir, traj_store)
    print(traj.final_message)
    return 0


def _cmd_grade(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    codex_config_path = _resolved_codex_config_path(args)
    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        docker_pull=args.docker_pull,
        difficulty_filter=_difficulty_filter(args),
    )
    harness = harness_store.get(args.harness)
    _audit_codex_config(run_dir, codex_config_path)
    agent = _build_agent(args, run_dir=run_dir)
    split = {
        "train": dataset.train,
        "val": dataset.val,
        "test": dataset.test,
    }[args.split]
    grades = grade_on_split(
        agent,
        harness,
        split,
        run_dir / "workdir",
        max_tasks=args.max_grading_tasks,
        traj_store=traj_store,
        stage=f"cli_{args.split}_grade",
        artifacts_root=run_dir / "workdir" / "grade_artifacts",
        max_workers=args.grade_workers,
        solve_workers=args.codex_concurrency,
    )
    _refresh_run_reports(run_dir, traj_store)
    print(json.dumps(_serialize_grades(grades), ensure_ascii=False, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    round_dir = Path(args.run_dir).resolve() / "rounds" / f"round_{args.round}"
    candidates_path = round_dir / "optimize_candidates.json"
    candidates_payload = (
        json.loads(candidates_path.read_text(encoding="utf-8"))
        if candidates_path.exists()
        else None
    )
    lines = [
        f"round: {args.round}",
        f"optimize_strategy: {config.get('optimize_strategy', 'diagnosis')}",
        f"input_harness_id: {(round_dir / 'input_harness_id').read_text(encoding='utf-8').strip()}",
        f"candidate_harness_id: {(round_dir / 'candidate_harness_id').read_text(encoding='utf-8').strip()}",
        f"accepted: {(round_dir / 'accepted').read_text(encoding='utf-8').strip()}",
        f"mean_score: {(round_dir / 'mean_score').read_text(encoding='utf-8').strip()}",
    ]
    if candidates_payload is not None:
        lines += [
            f"optimize_samples: {len(candidates_payload.get('samples', []))}",
            f"unique_candidate_count: {len(candidates_payload.get('unique_candidates', []))}",
            f"winner_sample_index: {candidates_payload.get('winner_sample_index')}",
        ]
        for candidate in candidates_payload.get("unique_candidates", []):
            lines.append(
                "  candidate {candidate_harness_id}: mean_score={mean_score:.2f} samples={samples} accepted={accepted}".format(
                    candidate_harness_id=candidate["candidate_harness_id"],
                    mean_score=candidate["mean_score"],
                    samples=candidate["sample_indices"],
                    accepted=str(candidate["accepted"]).lower(),
                )
            )
    print("\n".join(lines))
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    if args.selector in ("difficulty", "coverage") and args.k is None:
        print(f"--selector {args.selector} requires -k", file=sys.stderr)
        return 2

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = f"{timestamp}-select-{args.selector}-{_dataset_slug(args.dataset)}"
        run_dir = Path("runs") / slug
    else:
        run_dir = Path(args.run_dir)
    run_dir = run_dir.resolve()
    if (run_dir / "selection.json").exists():
        print(
            f"Refusing to overwrite existing selection at {run_dir / 'selection.json'}. "
            f"Pick a different --run-dir or remove it first.",
            file=sys.stderr,
        )
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        max_per_split=args.max_per_split,
        docker_pull=args.docker_pull,
    )
    pool = list({"train": dataset.train, "val": dataset.val, "test": dataset.test}[args.split])
    if args.task_filter is not None:
        pool = [task for task in pool if args.task_filter in task.id]
        if not pool:
            print(
                f"No tasks in split {args.split!r} match filter {args.task_filter!r}",
                file=sys.stderr,
            )
            return 1

    if args.k is not None and args.k > len(pool):
        print(
            f"warning: -k {args.k} exceeds pool size {len(pool)}; "
            f"selector will return all {len(pool)} task(s).",
            file=sys.stderr,
        )

    config = {
        "argv": args._raw_argv,
        "dataset_spec": args.dataset,
        "dataset_digest": _digest_dataset(args.dataset),
        "selector": args.selector,
        "split": args.split,
        "k": args.k,
        "seed": args.seed,
        "task_filter": args.task_filter,
        "max_per_split": args.max_per_split,
        "judge_model": args.judge_model,
        "selector_reasoning_effort": args.selector_reasoning_effort,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "initial_harness": args.initial_harness,
        "cache_mode": args.cache,
        "embedding_model": args.embedding_model,
        "no_cache": args.no_cache,
        "start_timestamp": _now_iso(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _capture_environment())

    from rho.selection import DEFAULT_CACHE_ROOT, DPPSelector as _DPP
    from rho.selection import DifficultySelector, build_selector
    from rho.selection.short_solve import short_solve_all

    if args.initial_harness is not None:
        p = Path(args.initial_harness)
        initial = harness_store.capture(p) if p.is_dir() else harness_store.get(args.initial_harness)
    else:
        initial = pool[0].harness
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")

    trajectories: dict | None = None
    if args.selector != "random":
        agent = _build_agent(args, run_dir=run_dir)
        print(
            f"[select] short-solve probe over {len(pool)} candidate(s) "
            f"on initial harness {initial.id} (selector={args.selector})"
        )
        trajectories = short_solve_all(
            pool,
            agent=agent,
            harness=initial,
            traj_store=traj_store,
            workdir=run_dir / "workdir" / "short_solve",
            max_workers=args.codex_concurrency,
        )

    selector = build_selector(
        args.selector,
        workdir=run_dir / "selector_calls",
        judge_model=args.judge_model,
        judge_reasoning=args.selector_reasoning_effort,
        embedding_model=args.embedding_model,
        cache_root=None if args.no_cache else DEFAULT_CACHE_ROOT,
        theta=args.theta,
        trajectories=trajectories,
    )

    selected = selector.select(pool, k=args.k, seed=args.seed)

    selection_record: dict[str, Any] = {
        "selector": args.selector,
        "k": args.k,
        "seed": args.seed,
        "all_candidate_ids": [task.id for task in pool],
        "selected_task_ids": [task.id for task in selected],
    }
    if isinstance(selector, (DifficultySelector, _DPP)):
        selection_record["difficulty_scores"] = {
            task_id: result.difficulty
            for task_id, result in selector.results().items()
        }
    if isinstance(selector, _DPP):
        selection_record["theta"] = selector.theta
    _fingerprints_path = run_dir / "selector_calls" / "fingerprints.json"
    if _fingerprints_path.exists():
        selection_record["fingerprints"] = json.loads(
            _fingerprints_path.read_text(encoding="utf-8")
        )
    if trajectories is not None:
        selection_record["short_solve_trajectory_ids"] = {
            tid: traj.id for tid, traj in trajectories.items()
        }
        selection_record["judge_input_token_estimate"] = {
            task_id: result.digest_token_estimate
            for task_id, result in selector.results().items()
        }
    _write_json(run_dir / "selection.json", selection_record)

    import json as _json

    from rho.selection.report import write_selection_report

    queries = {task.id: task.query() for task in pool}
    workdir = run_dir / "selector_calls"

    scores_arg: dict[str, float] | None = None
    fingerprints_arg: dict[str, str] | None = None
    gain_trace_arg: list[dict[str, Any]] | None = None
    similarity_arg = None
    candidate_ids_arg: list[str] | None = None

    if args.selector == "difficulty":
        scores_arg = selection_record.get("difficulty_scores")
        if isinstance(selector, DifficultySelector):
            fingerprints_arg = {
                task_id: result.fingerprint
                for task_id, result in selector.results().items()
            }
    elif args.selector == "coverage":
        fingerprint_path = workdir / "fingerprints.json"
        gain_path = workdir / "gain_trace.json"
        sim_path = workdir / "similarity.npy"
        ids_path = workdir / "candidate_ids.json"
        if fingerprint_path.exists():
            fingerprints_arg = _json.loads(
                fingerprint_path.read_text(encoding="utf-8")
            )
        if gain_path.exists():
            gain_trace_arg = _json.loads(gain_path.read_text(encoding="utf-8"))
        if sim_path.exists() and ids_path.exists():
            import numpy as np

            similarity_arg = np.load(sim_path)
            candidate_ids_arg = _json.loads(ids_path.read_text(encoding="utf-8"))
    elif args.selector == "dpp":
        scores_arg = selection_record.get("difficulty_scores")
        fingerprint_path = workdir / "fingerprints.json"
        trace_path = workdir / "dpp_trace.json"
        sim_path = workdir / "similarity.npy"
        ids_path = workdir / "candidate_ids.json"
        if fingerprint_path.exists():
            fingerprints_arg = _json.loads(
                fingerprint_path.read_text(encoding="utf-8")
            )
        if trace_path.exists():
            gain_trace_arg = _json.loads(trace_path.read_text(encoding="utf-8"))
        if sim_path.exists() and ids_path.exists():
            import numpy as np

            similarity_arg = np.load(sim_path)
            candidate_ids_arg = _json.loads(ids_path.read_text(encoding="utf-8"))

    write_selection_report(
        run_dir=run_dir,
        selection=selection_record,
        queries=queries,
        dataset_spec=args.dataset,
        split=args.split,
        scores=scores_arg,
        fingerprints=fingerprints_arg,
        gain_trace=gain_trace_arg,
        similarity=similarity_arg,
        candidate_ids=candidate_ids_arg,
    )

    print(f"Selected {len(selected)} / {len(pool)} tasks from {args.split} split")
    print(f"Run dir: {run_dir}")
    return 0


def _cmd_reasoningbank(args: argparse.Namespace) -> int:
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{timestamp}-reasoningbank-{_dataset_slug(args.dataset)}"
    else:
        run_dir = Path(args.run_dir)
    run_dir = run_dir.resolve()
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "workdir").mkdir(parents=True, exist_ok=True)
    (run_dir / "reasoningbank").mkdir(parents=True, exist_ok=True)
    (run_dir / "selector_calls").mkdir(parents=True, exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        max_per_split=args.max_per_split,
        docker_pull=args.docker_pull,
        difficulty_filter=_difficulty_filter(args),
    )

    train_pool = list(dataset.train)
    if args.task_filter is not None:
        train_pool = [task for task in train_pool if args.task_filter in task.id]
        if not train_pool:
            print(
                f"No train tasks match filter {args.task_filter!r}",
                file=sys.stderr,
            )
            return 1

    if args.initial_harness is not None:
        p = Path(args.initial_harness)
        initial = harness_store.capture(p) if p.is_dir() else harness_store.get(args.initial_harness)
    else:
        initial = train_pool[0].harness
    agent = _build_agent(args, run_dir=run_dir)

    selected_train, selection_record = _reasoningbank_select_train_tasks(
        args,
        train_pool,
        run_dir,
        agent=agent,
        initial_harness=initial,
        traj_store=traj_store,
    )
    if not selected_train:
        print("ReasoningBank selected zero train tasks", file=sys.stderr)
        return 1
    _write_json(run_dir / "selection.json", selection_record)

    eval_tasks = list(dataset.val)
    if args.max_grading_tasks is not None:
        eval_tasks = eval_tasks[: args.max_grading_tasks]

    first_task = selected_train[0]
    harness = initial  # use the resolved short-solve harness for the runner too

    memory_model = args.memory_model or _reasoningbank_default_memory_model(args.model)
    memory_path = run_dir / "reasoningbank" / "memory.jsonl"
    embedding_cache_path = run_dir / "reasoningbank" / "embeddings.jsonl"
    memory_store, retriever, memory_llm = _build_reasoningbank_components(
        args,
        memory_model=memory_model,
        memory_path=memory_path,
        embedding_cache_path=embedding_cache_path,
    )

    codex_config_path = _resolved_codex_config_path(args)
    config = {
        "argv": args._raw_argv,
        "baseline": "reasoningbank",
        "dataset_spec": args.dataset,
        "dataset_digest": _digest_dataset(args.dataset),
        "selector": args.selector,
        "selection_json": args.selection_json,
        "seed": args.seed,
        "task_filter": args.task_filter,
        "max_per_split": args.max_per_split,
        "max_train_tasks": args.max_train_tasks,
        "max_grading_tasks": args.max_grading_tasks,
        "eval_variant": args.eval_variant,
        "memory_n": args.memory_n,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "judge_model": args.judge_model,
        "selector_reasoning_effort": args.selector_reasoning_effort,
        "initial_harness": args.initial_harness,
        "memory_model": memory_model,
        "memory_reasoning_effort": args.memory_reasoning_effort,
        "embedding_provider": args.embedding_provider,
        "embedding_model": _reasoningbank_embedding_model_for_config(args),
        "memory_path": str(memory_path),
        "embedding_cache_path": str(embedding_cache_path),
        "grade_workers": args.grade_workers,
        "codex_concurrency": args.codex_concurrency,
        "docker_pull": args.docker_pull,
        "cache_mode": args.cache,
        "cache_dir": str(_agent_cache_dir(args, run_dir))
        if _agent_cache_dir(args, run_dir) is not None
        else None,
        "fidelity_notes": [
            "train stream follows ReasoningBank online append semantics",
            "frozen eval reads train memory/cache and skips val memory/cache updates",
            "online eval variant appends val memories for appendix analyses",
            "trajectory text is rendered from rho trajectory artifacts",
        ],
        "codex_isolation": _codex_isolation_config(),
        **_audit_codex_config(run_dir, codex_config_path),
        "start_timestamp": _now_iso(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _capture_environment())

    from rho.reasoningbank.runner import ReasoningBankRunner

    runner = ReasoningBankRunner(
        agent=agent,
        memory_llm=memory_llm,
        retriever=retriever,
        memory_store=memory_store,
        traj_store=traj_store,
        workdir=run_dir / "workdir",
        harness=harness,
        memory_n=args.memory_n,
        eval_variant=args.eval_variant,
        grade_workers=args.grade_workers,
        solve_workers=args.codex_concurrency,
        artifacts_root=run_dir / "workdir" / "grade_artifacts",
    )
    result = runner.run(train_tasks=selected_train, eval_tasks=eval_tasks)

    report_dir = run_dir / "reports"
    _write_json(
        report_dir / "train_grades.json",
        _serialize_reasoningbank_records(result.train_records),
    )
    _write_json(
        report_dir / "eval_grades.json",
        _serialize_reasoningbank_records(result.eval_records),
    )
    _write_json(
        report_dir / "final_val_grades.json",
        _serialize_reasoningbank_records(result.eval_records),
    )
    summary = {
        "baseline": "reasoningbank",
        "eval_variant": args.eval_variant,
        "train": result.train_summary,
        "eval": result.eval_summary,
        "selected_train_task_ids": [task.id for task in selected_train],
        "eval_task_ids": [task.id for task in eval_tasks],
        "memory_entry_count": len(memory_store.load()),
        "memory_path": str(memory_path),
        "embedding_cache_path": str(embedding_cache_path),
        "embedding_provider": args.embedding_provider,
        "embedding_model": _reasoningbank_embedding_model_for_config(args),
        "embedding_dimensions_observed": _observed_embedding_dimensions(
            embedding_cache_path
        ),
        "end_timestamp": _now_iso(),
    }
    _write_json(report_dir / "summary.json", summary)
    all_trajs = list(traj_store._iter_all())
    _write_json(report_dir / "usage_summary.json", usage_summary(all_trajs))
    _write_json(report_dir / "manifest.json", _build_manifest(run_dir, all_trajs, []))
    summary_text = _format_reasoningbank_summary(summary)
    (report_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


def _cmd_meta_harness(args: argparse.Namespace) -> int:
    from rho.meta_harness import run_meta_harness

    # --- run dir (copy from _cmd_reasoningbank) ---
    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{timestamp}-meta-harness-{_dataset_slug(args.dataset)}"
    else:
        run_dir = Path(args.run_dir)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir = run_dir / "meta_harness"
    run_meta_dir.mkdir(parents=True, exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")

    # --- agent + dataset (copy from _cmd_reasoningbank) ---
    agent = _build_agent(args, run_dir=run_dir)
    dataset = load_dataset(
        args.dataset,
        harness_store=harness_store,
        max_per_split=args.max_per_split,
        docker_pull=args.docker_pull,
        difficulty_filter=_difficulty_filter(args),
    )
    # NOTE: `_difficulty_filter(args)` is the exact expression _cmd_reasoningbank /
    # _cmd_evolve pass to load_dataset. If that helper has a different name in
    # cli.py, copy whatever those commands use verbatim.

    # --- fixed search set from train (or selection.json), held-out final set ---
    if args.selection_json is not None:
        selection_path = Path(args.selection_json).expanduser().resolve()
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_ids = list(payload["selected_task_ids"])
        if args.max_search_tasks is not None:
            selected_ids = selected_ids[: args.max_search_tasks]
        by_id = {task.id: task for task in dataset.train}
        missing = [task_id for task_id in selected_ids if task_id not in by_id]
        if missing:
            raise KeyError(
                f"selection_json references task ids absent from train split: {missing[:5]}"
            )
        search_tasks = [by_id[task_id] for task_id in selected_ids]
    else:
        search_tasks = _meta_harness_search_tasks(
            dataset.train,
            task_filter=args.task_filter,
            seed=args.seed,
            max_tasks=args.max_search_tasks,
        )
    if not search_tasks:
        print("error: meta-harness search set is empty after filtering", file=sys.stderr)
        return 1

    final_split_taskset = dataset.val if args.final_split == "val" else dataset.test
    test_tasks: list[Task] = []
    if args.max_test_tasks != 0:
        test_tasks = list(final_split_taskset)
        if args.max_test_tasks is not None:
            test_tasks = test_tasks[: args.max_test_tasks]

    # --- seed harness: explicit override, else dataset built-in ---
    if args.initial_harness is not None:
        seed_path = Path(args.initial_harness)
        seed_harness = (
            harness_store.capture(seed_path)
            if seed_path.is_dir()
            else harness_store.get(args.initial_harness)
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seed"
            search_tasks[0].harness.materialize(seed_dir)
            seed_harness = harness_store.capture(seed_dir)

    config = {
        "baseline": "meta-harness",
        "dataset": args.dataset,
        "iterations": args.iterations,
        "candidates_per_iter": args.candidates_per_iter,
        "search_trials": args.search_trials,
        "selection_json": str(Path(args.selection_json).expanduser().resolve())
        if args.selection_json is not None
        else None,
        "final_split": args.final_split,
        "search_task_ids": [task.id for task in search_tasks],
        "test_task_ids": [task.id for task in test_tasks],
        "seed_harness_id": seed_harness.id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        **_audit_codex_config(run_dir, _resolved_codex_config_path(args)),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _capture_environment())
    print(
        f"[meta-harness] dataset={args.dataset} iterations={args.iterations} "
        f"candidates_per_iter={args.candidates_per_iter} "
        f"search_tasks={len(search_tasks)} test_tasks={len(test_tasks)}"
    )

    result = run_meta_harness(
        agent=agent,
        search_tasks=search_tasks,
        test_tasks=test_tasks,
        seed_harness=seed_harness,
        harness_store=harness_store,
        traj_store=traj_store,
        run_meta_dir=run_meta_dir,
        workdir=run_dir / "workdir",
        iterations=args.iterations,
        candidates_per_iter=args.candidates_per_iter,
        search_trials=args.search_trials,
        solve_workers=args.codex_concurrency,
    )

    test_pass_rate = (
        sum(g.grade.passed for g in result.test_grades) / len(result.test_grades)
        if result.test_grades
        else None
    )
    summary = {
        "baseline": "meta-harness",
        "n_candidates": len(result.records),
        "best_harness_id": result.best.harness_id,
        "best_search_mean_score": result.best.mean_score,
        "best_search_pass_rate": result.best.pass_rate,
        "test_pass_rate": test_pass_rate,
    }
    _write_json(run_dir / "summary.json", summary)
    print(
        f"meta-harness: best={result.best.harness_id} "
        f"search_mean={result.best.mean_score:.4f} test_pass_rate={test_pass_rate}"
    )
    return 0


def _meta_harness_search_tasks(
    train: TaskSet,
    *,
    task_filter: str | None,
    seed: int | None,
    max_tasks: int | None,
) -> list[Task]:
    """Build the fixed search set: filter, optional seeded shuffle, then cap.

    `max_per_split` is already applied by load_dataset, so it is not re-applied here.
    """
    tasks = list(train)
    if task_filter is not None:
        tasks = [task for task in tasks if task_filter in task.id]
    if seed is not None:
        import random

        random.Random(seed).shuffle(tasks)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def _cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn

    from rho.webui import create_app

    runs_dir = Path(args.runs_dir).resolve()
    app = create_app(runs_root=runs_dir)
    print(f"Serving runs UI from {runs_dir} at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_tb2_cleanup(args: argparse.Namespace) -> int:
    from rho.datasets.terminal_bench_2.cleanup import cli_cleanup

    removed = cli_cleanup(all_tb2=args.all)
    print(f"Removed {removed} container(s).")
    return 0


def _reasoningbank_select_train_tasks(
    args: argparse.Namespace,
    train_pool: list[Task],
    run_dir: Path,
    *,
    agent: Agent | None = None,
    initial_harness: Harness | None = None,
    traj_store: TrajectoryStore | None = None,
) -> tuple[list[Task], dict[str, Any]]:
    # Selection-json bypass reuses a stored selection without short-solve.
    if args.selection_json is not None:
        selection_path = Path(args.selection_json).expanduser().resolve()
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_ids = list(payload["selected_task_ids"])
        if args.max_train_tasks is not None:
            selected_ids = selected_ids[: args.max_train_tasks]
        by_id = {task.id: task for task in train_pool}
        missing = [task_id for task_id in selected_ids if task_id not in by_id]
        if missing:
            raise KeyError(
                f"selection_json references task ids absent from train split: {missing[:5]}"
            )
        selection_record = dict(payload)
        selection_record["selection_json"] = str(selection_path)
        selection_record["selected_task_ids"] = selected_ids
        selection_record.setdefault("all_candidate_ids", [task.id for task in train_pool])
        return [by_id[task_id] for task_id in selected_ids], selection_record

    from rho.selection import CoverageSelector as _CS
    from rho.selection import DifficultySelector as _DS
    from rho.selection import DPPSelector as _DPP
    from rho.selection import build_selector
    from rho.selection.short_solve import short_solve_all

    trajectories: dict | None = None
    if args.selector != "random":
        assert agent is not None and initial_harness is not None and traj_store is not None
        print(
            f"[reasoningbank] short-solve probe over {len(train_pool)} candidate(s) "
            f"on initial harness {initial_harness.id} (selector={args.selector})"
        )
        trajectories = short_solve_all(
            train_pool,
            agent=agent,
            harness=initial_harness,
            traj_store=traj_store,
            workdir=run_dir / "workdir" / "short_solve",
            max_workers=args.codex_concurrency,
        )

    selector = build_selector(
        args.selector,
        workdir=run_dir / "selector_calls",
        judge_model=args.judge_model,
        judge_reasoning=args.selector_reasoning_effort,
        embedding_model=args.embedding_model,
        theta=args.theta,
        trajectories=trajectories,
    )
    selected = selector.select(train_pool, k=args.max_train_tasks, seed=args.seed)
    selection_record: dict[str, Any] = {
        "selector": args.selector,
        "k": args.max_train_tasks,
        "seed": args.seed,
        "all_candidate_ids": [task.id for task in train_pool],
        "selected_task_ids": [task.id for task in selected],
    }
    if isinstance(selector, (_DS, _DPP)):
        selection_record["difficulty_scores"] = {
            task_id: result.difficulty
            for task_id, result in selector.results().items()
        }
    if isinstance(selector, _DPP):
        selection_record["theta"] = selector.theta
    if isinstance(selector, (_CS, _DPP)):
        fingerprints_path = run_dir / "selector_calls" / "fingerprints.json"
        if fingerprints_path.exists():
            selection_record["fingerprints"] = json.loads(
                fingerprints_path.read_text(encoding="utf-8")
            )
    if trajectories is not None:
        selection_record["short_solve_trajectory_ids"] = {
            tid: traj.id for tid, traj in trajectories.items()
        }
        selection_record["judge_input_token_estimate"] = {
            task_id: result.digest_token_estimate
            for task_id, result in selector.results().items()
        }
    return selected, selection_record


def _build_reasoningbank_components(
    args: argparse.Namespace,
    *,
    memory_model: str,
    memory_path: Path,
    embedding_cache_path: Path,
):
    from rho.reasoningbank.llm import ReasoningBankLLM
    from rho.reasoningbank.retrieval import (
        CachedEmbeddingStore,
        GeminiReasoningBankEmbedder,
        ReasoningBankRetriever,
    )
    from rho.reasoningbank.store import ReasoningMemoryStore
    from rho.selection import DEFAULT_CACHE_ROOT
    from rho.selection import llm_client as llm_mod

    if args.embedding_provider == "official-gemini":
        rb_embedder = GeminiReasoningBankEmbedder()
    else:
        # `local:` model strings route to the on-machine ONNX encoder,
        # litellm-style prefixes to a remote API — same dispatch as the
        # task selector.
        rb_embedder = build_embedder(args.embedding_model, DEFAULT_CACHE_ROOT)

    memory_store = ReasoningMemoryStore(memory_path)
    retriever = ReasoningBankRetriever(
        embedder=rb_embedder,
        cache=CachedEmbeddingStore(embedding_cache_path),
        trace_dir=embedding_cache_path.parent / "retrieval",
    )
    memory_llm = ReasoningBankLLM(
        client=llm_mod.LiteLLMClient(),
        model=memory_model,
        judge_reasoning_effort=args.memory_reasoning_effort,
        extraction_reasoning_effort=args.memory_reasoning_effort,
        snapshot_dir=memory_path.parent / "memory_llm",
    )
    return memory_store, retriever, memory_llm


def _reasoningbank_default_memory_model(solver_model: str) -> str:
    if "/" in solver_model:
        return solver_model
    return f"openai/{solver_model}"


def _reasoningbank_embedding_model_for_config(args: argparse.Namespace) -> str:
    if args.embedding_provider == "official-gemini":
        return "gemini-embedding-001"
    return args.embedding_model


def _observed_embedding_dimensions(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return len(embedding)
    return None


def _serialize_reasoningbank_records(records) -> list[dict[str, Any]]:
    return [
        {
            "task_id": record.task.id,
            "harness_id": record.trajectory.harness_id,
            "trajectory_id": record.trajectory.id,
            "stage": record.stage,
            "selected_memory_task_ids": list(record.selected_memory_task_ids),
            "memory_status": record.memory_status.value
            if record.memory_status is not None
            else None,
            "memory_item_count": record.memory_item_count,
            "prediction": record.grade.details.get(
                "prediction",
                record.trajectory.final_message,
            ),
            "score": record.grade.score,
            "details": record.grade.details,
        }
        for record in records
    ]


def _format_reasoningbank_summary(summary: dict[str, Any]) -> str:
    lines = [
        "reasoningbank:",
        f"  eval_variant: {summary['eval_variant']}",
        "train:",
        "  mean_score={:.2f} (n={})".format(
            summary["train"]["mean_score"],
            summary["train"]["n"],
        ),
        "eval:",
        "  mean_score={:.2f} (n={})".format(
            summary["eval"]["mean_score"],
            summary["eval"]["n"],
        ),
        f"memory entries: {summary['memory_entry_count']}",
    ]
    return "\n".join(lines) + "\n"


def _build_agent(args: argparse.Namespace, *, run_dir: Path) -> Agent:
    configure_global_codex_pool(args.codex_concurrency)
    return build_default_agent(
        CodexAgent(
            codex_config_path=_resolved_codex_config_path(args),
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            binary=getattr(args, "codex_binary", "codex"),
            sandbox=getattr(args, "codex_sandbox", None),
            isolate_codex_home=True,
            ephemeral=True,
        ),
        mode=args.cache,
        cache_dir=_agent_cache_dir(args, run_dir),
    )


def _resolved_codex_config_path(args: argparse.Namespace) -> Path:
    raw = getattr(args, "codex_config", None)
    if raw is not None:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            print(f"--codex-config path not found: {path}", file=sys.stderr)
            raise SystemExit(2)
        return path
    path = DEFAULT_CODEX_CONFIG_PATH.expanduser().resolve()
    if not path.is_file():
        print(
            f"no codex config found at {path}; pass --codex-config PATH "
            f"or create {path}. See configs/ for examples.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return path


def _audit_codex_config(run_dir: Path, path: Path) -> dict[str, str]:
    """Copy the resolved codex config into run_dir and return audit fields."""
    config_bytes = path.read_bytes()
    sha = hashlib.sha256(config_bytes).hexdigest()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "codex_config.toml").write_bytes(config_bytes)
    return {"codex_config_path": str(path), "codex_config_sha256": sha}


def _agent_cache_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.cache == "off":
        return None
    if args.cache_dir is not None:
        return Path(args.cache_dir).expanduser().resolve()
    return run_dir / "agent-cache"


def _codex_isolation_config() -> dict[str, Any]:
    return {
        "codex_home_mode": "isolated",
        "inherits_user_config": False,
        "auth_source": str(default_codex_auth_home()),
        "subprocess_env": "minimal",
        "ephemeral": True,
    }


def _find_task(dataset: Dataset, task_id: str) -> Task:
    for task in itertools.chain(dataset.train, dataset.val, dataset.test):
        if task.id == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")


def _serialize_grades(grades) -> list[dict[str, Any]]:
    return [
        {
            "task_id": record.task.id,
            "harness_id": record.trajectory.harness_id,
            "trajectory_id": record.trajectory.id,
            "stage": record.stage,
            "prediction": record.grade.details.get("prediction", record.trajectory.final_message),
            "score": record.grade.score,
            "details": record.grade.details,
        }
        for record in grades
    ]


def _build_manifest(run_dir: Path, trajectories, _rounds) -> dict[str, Any]:
    report_dir = run_dir / "reports"
    report_files = sorted(
        path.relative_to(run_dir).as_posix() for path in report_dir.iterdir() if path.is_file()
    )
    if "reports/manifest.json" not in report_files:
        report_files.append("reports/manifest.json")
    return {
        "run_dir": str(run_dir),
        "report_files": sorted(report_files),
        "round_dirs": _round_dir_entries(run_dir),
        "trajectory_count": len(trajectories),
        "trajectory_ids": [trajectory.id for trajectory in trajectories],
        "trajectory_counts_by_kind": _counts_by_attr(trajectories, "kind"),
        "trajectory_counts_by_stage": _counts_by_attr(trajectories, "stage"),
    }


def _refresh_run_reports(run_dir: Path, traj_store: FilesystemTrajectoryStore) -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    trajectories = list(traj_store._iter_all())
    _write_json(report_dir / "usage_summary.json", usage_summary(trajectories))
    _write_json(report_dir / "manifest.json", _build_manifest(run_dir, trajectories, []))


def _round_dir_entries(run_dir: Path) -> list[str]:
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return []
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in rounds_dir.glob("round_*")
        if path.is_dir()
    )


def _counts_by_attr(items, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = getattr(item, attr) or "(none)"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _format_summary(summary: dict[str, Any]) -> str:
    initial_harness_id = summary.get("initial_harness_id", "(unknown)")
    final_harness_id = summary.get("final_harness_id", "(unknown)")
    lines = [
        f"initial harness: {initial_harness_id}",
        f"final harness: {final_harness_id}",
        "",
        "rounds:",
    ]
    for round_info in summary.get("rounds", []):
        lines.append(
            "  round {round_ix}: mean_score={mean_score:.2f} accepted={accepted} candidate={candidate} winner_sample={winner_sample} unique_candidates={unique_candidate_count}".format(
                round_ix=round_info.get("round_ix"),
                mean_score=round_info.get("mean_score", 0.0),
                accepted=str(round_info.get("accepted")).lower(),
                candidate=round_info.get("candidate_harness_id"),
                winner_sample=round_info.get("winner_sample_index"),
                unique_candidate_count=round_info.get("unique_candidate_count"),
            )
        )
    final_val = summary.get("final_val") or {}
    final_mean_score = final_val.get("mean_score")
    if final_mean_score is not None:
        lines += [
            "",
            "val:",
            "  final: mean_score={:.2f} (n={})".format(
                final_mean_score,
                final_val.get("n"),
            ),
        ]
    else:
        lines += ["", "val: skipped"]
    return "\n".join(lines) + "\n"


def _capture_environment() -> dict[str, Any]:
    return {
        "codex_version": _command_output(["codex", "--version"]),
        "path": os.environ.get("PATH", ""),
        "uname": platform.uname()._asdict(),
        "python_version": sys.version,
        "codex_home": os.environ.get("CODEX_HOME"),
        "cwd": str(Path.cwd()),
        "git_sha": _git_sha(Path.cwd()),
    }


def _command_output(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (proc.stdout or proc.stderr).strip()


def _git_sha(cwd: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dataset_slug(spec: str) -> str:
    if ":" in spec:
        scheme, _, payload = spec.partition(":")
    else:
        scheme, payload = "directory", spec
    basename = Path(payload).name or "dataset"
    return f"{scheme}-{basename}"


def _digest_dataset(spec: str) -> str:
    if ":" in spec:
        _, _, payload = spec.partition(":")
    else:
        payload = spec
    path = Path(payload).resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        return _sha256_tree(path)
    return ""


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
