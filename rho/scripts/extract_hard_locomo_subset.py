#!/usr/bin/env python3
"""Extract hard LoCoMo subsets by grading sampled train/val tasks.

Default shape:

  - sample 100 tasks from train and 100 from val
  - solve each split with bounded concurrency 20
  - write the lowest-scoring 20 tasks per split

The script is intentionally dry-run by default. Pass ``--yes`` to launch
Codex calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rho.agent.cache import build_default_agent
from rho.agent.codex import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_CHOICES,
    CodexAgent,
    default_codex_auth_home,
)
from rho.datasets.loader import load_dataset
from rho.observability import usage_summary
from rho.orchestrators.solve import solve
from rho.protocols import Harness, Task
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore

DEFAULT_DATASET = "locomo:data/locomo10.json"
DEFAULT_SPLITS = ("train", "val")
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_HARD_COUNT = 20
DEFAULT_CONCURRENCY = 20
DEFAULT_SAMPLE_SEED = 0
DEFAULT_CACHE_MODE = "on"
VALID_SPLITS = {"train", "val", "test"}


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    reports_dir = run_dir / "reports"
    workdir = run_dir / "workdir"
    reports_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    harness_store = FilesystemHarnessStore(run_dir / "harness")
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    dataset = load_dataset(args.dataset, harness_store=harness_store)
    harness, harness_source = _resolve_harness(args, dataset, harness_store)

    split_names = _parse_splits(args.splits)
    sample_plans = {
        split: _sample_split(
            list(getattr(dataset, split)),
            split=split,
            sample_size=args.sample_size,
            seed=args.sample_seed,
        )
        for split in split_names
    }
    planned_task_count = sum(len(plan) for plan in sample_plans.values())
    completed_task_count = sum(
        _count_completed_in_plan(reports_dir, split, sample_plans[split])
        for split in split_names
    )
    remaining_task_count = planned_task_count - completed_task_count

    config = _build_config(
        args,
        run_dir=run_dir,
        dataset_digest=_digest_dataset(args.dataset),
        harness_id=harness.id,
        harness_source=harness_source,
        split_names=split_names,
        planned_task_count=planned_task_count,
    )
    _ensure_compatible_or_write_config(run_dir / "config.json", config)
    _write_json(reports_dir / "sample_plan.json", _serialize_sample_plans(sample_plans))

    _print_plan(
        args,
        split_names=split_names,
        planned_task_count=planned_task_count,
        completed_task_count=completed_task_count,
        remaining_task_count=remaining_task_count,
        harness=harness,
        harness_source=harness_source,
        run_dir=run_dir,
    )
    if not args.yes:
        print("\nDry run only. Re-run with --yes to launch Codex calls.")
        return 0
    if remaining_task_count <= 0:
        print("\nAll sampled tasks already have rows. Rebuilding reports.")
    else:
        print(f"\nLaunching {remaining_task_count} remaining solve calls.")

    agent = build_default_agent(
        CodexAgent(
            codex_config_path=Path.home() / ".codex" / "config.toml",
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            isolate_codex_home=True,
            ephemeral=True,
        ),
        mode=args.cache,
        cache_dir=_agent_cache_dir(args, run_dir),
    )

    all_split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in split_names:
        rows = _grade_split(
            split,
            sample_plans[split],
            agent=agent,
            harness=harness,
            traj_store=traj_store,
            workdir=workdir,
            reports_dir=reports_dir,
            concurrency=args.concurrency,
        )
        all_split_rows[split] = rows
        _write_split_reports(reports_dir, split, rows, hard_count=args.hard_count)

    trajectories = list(traj_store._iter_all())
    _write_json(reports_dir / "usage_summary.json", usage_summary(trajectories))
    _write_json(
        reports_dir / "hard_subset_summary.json",
        _build_summary(
            config,
            all_split_rows,
            hard_count=args.hard_count,
            trajectory_count=len(trajectories),
        ),
    )
    print(f"\nDone. Reports written under {reports_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lowest-scoring LoCoMo tasks from sampled splits."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory for harness, trajectories, workdir, and reports.",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names. Default: train,val.",
    )
    parser.add_argument("--sample-size", type=_positive_int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--hard-count", type=_positive_int, default=DEFAULT_HARD_COUNT)
    parser.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=None,
        help="Existing run dir to read initial/final harness IDs from.",
    )
    parser.add_argument(
        "--harness",
        default="dataset",
        help=(
            "Harness to grade against: dataset, initial, final, a harness ID, "
            "or a harness directory path. Default: dataset."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model to use. Default: {DEFAULT_CODEX_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=f"Codex model reasoning effort. Default: {DEFAULT_REASONING_EFFORT}.",
    )
    parser.add_argument(
        "--cache",
        choices=["on", "off", "readonly", "refresh"],
        default=DEFAULT_CACHE_MODE,
        help=f"Agent response cache mode. Default: {DEFAULT_CACHE_MODE}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Agent response cache directory. Default: <run-dir>/agent-cache.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually launch Codex calls. Without this, only prints the plan.",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return n


def _parse_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not splits:
        raise SystemExit("--splits must name at least one split")
    invalid = [split for split in splits if split not in VALID_SPLITS]
    if invalid:
        raise SystemExit(f"Unknown split(s): {', '.join(invalid)}")
    if len(set(splits)) != len(splits):
        raise SystemExit("--splits contains duplicates")
    return splits


def _resolve_harness(
    args: argparse.Namespace,
    dataset,
    harness_store: FilesystemHarnessStore,
) -> tuple[Harness, str]:
    if args.harness == "dataset":
        task = next(iter(dataset.train), None)
        if task is None:
            raise SystemExit("Dataset train split is empty; cannot pick dataset harness")
        return task.harness, "dataset"

    if args.harness in {"initial", "final"}:
        if args.source_run is None:
            raise SystemExit(f"--harness {args.harness} requires --source-run")
        summary_path = args.source_run / "reports" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = f"{args.harness}_harness_id"
        harness_id = summary[key]
        return (
            harness_store.capture(_source_harness_path(args.source_run, harness_id)),
            f"{args.harness}:{harness_id}",
        )

    harness_path = Path(args.harness)
    if harness_path.is_dir():
        return harness_store.capture(harness_path), f"path:{harness_path.resolve()}"
    if args.source_run is not None:
        candidate = args.source_run / "harness" / args.harness
        if candidate.is_dir():
            return harness_store.capture(candidate), f"source-run:{args.harness}"
    return harness_store.get(args.harness), f"run-store:{args.harness}"


def _source_harness_path(source_run: Path, harness_id: str) -> Path:
    path = source_run / "harness" / harness_id
    if not path.is_dir():
        raise SystemExit(
            f"Harness {harness_id!r} not found under {source_run / 'harness'}"
        )
    return path


def _sample_split(
    tasks: list[Task],
    *,
    split: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    target = min(sample_size, len(tasks))
    groups: dict[int | str, list[Task]] = {}
    for task in sorted(tasks, key=lambda item: item.id):
        groups.setdefault(_task_category(task), []).append(task)

    allocations = _proportional_allocations(groups, target)
    rng = random.Random(f"{seed}:{split}")
    selected: list[Task] = []
    for category in sorted(groups, key=lambda value: str(value)):
        group = list(groups[category])
        rng.shuffle(group)
        selected.extend(group[: allocations[category]])
    rng.shuffle(selected)
    return [
        {
            "sample_rank": rank,
            "task": task,
            "task_id": task.id,
            "category": _task_category(task),
        }
        for rank, task in enumerate(selected)
    ]


def _proportional_allocations(
    groups: dict[int | str, list[Task]],
    target: int,
) -> dict[int | str, int]:
    total = sum(len(group) for group in groups.values())
    if target >= total:
        return {category: len(group) for category, group in groups.items()}

    fractional: dict[int | str, float] = {}
    allocations: dict[int | str, int] = {}
    for category, group in groups.items():
        quota = target * len(group) / total
        base = min(len(group), math.floor(quota))
        allocations[category] = base
        fractional[category] = quota - base

    remaining = target - sum(allocations.values())
    while remaining > 0:
        candidates = [
            category
            for category, group in groups.items()
            if allocations[category] < len(group)
        ]
        if not candidates:
            break
        category = max(
            candidates,
            key=lambda item: (fractional[item], len(groups[item]), str(item)),
        )
        allocations[category] += 1
        fractional[category] = 0.0
        remaining -= 1
    return allocations


def _task_category(task: Task) -> int | str:
    category = getattr(task, "_category", None)
    if isinstance(category, int):
        return category
    return "unknown"


def _grade_split(
    split: str,
    plan: list[dict[str, Any]],
    *,
    agent,
    harness: Harness,
    traj_store: FilesystemTrajectoryStore,
    workdir: Path,
    reports_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    jsonl_path = _jsonl_path(reports_dir, split)
    existing = _read_jsonl_by_task(jsonl_path)
    pending = [entry for entry in plan if entry["task_id"] not in existing]
    if not pending:
        print(f"{split}: all {len(plan)} sampled tasks already complete.")
        return _rows_in_plan_order(plan, existing)

    print(
        f"{split}: grading {len(pending)} remaining tasks "
        f"({len(existing)}/{len(plan)} already complete), concurrency={concurrency}"
    )
    with jsonl_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_entry = {
                pool.submit(
                    _solve_and_grade_one,
                    entry,
                    agent,
                    harness,
                    traj_store,
                    workdir,
                    split,
                ): entry
                for entry in pending
            }
            completed = 0
            for future in as_completed(future_to_entry):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing[row["task_id"]] = row
                completed += 1
                print(
                    f"{split}: {completed}/{len(pending)} "
                    f"{row['task_id']} score={row['score']:.3f}"
                )
    return _rows_in_plan_order(plan, existing)


def _solve_and_grade_one(
    entry: dict[str, Any],
    agent,
    harness: Harness,
    traj_store: FilesystemTrajectoryStore,
    workdir: Path,
    split: str,
) -> dict[str, Any]:
    task = entry["task"]
    traj = solve(
        agent,
        task,
        harness,
        workdir=workdir,
        stage=f"hard_subset_{split}",
    )
    traj_store.put(traj)
    grade = task.grade(traj)
    row = {
        "split": split,
        "sample_rank": entry["sample_rank"],
        "task_id": task.id,
        "category": grade.details.get("category", entry["category"]),
        "question": grade.details.get("question", ""),
        "gold": grade.details.get("gold", ""),
        "prediction": grade.details.get("prediction", traj.final_message),
        "score": float(grade.score),
        "passed": bool(grade.passed),
        "details": grade.details,
        "trajectory_id": traj.id,
        "harness_id": traj.harness_id,
        "wall_time_s": traj.wall_time_s,
        "timed_out": traj.timed_out,
        "exit_code": traj.exit_code,
        "stage": traj.stage,
        "model": traj.model,
        "reasoning_effort": traj.reasoning_effort,
        "cache_mode": traj.cache_mode,
    }
    return row


def _write_split_reports(
    reports_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
    *,
    hard_count: int,
) -> None:
    _write_json(reports_dir / f"{split}_sample_grades.json", rows)
    hard_rows = _hard_rows(rows, hard_count=hard_count)
    _write_json(reports_dir / f"{split}_hard{hard_count}.json", hard_rows)
    (reports_dir / f"{split}_hard{hard_count}_task_ids.txt").write_text(
        "\n".join(row["task_id"] for row in hard_rows) + ("\n" if hard_rows else ""),
        encoding="utf-8",
    )
    mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    cutoff = hard_rows[-1]["score"] if hard_rows else None
    print(
        f"{split}: wrote {len(rows)} sampled grades, "
        f"mean_score={mean:.3f}, hard_count={len(hard_rows)}, cutoff={cutoff}"
    )


def _hard_rows(rows: list[dict[str, Any]], *, hard_count: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            int(row["sample_rank"]),
            row["task_id"],
        ),
    )[:hard_count]


def _rows_in_plan_order(
    plan: list[dict[str, Any]],
    rows_by_task: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [
        entry["task_id"] for entry in plan if entry["task_id"] not in rows_by_task
    ]
    if missing:
        raise RuntimeError(
            f"Missing result rows for {len(missing)} task(s): {missing[:5]}"
        )
    return [rows_by_task[entry["task_id"]] for entry in plan]


def _jsonl_path(reports_dir: Path, split: str) -> Path:
    return reports_dir / f"{split}_sample_grades.jsonl"


def _read_jsonl_by_task(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            raise RuntimeError(f"{path}:{line_no} has no string task_id")
        rows[task_id] = row
    return rows


def _count_completed_in_plan(
    reports_dir: Path,
    split: str,
    plan: list[dict[str, Any]],
) -> int:
    existing = _read_jsonl_by_task(_jsonl_path(reports_dir, split))
    return sum(1 for entry in plan if entry["task_id"] in existing)


def _serialize_sample_plans(
    sample_plans: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            {
                "sample_rank": entry["sample_rank"],
                "task_id": entry["task_id"],
                "category": entry["category"],
            }
            for entry in plan
        ]
        for split, plan in sample_plans.items()
    }


def _build_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    dataset_digest: str,
    harness_id: str,
    harness_source: str,
    split_names: Iterable[str],
    planned_task_count: int,
) -> dict[str, Any]:
    cache_dir = _agent_cache_dir(args, run_dir)
    fingerprint_payload = {
        "dataset": args.dataset,
        "dataset_digest": dataset_digest,
        "splits": list(split_names),
        "sample_size": args.sample_size,
        "hard_count": args.hard_count,
        "sample_seed": args.sample_seed,
        "harness_id": harness_id,
        "harness_source": harness_source,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    return {
        "script": Path(__file__).name,
        "fingerprint": _stable_digest(fingerprint_payload),
        "fingerprint_payload": fingerprint_payload,
        "dataset_spec": args.dataset,
        "dataset_digest": dataset_digest,
        "splits": list(split_names),
        "sample_size": args.sample_size,
        "hard_count": args.hard_count,
        "sample_seed": args.sample_seed,
        "concurrency": args.concurrency,
        "planned_task_count": planned_task_count,
        "harness_id": harness_id,
        "harness_source": harness_source,
        "source_run": str(args.source_run.resolve()) if args.source_run else None,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "effective_cache_mode": args.cache,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "codex_isolation": {
            "codex_home_mode": "isolated",
            "inherits_user_config": False,
            "auth_source": str(default_codex_auth_home()),
            "subprocess_env": "minimal",
            "ephemeral": True,
        },
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _ensure_compatible_or_write_config(path: Path, config: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise SystemExit(
                f"{path} already exists with a different extraction fingerprint. "
                "Use a new --run-dir for a different sample or harness."
            )
        merged = dict(existing)
        merged.update(
            {
                "concurrency": config["concurrency"],
                "effective_cache_mode": config["effective_cache_mode"],
                "cache_dir": config["cache_dir"],
            }
        )
        _write_json(path, merged)
        return
    _write_json(path, config)


def _build_summary(
    config: dict[str, Any],
    all_split_rows: dict[str, list[dict[str, Any]]],
    *,
    hard_count: int,
    trajectory_count: int,
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split, rows in all_split_rows.items():
        hard = _hard_rows(rows, hard_count=hard_count)
        scores = [float(row["score"]) for row in rows]
        cutoff = float(hard[-1]["score"]) if hard else None
        split_summaries[split] = {
            "n": len(rows),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "score_histogram": _score_histogram(scores),
            "hard_count": len(hard),
            "cutoff_score": cutoff,
            "cutoff_tie_count": (
                sum(1 for score in scores if score == cutoff) if cutoff is not None else 0
            ),
            "hard_task_ids": [row["task_id"] for row in hard],
        }
    return {
        "config": config,
        "trajectory_count": trajectory_count,
        "splits": split_summaries,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _score_histogram(scores: list[float]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for score in scores:
        key = f"{score:.3f}"
        buckets[key] = buckets.get(key, 0) + 1
    return dict(sorted(buckets.items(), key=lambda item: float(item[0])))


def _print_plan(
    args: argparse.Namespace,
    *,
    split_names: tuple[str, ...],
    planned_task_count: int,
    completed_task_count: int,
    remaining_task_count: int,
    harness: Harness,
    harness_source: str,
    run_dir: Path,
) -> None:
    print("Hard LoCoMo subset extraction")
    print(f"  dataset:      {args.dataset}")
    print(f"  splits:       {', '.join(split_names)}")
    print(f"  sample_size:  {args.sample_size} per split")
    print(f"  hard_count:   {args.hard_count} per split")
    print(f"  concurrency:  {args.concurrency}")
    print(f"  sample_seed:  {args.sample_seed}")
    print(f"  harness:      {harness.id} ({harness_source})")
    print(f"  cache:        {args.cache}")
    print(f"  run_dir:      {run_dir}")
    print(f"  planned:      {planned_task_count} solve calls")
    print(f"  completed:    {completed_task_count} rows")
    print(f"  remaining:    {remaining_task_count} solve calls")


def _agent_cache_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.cache == "off":
        return None
    if args.cache_dir is not None:
        return args.cache_dir.expanduser().resolve()
    return run_dir / "agent-cache"


def _digest_dataset(spec: str) -> str:
    if ":" in spec:
        _, _, payload = spec.partition(":")
    else:
        payload = spec
    path = Path(payload).resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        lines: list[str] = []
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix()
            lines.append(f"{rel}:{hashlib.sha256(child.read_bytes()).hexdigest()}")
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return "(missing)"


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
