"""Opt-in LOCOMO end-to-end smoke across the three optimize strategies.

Each parametrized case runs a 1-round rho loop on 5 train / 5 val
LOCOMO QA pairs using a different OptimizeStrategy (query-only,
trajectory, diagnosis). The test confirms each strategy produces a
non-regressing evolved harness end-to-end; it is not a statistical
comparison at N=5. Use this for smoke/regression, not research
conclusions — at this sample size, final>=initial on val can hinge on
one or two tasks flipping.

The three cases share a session-scoped agent-response cache so that the
initial solve trajectories and the initial-val grading are computed once
and reused across strategies — only the diagnose/optimize/evaluate and
final-val stages differ per strategy.

Cold-cache runtime: ~10-20 minutes of real codex calls for all three
strategies combined (~180 unique calls total). Gated behind the
``locomo`` pytest marker (excluded by default in pyproject).

Pass conditions per case:
- ``final_mean_score >= initial_mean_score - VAL_REGRESSION_TOLERANCE``
  (soft non-regression on val; small slips are tolerated because at N=5
  a single task flip can swing the mean by ~0.2).
- Strategy-specific round artifacts present (diagnoses.json and
  diagnose_traj_ids.json only for the diagnosis case)

Note: we do NOT require ``final.id != initial.id``. At N=5 it is
legitimate for a strategy to propose only candidates that regress on
train (mean_score <= 0) and thus have nothing accepted — that is a
valid strategy outcome, not a test failure. The snapshot still records
``harness_changed`` in summary.json for post-hoc inspection.

Artifacts worth eyeballing are snapshotted to
``snapshots/locomo_e2e/<timestamp>/<strategy>/`` (under the repo root),
persisted even if the test fails. See ``_save_snapshot`` for layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rho.datasets.locomo import LocomoDataset
from rho.loop import run_evolution
from rho.protocols import Harness, TrajectoryStore
from rho.reporting import grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import build_optimize_strategy

pytestmark = [pytest.mark.locomo, pytest.mark.codex]


LOCOMO_PATH = Path(__file__).parents[2] / "data" / "locomo10.json"
REPO_ROOT = Path(__file__).parents[2]

# Soft non-regression tolerance on the val mean score. At N=5, a single
# task's judge-score flipping can move the mean by ~0.2, so a strict
# ``>= initial`` assertion is dominated by LLM noise. 0.05 rejects
# large regressions (clear overfit / harness damage) while tolerating
# stochastic dips; judge the real ablation with a larger N outside CI.
VAL_REGRESSION_TOLERANCE = 0.05


@pytest.fixture(scope="session")
def locomo_shared_cache_dir(tmp_path_factory) -> Path:
    """Session-scoped agent-response cache shared across strategy params
    AND across pytest-xdist workers.

    We use ``tmp_path_factory.getbasetemp().parent`` (the shared xdist
    session root) instead of the worker-local basetemp. Content-addressed
    cache is safe under concurrent multi-worker writes (atomic os.rename
    with race fallback — see src/rho/agent/cache.py::store).
    """
    base = tmp_path_factory.getbasetemp().parent / "rho_locomo_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def locomo_snapshot_root(tmp_path_factory) -> Path:
    """One root per pytest session, shared across xdist workers.

    Each strategy param writes its own sibling subdir (``query-only/``,
    ``trajectory/``, ``diagnosis/``) so a cross-strategy visual diff is
    one ``ls`` away.

    The session id is taken from ``tmp_path_factory.getbasetemp().parent.name``
    (e.g. ``pytest-123``) — xdist workers share the same parent dir, so
    all workers race-free compute the same snapshot root path.
    """
    session_tag = tmp_path_factory.getbasetemp().parent.name
    root = REPO_ROOT / "snapshots" / "locomo_e2e" / session_tag
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize(
    ("strategy_name", "build_kwargs", "expects_diagnosis_files"),
    [
        ("query-only", {}, False),
        ("trajectory", {"trajectories_per_task": 3}, False),
        ("diagnosis", {}, True),
    ],
)
def test_one_round_evolve_on_locomo_subset(
    codex_agent_factory,
    tmp_path: Path,
    locomo_shared_cache_dir: Path,
    locomo_snapshot_root: Path,
    strategy_name: str,
    build_kwargs: dict,
    expects_diagnosis_files: bool,
) -> None:
    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = LocomoDataset(
        LOCOMO_PATH,
        harness_store=harness_store,
        seed=0,
        max_per_split=5,
    )

    initial = next(iter(dataset.train)).harness
    assert len(dataset.train) == 5
    assert len(dataset.val) == 5
    assert len(dataset.test) == 5

    handle = codex_agent_factory(
        cache_mode="on",
        cache_dir=locomo_shared_cache_dir,
    )

    strategy = build_optimize_strategy(strategy_name, **build_kwargs)

    snapshot_dir = locomo_snapshot_root / strategy_name
    initial_summary: dict | None = None
    final_summary: dict | None = None
    final: Harness | None = None

    try:
        final, rounds = run_evolution(
            train=dataset.train,
            n_rounds=1,
            agent=handle.agent,
            harness_store=harness_store,
            traj_store=traj_store,
            workdir=tmp_path / "workdir",
            rounds_dir=tmp_path / "rounds",
            initial=initial,
            strategy=strategy,
        )

        round_dir = tmp_path / "rounds" / "round_0"
        assert (round_dir / "diagnoses.json").exists() is expects_diagnosis_files, (
            f"{strategy_name}: diagnoses.json presence={expects_diagnosis_files} mismatch"
        )
        assert (round_dir / "diagnose_traj_ids.json").exists() is expects_diagnosis_files, (
            f"{strategy_name}: diagnose_traj_ids.json presence={expects_diagnosis_files} mismatch"
        )
        assert (round_dir / "optimize_instructions.txt").exists(), (
            f"{strategy_name}: missing optimize_instructions.txt"
        )
        assert (round_dir / "optimize_input_tokens.json").exists(), (
            f"{strategy_name}: missing optimize_input_tokens.json"
        )

        initial_grades = grade_on_split(handle.agent, initial, dataset.val, tmp_path / "workdir")
        final_grades = grade_on_split(handle.agent, final, dataset.val, tmp_path / "workdir")
        initial_summary = summarize(initial_grades)
        final_summary = summarize(final_grades)

        delta = final_summary["mean_score"] - initial_summary["mean_score"]
        assert delta >= -VAL_REGRESSION_TOLERANCE, (
            f"{strategy_name}: regression beyond tolerance "
            f"{VAL_REGRESSION_TOLERANCE}: "
            f"initial={initial_summary['mean_score']:.4f} "
            f"final={final_summary['mean_score']:.4f} "
            f"delta={delta:+.4f}"
        )

        assert len(rounds) == 1
    finally:
        _save_snapshot(
            snapshot_dir,
            strategy_name=strategy_name,
            build_kwargs=build_kwargs,
            initial=initial,
            final=final,
            round_dir=tmp_path / "rounds" / "round_0",
            traj_store=traj_store,
            initial_summary=initial_summary,
            final_summary=final_summary,
        )
        print(f"[snapshot] {strategy_name}: {snapshot_dir}")


def _save_snapshot(
    snapshot_dir: Path,
    *,
    strategy_name: str,
    build_kwargs: dict,
    initial: Harness,
    final: Harness | None,
    round_dir: Path,
    traj_store: TrajectoryStore,
    initial_summary: dict | None,
    final_summary: dict | None,
) -> None:
    """Persist human-inspectable artifacts. Tolerant to partial state so
    a mid-test failure still produces whatever the run managed to create.

    Layout::

        <snapshot_dir>/
            summary.json          — top-level numbers (scores, harness ids)
            initial_harness/      — materialized copy
            final_harness/        — materialized copy (if run produced one)
            round_0/              — full copy of round_dir (includes
                                    optimize_instructions.txt,
                                    optimize_input_tokens.json,
                                    diagnoses.json [diagnosis only], etc.)
            optimize_messages/    — final_message.txt per optimize sample
            diagnose_messages/    — final_message.txt per diagnose task
                                    (diagnosis strategy only)
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    _materialize_if_possible(initial, snapshot_dir / "initial_harness")
    _materialize_if_possible(final, snapshot_dir / "final_harness")

    if round_dir.exists():
        dest = snapshot_dir / "round_0"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(round_dir, dest)

    _dump_trajectory_final_messages(
        round_dir / "optimize_traj_ids.json",
        traj_store,
        snapshot_dir / "optimize_messages",
        name_prefix="sample",
    )
    _dump_trajectory_final_messages(
        round_dir / "diagnose_traj_ids.json",
        traj_store,
        snapshot_dir / "diagnose_messages",
        name_prefix="task",
    )

    summary = {
        "strategy": strategy_name,
        "build_kwargs": build_kwargs,
        "initial_harness_id": initial.id,
        "final_harness_id": final.id if final is not None else None,
        "harness_changed": bool(final is not None and final.id != initial.id),
        "initial_val": initial_summary,
        "final_val": final_summary,
        "val_delta_mean_score": (
            final_summary["mean_score"] - initial_summary["mean_score"]
            if initial_summary is not None
            and final_summary is not None
            and initial_summary.get("mean_score") is not None
            and final_summary.get("mean_score") is not None
            else None
        ),
    }
    (snapshot_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _materialize_if_possible(harness: Harness | None, dest: Path) -> None:
    if harness is None:
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    harness.materialize(dest)


def _dump_trajectory_final_messages(
    traj_ids_path: Path,
    traj_store: TrajectoryStore,
    dest: Path,
    *,
    name_prefix: str,
) -> None:
    if not traj_ids_path.exists():
        return
    traj_ids = json.loads(traj_ids_path.read_text(encoding="utf-8"))
    if not isinstance(traj_ids, list) or not traj_ids:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for ix, traj_id in enumerate(traj_ids):
        try:
            trajectory = traj_store.get(traj_id)
        except (KeyError, FileNotFoundError):
            continue
        (dest / f"{name_prefix}_{ix:02d}.txt").write_text(
            trajectory.final_message,
            encoding="utf-8",
        )
