"""Opt-in SWE-bench Pro end-to-end evolve smoke test.

Runs one real Codex evolution round on five SWE-bench Pro training tasks,
with one optimize sample, then grades five validation tasks with the real
SWE-bench Pro Docker evaluator. The round must produce an accepted
candidate with positive loop-evaluator score, and the official Docker
validation score must not regress. This is intentionally excluded from the
default test suite via the ``swebench_pro`` marker.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rho.datasets.swebench_pro import SWEbenchProDataset
from rho.loop import run_evolution
from rho.reporting import grade_on_split, summarize
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy

pytestmark = [pytest.mark.swebench_pro, pytest.mark.codex]

SWEBENCH_PRO_SOURCE = "ScaleAI/SWE-bench_Pro"
SAMPLE_COUNT = 5


@pytest.fixture(scope="module")
def docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found on PATH")
    proc = subprocess.run(
        ["docker", "ps"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker daemon is not usable: {proc.stderr or proc.stdout}")


def test_one_round_evolve_on_swebench_pro_subset_real_codex(
    docker_available,
    codex_agent_factory,
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "datasets",
        reason="SWE-bench Pro tests require `uv run --extra swebench-pro ...`",
    )

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = SWEbenchProDataset(
        SWEBENCH_PRO_SOURCE,
        harness_store=harness_store,
        max_per_split=SAMPLE_COUNT,
        docker_pull="missing",
        seed=0,
    )

    initial = next(iter(dataset.train)).harness
    assert len(dataset.train) == SAMPLE_COUNT
    assert len(dataset.val) == SAMPLE_COUNT
    assert len(dataset.test) == SAMPLE_COUNT
    _prefetch_repos(dataset, tmp_path / "repo_prefetch")
    train_tasks = list(dataset.train)[:SAMPLE_COUNT]

    handle = codex_agent_factory(
        cache_mode="on",
        cache_dir=tmp_path / "agent-cache",
    )

    final, rounds = run_evolution(
        train=train_tasks,
        n_rounds=1,
        agent=handle.agent,
        harness_store=harness_store,
        traj_store=traj_store,
        workdir=tmp_path / "workdir",
        rounds_dir=tmp_path / "rounds",
        initial=initial,
        strategy=DiagnoseStrategy(),
        optimize_samples=1,
    )

    initial_grades = grade_on_split(
        handle.agent,
        initial,
        dataset.val,
        tmp_path / "workdir",
        max_tasks=SAMPLE_COUNT,
        traj_store=traj_store,
        stage="initial_val_grade",
        artifacts_root=tmp_path / "grade_artifacts",
    )
    final_grades = grade_on_split(
        handle.agent,
        final,
        dataset.val,
        tmp_path / "workdir",
        max_tasks=SAMPLE_COUNT,
        traj_store=traj_store,
        stage="final_val_grade",
        artifacts_root=tmp_path / "grade_artifacts",
    )
    initial_summary = summarize(initial_grades)
    final_summary = summarize(final_grades)

    assert len(rounds) == 1
    assert rounds[0].winner_sample_index == 0
    assert rounds[0].accepted is True, (
        "the one-round evolve smoke is only meaningful when the optimizer "
        "produces an accepted candidate"
    )
    assert rounds[0].mean_score > 0, (
        f"expected positive loop-evaluator improvement, got {rounds[0].mean_score:.4f}"
    )
    assert final.id != initial.id, (
        "optimize loop did not change the harness; inspect "
        f"{tmp_path / 'rounds' / 'round_0'}"
    )
    assert final_summary["mean_score"] >= initial_summary["mean_score"], (
        f"expected no regression on SWE-bench Pro val subset: "
        f"initial={initial_summary['mean_score']:.4f} "
        f"final={final_summary['mean_score']:.4f}; "
        f"run_dir={tmp_path}"
    )


def _prefetch_repos(dataset: SWEbenchProDataset, dest: Path) -> None:
    """Materialize task repos before spending real Codex calls."""
    for split_name, split in (
        ("train", dataset.train),
        ("val", dataset.val),
        ("test", dataset.test),
    ):
        for ix, task in enumerate(split):
            task.materialize(dest / split_name / str(ix))
