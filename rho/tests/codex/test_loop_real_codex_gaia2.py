"""Opt-in GAIA-2 end-to-end evolve smoke test.

Runs one real Codex evolution round on two GAIA-2 mini training scenarios
with a single optimize sample. Exercises the full reflect-and-update path:
solve -> diagnose (reflection produces a candidate harness) -> evaluate.

Excluded from the default suite via the ``gaia2`` marker. Needs the codex
CLI, a Hugging Face token (scenario data + lazy-loaded ARE assets), and an
Azure Foundry token (Codex auth + the GAIA-2 judge).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rho.datasets.gaia2.dataset import Gaia2Dataset
from rho.loop import run_round
from rho.stores.harness import FilesystemHarnessStore
from rho.stores.trajectory import FilesystemTrajectoryStore
from rho.strategies import DiagnoseStrategy

from tests.codex._az_helper import have_azure_foundry_token

pytestmark = [pytest.mark.gaia2, pytest.mark.codex]

GAIA2_PAYLOAD = "meta-agents-research-environments/gaia2#config=mini"


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.is_file():
        return cached.read_text(encoding="utf-8").strip() or None
    return None


def test_loop_real_codex_gaia2(
    codex_agent_factory, tmp_path: Path, monkeypatch
) -> None:
    hf_token = _hf_token()
    if hf_token is None:
        pytest.skip("no Hugging Face token (HF_TOKEN or ~/.cache/huggingface/token)")
    if not have_azure_foundry_token():
        pytest.skip("no Azure Foundry token (az account get-access-token failed)")

    # Forked ARE sidecars inherit these: HF_TOKEN for lazy asset loading,
    # the enable flag so the round's evaluate step runs the real judge.
    monkeypatch.setenv("HF_TOKEN", hf_token)
    monkeypatch.setenv("RHO_GAIA2_ENABLE_JUDGE", "1")

    harness_store = FilesystemHarnessStore(tmp_path / "harness")
    traj_store = FilesystemTrajectoryStore(tmp_path / "trajectories")
    dataset = Gaia2Dataset(GAIA2_PAYLOAD, harness_store=harness_store, max_per_split=2)
    tasks = list(dataset.train)
    assert tasks, "GAIA-2 mini train split is empty"

    handle = codex_agent_factory()
    round_dir = tmp_path / "rounds" / "round_0"
    result = run_round(
        0,
        harness_store.empty(),
        tasks,
        handle.agent,
        harness_store,
        traj_store,
        tmp_path / "workdir",
        round_dir,
        strategy=DiagnoseStrategy(),
        optimize_samples=1,
        solve_workers=2,
    )

    # Reflection ran: the round produced both solve and diagnose trajectories.
    kinds = {trajectory.kind for trajectory in traj_store._iter_all()}
    assert {"solve", "diagnose"} <= kinds
    assert (round_dir / "optimize_instructions.txt").exists()

    # The optimize step produced a candidate harness that materializes.
    materialized = tmp_path / "candidate"
    result.candidate.materialize(materialized)
    assert any(materialized.iterdir())

    # The round's evaluate step ran (or honestly produced no candidate).
    assert (round_dir / "scores.json").exists() or result.accepted is False
