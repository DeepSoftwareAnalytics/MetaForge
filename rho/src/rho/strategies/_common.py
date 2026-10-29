from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rho.agent.base import Agent
from rho.observability import annotate_trajectory
from rho.protocols import Harness, HarnessStore, Trajectory


def parallel_map(fn, items):
    """Run fn over items concurrently, return results in input order."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(fn, items))


def optimize_agent_call(
    agent: Agent,
    harness: Harness,
    harness_store: HarnessStore,
    *,
    workspace_builder: Callable[[Path], None],
    instructions: str,
    workdir: Path,
    stage: str,
    round_ix: int,
    sample_index: int,
) -> tuple[Trajectory, Harness | None]:
    """Run one optimize sample in a fresh workspace and capture a candidate harness."""
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(workdir), prefix="opt_", ignore_cleanup_errors=True
    ) as tmp:
        ws = Path(tmp)
        harness_dir = ws / "harness"
        harness_dir.mkdir()
        harness.materialize(harness_dir)
        (ws / ".sample_index").write_text(str(sample_index), encoding="utf-8")
        workspace_builder(ws)
        result = agent.run(
            ws,
            instructions,
            task_id="*",
            harness_id=harness.id,
            kind="optimize",
        )
        result = annotate_trajectory(
            result,
            agent=agent,
            stage=stage,
            round_ix=round_ix,
            sample_index=sample_index,
        )
        if result.exit_code != 0 or result.timed_out:
            return result, None
        candidate = harness_store.capture(harness_dir)
        if candidate.id == harness.id:
            return result, None
        return result, candidate
