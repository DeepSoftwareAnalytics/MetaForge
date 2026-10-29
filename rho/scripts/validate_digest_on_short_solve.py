"""§4.5 pre-implementation validation.

Run short_solve_one on N real tasks, render each through the digest
function, and verify the spec §4.5 thresholds:
  - All token counts <= 10,000
  - >= 4/5 trajectories contain agent text
  - >= 4/5 trajectories have non-empty final_message
  - Empty-trajectory rate < 50% (else stop and notify per §13.2)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tb2")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--run-dir", default="/tmp/digest-validation")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--codex-config", default=None)
    parser.add_argument("--codex-concurrency", type=int, default=5)
    parser.add_argument("--cache", default="off")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--docker-pull", default="missing")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    from rho.cli import _build_agent
    from rho.datasets import load_dataset
    from rho.stores.harness import FilesystemHarnessStore
    from rho.stores.trajectory import FilesystemTrajectoryStore
    from rho.selection.short_solve import short_solve_all
    from rho.selection.trajectory_digest import render_digest

    hs = FilesystemHarnessStore(run_dir / "harness")
    ds = load_dataset(args.dataset, harness_store=hs, max_per_split=args.n)
    tasks = list(ds.train)[: args.n]
    harness = tasks[0].harness
    traj_store = FilesystemTrajectoryStore(run_dir / "trajectories")
    agent = _build_agent(args, run_dir=run_dir)

    trajs = short_solve_all(
        tasks,
        agent=agent,
        harness=harness,
        traj_store=traj_store,
        workdir=run_dir / "workdir",
        max_workers=args.codex_concurrency,
    )

    print(f"\n=== §4.5 digest validation on {len(trajs)} short-solve trajectories ===")
    empty = 0
    have_agent_text = 0
    have_final_msg = 0
    max_tokens = 0
    for task_id, traj in trajs.items():
        text, tokens = render_digest(traj)
        max_tokens = max(max_tokens, tokens)
        has_agent = "[AGENT]" in text
        has_final = bool(traj.final_message)
        if not (has_agent or has_final):
            empty += 1
        if has_agent:
            have_agent_text += 1
        if has_final:
            have_final_msg += 1
        print(
            f"  {task_id}: tokens={tokens} agent_text={has_agent} "
            f"final_msg={has_final} exit={traj.exit_code} timed_out={traj.timed_out}"
        )
        assert tokens <= 10_000, f"DIGEST OVER BUDGET for {task_id}: {tokens}"

    empty_rate = empty / max(len(trajs), 1)
    print(
        f"\nmax_tokens={max_tokens}  agent_text_present={have_agent_text}/{len(trajs)}  "
        f"final_msg_present={have_final_msg}/{len(trajs)}  empty_rate={empty_rate:.2%}"
    )

    if empty_rate > 0.5:
        print("STOP: empty-trajectory rate exceeds 50% threshold (§13.2).", file=sys.stderr)
        return 2
    if have_agent_text < int(0.8 * len(trajs)):
        print("WARNING: agent-text presence below 4/5 threshold (§4.5).", file=sys.stderr)
        return 1
    if have_final_msg < int(0.8 * len(trajs)):
        print("WARNING: final-message presence below 4/5 threshold (§4.5).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
