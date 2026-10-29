#!/usr/bin/env python3
"""Export saved probe patches and check application to their original base."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from rho.datasets.swebench_pro.patching import extract_prediction_patch
from rho.datasets.swebench_pro.util import cache_root, safe_name
from rho.stores.trajectory import FilesystemTrajectoryStore


def git(args, cwd=None):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args],
        cwd=cwd, capture_output=True, text=True, check=True, timeout=600,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--round", type=int, help="Audit solve trajectories from an evolution round (zero-based)")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text())
    source = Path(config["dataset_spec"].split(":", 1)[1])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config.get("dataset_sha256", config.get("dataset_digest"))
    rows = {row["instance_id"]: row for line in source.read_text().splitlines()
            if line.strip() for row in [json.loads(line)]}
    if args.round is None:
        index = json.loads((run / "probe_index.json").read_text())
        entries = list(index.items())
    else:
        metadata = [json.loads(p.read_text()) for p in sorted((run / "trajectories").glob("*/meta.json"))]
        entries = [(m["task_id"], m["id"]) for m in metadata
                   if m.get("round_ix") == args.round and m.get("stage") in {"round_solve_before", "round_solve_after"}]
        assert entries, "No saved solve trajectories in the requested round"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = run / "patch_audits" / stamp
    out.mkdir(parents=True)
    (out / "trajectory_snapshot.json").write_text(json.dumps(entries, indent=2) + "\n")
    store = FilesystemTrajectoryStore(run / "trajectories")
    repos = cache_root() / "swebench-pro/repos"

    def audit(task_id, trajectory_id):
        tr = store.get(trajectory_id)
        row = rows[task_id]
        result = {"task_id": task_id, "trajectory_id": trajectory_id,
                  "exit_code": tr.exit_code, "timed_out": tr.timed_out,
                  "base_commit": row["base_commit"], "valid_patch": False,
                  "stage": tr.stage, "harness_id": tr.harness_id,
                  "agent_completed": tr.exit_code == 0 and not tr.timed_out
                      and bool(tr.final_message.strip())
                      and any(e.get("type") == "turn.completed" for e in tr.events)}
        target = out / trajectory_id
        target.mkdir()
        try:
            cached = repos / safe_name(row["repo"])
            assert cached.is_dir(), f"Missing local repo cache: {row['repo']}"
            git(["cat-file", "-e", f"{row['base_commit']}^{{commit}}"], cached)

            def materialize(dest):
                # Offline scratch checkout; never reset or clean shared repos.
                git(["clone", "--quiet", "--shared", "--no-checkout", str(cached), str(dest)])
                git(["checkout", "--quiet", "--detach", row["base_commit"]], dest)

            with tempfile.TemporaryDirectory(prefix="scratch-", dir=target) as scratch:
                artifacts = Path(scratch)
                patch = extract_prediction_patch(tr, materialize_repo=materialize,
                                                 artifacts_dir=artifacts)
                patch_path = target / "prediction.patch"
                patch_path.write_text(patch)
                result["patch_bytes"] = patch_path.stat().st_size
                result["patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
                skipped = artifacts / "skipped_large_files.txt"
                if skipped.exists():
                    (target / skipped.name).write_text(skipped.read_text())
                    result["skipped_large_files"] = skipped.read_text().splitlines()
                repo = artifacts / "repo"
                if not repo.exists():
                    materialize(repo)
                git(["reset", "--hard", "--quiet", row["base_commit"]], repo)
                git(["clean", "-fdx", "--quiet"], repo)
                # --index additionally checks the pristine base index, including modes.
                git(["apply", "--check", "--index", str(patch_path)], repo)
                result["numstat"] = git(["apply", "--numstat", str(patch_path)], repo).splitlines()
                result["valid_patch"] = bool(patch.strip())
        except Exception as exc:
            result["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                result["stderr"] = exc.stderr[-4000:]
        (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit, tid, trid) for tid, trid in entries]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(entries)} valid={result['valid_patch']} timeout={result['timed_out']} {result['trajectory_id']}", flush=True)
    counts = Counter()
    stages = {}
    for result in results:
        counts["audited"] += 1
        counts["valid_patches"] += result["valid_patch"]
        group = "timed_out" if result["timed_out"] else "normal_exit" if result["exit_code"] == 0 else "other_exit"
        counts[group] += 1
        counts[group + "_valid_patches"] += result["valid_patch"]
        counts["agent_completed_valid_patches"] += result["agent_completed"] and result["valid_patch"]
        stage = stages.setdefault(result["stage"], Counter())
        stage["attempts"] += 1
        stage["valid_patches"] += result["valid_patch"]
        stage["agent_completed_valid_patches"] += result["agent_completed"] and result["valid_patch"]
        stage["timed_out"] += result["timed_out"]
    report = {"snapshot_at_utc": stamp, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "definition": "Nonempty patch extracted with RHO and git apply --check --index against original base; no benchmark tests run.",
              "counts": dict(counts), "stage_counts": {k: dict(v) for k, v in stages.items()},
              "unique_task_count": len({r["task_id"] for r in results}),
              "results": sorted(results, key=lambda r: (r["task_id"], r["trajectory_id"]))}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(out / "report.json"), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
