#!/usr/bin/env python3
"""Read-only screening of solver command logs for potential answer leakage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

RULES = {
    "benchmark_data_or_answers": r"test\.jsonl|sweap_eval|gold(?:en)?[_ -]?patch|reference[_ -]?patch|test_patch|FAIL_TO_PASS|PASS_TO_PASS|before_repo_set_cmd|selected_test_files_to_run|dockerhub_tag|solution\.patch",
    "evaluation_files": r"grade_artifacts|patch_audits|final_val_grades|failed_grade_attempts|SWE-bench_Pro-os|swe_bench_pro_eval|/run_scripts/|helper_code/|evaluation_100|split[s]?/swebench|/grades/|/solves/|/trajectories/|/results/|patch\.diff",
    "host_repository_or_history": r"\.cache/rho|swebench-pro/experiment|git\s+(?:-[^\s]+\s+[^\s]+\s+)*(?:show|log|fetch|clone|remote)|git\s+--git-dir",
    "host_discovery": r"/home/|/mnt/|/proc/|/root/|\$HOME|~[/\s]|\.\./\.\./|(?:find|ls|rg|grep)\s+(?:-[^\s]+\s+)*\.\.(?:\s|/|['\"]|$)",
    "docker_or_network": r"\bdocker\b|\bcurl\b|\bwget\b|requests\.(?:get|post)|urllib\.request|https?://",
}
PATTERNS = {key: re.compile(value, re.I) for key, value in RULES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--include-evolution", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = run / "access_audits" / stamp
    out.mkdir(parents=True, mode=0o700)
    sources = []
    for arm in ("baseline", "final"):
        base = run / arm
        for p in (base / "trajectories").glob("*/events.jsonl"):
            meta = json.loads((p.parent / "meta.json").read_text())
            sources.append((p, arm, meta["task_id"], meta["id"], "saved"))
        for pattern in ("solve_*/.rho/live.stdout.jsonl", "solve_*/.rho/before_resume_*/live.stdout.jsonl"):
            for p in (base / "workdir").glob(pattern):
                sources.append((p, arm, "", p.parts[p.parts.index("workdir") + 1], "live_or_archive"))
    if args.include_evolution:
        for p in (args.include_evolution / "trajectories").glob("*/events.jsonl"):
            meta = json.loads((p.parent / "meta.json").read_text())
            if meta.get("stage") in ("round_solve_before", "round_solve_after", "round_optimize"):
                sources.append((p, meta["stage"], meta["task_id"], meta["id"], "saved"))
    counts, groups, seen = Counter(), {}, set()
    flagged = []
    with (out / "commands.jsonl").open("w") as all_commands:
        for path, arm, task_id, trajectory_id, source_type in sources:
            stat = path.stat()
            counts["source_files"] += 1
            with path.open(errors="replace") as f:
                for number, line in enumerate(f, 1):
                    try: event = json.loads(line)
                    except ValueError:
                        counts["unparsed_lines"] += 1
                        continue
                    item = event.get("item") or {}
                    if event.get("type") != "item.completed" or item.get("type") != "command_execution":
                        continue
                    command = item.get("command", "")
                    output = item.get("aggregated_output", "")
                    counts["command_records_including_duplicates"] += 1
                    identity = hashlib.sha256(json.dumps([arm, command, output, item.get("exit_code")]).encode()).hexdigest()
                    if identity in seen: continue
                    seen.add(identity)
                    command_rules = [key for key, pattern in PATTERNS.items() if pattern.search(command)]
                    output_rules = [key for key in ("benchmark_data_or_answers", "evaluation_files") if PATTERNS[key].search(output)]
                    record = {"arm": arm, "task_id": task_id, "trajectory_id": trajectory_id,
                        "source": str(path), "line": number, "source_type": source_type,
                        "source_size_at_scan": stat.st_size, "command": command,
                        "exit_code": item.get("exit_code"), "output_chars": len(output),
                        "command_rules": command_rules, "output_rules": output_rules}
                    all_commands.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["unique_command_records"] += 1
                    groups.setdefault(arm, Counter())["unique_command_records"] += 1
                    if command_rules or output_rules:
                        record["finding_id"] = len(flagged)
                        flagged.append(record)
                        for rule in command_rules: counts["command:" + rule] += 1
                        for rule in output_rules: counts["output:" + rule] += 1
    (out / "flagged.json").write_text(json.dumps(flagged, ensure_ascii=False, indent=2))
    summary = {"snapshot_utc": stamp, "run_dir": str(run), "counts": dict(counts),
        "groups": {k: dict(v) for k, v in groups.items()}, "flagged_count": len(flagged),
        "note": "Screening only; keyword matches require manual command/output review. No OS file-access audit is available."}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for p in out.iterdir(): p.chmod(0o600)
    print(json.dumps({"audit_dir": str(out), **summary}, ensure_ascii=False))


if __name__ == "__main__": main()
