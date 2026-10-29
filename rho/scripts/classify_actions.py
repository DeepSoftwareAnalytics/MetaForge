#!/usr/bin/env python3
"""Classify every `command_execution` event in the six runs into one of 6
behavior buckets, then emit a per-trajectory CSV.

Categories (semantic, dataset-agnostic):
  read     — file/state read (cat/sed -n/head/tail/jq read; ARE app_read)
  search   — pattern grep (rg/grep/ag/awk)
  navigate — structure listing + tool discovery (ls/find/which/--version;
             ARE schema/list/catalog)
  edit     — state-mutating writes (heredoc>/sed -i/tee/docker cp/apply_patch;
             ARE app_write — send_/create_/delete_/update_/...;
             send_message_to_user; git apply/commit)
  execute  — run scripts/binaries/build/wait (python file/bash script/make/gcc;
             python -c / python - <<PY; ARE poll/wait/sleep; nohup/&)
  verify   — explicit checks (pytest/diff/cmp/git diff/git status/sha256sum/
             sqlite integrity_check/py_compile)

The classifier inspects only the `command` string. Compound commands
(`a && b`) are classified by the *last* meaningful operator. Long commands
get checked against an ordered rule list — the first match wins. The order
matters: more specific patterns first.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

SPECS = [
    ("SWE-bench Pro", "vanilla", RUNS / "exp-vanilla-swebench"),
    ("SWE-bench Pro", "rho", RUNS / "exp-rho-swebench"),
    ("Terminal-Bench 2", "vanilla", RUNS / "exp-vanilla-tb2"),
    ("Terminal-Bench 2", "rho", RUNS / "20260520-rho-tb2-traj"),
    ("GAIA2", "vanilla", RUNS / "exp-vanilla-gaia2"),
    ("GAIA2", "rho", RUNS / "exp-rho-gaia2-patched"),
]

CATEGORIES = ["read", "search", "navigate", "edit", "execute", "verify"]

# ------------------------------------------------------------------ rules ----
# Each entry is (label, compiled_regex).  Earlier rules take priority.
# The regex is matched against the command string AFTER stripping the
# `/bin/bash -lc "..."` wrapper.

_VERB_WRITE = (
    r"send_|create_|add_|delete_|remove_|update_|edit_|cancel_|book_"
    r"|checkout|reply_|save_|order_|join_|leave_|like_|unlike_"
    r"|move_|copy_event|set_"
)
_VERB_READ = (
    r"get_|list_|search_|read_|lookup_|find_|view_|describe_|show_|fetch_"
)
_ARE = r"are(?:_helper)?\.py"

RULES: list[tuple[str, re.Pattern[str]]] = [
    # --- ARE / GAIA-2 specific (must come before generic rules) ----------
    ("edit",     re.compile(r"\bsend_message_to_user\b")),
    ("read",     re.compile(r"\bAgentUserInterface\s+(get_all_messages|get_last_unread)")),
    ("execute",  re.compile(rf"{_ARE}\s+(poll|wait)\b|\bwait_for_notification\b|\bSystemApp\s+get_current_time\b")),
    ("navigate", re.compile(rf"{_ARE}\s+(schema|list|state|catalog|status)\b")),
    ("edit",     re.compile(rf"{_ARE}\s+call\s+\w+\s+({_VERB_WRITE})")),
    ("read",     re.compile(rf"{_ARE}\s+call\s+\w+\s+({_VERB_READ})")),

    # --- Tests & explicit verification ----------------------------------
    ("verify",   re.compile(r"\b(pytest|go\s+test|cargo\s+test|npm\s+test|yarn\s+\S*\s*test|jest|bats|tox|mocha|pyflakes|mypy|ruff(\s+check)?)\b")),
    ("verify",   re.compile(r"\b(diff|cmp|sha256sum|md5sum|shasum|sqlite3\s+\S+\s+[\"'].*?integrity_check|integrity_check|expect_eq|assert_eq)\b")),
    ("verify",   re.compile(r"\bgit\s+(diff|status|log|show|blame)\b")),
    ("verify",   re.compile(r"\b(py_compile|compileall|tsc(\s+--noEmit)?|gofmt\s+-l|eslint|prettier\s+--check)\b")),
    ("verify",   re.compile(r"repair-verify\b|verify\.sh\b|\bcheck\.sh\b")),

    # --- Writes / patches / state-mutating ------------------------------
    ("edit",     re.compile(r"\bapply_patch\b")),
    ("edit",     re.compile(r"\bdocker\s+cp\b")),
    ("edit",     re.compile(r"\bsed\s+-i\b")),
    ("edit",     re.compile(r"\b(cat|tee)\b[^|]*>\s*[^|&]*<<")),       # cat > file <<EOF
    ("edit",     re.compile(r">\s*\S+\s*<<\s*['\"]?(EOF|PY|BASH|PERL|SCRIPT|PATCH|JS|TS|HTML|JSON)")),
    ("edit",     re.compile(r"\bgit\s+(apply|commit|add|reset|checkout|restore|stash|rm)\b")),
    ("edit",     re.compile(r"\b(touch|mkdir|cp|mv|rm|chmod|chown|ln|patch)\s")),

    # --- Build / execute / install / run / wait -------------------------
    ("execute",  re.compile(r"\bnohup\b|\bsystemctl\b|\bservice\s+\S+\s+start\b")),
    ("execute",  re.compile(r"\b(make|cargo\s+build|go\s+build|gcc|g\+\+|clang(\+\+)?|cmake|ninja|meson|cargo\s+run)\b")),
    ("execute",  re.compile(r"\b(pip\s+install|apt(-get)?\s+install|dpkg\s+-i|conda\s+install|npm\s+install|yarn\s+(add|install)\b|brew\s+install)\b")),
    ("execute",  re.compile(r"\bpython3?\s+\S*\.py\b")),
    ("execute",  re.compile(r"\bpython3?\s+-\s*<<")),
    ("execute",  re.compile(r"\bpython3?\s+-c\b|\bnode\s+-e\b")),
    ("execute",  re.compile(r"\bnode\s+\S*\.(?:js|mjs|ts)\b")),
    ("execute",  re.compile(r"\b(bash|sh|zsh)\s+\S*\.sh\b")),
    ("execute",  re.compile(r"\./\S+\.sh\b|\./solve\b|\./run\b")),
    ("execute",  re.compile(r"\bsleep\s+\d")),
    ("execute",  re.compile(r"\bcurl\b|\bwget\b")),

    # --- Search ---------------------------------------------------------
    ("search",   re.compile(r"\b(rg|ripgrep|ag|grep)\b")),
    ("search",   re.compile(r"\bawk\b\s+['\"]?/")),

    # --- Navigation / discovery -----------------------------------------
    ("navigate", re.compile(r"\bls\b|\bfind\b|\bfd\b|\btree\b|\bpwd\b")),
    ("navigate", re.compile(r"\bwhich\b|\bcommand\s+-v\b|\b\S+\s+--version\b|\bprintenv\b|\benv\b\s*$|\bdpkg(-query)?\b|\bapt\s+list\b")),
    ("navigate", re.compile(r"\b(ps|pgrep|ss|netstat|lsof|docker\s+(inspect|logs|ps))\b")),

    # --- Read -----------------------------------------------------------
    ("read",     re.compile(r"\b(cat|sed\s+-n|head|tail|less|more|wc|file|xxd|strings|objdump|readelf|nm|hexdump|od)\b")),
    ("read",     re.compile(r"\bjq\b")),                        # jq mostly reads
]


def classify(cmd: str) -> str:
    """Return the category label for a shell command string.

    Strategy:
      - Strip `/bin/bash -lc "..."` wrapper if present.
      - Match against ordered RULES; first match wins.
      - Default to "execute" (the most generic running thing) if nothing
        matches — better than dropping unrecognized commands.
    """
    s = cmd.strip()
    # strip /bin/bash -lc "..." wrapper
    m = re.match(r"^/bin/(?:bash|sh)\s+-lc\s+[\"'](.*)[\"']\s*$", s, re.DOTALL)
    if m:
        s = m.group(1)
    # strip leading `docker exec -i <name> bash -lc "..."` wrapper
    m = re.match(r"^docker\s+exec\s+(?:-\S+\s+)*\S+\s+(?:bash|sh)\s+-lc\s+[\"'](.*)[\"']\s*$", s, re.DOTALL)
    if m:
        s = m.group(1)
    # also strip leading `docker exec ... <prog>` plain
    s = re.sub(r"^docker\s+exec\s+(?:-\S+\s+)*\S+\s+", "", s)

    # for compound commands (`a && b ; c`), match each segment with last-wins
    # so the *outcome-producing* op dominates.
    segments = re.split(r"\s*(?:&&|;|\|\|)\s*", s)
    last_label = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        for label, pat in RULES:
            if pat.search(seg):
                last_label = label
                break
    return last_label or "execute"


def load_grades(run_dir: Path) -> list[dict]:
    grades_path = run_dir / "reports" / "final_val_grades.json"
    if grades_path.exists():
        return json.loads(grades_path.read_text())
    log = (run_dir / "run.log").read_text()
    start = log.index("[", log.index("\n"))
    return json.loads(log[start:])


def per_trajectory_counts(run_dir: Path, traj_id: str) -> dict[str, int]:
    """Return {category: count} for one trajectory."""
    counts = defaultdict(int)
    events_path = run_dir / "trajectories" / traj_id / "events.jsonl"
    if not events_path.exists():
        return counts
    with events_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "item.completed":
                continue
            item = e.get("item", {})
            if item.get("type") != "command_execution":
                continue
            counts[classify(item.get("command", ""))] += 1
    return counts


def main() -> None:
    rows: list[dict] = []
    for dataset, method, run_dir in SPECS:
        grades = load_grades(run_dir)
        total = defaultdict(int)
        n_traj = 0
        for g in grades:
            traj_id = g["trajectory_id"]
            counts = per_trajectory_counts(run_dir, traj_id)
            n_traj += 1
            row = {
                "dataset": dataset,
                "method": method,
                "task_id": g["task_id"],
                "trajectory_id": traj_id,
                "success": 1 if float(g["score"]) >= 1.0 else 0,
            }
            for cat in CATEGORIES:
                row[cat] = counts.get(cat, 0)
                total[cat] += counts.get(cat, 0)
            row["total"] = sum(counts.values())
            rows.append(row)
        tot = sum(total.values())
        breakdown = "  ".join(f"{c}={total[c]}({100*total[c]/max(1,tot):.0f}%)" for c in CATEGORIES)
        print(f"{dataset:18s} {method:8s}  n_traj={n_traj:3d}  total={tot:6d}  {breakdown}")

    out_path = REPO / "docs" / "paper" / "figures" / "action_distribution_per_trajectory.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "method", "task_id", "trajectory_id", "success",
                        *CATEGORIES, "total"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
