"""Compare original (gpt-5.5) vs new (Llama-3.3-70B) judge scores.

Per method: aggregate counts, flip lists, and categorize flips by the
original failure rationale (H4 = soft-judge wording reject; otherwise
structural failures like missing tool calls or count mismatches).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify_failure(rationale: str) -> str:
    if not rationale:
        return "unknown"
    r = rationale.lower()
    if "tool judge reject" in r:
        return "H4_soft_reject"
    if "counters do not match" in r:
        return "tool_count_mismatch"
    if "agent did not perform" in r:
        return "missing_oracle_call"
    if "unmatched" in r or "matching" in r:
        return "matching_other"
    return "other"


def analyze_one(regrade_file: Path) -> dict:
    d = json.loads(regrade_file.read_text())
    results = d["results"]
    n = len(results)
    summary = d["summary"]

    fail_to_pass = []
    pass_to_fail = []
    same_fail = []
    same_pass = []
    for r in results:
        if not r["old_passed"] and r["new_passed"]:
            fail_to_pass.append(r)
        elif r["old_passed"] and not r["new_passed"]:
            pass_to_fail.append(r)
        elif r["old_passed"] and r["new_passed"]:
            same_pass.append(r)
        else:
            same_fail.append(r)

    # Classify old failure rationale for each F->P task
    by_kind = Counter()
    for r in fail_to_pass:
        # old rationale lives in the original final_val_grades.json — load it lazily
        kind = classify_failure(_load_old_rationale(regrade_file, r["task_id"]))
        by_kind[kind] += 1

    return {
        "method": regrade_file.parent.parent.name,
        "n": n,
        "old_pass": summary["old_pass"],
        "new_pass": summary["new_pass"],
        "delta": summary["delta"],
        "fail_to_pass_n": len(fail_to_pass),
        "pass_to_fail_n": len(pass_to_fail),
        "same_pass": len(same_pass),
        "same_fail": len(same_fail),
        "errors": summary["errors"],
        "fail_to_pass_by_old_kind": dict(by_kind),
        "fail_to_pass_task_ids": [r["task_id"] for r in fail_to_pass],
        "pass_to_fail_task_ids": [r["task_id"] for r in pass_to_fail],
    }


_OLD_GRADE_CACHE: dict[Path, dict[str, str]] = {}


def _load_old_rationale(regrade_file: Path, task_id: str) -> str:
    grade_file = regrade_file.parent / "final_val_grades.json"
    if grade_file not in _OLD_GRADE_CACHE:
        _OLD_GRADE_CACHE[grade_file] = {}
        for e in json.loads(grade_file.read_text()):
            rationale = e.get("details", {}).get("validation", {}).get("rationale", "") or ""
            _OLD_GRADE_CACHE[grade_file][e["task_id"]] = str(rationale)
    return _OLD_GRADE_CACHE[grade_file].get(task_id, "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("regrade_files", nargs="+", type=Path,
                   help="Paths to regrade_*.json files")
    args = p.parse_args(argv)

    rows = [analyze_one(f) for f in args.regrade_files]

    # Header
    headers = ["method", "n", "old_pass", "new_pass", "delta",
               "F->P", "P->F", "errors"]
    print(f"{'method':<24} {'n':>4} {'old':>4} {'new':>4} {'delta':>6} "
          f"{'F->P':>5} {'P->F':>5} {'errs':>5}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<24} {r['n']:>4} {r['old_pass']:>4} "
              f"{r['new_pass']:>4} {r['delta']:>+6} "
              f"{r['fail_to_pass_n']:>5} {r['pass_to_fail_n']:>5} "
              f"{r['errors']:>5}")
    print()
    print("F->P breakdown by old failure kind:")
    for r in rows:
        print(f"  {r['method']:<24} {r['fail_to_pass_by_old_kind']}")
    print()
    print("Aggregate F->P task ids (would no longer fail under Llama judge):")
    for r in rows:
        print(f"  {r['method']:<24} {len(r['fail_to_pass_task_ids'])}")
        for tid in r['fail_to_pass_task_ids'][:5]:
            print(f"    - {tid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
