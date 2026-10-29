#!/usr/bin/env bash
# Tables 2-3 (diagnosis + selector) ablation runner — SWE-Bench Pro only.
# Plan: docs/superpowers/specs/2026-05-20-ablation-runner-design.md
#
# Reuses exp-rho-swebench as the shared "Full diagnosis + DPP" row in both
# tables — that cell is already at runs/exp-rho-swebench/reports/.
#
# 6 chunks to fill the remaining 3+3 cells:
#   diagnosis ablation (selector=dpp θ=0.7, vary strategy):
#     diag-noconsis   diag-noval   diag-queryonly
#   selector ablation (strategy=diagnosis, vary selector):
#     sel-random      sel-difficulty   sel-coverage
#
# Group aliases:  diag (3) | sel (3) | all (6, diag first for cache reuse)
#
# Cache: all 6 share runs/cache/abl-swebench so the 3 diag runs hit the
# (10 train × h_empty) solve trajectories on the 2nd/3rd pass.
# Selector ablation picks different tasks → low hit rate, intentional.
#
# Selection pinning: diag chunks pin to runs/exp-rho-swebench/selection.json
# via --selection-json. Sel chunks run their own selector. See spec §6.
#
# Env / .done / partial / FORCE / DRYRUN semantics identical to run-table1.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CODEX_CONCURRENCY="${CODEX_CONCURRENCY:-10}"
GRADE_WORKERS="${GRADE_WORKERS:-$CODEX_CONCURRENCY}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"
SHARED_CACHE_DIR="runs/cache/abl-swebench"
RHO_RUN="runs/exp-rho-swebench"
RHO_SELECTION="$RHO_RUN/selection.json"

die() { echo "error: $*" >&2; exit 1; }

preflight() {
  # Reused-artifact preflight: exp-rho-swebench must be intact.
  for f in "$RHO_RUN/config.json" "$RHO_SELECTION" \
           "$RHO_RUN/reports/final_val_grades.json"; do
    [[ -f "$f" ]] || die "preflight: missing $f (needed to reuse Full+DPP cell)"
  done
  # Strategy preflight: diagnosis-no-validation must be merged.
  if ! uv run --extra swebench-pro rho evolve --help 2>/dev/null \
       | grep -q diagnosis-no-validation; then
    die "preflight: 'diagnosis-no-validation' not in evolve CLI choices; merge the branch (spec §7.1)"
  fi
  # CLI preflight: --selection-json must exist on evolve.
  if ! uv run --extra swebench-pro rho evolve --help 2>/dev/null \
       | grep -q -- '--selection-json'; then
    die "preflight: evolve lacks --selection-json (spec §7.2)"
  fi
}

invoke() {
  local rundir="$1"; shift
  if [[ "$DRYRUN" == "1" ]]; then
    echo ">> DRYRUN $rundir"
    echo "   uv run --extra swebench-pro rho $*"
    return 0
  fi
  if [[ -f "$rundir/.done" && "$FORCE" != "1" ]]; then
    echo ">> SKIP  $rundir (already completed; set FORCE=1 to redo)"
    return 0
  fi
  if [[ -e "$rundir" && "$FORCE" != "1" ]]; then
    echo "!! $rundir exists but not marked complete (crashed/partial run)." >&2
    echo "!! Inspect, then 'rm -rf $rundir' to retry, or set FORCE=1." >&2
    return 1
  fi
  [[ "$FORCE" == "1" ]] && rm -rf "$rundir"
  mkdir -p "$rundir"
  echo ">> RUN   $rundir"
  uv run --extra swebench-pro rho "$@" 2>&1 | tee "$rundir/run.log"
  touch "$rundir/.done"
  echo ">> DONE  $rundir"
}

# _evolve_abl <tag> <selector> <theta-or-dash> <strategy> [--selection-json PATH]
_evolve_abl() {
  local tag="$1" selector="$2" theta="$3" strategy="$4"; shift 4
  local rundir="runs/exp-abl-$tag-swebench"
  local args=(evolve
    --dataset swebench-pro:ScaleAI/SWE-bench_Pro
    --rounds 1
    --max-evolve-tasks 10 --max-grading-tasks 100
    --optimize-strategy "$strategy" --optimize-samples 3
    --selector "$selector"
    --model gpt-5.5 --reasoning-effort high
    --seed 0 --docker-pull missing
    --grade-workers "$GRADE_WORKERS" --codex-concurrency "$CODEX_CONCURRENCY"
    --cache on --cache-dir "$SHARED_CACHE_DIR"
    --run-dir "$rundir" "$@")
  [[ "$selector" == "dpp" ]] && args+=(--theta "$theta")
  invoke "$rundir" "${args[@]}"
}

# Diagnosis ablations pin to exp-rho-swebench's 10 DPP-selected tasks.
run_diag_noconsis()  { _evolve_abl diag-noconsis  dpp        0.7 diagnosis-no-consistency --selection-json "$RHO_SELECTION"; }
run_diag_noval()     { _evolve_abl diag-noval     dpp        0.7 diagnosis-no-validation  --selection-json "$RHO_SELECTION"; }
run_diag_queryonly() { _evolve_abl diag-queryonly dpp        0.7 query-only               --selection-json "$RHO_SELECTION"; }
# Selector ablations run their own selector — that's the experimental variable.
run_sel_random()     { _evolve_abl sel-random     random     -   diagnosis; }
run_sel_difficulty() { _evolve_abl sel-difficulty difficulty -   diagnosis; }
run_sel_coverage()   { _evolve_abl sel-coverage   coverage   -   diagnosis; }

SINGLE_RUNS=(diag-noconsis diag-noval diag-queryonly
             sel-random    sel-difficulty sel-coverage)

run_one() { case "$1" in
  list) printf 'single:\n'; printf '  %s\n' "${SINGLE_RUNS[@]}"
        printf 'groups:\n  diag  sel  all\n'; return ;;
  all)  run_one diag; run_one sel; return ;;
  diag) run_one diag-noconsis; run_one diag-noval; run_one diag-queryonly; return ;;
  sel)  run_one sel-random; run_one sel-difficulty; run_one sel-coverage; return ;;
  diag-noconsis)  run_diag_noconsis ;;
  diag-noval)     run_diag_noval ;;
  diag-queryonly) run_diag_queryonly ;;
  sel-random)     run_sel_random ;;
  sel-difficulty) run_sel_difficulty ;;
  sel-coverage)   run_sel_coverage ;;
  *) die "unknown chunk: $1 (try 'list')" ;;
esac; }

[[ $# -ge 1 ]] || die "Usage: scripts/run-ablations.sh <chunk> ... (try 'list')"
[[ "$DRYRUN" == "1" || "$1" == "list" ]] || preflight
for chunk in "$@"; do echo "=== chunk: $chunk ==="; run_one "$chunk"; done
echo "=== all requested chunks finished ==="
