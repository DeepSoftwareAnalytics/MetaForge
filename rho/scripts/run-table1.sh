#!/usr/bin/env bash
#
# Table 1 baseline-comparison experiment runner.
# Plan: docs/experiments/2026-05-17-table1-baseline-comparison.md
#
# Runs RHO + 3 baselines (letta-sleep, dynamic-cheatsheet, reasoning-bank) plus
# a Vanilla empty-harness grade, on 3 datasets (SWE-bench Pro, Terminal-Bench 2,
# GAIA-2). 15 runs total. Runs in chunks — pass one or more chunk names.
#
# Usage:
#   scripts/run-table1.sh <chunk> [<chunk> ...]
#   scripts/run-table1.sh list          # print every chunk name
#
# Chunks — single runs (<method>-<dataset>):
#   rho-swebench   letta-swebench   dc-swebench   rb-swebench   vanilla-swebench
#   rho-tb2        letta-tb2        dc-tb2        rb-tb2        vanilla-tb2
#   rho-gaia2      letta-gaia2      dc-gaia2      rb-gaia2      vanilla-gaia2
#
# Chunks — group aliases (expand to the 5 runs of a dataset, RHO first):
#   swebench   tb2   gaia2   all
#
# Extra chunk:
#   gaia2-smoke   # 5-task Vanilla grade on GAIA-2 to confirm non-zero scores
#
# Ordering: within a dataset, RHO runs first (it warms the DPP judge cache and
# writes selection.json, which reasoning-bank reuses). The strict-sequential
# constraint from the plan is satisfied because chunks run one at a time.
#
# Resumability: a finished run-dir gets a `.done` marker and is skipped on
# re-run. A run-dir that exists without `.done` (crashed / partial) is NOT
# silently skipped — the script stops and asks you to inspect and remove it.
#
# Env overrides:
#   CODEX_CONCURRENCY  concurrent codex exec subprocesses = concurrent requests
#                      to the Azure OpenAI backend (default 12)
#   GRADE_WORKERS      concurrent dataset grade() calls — Docker grading (SWE/TB2)
#                      or ARE sidecars (GAIA-2). Default: tracks CODEX_CONCURRENCY.
#   FORCE=1            wipe and re-run even a completed run-dir
#   DRYRUN=1           print the commands that would run, execute nothing
#   HF_TOKEN           GAIA-2 HuggingFace auth. Optional if HUGGINGFACEHUB_API_TOKEN
#                      is set or ~/.cache/huggingface/token exists.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CODEX_CONCURRENCY="${CODEX_CONCURRENCY:-12}"
GRADE_WORKERS="${GRADE_WORKERS:-$CODEX_CONCURRENCY}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

die() { echo "error: $*" >&2; exit 1; }

# ---- per-dataset config -------------------------------------------------
ds_spec() { case "$1" in
  swebench) echo "swebench-pro:ScaleAI/SWE-bench_Pro" ;;
  tb2)      echo "terminal-bench-2:$HOME/.cache/rho/tb2-repo" ;;
  gaia2)    echo "gaia2:meta-agents-research-environments/gaia2#config=mini" ;;
  *) die "unknown dataset key: $1" ;;
esac; }

# uv extra needed to import the dataset adapter (TB2 needs none).
ds_extra() { case "$1" in
  swebench) echo "--extra swebench-pro" ;;
  tb2)      echo "" ;;
  gaia2)    echo "--extra gaia2" ;;
esac; }

# Held-out grading cap. TB2 has only 59 val tasks → grade them all (empty = no cap).
ds_maxgrade() { case "$1" in
  swebench) echo "100" ;;
  tb2)      echo "" ;;
  gaia2)    echo "100" ;;
esac; }

# ---- core invoke --------------------------------------------------------
# invoke <dataset-key> <run-dir> <rho args...>
invoke() {
  local dskey="$1" rundir="$2"; shift 2

  local extra; extra="$(ds_extra "$dskey")"
  local env_prefix=()
  [[ "$dskey" == "gaia2" ]] && env_prefix=(env RHO_GAIA2_ENABLE_JUDGE=1)

  # DRYRUN: pure print, no filesystem mutation, no checks.
  if [[ "$DRYRUN" == "1" ]]; then
    echo ">> DRYRUN $rundir"
    echo "   ${env_prefix[*]} uv run $extra rho $*"
    return 0
  fi

  if [[ "$dskey" == "gaia2" ]]; then
    if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACEHUB_API_TOKEN:-}" \
          && ! -f "$HOME/.cache/huggingface/token" ]]; then
      die "GAIA-2 needs HuggingFace auth: set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN, or 'huggingface-cli login'"
    fi
  fi

  if [[ -f "$rundir/.done" && "$FORCE" != "1" ]]; then
    echo ">> SKIP  $rundir (already completed; set FORCE=1 to redo)"
    return 0
  fi
  if [[ -e "$rundir" && "$FORCE" != "1" ]]; then
    echo "!! $rundir exists but is not marked complete (crashed/partial run)." >&2
    echo "!! Inspect it, then 'rm -rf $rundir' to retry, or set FORCE=1." >&2
    return 1
  fi
  [[ "$FORCE" == "1" ]] && rm -rf "$rundir"
  mkdir -p "$rundir"

  echo ">> RUN   $rundir"
  # shellcheck disable=SC2086
  "${env_prefix[@]}" uv run $extra rho "$@" 2>&1 | tee "$rundir/run.log"
  touch "$rundir/.done"
  echo ">> DONE  $rundir"
}

# ---- method runners -----------------------------------------------------
# _evolve <dataset-key> <strategy> <optimize-samples> <method-tag>
_evolve() {
  local dskey="$1" strat="$2" m="$3" tag="$4"
  local rundir="runs/exp-$tag-$dskey"
  local mg; mg="$(ds_maxgrade "$dskey")"
  local args=(evolve
    --dataset "$(ds_spec "$dskey")"
    --rounds 1 --max-evolve-tasks 10
    --optimize-strategy "$strat" --optimize-samples "$m"
    --selector dpp --theta 0.7
    --model gpt-5.5 --reasoning-effort high
    --cache off --seed 0 --docker-pull missing
    --grade-workers "$GRADE_WORKERS" --codex-concurrency "$CODEX_CONCURRENCY"
    --run-dir "$rundir")
  [[ -n "$mg" ]] && args+=(--max-grading-tasks "$mg")
  invoke "$dskey" "$rundir" "${args[@]}"
}

run_rho()   { _evolve "$1" diagnosis          3 rho;   }
run_letta() { _evolve "$1" letta-sleep        3 letta; }
run_dc()    { _evolve "$1" dynamic-cheatsheet 1 dc;    }

run_rb() {
  local dskey="$1"
  local rundir="runs/exp-rb-$dskey"
  local seljson="runs/exp-rho-$dskey/selection.json"
  if [[ "$DRYRUN" != "1" && ! -f "$seljson" ]]; then
    die "$seljson not found — run 'rho-$dskey' first"
  fi
  local mg; mg="$(ds_maxgrade "$dskey")"
  local args=(reasoningbank
    --dataset "$(ds_spec "$dskey")"
    --selection-json "$seljson"
    --max-train-tasks 10
    --eval-variant frozen --memory-n 1
    --model gpt-5.5 --reasoning-effort high
    --memory-model openai/gpt-5.5 --memory-reasoning-effort medium
    --cache off --seed 0 --docker-pull missing
    --grade-workers "$GRADE_WORKERS" --codex-concurrency "$CODEX_CONCURRENCY"
    --run-dir "$rundir")
  [[ -n "$mg" ]] && args+=(--max-grading-tasks "$mg")
  invoke "$dskey" "$rundir" "${args[@]}"
}

run_vanilla() {
  local dskey="$1"
  local rundir="runs/exp-vanilla-$dskey"
  local mg; mg="$(ds_maxgrade "$dskey")"
  local args=(grade
    --dataset "$(ds_spec "$dskey")"
    --split val --harness h_empty
    --model gpt-5.5 --reasoning-effort high --docker-pull missing
    --grade-workers "$GRADE_WORKERS" --codex-concurrency "$CODEX_CONCURRENCY"
    --run-dir "$rundir")
  [[ -n "$mg" ]] && args+=(--max-grading-tasks "$mg")
  invoke "$dskey" "$rundir" "${args[@]}"
}

run_gaia2_smoke() {
  local rundir="runs/exp-gaia2-smoke"
  local args=(grade
    --dataset "$(ds_spec gaia2)"
    --split val --harness h_empty --max-grading-tasks 5
    --model gpt-5.5 --reasoning-effort high
    --grade-workers "$GRADE_WORKERS" --codex-concurrency "$CODEX_CONCURRENCY"
    --run-dir "$rundir")
  invoke gaia2 "$rundir" "${args[@]}"
}

# ---- dispatch -----------------------------------------------------------
SINGLE_RUNS=(
  rho-swebench letta-swebench dc-swebench rb-swebench vanilla-swebench
  rho-tb2      letta-tb2      dc-tb2      rb-tb2      vanilla-tb2
  rho-gaia2    letta-gaia2    dc-gaia2    rb-gaia2    vanilla-gaia2
)

run_one() {
  local chunk="$1"
  case "$chunk" in
    list)
      printf 'single runs:\n'; printf '  %s\n' "${SINGLE_RUNS[@]}"
      printf 'groups:\n  swebench  tb2  gaia2  all\n'
      printf 'extra:\n  gaia2-smoke\n'
      return ;;
    all)
      run_one swebench; run_one tb2; run_one gaia2; return ;;
    swebench|tb2|gaia2)
      run_one "rho-$chunk";   run_one "letta-$chunk"; run_one "dc-$chunk"
      run_one "rb-$chunk";    run_one "vanilla-$chunk"; return ;;
    gaia2-smoke)
      run_gaia2_smoke; return ;;
  esac

  local method="${chunk%-*}" dskey="${chunk##*-}"
  case "$dskey" in swebench|tb2|gaia2) ;; *) die "unknown chunk: $chunk (try 'list')" ;; esac
  case "$method" in
    rho)     run_rho     "$dskey" ;;
    letta)   run_letta   "$dskey" ;;
    dc)      run_dc      "$dskey" ;;
    rb)      run_rb      "$dskey" ;;
    vanilla) run_vanilla "$dskey" ;;
    *) die "unknown chunk: $chunk (try 'list')" ;;
  esac
}

[[ $# -ge 1 ]] || die "no chunk given. Usage: scripts/run-table1.sh <chunk> ... (try 'list')"
for chunk in "$@"; do
  echo "=== chunk: $chunk ==="
  run_one "$chunk"
done
echo "=== all requested chunks finished ==="
