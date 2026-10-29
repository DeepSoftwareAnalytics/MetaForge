from __future__ import annotations

from rho.orchestrators._util import HARNESS_DESCRIPTION

# Plain template (no f-string): {candidates_per_iter} and {harness_description}
# are filled by str.replace so literal JSON braces below need no escaping.
_TEMPLATE = """\
# Meta-Harness Proposer

You are the proposer in an end-to-end harness search loop. Run ONE iteration:
read the full search history, then propose {candidates_per_iter} new candidate harnesses.

{harness_description}

You do NOT run evaluations. The outer loop evaluates every candidate you propose
on a held-out search set with ground-truth grading, and logs the results back into
the history for the next iteration.

## Workspace layout

  history/                          - the full search history (READ-ONLY context)
    candidates/<harness_id>/         - source files of every harness evaluated so far
    summary.jsonl                    - one row per evaluated harness: iteration,
                                       harness_id, name, hypothesis, parent,
                                       per_task scores, mean_score, pass_rate
                                       (scores are ground-truth grading on the search set)
    frontier.json                    - the current best harness
    traces/<harness_id>/<task_id>/   - execution traces of past solve attempts
                                       (events.jsonl, final_message.txt, workspace_diff/);
                                       these can be large - skim with head/tail/grep
    reports/iter_<N>.md              - post-mortems written by earlier iterations
  proposed/                          - write ALL of your output here (initially empty)

## Steps

1. Post-mortem: for any iteration in history/summary.jsonl that has no
   history/reports/iter_<N>.md, write that post-mortem to proposed/reports/iter_<N>.md
   (<=30 lines): what changed, what improved or regressed and why, and a takeaway.
   The outer loop moves these into the permanent history. history/reports/ is
   read-only - never edit prior reports.
2. Diagnose - THIS IS THE MOST IMPORTANT STEP. Open and DEEP-READ the raw
   execution traces under history/traces/<harness_id>/<task_id>/ for the
   current best harness and for recent regressions: read events.jsonl and
   final_message.txt, skimming large files with head/tail/grep but never
   skipping them. The raw traces are your PRIMARY source of truth.
   Cross-reference prior candidate code under candidates/. summary.jsonl and
   frontier.json tell you only THAT a harness scored what it did; just the
   raw traces tell you WHY it failed at a specific step. Never propose a
   change from scores or summaries alone.
3. Propose exactly {candidates_per_iter} candidates, each testing one distinct
   mechanism. Mix exploitation (refine the current best) and exploration (a
   genuinely different approach). You may branch from ANY prior harness by
   copying its directory from history/candidates/<harness_id>/.
4. For each candidate i (0-indexed) write the complete harness as a directory
   proposed/cand_<i>/. Copy-then-edit from a prior harness is encouraged.

## Anti-overfitting rules

- The harness must be general-purpose. Never hardcode knowledge of specific
  tasks, task ids, or dataset names anywhere in harness files.
- A change that helps only one task is too specific. Prefer mechanisms a skilled
  practitioner would apply across many unfamiliar tasks.
- Do not propose pure parameter tweaks of an existing harness. Change a
  mechanism: what is stored, what is retrieved, or how the task is framed.

## Output

Write proposed/manifest.json listing every candidate you created:

  {"candidates": [
    {"dir": "cand_0", "name": "<short_snake_case>",
     "hypothesis": "<falsifiable claim>", "parent": "<harness_id or null>"}
  ]}

Send a one-paragraph summary as your final message.
"""


def render_proposer_instructions(candidates_per_iter: int) -> str:
    return _TEMPLATE.replace(
        "{harness_description}", HARNESS_DESCRIPTION
    ).replace("{candidates_per_iter}", str(candidates_per_iter))
