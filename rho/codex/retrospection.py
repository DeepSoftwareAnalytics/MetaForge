#!/usr/bin/env python3
"""Retrospection for the OpenAI Codex CLI — RHO (arXiv:2606.05922) on Codex's native harness.

One run is one retrospection cycle: mine your past `codex` sessions for a project,
diagnose recurring failures with label-free signals (self-validation + cross-session
self-consistency), propose N candidate harnesses (AGENTS.md + .agents/skills/ +
helper scripts), score them by pairwise self-preference on replayable probe tasks,
and apply the winner only if its mean score is positive — with a full backup first.

No labels, no validation set: the trajectories are the rollout files Codex already
wrote under ~/.codex/sessions/. Stdlib-only; the only dependency is the `codex` CLI.

Usage:
    python3 retrospection.py [--project DIR] [--model M] [--n 2] [--probes 4] ...
    python3 retrospection.py --dry-run        # just list the sessions it would mine
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Codex rollout format — verified against codex-cli 0.134.0.
# Sessions live at $CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl.
# Line 1 is {"type":"session_meta","payload":{"cwd":...,"git":...}}; user turns are
# {"type":"event_msg","payload":{"type":"user_message","message":...}}; final answers
# are agent_message events / task_complete.last_agent_message. session_index.jsonl is
# stale on real installs — always scan the rollout tree, never trust the index.
# ---------------------------------------------------------------------------
ROLLOUT_HOWTO = (
    "Codex CLI stores one transcript per session at ~/.codex/sessions/YYYY/MM/DD/"
    "rollout-<timestamp>-<uuid>.jsonl. Each line is JSON: line 1 (type=session_meta) has "
    "payload.cwd; user prompts are event_msg lines with payload.type=user_message; final "
    "assistant messages are event_msg lines with payload.type=agent_message (or "
    "task_complete.last_agent_message); tool calls are response_item lines with "
    "payload.type=function_call (arguments JSON has cmd/workdir) and their outputs carry "
    "exit codes. CAUTION: AGENTS.md text is injected as a user-role response_item titled "
    "'# AGENTS.md instructions ...' — that is harness text, not a real user message. "
    "Transcripts can be many MB — extract with jq/grep/python, never read whole files raw."
)

DIGEST_SCHEMA = {
    "type": "object",
    "required": ["sessions"],
    "additionalProperties": False,
    "properties": {
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "taskSummary", "query", "difficulty", "fingerprint", "outcome", "friction", "replayable"],
                "additionalProperties": False,
                "properties": {
                    "file": {"type": "string"},
                    "taskSummary": {"type": "string"},
                    "query": {"type": "string", "description": "self-contained restatement of the task, runnable without this transcript"},
                    "difficulty": {"type": "number", "description": "0-10: dead ends, retries, user corrections, long error chains"},
                    "fingerprint": {"type": "array", "items": {"type": "string"}, "description": "5-10 lowercase keywords abstracting the task type, NOT specific filenames"},
                    "outcome": {"type": "string", "enum": ["success", "partial", "failure", "unclear"]},
                    "friction": {"type": "array", "items": {"type": "string"}},
                    "replayable": {"type": "boolean"},
                },
            },
        },
    },
}

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "required": ["severity", "validationIssues", "failureModes", "inconsistency", "improvementDirection", "probe"],
    "additionalProperties": False,
    "properties": {
        "severity": {"type": "number", "description": "0.0-1.0 soft attention weight for the optimizer"},
        "validationIssues": {"type": "array", "items": {"type": "string"}},
        "failureModes": {"type": "string"},
        "inconsistency": {"type": "string", "description": "empty if single session"},
        "improvementDirection": {"type": "string", "description": "ONE high-level, task-agnostic direction"},
        "probe": {
            "type": "object",
            "required": ["replayable", "query", "originalSession"],
            "additionalProperties": False,
            "properties": {
                "replayable": {"type": "boolean"},
                "query": {"type": "string"},
                "originalSession": {"type": "string"},
            },
        },
    },
}

SCORE_SCHEMA = {
    "type": "object",
    "required": ["value", "rationale"],
    "additionalProperties": False,
    "properties": {
        "value": {"type": "integer", "description": "[-10, 10]; positive iff trajectory A is better"},
        "rationale": {"type": "string"},
    },
}


def log(msg: str) -> None:
    print(f"[retrospection {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# codex exec wrapper. Lessons baked in (from rho's production CodexAgent):
# prompt as final positional arg with stdin redirected from /dev/null; --json so
# stdout is parseable; --output-last-message because stdout interleaves events;
# subprocess-level timeout (codex has no wall-clock flag); exit 0 can still hide
# sandbox failures on Linux (bwrap) — scan output; one retry on failure.
# --ephemeral keeps orchestration calls out of the session store (they must not
# become "trajectories" for the next cycle) and -c features.memories=false stops
# the experimental memories feature from mutating hidden state mid-experiment.
# ---------------------------------------------------------------------------
class Codex:
    def __init__(self, model: str | None, effort: str | None, extra: list[str], run_dir: Path):
        self.model = model
        self.effort = effort
        self.extra = extra
        self.scratch = run_dir / "agents"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def run(self, prompt: str, *, label: str, cd: Path, sandbox: str, schema: dict | None = None,
            timeout_s: int = 900) -> str | dict | None:
        self._counter += 1
        slot = self.scratch / f"{self._counter:03d}-{re.sub(r'[^a-zA-Z0-9_-]', '_', label)}"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "prompt.md").write_text(prompt, encoding="utf-8")
        last_msg = slot / "last_message.txt"
        cmd = [
            "codex", "exec",
            "--cd", str(cd),
            "--json",
            "--sandbox", sandbox,
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-last-message", str(last_msg),
            "-c", "features.memories=false",
        ]
        if self.model:
            cmd += ["-m", self.model]
        if self.effort:
            cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
        if schema is not None:
            schema_path = slot / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            cmd += ["--output-schema", str(schema_path)]
        cmd += self.extra
        cmd += [prompt]

        for attempt in (1, 2):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=False,
                    timeout=timeout_s, stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                log(f"  {label}: timed out after {timeout_s}s (attempt {attempt})")
                continue
            (slot / "stdout.jsonl").write_text(proc.stdout or "", encoding="utf-8")
            (slot / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            sandbox_broke = "bwrap:" in (proc.stdout + proc.stderr) or "Operation not permitted" in (proc.stderr or "")
            if proc.returncode != 0 or (sandbox_broke and not last_msg.exists()):
                log(f"  {label}: codex exec failed (exit {proc.returncode}, attempt {attempt})")
                continue
            if not last_msg.exists():
                log(f"  {label}: no final message (attempt {attempt})")
                continue
            text = last_msg.read_text(encoding="utf-8").strip()
            if schema is None:
                return text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # --output-schema should prevent this; tolerate a fenced block once.
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                log(f"  {label}: unparseable JSON (attempt {attempt})")
        return None


# ---------------------------------------------------------------------------
# Phase 1 — Bootstrap (deterministic): find rollouts whose session_meta.cwd is
# this project (or a subdirectory / worktree of it), snapshot harness h_0.
# ---------------------------------------------------------------------------
def find_sessions(codex_home: Path, project: Path, max_sessions: int) -> list[dict]:
    sessions_root = codex_home / "sessions"
    if not sessions_root.is_dir():
        return []
    project_s = str(project)
    now = time.time()
    found = []
    for f in sorted(sessions_root.rglob("rollout-*.jsonl")):
        try:
            st = f.stat()
            if st.st_size < 5 * 1024 or now - st.st_mtime < 600:
                continue  # trivial, or likely a live session
            with f.open(encoding="utf-8") as fh:
                meta = json.loads(fh.readline())
            if meta.get("type") != "session_meta":
                continue
            cwd = meta.get("payload", {}).get("cwd", "")
            if cwd != project_s and not cwd.startswith(project_s + os.sep):
                continue
            found.append({"file": str(f), "sizeKb": st.st_size // 1024, "mtime": st.st_mtime, "cwd": cwd})
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    found.sort(key=lambda s: -s["sizeKb"])
    return found[:max_sessions]


def snapshot_harness(project: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    agents_md = project / "AGENTS.md"
    (dest / "AGENTS.md").write_text(
        agents_md.read_text(encoding="utf-8") if agents_md.is_file() else "", encoding="utf-8")
    skills = project / ".agents" / "skills"
    if skills.is_dir():
        shutil.copytree(skills, dest / "skills", dirs_exist_ok=True)
    else:
        (dest / "skills").mkdir(exist_ok=True)
    inventory = []
    for d in ("scripts", "bin"):
        p = project / d
        if p.is_dir():
            inventory += [str(x.relative_to(project)) for x in sorted(p.iterdir()) if x.is_file()]
    (dest / "scripts-inventory.md").write_text(
        "Helper scripts present in the project:\n" + "\n".join(f"- {x}" for x in inventory) + "\n",
        encoding="utf-8")


# ---------------------------------------------------------------------------
# Coreset selection — greedy MAP on the paper's DPP kernel L = diag(r) S diag(r)
# with a Jaccard fingerprint kernel; alpha = theta / (2(1-theta)).
# ---------------------------------------------------------------------------
def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {x.lower() for x in a}, {x.lower() for x in b}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def select_coreset(digests: list[dict], k: int, theta: float) -> list[dict]:
    alpha = theta / (2 * max(1 - theta, 1e-6))
    r = [(max(d["difficulty"], 1) / 10) ** alpha for d in digests]
    picked: list[int] = []
    while len(picked) < min(k, len(digests)):
        best, best_gain = -1, float("-inf")
        for i in range(len(digests)):
            if i in picked:
                continue
            max_sim = max((jaccard(digests[i]["fingerprint"], digests[j]["fingerprint"]) for j in picked), default=0.0)
            gain = r[i] * r[i] * (1 - max_sim)
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0:
            break
        picked.append(best)
    return [digests[i] for i in picked]


def group_similar(coreset: list[dict], threshold: float = 0.4, cap: int = 3) -> list[list[dict]]:
    groups, used = [], set()
    for i, d in enumerate(coreset):
        if i in used:
            continue
        g = [d]
        used.add(i)
        for j in range(i + 1, len(coreset)):
            if j not in used and len(g) < cap and jaccard(d["fingerprint"], coreset[j]["fingerprint"]) >= threshold:
                g.append(coreset[j])
                used.add(j)
        groups.append(g)
    return groups


# ---------------------------------------------------------------------------
# Probe isolation: a git worktree of the project (or a plain copy when not git).
# ---------------------------------------------------------------------------
def make_isolated_copy(project: Path, dest: Path) -> bool:
    if (project / ".git").exists():
        r = subprocess.run(["git", "-C", str(project), "worktree", "add", "--detach", str(dest)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
    shutil.copytree(project, dest, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "node_modules", ".venv", "__pycache__"))
    return False


def remove_isolated_copy(project: Path, dest: Path, is_worktree: bool) -> None:
    if is_worktree:
        subprocess.run(["git", "-C", str(project), "worktree", "remove", "--force", str(dest)],
                       capture_output=True, text=True)
    shutil.rmtree(dest, ignore_errors=True)


def install_candidate_into(copy_dir: Path, candidate_dir: Path) -> None:
    """Make the candidate the active harness inside the isolated copy: Codex will
    pick up AGENTS.md and .agents/skills/ from the (copied) project root natively."""
    cand_agents = candidate_dir / "AGENTS.md"
    if cand_agents.is_file():
        (copy_dir / "AGENTS.md").write_text(cand_agents.read_text(encoding="utf-8"), encoding="utf-8")
    cand_skills = candidate_dir / "skills"
    if cand_skills.is_dir():
        dest = copy_dir / ".agents" / "skills"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(cand_skills, dest)
    cand_scripts = candidate_dir / "scripts"
    if cand_scripts.is_dir():
        for f in cand_scripts.rglob("*"):
            if f.is_file():
                rel = _script_destination(f) or Path("scripts") / f.relative_to(cand_scripts)
                target = copy_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                target.chmod(target.stat().st_mode | 0o111)


def _script_destination(script: Path) -> Path | None:
    """Scripts state their intended project-relative path in a 'DESTINATION:' header comment."""
    try:
        head = script.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return None
    m = re.search(r"DESTINATION:\s*(\S+)", head)
    if m:
        rel = Path(m.group(1))
        if not rel.is_absolute() and ".." not in rel.parts:
            return rel
    return None


# ---------------------------------------------------------------------------
# Apply (deterministic): backup, then copy the winning candidate into the project.
# ---------------------------------------------------------------------------
def apply_candidate(project: Path, candidate_dir: Path, backup_dir: Path) -> list[str]:
    changed: list[str] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    agents_md = project / "AGENTS.md"
    if agents_md.is_file():
        shutil.copy2(agents_md, backup_dir / "AGENTS.md")
    skills = project / ".agents" / "skills"
    if skills.is_dir():
        shutil.copytree(skills, backup_dir / "skills", dirs_exist_ok=True)

    cand_agents = candidate_dir / "AGENTS.md"
    if cand_agents.is_file():
        agents_md.write_text(cand_agents.read_text(encoding="utf-8"), encoding="utf-8")
        changed.append("AGENTS.md")
    cand_skills = candidate_dir / "skills"
    if cand_skills.is_dir():
        for f in cand_skills.rglob("*"):
            if f.is_file():
                rel = Path(".agents") / "skills" / f.relative_to(cand_skills)
                target = project / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                changed.append(str(rel))
    cand_scripts = candidate_dir / "scripts"
    if cand_scripts.is_dir():
        for f in cand_scripts.rglob("*"):
            if f.is_file():
                rel = _script_destination(f) or Path("scripts") / f.relative_to(cand_scripts)
                target = project / rel
                if target.is_file():
                    bk = backup_dir / rel
                    bk.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, bk)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                target.chmod(target.stat().st_mode | 0o111)
                changed.append(str(rel))
    return changed


# ---------------------------------------------------------------------------
# The cycle.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="One RHO retrospection cycle over your past Codex CLI sessions.")
    ap.add_argument("--project", default=".", help="project to evolve (default: cwd)")
    ap.add_argument("--model", default=None, help="codex model override (default: your config.toml)")
    ap.add_argument("--reasoning-effort", default=None, choices=["minimal", "low", "medium", "high", "xhigh"])
    ap.add_argument("--k", type=int, default=8, help="coreset size (paper: 10)")
    ap.add_argument("--n", type=int, default=2, help="candidate harnesses (paper: 3)")
    ap.add_argument("--probes", type=int, default=4, help="self-preference probe tasks")
    ap.add_argument("--max-sessions", type=int, default=36)
    ap.add_argument("--batch", type=int, default=3, help="sessions per digest agent")
    ap.add_argument("--theta", type=float, default=0.7, help="DPP difficulty/diversity trade-off in [0,1]")
    ap.add_argument("--concurrency", type=int, default=6, help="max parallel codex exec processes")
    ap.add_argument("--no-apply", action="store_true", help="stage the winner, don't touch live files")
    ap.add_argument("--dry-run", action="store_true", help="list the sessions that would be mined, then exit")
    ap.add_argument("--codex-arg", action="append", default=[], metavar="ARG",
                    help="extra flag passed through to every codex exec (repeatable), e.g. --codex-arg=-c --codex-arg=model_provider=openai")
    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        ap.error(f"--project {project} is not a directory")
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    if shutil.which("codex") is None:
        ap.error("`codex` CLI not found on PATH")

    sessions = find_sessions(codex_home, project, args.max_sessions)
    if args.dry_run:
        print(f"{len(sessions)} session(s) for {project}:")
        for s in sessions:
            print(f"  {s['sizeKb']:>6} KB  {datetime.fromtimestamp(s['mtime']).isoformat(' ', 'minutes')}  {s['file']}")
        return 0
    if not sessions:
        log(f"no usable Codex sessions found for {project} under {codex_home}/sessions — use codex in this project for a while, then rerun.")
        return 1

    run_dir = Path.home() / ".codex" / "retrospection-runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + project.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run dir: {run_dir}; {len(sessions)} sessions for {project}")
    snapshot_harness(project, run_dir / "harness_0")
    cx = Codex(args.model, args.reasoning_effort, args.codex_arg, run_dir)
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    # --- Phase 2: digest -----------------------------------------------------
    batches = [sessions[i:i + args.batch] for i in range(0, len(sessions), args.batch)]

    def digest(bi_batch):
        bi, batch = bi_batch
        prompt = (
            f"You are digesting past Codex CLI sessions of the project {project} for a harness-evolution "
            f"workflow.\n\n{ROLLOUT_HOWTO}\n\nSessions to digest (read each, in full but efficiently — extract "
            "user messages, final assistant messages, tool errors, and signs of struggle):\n"
            + "\n".join(f"- {s['file']}" for s in batch)
            + "\n\nFor each session produce: a task summary; a self-contained query restatement; difficulty 0-10 "
            "(how much the agent struggled: dead ends, retries, user corrections, long error chains); a fingerprint "
            "of 5-10 abstract task-type keywords (lowercase; the KIND of work, not specific file names); outcome; "
            "concrete friction points; and whether the task is replayable in the current project state (a fresh "
            "agent could meaningfully re-attempt it now). Give empty or pure-chit-chat sessions difficulty 0."
        )
        return cx.run(prompt, label=f"digest-b{bi}", cd=project, sandbox="read-only",
                      schema=DIGEST_SCHEMA, timeout_s=900)

    log(f"digesting {len(sessions)} sessions in {len(batches)} batches ...")
    digest_results = list(pool.map(digest, enumerate(batches)))
    digests = [s for r in digest_results if r for s in r.get("sessions", []) if s.get("difficulty", 0) > 0]
    if not digests:
        log("all sessions digested as trivial; nothing to evolve from.")
        return 1
    (run_dir / "digests.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")

    # --- Coreset + grouping (deterministic) ----------------------------------
    coreset = select_coreset(digests, args.k, args.theta)
    groups = group_similar(coreset)
    log(f"coreset: {len(coreset)} sessions in {len(groups)} diagnosis groups (theta={args.theta})")

    # --- Phase 3: diagnose ----------------------------------------------------
    def diagnose(gi_group):
        gi, g = gi_group
        consistency = (
            "2. SELF-CONSISTENCY (across sessions): these sessions are similar task types — identify consequential "
            "divergences in approach, tool sequence, or conclusions, and what knowledge would have prevented them."
            if len(g) >= 2 else
            "2. SELF-CONSISTENCY: only one session in this group — leave inconsistency empty."
        )
        prompt = (
            f"You are the diagnoser in a harness-evolution workflow for the project {project}. The \"harness\" is "
            "everything persistent that shapes future Codex sessions: AGENTS.md, the .agents/skills/ directory, and "
            f"helper scripts. The current harness snapshot is at {run_dir}/harness_0/.\n\n{ROLLOUT_HOWTO}\n\n"
            f"Analyze {len(g)} past session(s) on similar task types:\n"
            + "\n".join(
                f"- {d['file']}\n  summary: {d['taskSummary']}\n  outcome: {d['outcome']}; "
                f"friction: {'; '.join(d.get('friction', [])) or 'none noted'}" for d in g)
            + "\n\nSteps:\n"
            "1. SELF-VALIDATION (per session): read the transcript and judge from internal evidence whether the final "
            "result was actually correct/complete — wrong assumptions, skipped verification, premature stops, errors "
            f"papered over.\n{consistency}\n"
            "3. Derive ONE high-level improvement direction for the harness. It must be TASK-AGNOSTIC: a reusable "
            "rule, a fact worth persisting as a skill, or a helper script worth adding — not a patch for this exact "
            "task instance.\n"
            "4. Set severity 0.0-1.0 (recurring, costly failure modes high; cosmetic friction low).\n"
            "5. Decide a probe: if any of these tasks is replayable in the project's current state, set "
            "probe.replayable=true with a self-contained probe.query and probe.originalSession (the rollout path it "
            "replays). Prefer probes finishable well under an hour; verification-style tasks also count."
        )
        return cx.run(prompt, label=f"diagnose-g{gi}", cd=project, sandbox="read-only",
                      schema=DIAGNOSIS_SCHEMA, timeout_s=1200)

    log(f"diagnosing {len(groups)} groups ...")
    diagnoses = [d for d in pool.map(diagnose, enumerate(groups)) if d]
    if not diagnoses:
        log("all diagnose agents failed.")
        return 1
    diagnoses.sort(key=lambda d: -d.get("severity", 0))
    diag_dir = run_dir / "diagnoses"
    diag_dir.mkdir(exist_ok=True)
    for i, d in enumerate(diagnoses):
        (diag_dir / f"group_{i}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    log(f"{len(diagnoses)} diagnoses; top severity {diagnoses[0].get('severity')}")

    # --- Phase 4: optimize (N candidates, staged outside the project) ---------
    def optimize(j: int):
        cand = run_dir / f"candidate_{j}"
        cand.mkdir(exist_ok=True)
        directions = "\n".join(
            f"- [severity {d.get('severity')}] {d.get('improvementDirection', '')}" for d in diagnoses)
        prompt = (
            f"You are the harness optimizer (sample {j + 1} of {args.n} — independent attempt; take your own angle "
            f"on the diagnoses).\n\nCurrent harness h_0: {run_dir}/harness_0/ (AGENTS.md, skills/, "
            f"scripts-inventory.md). Diagnoses: {diag_dir}/*.json, with severities as soft attention weights:\n"
            f"{directions}\n\nYour working directory IS the staging area. Build an improved FULL candidate harness "
            "here:\n"
            "- ./AGENTS.md — complete evolved file (start from h_0's copy; surgical edits, no bloat: AGENTS.md is "
            "loaded into every future session and capped at 32KB, so each line must earn its context cost)\n"
            "- ./skills/<name>/SKILL.md — evolved skill set (Codex loads .agents/skills/*/SKILL.md; frontmatter "
            "MUST be valid YAML with name + description; one reusable fact/procedure per skill; update or prune "
            "stale ones from h_0, add what the diagnoses show was missing)\n"
            "- ./scripts/ — new or improved helper scripts, if any diagnosis calls for a repeatable procedure; each "
            "script MUST state its intended project-relative path in a header comment line 'DESTINATION: <path>'\n"
            "- ./CHANGES.md — what changed and why, mapped to diagnoses\n\n"
            "Rules: keep everything TASK-AGNOSTIC (rules and knowledge that transfer to future tasks, not patches "
            "for past task instances). Prioritize high-severity recurring failure modes and inconsistency root "
            f"causes. You may READ the target project ({project}) for context, but write ONLY in your working "
            "directory."
        )
        msg = cx.run(prompt, label=f"optimize-{j}", cd=cand, sandbox="workspace-write", timeout_s=1800)
        return j if msg is not None and any(cand.iterdir()) else None

    log(f"optimizing: {args.n} candidates ...")
    candidates = [j for j in pool.map(optimize, range(args.n)) if j is not None]
    if not candidates:
        log("all optimize agents failed.")
        return 1

    # --- Phase 5: probe + pairwise self-preference ----------------------------
    probe_tasks = [d for d in diagnoses
                   if d.get("probe", {}).get("replayable") and d["probe"].get("query")][: args.probes]
    scores: dict[int, list[dict]] = {j: [] for j in candidates}

    if probe_tasks:
        def probe_pair(pair):
            j, t = pair
            d = probe_tasks[t]
            wt = run_dir / "wt" / f"c{j}t{t}"
            wt.parent.mkdir(parents=True, exist_ok=True)
            is_wt = make_isolated_copy(project, wt)
            try:
                install_candidate_into(wt, run_dir / f"candidate_{j}")
                account = cx.run(
                    "Re-attempt this past task end-to-end in the current project (you are in an isolated copy — "
                    "work freely). Follow your AGENTS.md and skills. Task:\n\n" + d["probe"]["query"] +
                    "\n\nWhen done, your final message must be a concise account of what you did, key decisions, "
                    "and the final result.",
                    label=f"solve-c{j}t{t}", cd=wt, sandbox="workspace-write", timeout_s=2400)
                if account is None:
                    return None
                acc_path = run_dir / "probe" / f"cand_{j}_task_{t}.md"
                acc_path.parent.mkdir(exist_ok=True)
                acc_path.write_text(account, encoding="utf-8")
                verdict = cx.run(
                    "You are a strict pairwise judge in a harness-evolution workflow. Compare two attempts at the "
                    "same task and return an integer score in [-10, 10]: positive iff trajectory A is better, "
                    f"magnitude = how decisive.\n\nTask: {d['probe']['query']}\n\n"
                    f"Trajectory A (fresh attempt): read {acc_path}\n"
                    f"Trajectory B (past session): {d['probe']['originalSession']}\n{ROLLOUT_HOWTO}\n\n"
                    "Judge PROCESS QUALITY, not surface polish: correct approach, fewer dead ends and wrong "
                    "assumptions, verification of the result, efficient use of project knowledge. The project may "
                    "have evolved since trajectory B — do not penalize either side for state drift. Be skeptical of "
                    "A: confident-sounding but unverified work scores negative.",
                    label=f"eval-c{j}t{t}", cd=project, sandbox="read-only",
                    schema=SCORE_SCHEMA, timeout_s=900)
                return (j, verdict) if verdict else None
            finally:
                remove_isolated_copy(project, wt, is_wt)

        pairs = [(j, t) for j in candidates for t in range(len(probe_tasks))]
        log(f"probing: {len(probe_tasks)} tasks x {len(candidates)} candidates ...")
        for res in pool.map(probe_pair, pairs):
            if res:
                scores[res[0]].append(res[1])
    else:
        # Fallback: harness-level preference judging (weaker signal; flagged in report).
        log("no replayable probes — falling back to harness-level preference judging")

        def judge(j: int):
            verdict = cx.run(
                f"Pairwise-judge two harnesses for the project {project} against the diagnosed failure modes "
                f"({diag_dir}/*.json). Harness A: {run_dir}/candidate_{j}/. Harness B (current): "
                f"{run_dir}/harness_0/. Return integer in [-10, 10], positive iff A would prevent the diagnosed "
                "failures better WITHOUT bloating context or overfitting to past task instances. Penalize "
                "task-specific patches and AGENTS.md bloat harshly.",
                label=f"judge-c{j}", cd=project, sandbox="read-only", schema=SCORE_SCHEMA, timeout_s=900)
            return (j, verdict) if verdict else None

        for res in pool.map(judge, candidates):
            if res:
                scores[res[0]].append(res[1])

    # --- Phase 6: select + apply (deterministic) -------------------------------
    means = {j: (sum(s["value"] for s in v) / len(v)) if v else float("-inf") for j, v in scores.items()}
    winner = max(means, key=lambda j: means[j])
    accepted = means[winner] > 0
    log("scores: " + ", ".join(f"c{j}={means[j]:.2f}(n={len(scores[j])})" for j in candidates)
        + f" -> {'accept c' + str(winner) if accepted else 'reject all (keep h_0)'}")

    changed: list[str] = []
    if accepted and not args.no_apply:
        changed = apply_candidate(project, run_dir / f"candidate_{winner}", run_dir / "backup")
        log(f"applied candidate_{winner}: {len(changed)} file(s) changed (backup in {run_dir}/backup)")
    elif accepted:
        log(f"--no-apply: winner candidate_{winner} staged at {run_dir}/candidate_{winner}, live files untouched")

    report = {
        "project": str(project), "runDir": str(run_dir), "coresetSize": len(coreset),
        "probeMode": "trajectory" if probe_tasks else "harness-level-fallback",
        "diagnoses": [{"severity": d.get("severity"), "direction": d.get("improvementDirection")} for d in diagnoses],
        "scores": [{"candidate": j, "mean": means[j], "n": len(scores[j]),
                    "rationales": [s["rationale"] for s in scores[j]]} for j in candidates],
        "winner": winner, "accepted": accepted, "applied": bool(changed), "changedFiles": changed,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Retrospection report — {project.name}", "",
             f"- run dir: `{run_dir}`", f"- coreset: {len(coreset)} sessions, {len(groups)} diagnosis groups",
             f"- preference mode: {report['probeMode']}",
             f"- scores: " + ", ".join(f"candidate_{j} = {means[j]:.2f} (n={len(scores[j])})" for j in candidates),
             f"- decision: {'ACCEPTED candidate_' + str(winner) if accepted else 'REJECTED (kept h_0)'}"
             + (f", {len(changed)} files applied" if changed else ""), "", "## Diagnoses", ""]
    lines += [f"- **[{d.get('severity')}]** {d.get('improvementDirection')}" for d in diagnoses]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {run_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
