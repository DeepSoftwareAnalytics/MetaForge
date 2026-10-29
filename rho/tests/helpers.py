from __future__ import annotations

import json
from pathlib import Path

from rho.agent.fake import FakeAgent, FakeResponse

# Toy-fixture facts. Chosen so that a "too-smart" solver (e.g. real codex)
# cannot independently verify them from the live system: there is no
# `project code name` file or environment variable, no way to find the
# `oncall rotation` by shell inspection, and the deploy script path is a
# made-up internal convention. The solver MUST read the harness.
FACTS = {
    "project code name": "team project code name is Phoenix",
    "oncall rotation": "primary oncall rotation is backend-alpha",
    "deploy script": "canonical deploy script is tools/deploy-prod-v3.sh",
}
FACT_ORDER = ["project code name", "oncall rotation", "deploy script"]
FACT_PREFIXES = {
    "project code name": "team project code name is ",
    "oncall rotation": "primary oncall rotation is ",
    "deploy script": "canonical deploy script is ",
}


def make_fake_agent(mode: str = "good") -> FakeAgent:
    def solve_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        prompt = (workspace / "task" / "prompt.md").read_text(encoding="utf-8")
        harness_text = _read_harness_text(workspace / "harness")
        requested = _facts_requested(prompt)
        lines = [_answer_line(fact, harness_text) for fact in requested]
        return FakeResponse(final_message="\n".join(lines))

    def evaluate_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        prompt = (workspace / "task" / "prompt.md").read_text(encoding="utf-8")
        requested = _facts_requested(prompt)
        option_a = (workspace / "trajectory_A" / "final_message.txt").read_text(
            encoding="utf-8"
        )
        option_b = (workspace / "trajectory_B" / "final_message.txt").read_text(
            encoding="utf-8"
        )
        delta = 0
        for fact in requested:
            delta += _status(option_b, fact) - _status(option_a, fact)
        value = max(-10, min(10, delta * 3))
        rationale = f"delta={delta} requested={requested}"
        return FakeResponse(final_message=json.dumps({"value": value, "rationale": rationale}))

    def diagnose_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        prompt = (workspace / "task" / "prompt.md").read_text(encoding="utf-8")
        requested = _facts_requested(prompt)
        # Read final messages from all 3 trajectories
        messages = []
        for ix in range(3):
            traj_dir = workspace / f"trajectory_{ix}"
            fm_path = traj_dir / "final_message.txt"
            if fm_path.exists():
                messages.append(fm_path.read_text(encoding="utf-8"))
            else:
                messages.append("")
        # Assess each trajectory: did it answer correctly?
        pass_count = 0
        trajectory_analyses = []
        for msg in messages:
            all_correct = all(FACTS[fact] in msg for fact in requested)
            if all_correct:
                pass_count += 1
        for ix, msg in enumerate(messages):
            missing_facts = [
                fact for fact in requested
                if FACTS[fact] not in msg
            ]
            trajectory_analyses.append(
                {
                    "trajectory": f"trajectory_{ix}",
                    "successful": 0 if missing_facts else 1,
                    "quality_analysis": (
                        "Completed the task accurately and efficiently."
                        if not missing_facts
                        else "Did not fully complete the task accurately."
                    ),
                    "issues": (
                        f"Missing facts: {', '.join(missing_facts)}"
                        if missing_facts
                        else ""
                    ),
                }
            )
        # Check consistency: are all answers the same?
        unique_answers = len(set(messages))
        consistent = unique_answers == 1
        # Failure analysis
        failure_analysis = ""
        if pass_count < 3:
            missing_facts = [
                fact for fact in requested
                if any(f"I don't know {fact}" in msg for msg in messages)
            ]
            if missing_facts:
                failure_analysis = f"Missing facts: {', '.join(missing_facts)}"
        divergence_analysis = ""
        if not consistent:
            divergence_analysis = f"Found {unique_answers} distinct answers"
        suggested = ""
        if pass_count < 3:
            missing_facts = [
                fact for fact in requested
                if any(f"I don't know {fact}" in msg for msg in messages)
            ]
            if missing_facts:
                suggested = f"Add facts to harness: {', '.join(missing_facts)}"
        severity = 1.0 if pass_count < 3 else (0.5 if not consistent else 0.0)
        # Determine task_id from prompt filename
        task_id = ""
        prompt_path = workspace / "task" / "prompt.md"
        if prompt_path.exists():
            # Try to extract from workspace path
            task_id = workspace.name
        diag = {
            "task_id": task_id,
            "severity": severity,
            "trajectory_analyses": trajectory_analyses,
            "failure_mode_analysis": failure_analysis,
            "inconsistency_analysis": divergence_analysis,
            "harness_improvement_direction": suggested,
        }
        return FakeResponse(final_message=json.dumps(diag))

    def optimize_script(workspace: Path, instructions: str, output_schema: dict | None) -> FakeResponse:
        del instructions, output_schema
        harness_dir = workspace / "harness"
        existing_lines = _read_harness_lines(harness_dir)
        if mode == "noop":
            return FakeResponse(final_message="no changes")

        if mode == "harmful":
            lines = list(existing_lines)
            bad = "team project code name is Wrongname"
            if bad not in lines:
                lines.append(bad)
            content = "\n".join(lines).strip() + "\n"
            return FakeResponse(
                final_message="wrote harmful fact",
                workspace_edits={"harness/notes.md": content.encode("utf-8")},
            )

        if mode == "sampled":
            sample_index = _read_sample_index(workspace)
            fact = FACT_ORDER[sample_index % len(FACT_ORDER)]
            if FACTS[fact] in existing_lines:
                return FakeResponse(final_message=f"{fact} already present")
            new_lines = list(existing_lines)
            new_lines.append(FACTS[fact])
            content = "\n".join(new_lines).strip() + "\n"
            return FakeResponse(
                final_message=f"added {fact} from sample {sample_index}",
                workspace_edits={"harness/notes.md": content.encode("utf-8")},
            )

        missing = _missing_facts_from_diagnoses(workspace / "diagnoses")
        if not missing:
            missing = _missing_facts_from_task_prompts(workspace / "tasks")
        known = {fact for fact, answer in FACTS.items() if answer in existing_lines}
        to_add = next((fact for fact in FACT_ORDER if fact in missing and fact not in known), None)
        if to_add is None:
            return FakeResponse(final_message="no changes")
        new_lines = list(existing_lines)
        if FACTS[to_add] not in new_lines:
            new_lines.append(FACTS[to_add])
        content = "\n".join(new_lines).strip() + "\n"
        return FakeResponse(
            final_message=f"added {to_add}",
            workspace_edits={"harness/notes.md": content.encode("utf-8")},
        )

    return FakeAgent(
        {
            "solve": solve_script,
            "evaluate": evaluate_script,
            "diagnose": diagnose_script,
            "optimize": optimize_script,
        }
    )


def _facts_requested(prompt: str) -> list[str]:
    requested: list[str] = []
    for fact in FACT_ORDER:
        if fact in prompt.lower():
            requested.append(fact)
    return requested


def _read_harness_text(harness_dir: Path) -> str:
    texts: list[str] = []
    if not harness_dir.exists():
        return ""
    for path in sorted(harness_dir.rglob("*.md")):
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


def _read_harness_lines(harness_dir: Path) -> list[str]:
    text = _read_harness_text(harness_dir)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _answer_line(fact: str, harness_text: str) -> str:
    correct = FACTS[fact]
    if correct in harness_text:
        return correct
    prefix = FACT_PREFIXES[fact]
    for line in harness_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    return f"I don't know {fact}"


def _status(message: str, fact: str) -> int:
    if FACTS[fact] in message:
        return 2
    if f"I don't know {fact}" in message:
        return 1
    return 0


def _missing_facts_from_trajectories(trajectories_dir: Path) -> set[str]:
    missing: set[str] = set()
    for path in sorted(trajectories_dir.iterdir()):
        final_message_path = path / "final_message.txt"
        if not final_message_path.exists():
            continue
        text = final_message_path.read_text(encoding="utf-8")
        for fact in FACT_ORDER:
            if f"I don't know {fact}" in text:
                missing.add(fact)
    return missing


def _missing_facts_from_diagnoses(diagnoses_dir: Path) -> set[str]:
    """Extract missing facts from diagnosis.md files in the diagnoses/ directory."""
    missing: set[str] = set()
    if not diagnoses_dir.exists():
        return missing
    for path in sorted(diagnoses_dir.iterdir()):
        diag_path = path / "diagnosis.md"
        if not diag_path.exists():
            continue
        text = diag_path.read_text(encoding="utf-8")
        # Only consider diagnoses that indicate problems.
        if "**Successful:** 0" in text or "Missing facts:" in text:
            # Look for "Missing facts: X, Y" in the failure analysis section
            for fact in FACT_ORDER:
                if f"Missing facts:" in text and fact in text:
                    missing.add(fact)
        if "## Harness improvement direction" in text and "Add facts to harness:" in text:
            for fact in FACT_ORDER:
                if fact in text:
                    missing.add(fact)
    return missing


def _missing_facts_from_task_prompts(tasks_dir: Path) -> set[str]:
    missing: set[str] = set()
    if not tasks_dir.exists():
        return missing
    for prompt_path in sorted(tasks_dir.glob("task_*/prompt.md")):
        prompt = prompt_path.read_text(encoding="utf-8")
        for fact in _facts_requested(prompt):
            missing.add(fact)
    return missing


def _read_sample_index(workspace: Path) -> int:
    sample_index_path = workspace / ".sample_index"
    if not sample_index_path.exists():
        return 0
    return int(sample_index_path.read_text(encoding="utf-8").strip())


def make_meta_harness_fake_agent() -> FakeAgent:
    """FakeAgent for Meta-Harness tests.

    Reuses the toy-dataset `solve` script from make_fake_agent("good") and swaps
    in an `optimize` script that acts as the Meta-Harness proposer: it writes one
    candidate harness containing every toy fact, plus a manifest and a post-mortem
    report, all under proposed/.
    """
    all_facts = "\n".join(FACTS.values()) + "\n"

    def proposer_script(
        workspace: Path, instructions: str, output_schema: dict | None
    ) -> FakeResponse:
        del workspace, instructions, output_schema
        manifest = json.dumps(
            {
                "candidates": [
                    {
                        "dir": "cand_0",
                        "name": "all_facts",
                        "hypothesis": "store every fact in the harness",
                        "parent": None,
                    }
                ]
            }
        )
        return FakeResponse(
            final_message="proposed 1 candidate",
            workspace_edits={
                "proposed/cand_0/notes.md": all_facts.encode("utf-8"),
                "proposed/manifest.json": manifest.encode("utf-8"),
                "proposed/reports/iter_0.md": b"seed evaluation complete\n",
            },
        )

    agent = make_fake_agent("good")
    agent.scripts["optimize"] = proposer_script
    return agent
