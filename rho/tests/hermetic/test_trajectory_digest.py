from __future__ import annotations

from rho.protocols import Trajectory
from rho.selection.trajectory_digest import render_digest


def _traj(events, *, final_message="done", exit_code=0, timed_out=False) -> Trajectory:
    return Trajectory(
        id="traj_test",
        kind="solve",
        task_id="task_x",
        harness_id="h_0",
        instructions="",
        events=events,
        final_message=final_message,
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=exit_code,
        wall_time_s=12.5,
        timed_out=timed_out,
    )


def test_render_drops_raw_stderr() -> None:
    events = [
        {"type": "raw_stderr", "text": "x" * 10_000},
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "looking now"}},
    ]
    text, tokens = render_digest(_traj(events))
    assert "x" * 10 not in text
    assert "looking now" in text
    assert tokens > 0


def test_header_counts_and_no_wall_time() -> None:
    events = [
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "a"}},
        {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "b"}},
        {"type": "item.completed", "item": {"id": "i3", "type": "command_execution",
                                            "command": "/bin/bash -lc \"ls\"",
                                            "aggregated_output": "out", "exit_code": 0}},
        {"type": "item.completed", "item": {"id": "i4", "type": "file_change",
                                            "changes": [{"path": "a/b.py", "kind": "update"}]}},
        {"type": "item.completed", "item": {"id": "i5", "type": "web_search",
                                            "query": "py", "action": "search"}},
    ]
    text, _ = render_digest(_traj(events))
    assert "## Summary: 2 agent msgs, 1 cmds, 1 file edits, 1 searches | exit=0 timed_out=false" in text
    assert "wall=" not in text


def test_final_message_appears_at_end() -> None:
    text, _ = render_digest(_traj(events=[], final_message="My final answer"))
    assert text.rstrip().endswith("My final answer")
    assert "## Final message" in text


def test_empty_final_message_renders_placeholder() -> None:
    text, _ = render_digest(_traj(events=[], final_message=""))
    assert "## Final message" in text
    assert "<empty>" in text


def test_bash_lc_wrapper_stripped() -> None:
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "/bin/bash -lc \"ls -la /tmp\"",
        "aggregated_output": "total 0", "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "[CMD exit=0] ls -la /tmp" in text
    assert "/bin/bash -lc" not in text


def test_bash_lc_wrapper_single_quotes_stripped() -> None:
    # Codex agents sometimes use single-quote wrapping for commands that
    # contain double quotes internally; the strip rule must handle both
    # quote styles. Bug surfaced during §10.4 toy-dataset smoke run.
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "/bin/bash -lc 'find harness -maxdepth 2 -type f -print'",
        "aggregated_output": "harness/README.md", "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "[CMD exit=0] find harness -maxdepth 2 -type f -print" in text
    assert "/bin/bash -lc" not in text


def test_task_prompt_md_read_output_redacted() -> None:
    # Surfaced by §10.5 grounding self-check: short-solve agents typically
    # read task/prompt.md as their first action, leaking the entire task
    # description into the digest verbatim. Replace the output with a
    # structural marker so the judge can't paraphrase the prompt instead
    # of writing an abstract fingerprint.
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "/bin/bash -lc \"sed -n '1,220p' task/prompt.md\"",
        "aggregated_output": (
            "What are the team's project code name, primary oncall "
            "rotation, and canonical deploy script? Reply with exactly "
            "these lines in order..."
        ),
        "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "[CMD exit=0] sed -n '1,220p' task/prompt.md" in text
    assert "team's project code name" not in text
    assert "<task content read," in text
    assert "tokens elided>" in text


def test_task_expected_json_read_output_redacted() -> None:
    # Reading task/expected.json leaks ground-truth into the digest.
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "cat task/expected.json",
        "aggregated_output": '{"answer": "Phoenix is the project name"}',
        "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "Phoenix" not in text
    assert "<task content read," in text


def test_unrelated_task_path_not_redacted() -> None:
    # A read of `task/repo/src/foo.py` is normal codebase exploration,
    # not a task-content read, and must NOT be redacted.
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "cat task/repo/src/foo.py",
        "aggregated_output": "def foo(): return 42",
        "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "def foo(): return 42" in text
    assert "<task content read" not in text


def test_file_change_rendered_as_kind_path() -> None:
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "file_change",
        "changes": [{"path": "src/foo.py", "kind": "update"}, {"path": "bar.py", "kind": "create"}],
    }}]
    text, _ = render_digest(_traj(events))
    assert "[FILE update:src/foo.py]" in text
    assert "[FILE create:bar.py]" in text


def test_token_estimate_is_positive_int() -> None:
    text, tokens = render_digest(_traj(events=[], final_message="hello world"))
    assert isinstance(tokens, int)
    assert tokens >= 1


def test_long_command_output_head_tail_truncated() -> None:
    long_output = "X " * 5000  # ~5000 tokens, well over the 300-tok budget
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "echo big", "aggregated_output": long_output, "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "tokens elided" in text
    # The body line for this command must be much smaller than the raw output.
    assert text.count("X") < 1000


def test_short_command_output_not_truncated() -> None:
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "ls", "aggregated_output": "a\nb\nc", "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "tokens elided" not in text
    assert "a\n  b\n  c" in text


def test_empty_command_output_skipped() -> None:
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "command_execution",
        "command": "true", "aggregated_output": "", "exit_code": 0,
    }}]
    text, _ = render_digest(_traj(events))
    assert "[CMD exit=0] true" in text
    # Empty body line should not produce a stray bullet-indent line.
    assert "\n  \n" not in text


def test_long_agent_message_capped_500_tokens() -> None:
    msg = "word " * 1000   # ~1000 tokens
    events = [{"type": "item.completed", "item": {
        "id": "i1", "type": "agent_message", "text": msg,
    }}]
    text, _ = render_digest(_traj(events))
    # After the cap the body chunk should be near 500 tokens, plus marker.
    # Pull the [AGENT] line out:
    agent_line = [ln for ln in text.splitlines() if ln.startswith("[AGENT]")][0]
    body = agent_line[len("[AGENT] "):]
    from rho.selection.trajectory_digest import _count_tokens
    assert _count_tokens(body) <= 550   # 500 + small marker
    assert "truncated" in body.lower()


def test_global_cap_keeps_header_trailer_clips_body() -> None:
    # 50 agent messages of 200 tokens each = ~10k tokens, plus header + trailer
    # pushes us over budget. Renderer must clip body but keep header + trailer.
    events = [
        {"type": "item.completed", "item": {"id": f"i{i}", "type": "agent_message",
                                            "text": ("word " * 200).strip()}}
        for i in range(50)
    ]
    text, tokens = render_digest(_traj(events, final_message="final summary"))
    assert tokens <= 10_000
    assert "## Summary:" in text          # header survived
    assert "final summary" in text          # trailer survived
    assert "## Final message" in text


def test_unmatched_in_flight_command_rendered_in_trailer() -> None:
    # A command_execution item.started without matching item.completed
    # (trajectory killed mid-command). We expect a dedicated trailer section.
    events = [
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "trying"}},
        {"type": "item.started", "item": {"id": "i2", "type": "command_execution",
                                           "command": "/bin/bash -lc \"sleep 9999\""}},
        # no item.completed for i2
    ]
    text, _ = render_digest(_traj(events, exit_code=124, timed_out=True))
    assert "## Unmatched in-flight commands" in text
    assert "sleep 9999" in text
    assert "/bin/bash -lc" not in text


def test_matched_item_started_is_dropped() -> None:
    # When item.started has a matching item.completed for the same id,
    # the .started form is dropped (renderer uses .completed only).
    events = [
        {"type": "item.started", "item": {"id": "i1", "type": "command_execution",
                                           "command": "/bin/bash -lc \"ls\""}},
        {"type": "item.completed", "item": {"id": "i1", "type": "command_execution",
                                             "command": "/bin/bash -lc \"ls\"",
                                             "aggregated_output": "a", "exit_code": 0}},
    ]
    text, _ = render_digest(_traj(events))
    assert text.count("[CMD exit=0] ls") == 1
