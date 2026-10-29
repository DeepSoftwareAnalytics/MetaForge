import json
from pathlib import Path

from rho.orchestrators._util import dump_trajectory
from rho.protocols import Trajectory


def _trajectory_with_large_output() -> Trajectory:
    return Trajectory(
        id="traj_large",
        kind="solve",
        task_id="task_large",
        harness_id="h_empty",
        instructions="solve",
        events=[
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python task/tools/are.py call Files read_file",
                    "aggregated_output": "A" * 20_000,
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "I found the key fact.",
                },
            },
        ],
        final_message="Final answer is still complete.",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )


def _trajectory_with_many_noisy_events() -> Trajectory:
    events = [{"type": "raw_stderr", "data": "warning " * 1000} for _ in range(300)]
    events.append(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python task/tools/are.py poll",
                "aggregated_output": '{"ok": true}',
                "exit_code": 0,
                "status": "completed",
            },
        }
    )
    return Trajectory(
        id="traj_noisy",
        kind="solve",
        task_id="task_noisy",
        harness_id="h_empty",
        instructions="solve",
        events=events,
        final_message="Final answer.",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )


def _trajectory_with_started_and_completed_command() -> Trajectory:
    events = [
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "cat task/prompt.md",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat task/prompt.md",
                "aggregated_output": "prompt",
                "exit_code": 0,
                "status": "completed",
            },
        },
    ]
    return Trajectory(
        id="traj_started",
        kind="solve",
        task_id="task_started",
        harness_id="h_empty",
        instructions="solve",
        events=events,
        final_message="Final answer.",
        stdout="",
        stderr="",
        workspace_diff={},
        workspace_deletions=frozenset(),
        exit_code=0,
        wall_time_s=1.0,
    )


def test_dump_trajectory_writes_full_event_payloads(tmp_path) -> None:
    dump_trajectory(tmp_path / "trajectory", None, _trajectory_with_large_output())

    events_path = tmp_path / "trajectory" / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(lines[0])

    assert first_event["item"]["aggregated_output"] == "A" * 20_000
    assert "python task/tools/are.py call Files read_file" in lines[0]
    assert (tmp_path / "trajectory" / "final_message.txt").read_text(
        encoding="utf-8"
    ) == "Final answer is still complete."
    assert _dumped_filenames(tmp_path / "trajectory") == [
        "events.jsonl",
        "final_message.txt",
        "workspace_diff",
    ]


def test_dump_trajectory_keeps_raw_stream_events(tmp_path) -> None:
    dump_trajectory(tmp_path / "trajectory", None, _trajectory_with_many_noisy_events())

    events_text = (tmp_path / "trajectory" / "events.jsonl").read_text(encoding="utf-8")

    assert events_text.count("raw_stderr") == 300
    assert "python task/tools/are.py poll" in events_text
    assert _dumped_filenames(tmp_path / "trajectory") == [
        "events.jsonl",
        "final_message.txt",
        "workspace_diff",
    ]


def test_dump_trajectory_keeps_started_command_events(tmp_path) -> None:
    dump_trajectory(
        tmp_path / "trajectory",
        None,
        _trajectory_with_started_and_completed_command(),
    )

    events_text = (tmp_path / "trajectory" / "events.jsonl").read_text(encoding="utf-8")

    assert "item.started" in events_text
    assert '"exit_code": null' in events_text
    assert '"exit_code": 0' in events_text
    assert events_text.count("cat task/prompt.md") == 2
    assert _dumped_filenames(tmp_path / "trajectory") == [
        "events.jsonl",
        "final_message.txt",
        "workspace_diff",
    ]


def _dumped_filenames(path: Path) -> list[str]:
    return sorted(child.name for child in path.iterdir())
