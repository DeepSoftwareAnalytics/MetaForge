from __future__ import annotations

from rho.protocols import Trajectory


def fake_trajectory(task_id: str, *, harness_id: str = "h") -> Trajectory:
    return Trajectory(
        id=f"traj_{task_id}", kind="solve", task_id=task_id,
        harness_id=harness_id, instructions="",
        events=[{"type": "item.completed", "item": {
            "id": "i1", "type": "agent_message", "text": f"thinking about {task_id}",
        }}],
        final_message=f"final {task_id}", stdout="", stderr="",
        workspace_diff={}, workspace_deletions=frozenset(),
        exit_code=0, wall_time_s=1.0,
    )


def trajectories_for(task_ids):
    return {tid: fake_trajectory(tid) for tid in task_ids}
