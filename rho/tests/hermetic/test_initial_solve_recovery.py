import json
from dataclasses import replace
from pathlib import Path

import pytest

from rho.agent.initial_solve_recovery import InitialSolveRecovery
from rho.agent.request_budget import write_json
from rho.protocols import Trajectory
from rho.stores.trajectory import FilesystemTrajectoryStore


def fixture_run(tmp_path):
    run = tmp_path / 'run'
    for name in ('trajectories', 'rounds/round_0', 'runtime/calls'):
        (run / name).mkdir(parents=True)
    write_json(run / 'config.json', {'n_rounds': 1, 'initial_harness': None,
                                    'model': 'test', 'reasoning_effort': 'high'})
    selection = tmp_path / 'selection.json'
    images = tmp_path / 'images.json'
    write_json(selection, {'selected_task_ids': ['done', 'failed', 'unstarted']})
    write_json(images, {t: {'image_id': 'sha256:base'} for t in ['done', 'failed', 'unstarted']})
    store = FilesystemTrajectoryStore(run / 'runtime/completed_trajectories')
    for i in range(3):
        tr = Trajectory(id=f'traj_{i}', kind='solve', task_id='done', harness_id='h_empty',
            instructions='solve', events=[], final_message='done', stdout='', stderr='',
            workspace_diff={'task/repo/code.py': str(i).encode()}, workspace_deletions=frozenset(),
            exit_code=0, wall_time_s=1, model='test', reasoning_effort='high')
        store.put(tr)
        write_json(run / f'runtime/calls/{i}/isolation.json', {'call_id': str(i), 'kind': 'solve',
            'task_id': 'done', 'harness_id': 'h_empty', 'image_id': 'sha256:base',
            'phase': 'completed', 'trajectory_id': tr.id, 'started_at': 10 - i})
    write_json(run / 'runtime/calls/failed/isolation.json', {'call_id': 'failed', 'kind': 'solve',
        'task_id': 'failed', 'harness_id': 'h_empty', 'image_id': 'sha256:base', 'phase': 'failed'})
    return run, selection, images


def test_reuses_checkpoints_once_and_leaves_failed_and_unstarted_for_execution(tmp_path):
    run, selection, images = fixture_run(tmp_path)
    recovery = InitialSolveRecovery(run, selection, images)
    ws = tmp_path / 'ws'; ws.mkdir()
    for sample, expected in enumerate(['traj_2', 'traj_1', 'traj_0']):
        (ws / '.sample_index').write_text(str(sample))
        for task in ('failed', 'unstarted'):
            assert recovery.take(ws, 'solve', task_id=task, harness_id='h_empty') is None
        assert recovery.take(ws, 'solve', task_id='done', harness_id='candidate') is None
        with pytest.raises(RuntimeError, match='instructions'):
            recovery.take(ws, 'different', task_id='done', harness_id='h_empty')
        tr = recovery.take(ws, 'solve', task_id='done', harness_id='h_empty')
        assert tr.id == expected and tr.workspace_diff
        assert recovery.take(ws, 'solve', task_id='done', harness_id='h_empty') is None
    audit = json.loads((run / 'initial_solve_recovery.json').read_text())
    assert audit['reused'] == 3 and audit['remaining'] == 6
    assert {e['sample_index_basis'] for e in audit['entries']} == {'legacy_launch_order_assignment'}
    again = InitialSolveRecovery(run, selection, images)
    assert {k: t.id for k, t in again.lookup.items()} == {k: t.id for k, t in recovery.lookup.items()}


def test_refuses_later_phase_and_changed_inputs(tmp_path):
    run, selection, images = fixture_run(tmp_path)
    InitialSolveRecovery(run, selection, images)
    images.write_text(images.read_text() + '\n')
    with pytest.raises(RuntimeError, match='manifests changed'):
        InitialSolveRecovery(run, selection, images)
    (run / 'rounds/round_0/solve_before_traj_ids.json').write_text('[]')
    with pytest.raises(RuntimeError, match='before diagnosis'):
        InitialSolveRecovery(run, selection, images)


def test_timeout_patch_is_retained_but_not_reused_as_success(tmp_path):
    run, selection, images = fixture_run(tmp_path)
    store = FilesystemTrajectoryStore(run / 'runtime/completed_trajectories')
    store.put(replace(store.get('traj_0'), id='timeout_patch', task_id='failed',
                      timed_out=True, exit_code=-1, sample_index=0))
    write_json(run / 'runtime/calls/timeout/isolation.json', {
        'call_id': 'timeout', 'kind': 'solve', 'task_id': 'failed', 'harness_id': 'h_empty',
        'image_id': 'sha256:base', 'phase': 'completed', 'trajectory_id': 'timeout_patch',
        'started_at': 20, 'sample_index': 0, 'exit_code': -1, 'timed_out': True})
    recovery = InitialSolveRecovery(run, selection, images)
    assert len(recovery.lookup) == 3 and ('failed', 0) not in recovery.lookup
    assert store.get('timeout_patch').workspace_diff
