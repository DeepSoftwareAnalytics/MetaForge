from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from rho.agent.retry_queue import ExecutionRetryQueue
from rho.protocols import Trajectory


def result(**changes):
    return replace(Trajectory(id='ok', kind='solve', task_id='task', harness_id='h_empty',
        instructions='solve', events=[], final_message='', stdout='', stderr='',
        workspace_diff={}, workspace_deletions=frozenset(), exit_code=0, wall_time_s=1), **changes)


def test_timeout_and_exception_wait_outside_admission_and_preserve_harness(tmp_path):
    ws = tmp_path / 'workspace'; (ws / 'harness').mkdir(parents=True)
    (ws / 'harness/original').write_text('original')
    (ws / '.sample_index').write_text('2')
    active = [0]; delays = []; attempts = []
    def sleep(delay):
        assert active[0] == 0
        state = json.loads(next((tmp_path / 'queue').glob('*.json')).read_text())
        assert state['state'] == 'waiting_retry'
        delays.append(delay)
    def execute(workspace, instructions, **kwargs):
        active[0] += 1
        try:
            assert (workspace / 'harness/original').read_text() == 'original'
            attempts.append(1)
            if len(attempts) == 1:
                (workspace / 'harness/original').write_text('partial failed optimize')
                return result(id='timeout', timed_out=True, exit_code=-1)
            if len(attempts) == 2:
                raise ConnectionError('temporary provider outage')
            return result()
        finally:
            active[0] -= 1
    queue = ExecutionRetryQueue(tmp_path / 'queue', sleep=sleep)
    tr = queue.run(execute, ws, 'solve', task_id='task', kind='solve', harness_id='h_empty')
    assert tr.id == 'ok' and delays == [60, 120]
    state = json.loads(next((tmp_path / 'queue').glob('*.json')).read_text())
    assert state['state'] == 'completed' and state['attempts'] == 3
    assert state['sample_index'] == 2 and len(state['failures']) == 2


def test_normal_empty_patch_is_not_retried_and_interrupt_is_not_swallowed(tmp_path):
    queue = ExecutionRetryQueue(tmp_path / 'queue', sleep=lambda _: pytest.fail('unexpected retry'))
    tr = queue.run(lambda *a, **kw: result(), tmp_path, 'solve')
    assert not tr.workspace_diff
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        queue.run(interrupted, tmp_path, 'interrupted')


def test_exhausted_solve_keeps_partial_patch_without_retrying(tmp_path):
    queue = ExecutionRetryQueue(tmp_path / 'queue', sleep=lambda _: pytest.fail('budget must not reset'))
    exhausted = result(exit_code=1, workspace_diff={'task/repo/f.py': b'partial repair'},
        events=[{'type': 'model_request_budget', 'maximum': 170,
                 'requests_sent': 170, 'blocked_requests': 1}])
    actual = queue.run(lambda *a, **kw: exhausted, tmp_path, 'solve', kind='solve')
    assert actual.workspace_diff == exhausted.workspace_diff and actual.exit_code == 1
    state = json.loads(next((tmp_path / 'queue').glob('*.json')).read_text())
    assert state['attempts'] == 1 and state['terminal_reason'] == 'request_budget_reached'


def test_parallel_recovery_shares_slots_with_existing_workers():
    import runpy
    from pathlib import Path
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/parallel-swebench-initial-recovery.py'))
    external = [3]
    gate = module['SharedInitialAdmission'](lambda: external[0])
    counts = [0, 0]
    lock = threading.Lock()
    def job(_):
        with gate.slot():
            with lock:
                counts[0] += 1
                counts[1] = max(counts[1], counts[0])
                assert counts[0] + external[0] <= 8
            time.sleep(.02)
            with lock:
                counts[0] -= 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(job, range(16)))
    assert counts[1] <= 5
    external[0] = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(job, range(16)))
    assert 5 < counts[1] <= 8
