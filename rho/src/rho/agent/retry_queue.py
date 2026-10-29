"""Persist execution failures and retry with backoff outside admission slots."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from rho.agent.request_budget import write_json


class ExecutionRetryQueue:
    def __init__(self, root, *, sleep=time.sleep, initial_delay=60, max_delay=600):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def run(self, execute, workspace, instructions, **kwargs):
        workspace = Path(workspace)
        sample = workspace / '.sample_index'
        identity = {'kind': kwargs.get('kind', 'solve'), 'task_id': kwargs.get('task_id', ''),
                    'harness_id': kwargs.get('harness_id', ''),
                    'sample_index': int(sample.read_text()) if sample.exists() else None,
                    'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.root / (key + '.json')
        state = json.loads(path.read_text()) if path.exists() else {**identity, 'attempts': 0, 'failures': []}
        with tempfile.TemporaryDirectory(prefix='rho-retry-harness-') as tmp:
            backup = Path(tmp) / 'harness'
            harness = workspace / 'harness'
            if harness.exists():
                shutil.copytree(harness, backup, symlinks=True)
            while True:
                state.update(state='running', attempts=state['attempts'] + 1, updated_at=time.time())
                write_json(path, state)
                trajectory = None
                try:
                    trajectory = execute(workspace, instructions, **kwargs)
                    budget = next((e for e in reversed(trajectory.events)
                                   if e.get('type') == 'model_request_budget'), {})
                    if (kwargs.get('kind', 'solve') == 'solve'
                            and budget.get('maximum') is not None
                            and budget.get('requests_sent', 0) >= budget['maximum']):
                        # Budget exhaustion is a terminal solve outcome, including
                        # its partial patch. Retrying would exceed the experiment budget.
                        state.update(state='completed', trajectory_id=trajectory.id,
                                     terminal_reason='request_budget_reached', updated_at=time.time())
                        write_json(path, state)
                        return trajectory
                    if trajectory.timed_out:
                        reason = 'timeout'
                    elif trajectory.exit_code != 0:
                        reason = 'nonzero_exit:' + str(trajectory.exit_code)
                    else:
                        # A normal empty patch or failing tests are solve outcomes,
                        # not grounds for additional attempts or cherry-picking.
                        state.update(state='completed', trajectory_id=trajectory.id, updated_at=time.time())
                        write_json(path, state)
                        return trajectory
                except Exception as exc:
                    reason = type(exc).__name__ + ': ' + str(exc)[:1800]
                if backup.exists():
                    shutil.rmtree(harness, ignore_errors=True)
                    shutil.copytree(backup, harness, symlinks=True)
                delay = min(self.max_delay, self.initial_delay * 2 ** min(len(state['failures']), 10))
                state['failures'].append({'attempt': state['attempts'], 'reason': reason,
                    'trajectory_id': trajectory.id if trajectory else None, 'at': time.time()})
                state.update(state='waiting_retry', retry_after=time.time() + delay, updated_at=time.time())
                write_json(path, state)
                self.sleep(delay)
