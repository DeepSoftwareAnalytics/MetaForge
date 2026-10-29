#!/usr/bin/env python3
"""Continue incomplete held-out arms after the current owner releases them."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/swebench-v1-clean-deps'
OWNER = 'rho-clean-pairwise-fixed-20261028.service'


def arm_complete(root, arm):
    path = root / arm / 'summary.json'
    return path.exists() and json.loads(path.read_text()).get('n') == 100


def continue_arms(complete, execute, report, sleep=time.sleep):
    """A failed baseline backlog must not prevent the final arm's first pass."""
    cycle = 0
    while not all(complete(arm) for arm in ('baseline', 'final')):
        cycle += 1
        # Baseline has already had its full pass under the original owner.
        # Visit both arms before returning to any failed tasks.
        for arm in ('final', 'baseline'):
            if complete(arm):
                continue
            report('running', arm=arm, cycle=cycle)
            code = execute(arm)
            report('pass_finished', arm=arm, cycle=cycle, exit_code=code,
                   completed=complete(arm))
        if not all(complete(arm) for arm in ('baseline', 'final')):
            report('waiting_retry', cycle=cycle)
            sleep(60)
    report('complete', cycle=cycle)


def main():
    with (RUN / 'heldout_continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def report(phase, **extra):
            write_json(RUN / 'heldout_continuation_status.json', {
                'phase': phase, 'updated_at': time.time(), **extra})

        report('waiting_for_current_owner', owner=OWNER)
        while True:
            state = subprocess.run(['systemctl', '--user', 'show', OWNER,
                '--property=ActiveState', '--value'], capture_output=True, text=True)
            if state.stdout.strip() not in {'active', 'activating', 'deactivating', 'reloading'}:
                break
            time.sleep(10)

        policy_path = RUN / 'heldout_policy.json'
        policy = json.loads(policy_path.read_text())
        assert policy['solve_workers'] == policy['grade_workers'] == 8
        assert policy['max_model_requests_per_task'] == 120
        output = ROOT / policy['output_root']

        def execute(arm):
            write_json(RUN / 'pipeline_status.json', {
                'phase': 'heldout_' + arm, 'updated_at': time.time(),
                'solve_workers': 8, 'grade_workers': 8,
                'evaluation_model_requests': 120, 'evaluation_wall_timeout': None,
                'active_service': 'rho-clean-heldout-continuation-20261029.service'})
            with (RUN / 'heldout.log').open('a') as out:
                return subprocess.run([sys.executable,
                    str(ROOT / 'scripts/run-swebench-v1-evaluation.py'),
                    '--policy', str(policy_path), '--arm', arm, '--execute'],
                    cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode

        continue_arms(lambda arm: arm_complete(output, arm), execute, report)
        write_json(RUN / 'pipeline_status.json', {
            'phase': 'complete', 'updated_at': time.time(),
            'solve_workers': 8, 'grade_workers': 8,
            'evaluation_model_requests': 120, 'evaluation_wall_timeout': None})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
