#!/usr/bin/env python3
"""Drain the current initial-solve workers, then enable durable execution retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--previous-service', required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    status_path = run / 'retry_handoff.json'
    state = {'previous_service': args.previous_service, 'started_at': time.time()}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=time.time(), **extra)
        write_json(status_path, state)
        print(json.dumps(state), flush=True)
    main_pid = int(subprocess.check_output(['systemctl', '--user', 'show', args.previous_service,
                                           '--property=MainPID', '--value'], text=True).strip())
    if main_pid <= 0:
        raise RuntimeError('Previous pipeline is not running; inspect before handoff')
    if any((run / 'evolution/trajectories').iterdir()):
        raise RuntimeError('Initial phase already published; refusing initial-phase handoff')
    candidates = []
    for child in Path('/proc').iterdir():
        if not child.name.isdigit():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (child / 'status').read_text().splitlines() if ':' in line)
            cmd = (child / 'cmdline').read_bytes().split(b'\0')
            if int(fields['PPid']) == main_pid and str(ROOT / 'scripts/run-swebench-v1-evolution.py').encode() in cmd:
                candidates.append(int(child.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    if len(candidates) != 1:
        raise RuntimeError('Cannot unambiguously identify current evolution child')
    calls = [json.loads(p.read_text()) for p in (run / 'evolution/runtime/calls').glob('*/isolation.json')]
    if any(d['kind'] != 'solve' or d['harness_id'] != 'h_empty' for d in calls):
        raise RuntimeError('Later phase started; refusing initial-phase handoff')
    status('draining_current_calls', evolution_pid=candidates[0],
           note='SIGINT to coordinator only; ThreadPoolExecutor waits for current workers to save results.')
    # The coordinator leaves pool.map; executor shutdown waits for active
    # workers. Do not signal the service/group or its Docker subprocesses.
    os.kill(candidates[0], signal.SIGINT)
    while True:
        snapshots = [json.loads(p.read_text()) for p in (run / 'evolution/runtime/calls').glob('*/isolation.json')]
        successful = {(d['task_id'], d.get('sample_index')) for d in snapshots
                      if d['phase'] == 'completed' and d.get('exit_code') == 0 and not d.get('timed_out')}
        pending = []
        for d in snapshots:
            if d.get('sample_index') is None or (d['task_id'], d['sample_index']) in successful:
                continue
            if d['phase'] == 'failed' or (d['phase'] == 'completed' and (d.get('timed_out') or d.get('exit_code') != 0)):
                pending.append({k: d.get(k) for k in ('call_id', 'task_id', 'sample_index', 'timed_out', 'exit_code', 'error')})
        write_json(run / 'evolution/retry_queue/draining_failures.json', {
            'state': 'waiting_for_current_workers', 'pending': pending, 'updated_at': time.time()})
        result = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.previous_service])
        if result.returncode:
            break
        time.sleep(5)
    calls = [json.loads(p.read_text()) for p in (run / 'evolution/runtime/calls').glob('*/isolation.json')]
    unfinished = [d['call_id'] for d in calls if d['phase'] not in {'completed', 'failed'}]
    if unfinished:
        status('blocked_unfinished_calls', call_ids=unfinished)
        raise RuntimeError('Unfinished calls remain; refusing duplicate execution')
    write_json(run / 'evolution/retry_queue/draining_failures.json', {
        'state': 'transferred_to_initial_recovery', 'pending': pending, 'updated_at': time.time()})
    status('restarting_with_retry_queue')
    command = [sys.executable, str(ROOT / 'scripts/run-swebench-v1-clean.py'),
               '--run-root', str(run), '--resume-initial']
    with (run / 'pipeline.log').open('a') as out:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        status('retry_pipeline_running', pipeline_pid=process.pid)
        code = process.wait()
    status('complete' if code == 0 else 'pipeline_failed', exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
