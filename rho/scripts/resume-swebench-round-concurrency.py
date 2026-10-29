#!/usr/bin/env python3
"""Resume the current round with eight shared admission slots and checkpoints."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/swebench-v1-clean-deps'


def main():
    evolution = RUN / 'evolution'
    manifest_path = evolution / 'concurrency_handoff.json'
    manifest = json.loads(manifest_path.read_text())
    pid = manifest['previous_evolution_pid']
    expected = str(ROOT / 'scripts/run-swebench-v1-evolution.py').encode()
    if expected not in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('Evolution process changed; refusing to signal')
    os.kill(pid, signal.SIGINT)
    write_json(RUN / 'concurrency_transition.json', {'phase': 'resuming', 'limit': 8,
        'old_pid': pid, 'updated_at': time.time(), 'preserve_existing_workers': True})
    command = [sys.executable, str(ROOT / 'scripts/run-swebench-v1-evolution.py'),
        '--selection-json', str(RUN / 'selection.json'), '--run-dir', str(evolution),
        '--container-images', str(RUN / 'images.json'), '--resume-round', str(manifest_path)]
    with (RUN / 'evolution.log').open('a') as out:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        write_json(RUN / 'concurrency_transition.json', {'phase': 'running', 'limit': 8,
            'old_pid': pid, 'new_pid': process.pid, 'updated_at': time.time()})
        code = process.wait()
    if code:
        write_json(RUN / 'concurrency_transition.json', {'phase': 'failed', 'exit_code': code, 'updated_at': time.time()})
        return code
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                          'rho-clean-deps-parallel-20261028.service']).returncode == 0:
        time.sleep(2)
    # Completed summary makes the clean pipeline skip evolution and proceed
    # directly to the already-configured held-out evaluation arms.
    with (RUN / 'pipeline.log').open('a') as out:
        return subprocess.run([sys.executable, str(ROOT / 'scripts/run-swebench-v1-clean.py'),
                               '--run-root', str(RUN)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
