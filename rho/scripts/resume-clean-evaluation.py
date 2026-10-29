#!/usr/bin/env python3
"""Replay retained evolution work, finish fresh pairwise reviews, then held-out."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/swebench-v1-clean-deps'


def main():
    with (RUN / 'evaluation_restart.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evolution = RUN / 'evolution'
        write_json(RUN / 'pipeline_status.json', {
            'phase': 'evolution', 'stage': 'pairwise_evaluation',
            'updated_at': time.time(), 'solve_workers': 8, 'grade_workers': 8,
            'evaluation_model_requests': 120, 'evaluation_wall_timeout': None,
            'pairwise_score_range': [-10, 10], 'pairwise_requires_valid_result': True,
            'active_service': 'rho-clean-pairwise-fixed-20261028.service'})
        if not (evolution / 'reports/summary.json').exists():
            command = [sys.executable, str(ROOT / 'scripts/run-swebench-v1-evolution.py'),
                '--selection-json', str(RUN / 'selection.json'), '--run-dir', str(evolution),
                '--container-images', str(RUN / 'images.json'),
                '--resume-round', str(evolution / 'concurrency_handoff.json')]
            with (RUN / 'evolution.log').open('a') as out:
                code = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
            if code:
                write_json(RUN / 'evaluation_restart_status.json', {
                    'phase': 'failed', 'exit_code': code, 'updated_at': time.time()})
                return code
        with (RUN / 'pipeline.log').open('a') as out:
            return subprocess.run([sys.executable, str(ROOT / 'scripts/run-swebench-v1-clean.py'),
                '--run-root', str(RUN)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
