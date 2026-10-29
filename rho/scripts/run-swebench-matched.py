#!/usr/bin/env python3
"""Fresh matched-budget evolution and both held-out arms in one isolated run root."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--max-model-requests', type=int, default=170)
    args = parser.parse_args()
    run = args.run_root.resolve()
    maximum = args.max_model_requests
    assert maximum > 0 and run.is_relative_to(ROOT / 'runs')
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'evolution/config.json').exists():
        raise RuntimeError('This launcher starts fresh experiments only; use audited recovery for interruptions')
    selection = json.loads((run / 'selection.json').read_text())
    manifest = json.loads((run / 'split_manifest.json').read_text())
    core, heldout = selection['selected_task_ids'], manifest['groups']['evaluation_100']
    assert len(set(core)) == 10 and len(set(heldout)) == 100 and not set(core) & set(heldout)
    images = json.loads((run / 'images.json').read_text())
    available = set(subprocess.check_output(['docker', 'image', 'ls', '-aq', '--no-trunc'], text=True).split())
    assert all(images[t]['image_id'] in available for t in core + heldout), 'Pinned image missing'

    def status(phase, **extra):
        write_json(run / 'pipeline_status.json', dict(phase=phase, updated_at=time.time(),
            solve_workers=8, grade_workers=8, evolution_model_requests=maximum,
            evaluation_model_requests=maximum, solve_wall_timeout=None, **extra))
        print(phase, extra, flush=True)

    def execute(script, argv, log):
        with (run / log).open('a') as out:
            return subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *argv],
                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode

    status('core_environment_preflight')
    check_args = ['--run-root', str(run), '--workers', '4']
    code = execute('check-swebench-container-environments.py', check_args + ['--core-only'], 'environment_preflight.log')
    if code:
        status('preflight_failed', exit_code=code)
        return code
    with (run / 'environment_preflight.log').open('a') as out:
        environment = subprocess.Popen([sys.executable, str(ROOT / 'scripts/check-swebench-container-environments.py'),
            *check_args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    evolution = run / 'evolution'
    status('evolution', core_tasks=10, initial_rollouts=30, optimize_samples=3,
           evaluation_environment_pid=environment.pid)
    code = execute('run-swebench-v1-evolution.py', [
        '--selection-json', str(run / 'selection.json'), '--run-dir', str(evolution),
        '--container-images', str(run / 'images.json'), '--max-model-requests', str(maximum),
        '--solve-timeout-seconds', '0'], 'evolution.log')
    if code:
        status('evolution_failed', exit_code=code)
        return code
    if environment.wait():
        status('evaluation_preflight_failed')
        return 1
    for tid in core + heldout:
        report = json.loads((run / 'environment_preflight/results' /
            (hashlib.sha256(tid.encode()).hexdigest() + '.json')).read_text())
        assert report['passed'] and report['image_id'] == images[tid]['image_id']
    policy = {
        'source_run': str(evolution.relative_to(ROOT)),
        'pairwise_results': 'reports/summary.json', 'pairwise_results_format': 'rho_summary',
        'split_manifest': str((run / 'split_manifest.json').relative_to(ROOT)),
        'output_root': str((run / 'heldout').relative_to(ROOT)),
        'container_images': str((run / 'images.json').relative_to(ROOT)),
        'arms': ['baseline', 'final'], 'max_model_requests_per_task': maximum,
        'request_counting': 'outbound_model_requests_including_retries_and_compaction',
        'solve_timeout_seconds': None, 'solve_workers': 8, 'grade_workers': 8,
        'max_grading_tasks': 100, 'solve_attempts': 1, 'model': 'deepseek-v4-flash-0731',
        'reasoning_effort': 'high', 'codex_version': '0.163.0', 'cache': 'off',
        'grading_environment': json.loads((run / 'grading_environment.json').read_text()),
    }
    write_json(run / 'heldout_policy.json', policy)

    def complete(arm):
        path = run / 'heldout' / arm / 'summary.json'
        return path.exists() and json.loads(path.read_text()).get('n') == 100

    cycle = 0
    while not all(complete(a) for a in policy['arms']):
        cycle += 1
        # Finish each arm's pass before retrying infrastructure failures, so a
        # single blocked baseline task cannot prevent the final arm from starting.
        for arm in policy['arms']:
            if complete(arm):
                continue
            status('heldout_' + arm, cycle=cycle)
            code = execute('run-swebench-v1-evaluation.py', [
                '--policy', str(run / 'heldout_policy.json'), '--arm', arm, '--execute'], 'heldout.log')
            status('heldout_pass_finished', arm=arm, cycle=cycle, exit_code=code)
        if not all(complete(a) for a in policy['arms']):
            status('waiting_infrastructure_retry', cycle=cycle)
            time.sleep(60)
    status('complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
