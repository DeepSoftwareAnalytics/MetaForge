#!/usr/bin/env python3
"""Fill free initial-solve slots alongside the draining original workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time

from rho.agent.container_codex import ContainerCodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.agent.request_budget import write_json
from rho.agent.retry_queue import ExecutionRetryQueue
from rho.datasets.swebench_pro.dataset import SWEbenchProDataset
from rho.orchestrators.solve import SOLVE_INSTRUCTIONS
from rho.stores.harness import FilesystemHarnessStore

ROOT = Path(__file__).resolve().parents[1]


class SharedInitialAdmission:
    """Original workers are draining and cannot admit new calls."""
    def __init__(self, external_active, limit=8):
        self.external_active = external_active
        self.limit = limit
        self.inflight = 0
        self.condition = threading.Condition()

    @contextmanager
    def slot(self):
        with self.condition:
            while self.inflight + self.external_active() >= self.limit:
                self.condition.wait(timeout=1)
            self.inflight += 1
        try:
            yield
        finally:
            with self.condition:
                self.inflight -= 1
                self.condition.notify_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--previous-service', required=True)
    args = parser.parse_args(); root = args.run_root.resolve(); run = root / 'evolution'
    lock = (root / 'parallel_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((run / 'trajectories').iterdir()):
        raise RuntimeError('Only initial solves may be recovered concurrently')
    def records():
        return [json.loads(p.read_text()) for p in (run / 'runtime/calls').glob('*/isolation.json')]
    original = records()
    if any(d['kind'] != 'solve' or d['harness_id'] != 'h_empty' for d in original):
        raise RuntimeError('Later evolution phase already started')
    write_json(root / 'retry_handoff.json', {'phase': 'parallel_initial_running',
        'updated_at': time.time(), 'concurrency_limit': 8,
        'note': 'Existing and retry/new solves share eight slots; no drain barrier.'})
    external_ids = {d['call_id'] for d in original if d['phase'] not in {'completed', 'failed'}}
    def external_active():
        return sum(d['call_id'] in external_ids and d['phase'] not in {'completed', 'failed'} for d in records())
    admission = SharedInitialAdmission(external_active)
    selected = json.loads((root / 'selection.json').read_text())['selected_task_ids']
    expected = {(task, sample) for task in selected for sample in range(3)}
    legacy = {e['call_id']: (e['task_id'], e['sample_index'])
              for e in json.loads((run / 'initial_solve_recovery.json').read_text())['entries']}
    def key(d):
        if d.get('sample_index') is not None:
            return d['task_id'], d['sample_index']
        return legacy.get(d['call_id'])
    launcher = runpy.run_path(str(ROOT / 'scripts/run-swebench-v1-evolution.py'))
    runner = launcher['load_selection_runner']()
    ds = runpy.run_path(str(ROOT / 'scripts/rho-ds.py')); ds['configure_environment']()
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ[k] = 'http://127.0.0.2:7896'
    for k in ('NO_PROXY', 'no_proxy'):
        os.environ[k] = 'localhost,127.0.0.1,127.0.0.2'
    store = FilesystemHarnessStore(run / 'harness'); harness = store.empty()
    dataset = SWEbenchProDataset(str(runner.SOURCE), harness_store=store,
                                repo_cache=runner.HistoryFreeRepoCache(), docker_pull='never')
    tasks = {task.id: task for task in dataset.train if task.id in selected}
    assert set(tasks) == set(selected)
    configure_global_codex_pool(8)
    agent = ContainerCodexAgent(ROOT / 'configs/codex.dashscope.toml', images_path=root / 'images.json',
        audit_root=run / 'runtime', model=ds['MODEL'], reasoning_effort='high', default_timeout_s=3600,
        binary=ROOT.parent / '.codex/packages/standalone/releases/0.163.0-x86_64-unknown-linux-musl/bin/codex')
    queue = ExecutionRetryQueue(run / 'retry_queue')
    def execute(*a, **kw):
        with admission.slot():
            return agent.run(*a, **kw)
    def solve(job):
        task_id, sample = job
        with tempfile.TemporaryDirectory(prefix='parallel_initial_', dir=run / 'workdir') as tmp:
            ws = Path(tmp); (ws / 'task/repo').mkdir(parents=True); (ws / 'harness').mkdir()
            harness.materialize(ws / 'harness')
            (ws / 'task/prompt.md').write_text(launcher['container_task_prompt'](tasks[task_id]))
            (ws / '.sample_index').write_text(str(sample))
            return queue.run(execute, ws, SOLVE_INSTRUCTIONS, task_id=task_id, harness_id='h_empty',
                             kind='solve', timeout_s=3600)
    futures = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            calls = records()
            successful = {key(d) for d in calls if d['phase'] == 'completed'
                          and d.get('exit_code') == 0 and not d.get('timed_out')}
            active = {key(d) for d in calls if d['phase'] not in {'completed', 'failed'}}
            for job, future in list(futures.items()):
                if future.done():
                    future.result(); del futures[job]
            # A just-finished result may be absent from this scan. Re-read
            # before admitting anything, so a successful sample is never rerun.
            calls = records()
            successful = {key(d) for d in calls if d['phase'] == 'completed'
                          and d.get('exit_code') == 0 and not d.get('timed_out')}
            active = {key(d) for d in calls if d['phase'] not in {'completed', 'failed'}}
            for job in sorted(expected - successful - active - set(futures)):
                futures[job] = pool.submit(solve, job)
            write_json(root / 'parallel_recovery_status.json', {
                'phase': 'running_initial', 'concurrency_limit': 8,
                'completed': len(expected & successful), 'active': len(expected & active),
                'pending_or_submitted': len(expected - successful - active), 'updated_at': time.time()})
            if expected <= successful and not futures:
                break
            time.sleep(2)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.previous_service]).returncode == 0:
        time.sleep(2)
    write_json(root / 'retry_handoff.json', {'phase': 'parallel_initial_complete_resuming_evolution',
        'updated_at': time.time(), 'completed': 30, 'concurrency_limit': 8})
    with (root / 'pipeline.log').open('a') as out:
        code = subprocess.run([sys.executable, str(ROOT / 'scripts/run-swebench-v1-clean.py'),
            '--run-root', str(root), '--resume-initial'], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    return code


if __name__ == '__main__':
    raise SystemExit(main())
