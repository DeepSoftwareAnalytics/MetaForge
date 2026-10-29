#!/usr/bin/env python3
"""Regrade saved patches symmetrically, preserving original evidence and budgets."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from rho.agent.request_budget import write_json
from rho.datasets.swebench_pro import evaluator as evaluator_module
from rho.datasets.swebench_pro.evaluator import SWEbenchProDockerEvaluator

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/swebench-v1-clean-deps'
HELDOUT = RUN / 'heldout'
OUT = HELDOUT / 'environment_regrade'
SCHEMA_URL = 'https://raw.githubusercontent.com/internetarchive/openlibrary-client/master/olclient/schemata/import.schema.json'


def repairs(task_id):
    settings = {'workers': 2}
    if 'teleport-b1bcd8' in task_id or 'teleport-bb562408' in task_id:
        settings['install_linux_headers'] = True
    if 'flipt-1737085488' in task_id:
        settings.update(test_fixture_commit='1737085488ecdcd3299c8e61af45a8976d457b7e',
            test_fixtures=['internal/ext/testdata/import_flag_complex_metadata.' + ext for ext in ('json', 'yml')])
    if 'openlibrary-8a9d9d323' in task_id:
        settings['static_resources'] = {SCHEMA_URL: str(OUT / 'import.schema.json')}
    return settings


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            os.environ[key] = 'http://127.0.0.2:7896'
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1,127.0.0.2'
        review = json.loads((HELDOUT / 'comparison/environment_failure_review.json').read_text())
        ids = sorted({x['task_id'] for x in review['records']})
        config = json.loads((RUN / 'evolution/config.json').read_text())
        source = Path(config['dataset_spec'].split(':', 1)[1])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == config['dataset_digest']
        rows = {x['instance_id']: x for x in map(json.loads, source.read_text().splitlines()) if x['instance_id'] in ids}
        images = json.loads((RUN / 'images.json').read_text())
        available = set(subprocess.check_output(['docker', 'image', 'ls', '-aq', '--no-trunc'], text=True).split())
        assert all(images[tid]['image_id'] in available for tid in ids), 'Pinned image missing'
        probe = SWEbenchProDockerEvaluator(docker_pull='never')
        assets = evaluator_module.ensure_official_assets(probe.assets_root)
        # Resolve/pin official assets once for this immutable batch.
        evaluator_module.ensure_official_assets = lambda root: assets
        original = OUT / 'original'
        original.mkdir(exist_ok=True)
        for arm in ('baseline', 'final'):
            for name in ('summary.json', 'status.json'):
                dest = original / arm / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): shutil.copy2(HELDOUT / arm / name, dest)
        jobs = [(arm, tid) for tid in ids for arm in ('baseline', 'final')]
        state = {'phase': 'running', 'total': len(jobs), 'workers': 8, 'model_requests': 0,
                 'active': {}, 'completed': {}, 'errors': {}, 'started_at': time.time()}
        mutex = threading.Lock()
        def update():
            state['updated_at'] = time.time()
            write_json(OUT / 'status.json', state)
        update()

        def one(arm, tid):
            key = hashlib.sha256(tid.encode()).hexdigest()
            job = arm + '/' + key
            saved = OUT / 'results' / arm / (key + '.json')
            old_path = HELDOUT / arm / 'grades' / (key + '.json')
            old_copy = original / arm / 'grades' / (key + '.json')
            old_copy.parent.mkdir(parents=True, exist_ok=True)
            if not old_copy.exists(): shutil.copy2(old_path, old_copy)
            old = json.loads(old_copy.read_text())
            if saved.exists():
                result = json.loads(saved.read_text())
                assert result['trajectory_id'] == old['trajectory_id']
                with mutex: state['completed'][job] = result['grade']['score']; update()
                return
            with mutex: state['active'][job] = tid; update()
            try:
                if old['grade']['details'].get('outcome') == 'no_patch':
                    result = {**old, 'environment_regrade': {'outcome': 'no_patch_unchanged', 'model_requests': 0}}
                else:
                    patch_path = HELDOUT / arm / 'grade_artifacts' / key / 'docker/patch.diff'
                    patch = patch_path.read_text()
                    patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
                    old_patch = original / arm / 'patches' / (key + '.diff')
                    old_patch.parent.mkdir(parents=True, exist_ok=True)
                    if not old_patch.exists(): shutil.copy2(patch_path, old_patch)
                    assert old_patch.read_bytes() == patch_path.read_bytes()
                    preflight = json.loads((RUN / 'environment_preflight/results' / (key + '.json')).read_text())
                    row = dict(rows[tid], dockerhub_tag='', image_name=images[tid]['image_id'],
                        _rho_network_disabled=True, _rho_dependency_policy=preflight['dependency_policy'],
                        _rho_grading_environment=repairs(tid))
                    attempts = OUT / 'artifacts' / arm / key
                    for number in range(1, 4):
                        art = attempts / ('attempt_' + str(time.time_ns()))
                        evaluation = SWEbenchProDockerEvaluator(docker_pull='never', assets_root=assets).evaluate(
                            row=row, patch=patch, artifacts_dir=art)
                        write_json(art / 'evaluation.json', asdict(evaluation))
                        if not evaluation.details.get('error'): break
                        if number == 3: raise RuntimeError(json.dumps(evaluation.details)[:1800])
                        time.sleep(10)
                    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == patch_hash
                    result = {**old, 'grade': asdict(evaluation), 'environment_regrade': {
                        'patch_sha256': patch_hash, 'original_grade': str(old_copy),
                        'settings': repairs(tid), 'model_requests': 0, 'completed_at': time.time()}}
                write_json(saved, result)
                with mutex: state['completed'][job] = result['grade']['score']
                print('REGRADED', arm, tid, result['grade']['score'], flush=True)
            except Exception as exc:
                with mutex: state['errors'][job] = type(exc).__name__ + ': ' + str(exc)
                print('REGRADING_ERROR', arm, tid, type(exc).__name__, str(exc)[:300], flush=True)
            finally:
                with mutex: state['active'].pop(job, None); update()

        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in as_completed([pool.submit(one, *job) for job in jobs]): result.result()
        if state['errors']:
            state['phase'] = 'incomplete'; update(); return 1
        # Publish only after every selected comparison has a valid outcome.
        changes = []
        for arm, tid in jobs:
            key = hashlib.sha256(tid.encode()).hexdigest()
            new = json.loads((OUT / 'results' / arm / (key + '.json')).read_text())
            old = json.loads((original / arm / 'grades' / (key + '.json')).read_text())
            write_json(HELDOUT / arm / 'grades' / (key + '.json'), new)
            changes.append({'arm': arm, 'task_id': tid, 'before': old['grade']['score'], 'after': new['grade']['score']})
        summaries = {}
        for arm in ('baseline', 'final'):
            records = [json.loads(p.read_text()) for p in (HELDOUT / arm / 'grades').glob('*.json')]
            assert len(records) == 100 and not any(x['grade']['details'].get('error') for x in records)
            summary = json.loads((original / arm / 'summary.json').read_text())
            summary.update(mean_score=sum(x['grade']['score'] for x in records)/100,
                           environment_regrade=str(OUT / 'summary.json'))
            summaries[arm] = summary
            write_json(HELDOUT / arm / 'summary.json', summary)
        write_json(HELDOUT / 'summary.json', summaries)
        status_path = HELDOUT / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text()); status['summaries'] = summaries
            write_json(status_path, status)
        write_json(OUT / 'summary.json', {'changes': changes, 'summaries': summaries,
                                         'model_requests': 0, 'finished_at': time.time()})
        state['phase'] = 'complete'; update()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
