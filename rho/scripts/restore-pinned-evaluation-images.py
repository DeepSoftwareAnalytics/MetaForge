#!/usr/bin/env python3
"""Restore missing images without changing the experiment's pinned image IDs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from rho.agent.request_budget import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/swebench-v1-clean-deps'
SWE = ROOT.parent / 'swebench-pro'
CRANE = SWE / 'experiment/bin/crane'


def inventory(images):
    available = set(subprocess.check_output(
        ['docker', 'image', 'ls', '-aq', '--no-trunc'], text=True).split())
    return {k: v for k, v in images.items() if v['image_id'] not in available}


def restore(task_id, item):
    env = dict(os.environ, HTTP_PROXY='http://127.0.0.2:7896',
               HTTPS_PROXY='http://127.0.0.2:7896', NO_PROXY='localhost,127.0.0.1,127.0.0.2')
    folder = RUN / 'image_restoration'
    folder.mkdir(exist_ok=True)
    key = item['image_id'].split(':')[1]
    archive = folder / (key + '.tar')
    with (folder / (key + '.log')).open('a') as log:
        # Resolve once to an immutable registry manifest, then verify its config
        # matches the locally pinned Docker image ID before downloading/loading.
        digest = subprocess.check_output([str(CRANE), 'digest', item['image']],
            env=env, text=True, timeout=180).strip()
        reference = item['image'].rsplit(':', 1)[0] + '@' + digest
        manifest = json.loads(subprocess.check_output([str(CRANE), 'manifest', reference],
            env=env, text=True, timeout=180))
        if manifest.get('config', {}).get('digest') != item['image_id']:
            raise RuntimeError('Registry image differs from the preflight pin: ' + task_id)
        subprocess.run([str(CRANE), 'pull', '--cache_path', str(SWE / 'experiment/layer-cache'),
            reference, str(archive)], env=env, stdout=log, stderr=log, check=True, timeout=3600)
        with tarfile.open(archive) as tar:
            saved = json.load(tar.extractfile('manifest.json'))
            assert len(saved) == 1
            config = saved[0]['Config']
            import hashlib
            config_digest = 'sha256:' + hashlib.sha256(tar.extractfile(config).read()).hexdigest()
            assert config_digest == item['image_id'], 'Archive image ID changed'
        subprocess.run(['docker', 'load', '-i', str(archive)], stdout=log, stderr=log,
                       check=True, timeout=3600)
        actual = subprocess.check_output(['docker', 'image', 'inspect', item['image_id'],
            '--format', '{{.Id}}'], text=True).strip()
        assert actual == item['image_id']
        archive.unlink()
        return {'task_id': task_id, 'image_id': actual, 'registry_digest': digest,
                'restored_at': time.time()}


def main():
    with (RUN / 'image_restoration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        images = json.loads((RUN / 'images.json').read_text())
        missing = inventory(images)
        state = {'started_at': time.time(), 'initial_missing': missing,
                 'restored': {}, 'errors': {}, 'phase': 'restoring'}
        write_json(RUN / 'image_restoration_status.json', state)
        ordered = sorted(missing.items(), key=lambda kv: 'b65e76293a917' not in kv[0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(restore, task, item): task for task, item in ordered}
            for future in as_completed(pending):
                task = pending[future]
                try:
                    state['restored'][task] = future.result()
                    print('RESTORED', task, flush=True)
                except Exception as exc:
                    state['errors'][task] = type(exc).__name__ + ': ' + str(exc)
                    print('RESTORE_FAILED', task, type(exc).__name__, flush=True)
                state['updated_at'] = time.time()
                write_json(RUN / 'image_restoration_status.json', state)
        state['remaining_missing'] = inventory(images)
        state['phase'] = 'complete' if not state['remaining_missing'] else 'incomplete'
        write_json(RUN / 'image_restoration_status.json', state)
        return 0 if state['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
