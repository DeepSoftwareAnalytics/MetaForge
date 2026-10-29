"""Explicit checkpoint recovery restricted to the unfinished initial solve phase."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

from rho.agent.request_budget import write_json
from rho.stores.trajectory import FilesystemTrajectoryStore


class InitialSolveRecovery:
    def __init__(self, run: Path, selection: Path, images: Path):
        self.run = run
        self.lock = threading.Lock()
        self.consumed = set()
        selected = json.loads(selection.read_text())['selected_task_ids']
        pinned = json.loads(images.read_text())
        # Do not restart a later phase or overwrite any published trajectory.
        if any((run / 'trajectories').iterdir()):
            raise RuntimeError('Recovery is only supported before initial trajectories are published')
        if any(p.name != 'input_harness_id' for p in (run / 'rounds').rglob('*') if p.is_file()):
            raise RuntimeError('Recovery is only supported before diagnosis and candidate generation')
        self.store = FilesystemTrajectoryStore(run / 'runtime/completed_trajectories')
        manifest_path = run / 'initial_solve_recovery.json'
        signature = {'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                     'images_sha256': hashlib.sha256(images.read_bytes()).hexdigest()}
        config = json.loads((run / 'config.json').read_text())
        if config['n_rounds'] != 1 or config['initial_harness'] is not None:
            raise RuntimeError('Recovery expects one round starting with h_empty')
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous and previous['inputs'] != signature:
            raise RuntimeError('Recovery input manifests changed')
        assigned = {(e['task_id'], e['trajectory_id']): e for e in previous['entries']} if previous else {}
        groups = defaultdict(list)
        for path in (run / 'runtime/calls').glob('*/isolation.json'):
            record = json.loads(path.read_text())
            if record['kind'] != 'solve' or record['harness_id'] != 'h_empty':
                raise RuntimeError('Found calls beyond the initial solve phase')
            if record['task_id'] not in selected or record['image_id'] != pinned[record['task_id']]['image_id']:
                raise RuntimeError('Checkpoint task/image does not match this experiment')
            if record['phase'] == 'completed':
                tr = self.store.get(record['trajectory_id'])
                if tr.exit_code == 0 and not tr.timed_out:
                    groups[record['task_id']].append(record)
            elif record['phase'] != 'failed':
                raise RuntimeError('Unfinished call found; stop or inspect it before recovery')
        entries = []
        self.lookup = {}
        for task_id, records in groups.items():
            records.sort(key=lambda r: (r['started_at'], r['call_id']))
            if len(records) > 3:
                raise RuntimeError('Ambiguous duplicate initial solve checkpoints')
            # Legacy checkpoints omitted sample_index. Only accept a complete
            # group of three, assign by launch time, and disclose this in audit.
            if any(r.get('sample_index') is None for r in records) and len(records) != 3:
                raise RuntimeError('Cannot assign an incomplete legacy sample group')
            for index, record in enumerate(records):
                tr = self.store.get(record['trajectory_id'])
                if (tr.kind, tr.task_id, tr.harness_id, tr.model, tr.reasoning_effort) != (
                    'solve', task_id, 'h_empty', config['model'], config['reasoning_effort']):
                    raise RuntimeError('Checkpoint metadata mismatch')
                if tr.exit_code != 0 or tr.timed_out:
                    raise RuntimeError('Unsuccessful checkpoint cannot be reused')
                entry = assigned.get((task_id, tr.id)) or {
                    'task_id': task_id, 'trajectory_id': tr.id, 'call_id': record['call_id'],
                    'sample_index': record.get('sample_index') if record.get('sample_index') is not None else index,
                    'sample_index_basis': 'recorded' if record.get('sample_index') is not None else 'legacy_launch_order_assignment',
                }
                key = (task_id, entry['sample_index'])
                if key in self.lookup:
                    raise RuntimeError('Duplicate recovered sample index')
                self.lookup[key] = tr
                entries.append(entry)
        backup = run / 'before_initial_recovery'
        backup.mkdir(exist_ok=True)
        for name in ('config.json', 'environment.json', 'launch.json', 'selection.json'):
            if (run / name).exists() and not (backup / name).exists():
                shutil.copy2(run / name, backup / name)
        write_json(manifest_path, {'inputs': signature, 'updated_at': time.time(), 'entries': entries,
            'reused': len(entries), 'remaining': len(selected) * 3 - len(entries),
            'legacy_note': 'Original sample indices were not persisted. Complete legacy groups are assigned 0/1/2 by launch time, without inspecting patch quality. This may change the sample-0 pairwise reference; all three trajectories remain available for diagnosis.'})

    def take(self, workspace, instructions, *, task_id='', harness_id='', kind='solve', **_unused):
        sample = Path(workspace) / '.sample_index'
        if kind != 'solve' or harness_id != 'h_empty' or not sample.exists():
            return None
        key = (task_id, int(sample.read_text()))
        with self.lock:
            tr = self.lookup.get(key)
            if tr is None or key in self.consumed:
                return None
            if tr.instructions != instructions:
                raise RuntimeError('Recovered solve instructions changed')
            self.consumed.add(key)
            with (self.run / 'reused_calls.jsonl').open('a') as f:
                f.write(json.dumps({'task_id': task_id, 'sample_index': key[1],
                                    'trajectory_id': tr.id, 'reused_at': time.time()}) + '\n')
            return tr
