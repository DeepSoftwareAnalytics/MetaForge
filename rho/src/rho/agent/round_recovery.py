"""Replay a fixed single-round experiment while existing workers finish."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
import shutil

from rho.agent.request_budget import write_json
from rho.agent.judge_output import usable_judge_output
from rho.stores.trajectory import FilesystemTrajectoryStore


class ReplayTrajectoryStore(FilesystemTrajectoryStore):
    def put(self, trajectory):
        if (self.root / trajectory.id).exists():
            previous = self.get(trajectory.id)
            for field in ('kind', 'task_id', 'harness_id', 'instructions', 'final_message',
                          'workspace_diff', 'workspace_deletions', 'exit_code', 'timed_out'):
                if getattr(previous, field) != getattr(trajectory, field):
                    raise RuntimeError('Refusing changed checkpoint: ' + trajectory.id)
            return
        return super().put(trajectory)


class RoundRecovery:
    def __init__(self, run, handoff):
        self.run = Path(run)
        self.handoff = json.loads(Path(handoff).read_text())
        self.external = {tuple(j) for j in self.handoff['draining_jobs']}
        self.old_limit = self.handoff['previous_limit']
        self.condition = threading.Condition()
        self.inflight = 0
        self.checkpoints = FilesystemTrajectoryStore(self.run / 'runtime/completed_trajectories')
        self.published = FilesystemTrajectoryStore(self.run / 'trajectories')
        self.legacy = {e['trajectory_id']: e['sample_index'] for e in
                       json.loads((self.run / 'initial_solve_recovery.json').read_text())['entries']}

    def checkpoint_map(self):
        result = {}
        priorities = {}
        accepted = set()
        for queue_path in (self.run / 'retry_queue').glob('*.json'):
            state = json.loads(queue_path.read_text())
            if state.get('state') == 'completed' and state.get('trajectory_id'):
                accepted.add(state['trajectory_id'])
        for path in (self.run / 'runtime/completed_trajectories').glob('*/meta.json'):
            meta = json.loads(path.read_text())
            if meta['exit_code'] != 0 or meta.get('timed_out'):
                continue
            published = self.published.root / meta['id'] / 'meta.json'
            is_published = published.exists()
            source = published if is_published else path
            if is_published:
                meta = json.loads(published.read_text())
            is_judge = meta['kind'] in {'evaluate', 'diagnose'}
            if is_judge and not usable_judge_output(
                    meta['kind'], (source.parent / 'final_message.txt').read_text(),
                    meta['exit_code'], meta.get('timed_out', False)):
                continue
            sample = meta.get('sample_index')
            if sample is None and meta['id'] in self.legacy:
                sample = self.legacy[meta['id']]
            key = (meta['kind'], meta['task_id'], meta['harness_id'], sample)
            priority = (0 if is_published else 1 if meta['id'] in accepted else 2,
                        path.stat().st_mtime_ns, meta['id'])
            if key in result and result[key] != meta['id']:
                if not is_judge:
                    raise RuntimeError('Ambiguous successful checkpoints for ' + str(key))
                # Judge retries are execution records, not competing solutions.
                # Prefer a published/accepted result, otherwise the first valid
                # completion. Never select according to the score's magnitude.
                if priority >= priorities[key]:
                    continue
            result[key] = meta['id']
            priorities[key] = priority
        return result

    def take(self, workspace, instructions, *, kind='solve', task_id='', harness_id='', output_schema=None, **kwargs):
        path = Path(workspace) / '.sample_index'
        key = (kind, task_id, harness_id, int(path.read_text()) if path.exists() else None)
        while True:
            tid = self.checkpoint_map().get(key)
            if tid:
                store = self.published if (self.published.root / tid).exists() else self.checkpoints
                tr = store.get(tid)
                expected = instructions
                if output_schema is not None:
                    expected += '\nYour final reply must be exactly one JSON value matching this schema, without markdown:\n' + json.dumps(output_schema)
                if tr.instructions != expected:
                    raise RuntimeError('Recovery instructions changed for ' + tid)
                if kind == 'optimize':
                    for name in tr.workspace_deletions:
                        if not name.startswith('harness/') or '..' in Path(name).parts:
                            raise RuntimeError('Unsafe optimize checkpoint path')
                        (Path(workspace) / name).unlink(missing_ok=True)
                    for name, content in tr.workspace_diff.items():
                        if not name.startswith('harness/') or '..' in Path(name).parts:
                            raise RuntimeError('Unsafe optimize checkpoint path')
                        target = Path(workspace) / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                    from rho.stores.harness import FilesystemHarnessStore, _tree_digest
                    harness_dir = Path(workspace) / 'harness'
                    digest = _tree_digest(harness_dir)
                    stored = self.run / 'harness' / ('h_' + digest[:12])
                    if not stored.exists():
                        raise RuntimeError('Recovered candidate is absent from harness store')
                    shutil.rmtree(harness_dir)
                    shutil.copytree(stored, harness_dir)
                return tr
            if key not in self.external:
                return None
            # Failed old work is eligible for a fresh attempt only after its
            # old queue wrapper has finished, avoiding simultaneous duplicates.
            queue_states = [json.loads(p.read_text()) for p in (self.run / 'retry_queue').glob('*.json')]
            if any((d.get('kind'), d.get('task_id'), d.get('harness_id'), d.get('sample_index')) == key
                   and d.get('state') == 'completed' for d in queue_states):
                raise RuntimeError('Completed old job has no usable checkpoint')
            time.sleep(2)

    @contextmanager
    def slot(self):
        with self.condition:
            while self.inflight + min(self.old_limit, len(self.external - set(self.checkpoint_map()))) >= 8:
                self.condition.wait(timeout=1)
            self.inflight += 1
        try:
            yield
        finally:
            with self.condition:
                self.inflight -= 1
                self.condition.notify_all()
