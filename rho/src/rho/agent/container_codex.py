"""Codex and every tool run in a fresh, network-disabled SWE-bench container.

Only four read-only bind mounts: the pinned binary, bootstrap, and per-call
Unix model/dependency sockets. No host checkout, dataset, credentials, Docker socket, or
other run directory is exposed. Official grading runs separately afterwards.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import tomllib
import uuid

from rho.agent.codex import _snapshot, _diff
from rho.agent.codex_pool import global_codex_pool
from rho.agent.request_budget import ModelRequestBudget, write_json
from rho.agent.dependency_proxy import DependencyProxy, POLICY_VERSION
from rho.agent.image_diff import inspect_image_diff
from rho.agent.judge_output import usable_judge_output
from rho.protocols import Trajectory
from rho.stores.trajectory import FilesystemTrajectoryStore

ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / 'scripts/container-runtime/runtime.py'


def command(args, *, timeout=120, input=None):
    result = subprocess.run(args, input=input, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[:3]} failed ({result.returncode}): {result.stderr.decode(errors="replace")[-2000:]}')
    return result.stdout


def workspace_archive(workspace):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as archive:
        for item in sorted(workspace.iterdir()):
            def allowed(member):
                if member.name == 'task/repo' or member.name.startswith('task/repo/'):
                    return None  # Use the repository and dependencies in the task image.
                if '.git' in Path(member.name).parts:
                    return None
                if member.issym() or member.islnk():
                    raise ValueError('Unexpected link in agent input: ' + member.name)
                return member
            archive.add(item, arcname=item.name, filter=allowed)
    return buf.getvalue()


def restore_harness(data, target):
    # Agent-controlled output cannot create links or escape the designated directory.
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe harness output member: ' + member.name)
        shutil.rmtree(target)
        target.mkdir()
        archive.extractall(target, filter='data')


def budget_identity(task_id, harness_id, *, kind='solve', sample_index=None, per_sample=False):
    if not per_sample:
        identity = task_id + '\0' + harness_id
    else:
        identity = json.dumps([kind, task_id, harness_id, sample_index])
    return hashlib.sha256(identity.encode()).hexdigest()


class ContainerCodexAgent:
    def __init__(self, codex_config_path, *, images_path, audit_root, maximum=None,
                 solve_maximum=None,
                 model=None, reasoning_effort='high', binary='codex',
                 default_timeout_s=3600, **_unused):
        self.config_path = Path(codex_config_path)
        self.config = tomllib.loads(self.config_path.read_text())
        self.provider = self.config['model_provider']
        provider = self.config['model_providers'][self.provider]
        self.upstream = provider['base_url']
        self.env_key = provider['env_key']
        self.model = model or self.config['model']
        self.reasoning_effort = reasoning_effort
        self.binary = Path(binary).resolve()
        if not self.binary.is_file(): raise FileNotFoundError(self.binary)
        self.images_path = Path(images_path)
        self.audit_root = Path(audit_root)
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.maximum = maximum
        self.solve_maximum = solve_maximum
        self.bootstrap_content = BOOTSTRAP.read_bytes()
        self.default_timeout_s = default_timeout_s
        self.cache_mode = 'off'
        self.event_lock = threading.Lock()
        self.checkpoints = FilesystemTrajectoryStore(self.audit_root / 'completed_trajectories')

    @property
    def cache_env_signature(self):
        return {'agent_class': 'ContainerCodexAgent', 'model': self.model,
                'network': 'none', 'maximum': self.maximum,
                'solve_maximum': self.solve_maximum, 'cache': 'off'}

    def run(self, workspace, instructions, *, output_schema=None, task_id='', harness_id='',
            kind='solve', timeout_s=None, env=None):
        # This semaphore covers containers, not just inference, so never start
        # dozens of idle judge containers outside the requested parallelism.
        with global_codex_pool().acquire():
            for attempt in range(3):
                tr = self._run(Path(workspace), instructions, output_schema=output_schema,
                               task_id=task_id, harness_id=harness_id, kind=kind, timeout_s=timeout_s)
                if kind not in {'evaluate', 'diagnose'}:
                    return tr
                usable = usable_judge_output(kind, tr.final_message, tr.exit_code, tr.timed_out)
                if usable: return tr
                if attempt < 2: time.sleep(30)
            raise RuntimeError(f'{kind} remained unusable after 3 isolated attempts; refusing a fabricated score')

    def _run(self, workspace, instructions, *, output_schema, task_id, harness_id, kind, timeout_s):
        images = json.loads(self.images_path.read_text())
        if kind != 'optimize' and task_id not in images:
            raise RuntimeError('No pinned local image for task: ' + task_id)
        image = images.get(task_id) or next(iter(images.values()))
        call_id = uuid.uuid4().hex
        audit = self.audit_root / 'calls' / call_id
        audit.mkdir(parents=True)
        runtime_copy = audit / 'bootstrap.py'
        runtime_copy.write_bytes(self.bootstrap_content)
        meta = workspace / '.rho'
        meta.mkdir(exist_ok=True)
        (meta / 'instructions.md').write_text(instructions)
        if output_schema is not None:
            (meta / 'output_schema.json').write_text(json.dumps(output_schema))
            instructions += '\nYour final reply must be exactly one JSON value matching this schema, without markdown:\n' + json.dumps(output_schema)
        before = _snapshot(workspace / 'harness', exclude={'.rho'}) if (workspace / 'harness').exists() else {}
        container = 'rho-clean-' + call_id[:16]
        record = {'call_id': call_id, 'kind': kind, 'task_id': task_id, 'harness_id': harness_id,
                  'container': container, 'image_id': image['image_id'], 'network': 'none',
                  'started_at': time.time(), 'phase': 'preparing'}
        sample_path = workspace / '.sample_index'
        sample_index = int(sample_path.read_text()) if sample_path.exists() else None
        record['sample_index'] = sample_index
        record['bootstrap_sha256'] = hashlib.sha256(self.bootstrap_content).hexdigest()
        write_json(audit / 'isolation.json', record)
        started = time.monotonic()
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        if kind == 'evaluate': timeout = 3600  # Explicit user setting for pairwise review.
        maximum = self.solve_maximum if kind == 'solve' and self.solve_maximum is not None else self.maximum
        limit = maximum or 1000000
        # Held-out calls reuse a durable per-task budget on retries; evolution
        # needs distinct ledgers for its three independent samples.
        budget_name = (budget_identity(task_id, harness_id, kind=kind, sample_index=sample_index,
                                      per_sample=self.solve_maximum is not None)
                       if maximum is not None else call_id)
        record.update(max_model_requests=maximum, budget_id=budget_name, timeout_seconds=timeout)
        with tempfile.TemporaryDirectory(prefix='rho-socket-', dir='/tmp') as socket_dir:
            with ModelRequestBudget(self.upstream, limit, self.audit_root / 'budgets' / (budget_name + '.json'),
                                    unix_socket=Path(socket_dir) / 'model.sock',
                                    authorization='Bearer ' + os.environ[self.env_key]) as gateway, \
                 DependencyProxy(Path(socket_dir) / 'deps.sock', audit / 'dependencies') as dependencies:
                try:
                    command(['docker', 'run', '-d', '--pull=never', '--name', container, '--hostname', 'task',
                             '--network=none', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                             '--pids-limit=1024', '--memory=12g', '--cpus=4', '--init',
                             '--tmpfs', '/tmp:rw,exec,nosuid,size=4g', '--tmpfs', '/run:rw,nosuid,size=64m',
                             '--label', 'rho.clean=true',
                             '--mount', f'type=bind,src={self.binary},dst=/rho-codex,readonly',
                             '--mount', f'type=bind,src={runtime_copy.resolve()},dst=/rho-runtime.py,readonly',
                             '--mount', f'type=bind,src={gateway.unix_socket},dst=/rho-model.sock,readonly',
                             '--mount', f'type=bind,src={dependencies.socket_path},dst=/rho-dependency.sock,readonly',
                             '--entrypoint', '/bin/sh', image['image_id'], '-c', 'exec sleep infinity'])
                    command(['docker', 'exec', container, 'mkdir', '-p', '/workspace'])
                    command(['docker', 'exec', '-i', container, 'tar', '--no-same-owner', '-xf', '-', '-C', '/workspace'],
                            input=workspace_archive(workspace), timeout=300)
                    prepared = json.loads(command(['docker', 'exec', container, 'python3', '/rho-runtime.py',
                                                   'prepare', image['base_commit']], timeout=600))
                    project_match = re.match(r'^instance_(.+?)__(.+?)-[0-9a-f]{40}', task_id)
                    projects = set(prepared['project_packages'])
                    if project_match:
                        projects.add(project_match[2])
                        if project_match[2].lower() == 'ansible': projects.add('ansible-core')
                    dependencies.set_projects(projects, prepared['project_modules'])
                    command(['docker','cp',str(dependencies.ca),container+':/rho-dependency-ca.crt'])
                    dependency_env = json.loads(command(['docker','exec',container,'python3','/rho-runtime.py','dependency-setup']))
                    command(['docker','exec','-d',container,'python3','/rho-runtime.py','dependency-relay'])
                    record['dependency_policy'] = {'version':POLICY_VERSION,'blocked_projects':sorted(projects),
                                                   'blocked_modules':prepared['project_modules']}
                    command(['docker','exec','-i',container,'sh','-c','cat > /rho-dependency-policy.json'],
                            input=json.dumps(record['dependency_policy']).encode())
                    command(['docker', 'cp', str(self.config_path), container + ':/rho-home/.codex/config.toml'])
                    record['base_comparison'] = inspect_image_diff(container,audit,prepared['source_comparison'])
                    inspection = json.loads(command(['docker', 'inspect', container]))[0]
                    expected_mounts = {'/rho-codex', '/rho-runtime.py', '/rho-model.sock', '/rho-dependency.sock'}
                    assert {m['Destination'] for m in inspection['Mounts']} == expected_mounts
                    assert all(not m['RW'] for m in inspection['Mounts'])
                    hostconfig = inspection['HostConfig']
                    assert hostconfig['NetworkMode'] == 'none' and not hostconfig['Privileged']
                    assert not hostconfig['PidMode'] and not hostconfig['IpcMode'] == 'host'
                    checks = json.loads(command(['docker', 'exec', container, 'python3', '-c',
                        'import json,os,socket; from pathlib import Path; '
                        'print(json.dumps({"host_home_absent":not Path("/mnt/sdc/home/huangy956").exists(),'
                        '"host_home_alias_absent":not Path("/home/huangy956").exists(),'
                        '"docker_socket_absent":not Path("/var/run/docker.sock").exists(),'
                        '"interfaces":os.listdir("/sys/class/net"),'
                        '"python":__import__("sys").executable}))']))
                    assert checks['host_home_absent'] and checks['host_home_alias_absent'] and checks['docker_socket_absent']
                    assert checks['interfaces'] == ['lo']
                    fixing = re.search(r'-([0-9a-f]{40})(?:-v.*)?$', task_id)
                    if fixing:
                        probe = subprocess.run(['docker','exec',container,'git','-C','/app','cat-file','-e',fixing.group(1)],
                                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
                        assert probe.returncode != 0, 'Fixing commit is accessible in solver container'
                        checks['fixing_commit_absent'] = True
                    record.update(phase='running', preparation=prepared, checks=checks,
                                  mounts=[{'destination':m['Destination'], 'read_only':not m['RW']} for m in inspection['Mounts']])
                    write_json(audit / 'isolation.json', record)
                    command(['docker', 'exec', '-d', container, 'python3', '/rho-runtime.py', 'relay'])
                    command(['docker', 'exec', container, 'python3', '-c',
                             'import socket,time\nfor _ in range(100):\n try:\n  socket.create_connection(("127.0.0.1",18080),.2).close(); break\n except OSError: time.sleep(.05)\nelse: raise RuntimeError("relay not ready")'])
                    cmd = ['docker', 'exec']
                    for key,value in dependency_env.items(): cmd += ['-e',key+'='+value]
                    cmd += ['-e', 'HOME=/rho-home', '-e', 'CODEX_HOME=/rho-home/.codex',
                           '-e', self.env_key + '=local-session-only',
                           container, '/rho-codex', 'exec', '--cd', '/workspace', '--json',
                           '--sandbox', 'danger-full-access', '--skip-git-repo-check', '--ephemeral',
                           '--output-last-message', '/workspace/.rho/last_message.txt', '-m', self.model,
                           '-c', 'shell_environment_policy.inherit="all"',
                           '-c', f'model_reasoning_effort="{self.reasoning_effort}"',
                           '-c', f'model_providers.{self.provider}.base_url="http://127.0.0.1:18080/{gateway.token}"',
                           '-c', f'model_providers.{self.provider}.supports_websockets=false']
                    # The CLI retries rate-limited requests with backoff inside
                    # the same task. Every outbound retry still consumes budget.
                    cmd += ['-c', f'model_providers.{self.provider}.request_max_retries=100',
                            '-c', f'model_providers.{self.provider}.stream_max_retries=100']
                    if output_schema is not None: cmd += ['--output-schema', '/workspace/.rho/output_schema.json']
                    cmd.append(instructions)
                    with (audit / 'stdout.jsonl').open('w') as out, (audit / 'stderr.log').open('w') as err:
                        try:
                            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout)
                            code, timed_out = proc.returncode, False
                        except subprocess.TimeoutExpired:
                            code, timed_out = -1, True
                    command(['docker','exec',container,'python3','/rho-runtime.py','stop-tools'],timeout=30)
                    final = command(['docker', 'exec', container, 'sh', '-c', 'cat /workspace/.rho/last_message.txt 2>/dev/null || true']).decode(errors='replace')
                    changes, deleted = {}, []
                    if kind == 'solve':
                        exported = json.loads(command(['docker', 'exec', container, 'python3', '/rho-runtime.py', 'export', prepared['snapshot_commit']], timeout=180))
                        changes = {k: base64.b64decode(v) for k,v in exported['changes'].items()}
                        deleted = exported['deleted']
                        # Pre-existing untracked build/database files are not
                        # production submissions, just as in the original repo.
                        artifacts = {'task/repo/'+p for p in prepared['image_untracked_files']}
                        changes = {k:v for k,v in changes.items() if k not in artifacts}
                        deleted = [k for k in deleted if k not in artifacts]
                    elif kind == 'optimize':
                        data = command(['docker', 'exec', container, 'tar', '-cf', '-', '-C', '/workspace/harness', '.'])
                        restore_harness(data, workspace / 'harness')
                        diff, removed = _diff(before, _snapshot(workspace / 'harness', exclude={'.rho'}))
                        changes = {'harness/' + k:v for k,v in diff.items()}
                        deleted = ['harness/' + k for k in removed]
                    stdout = (audit / 'stdout.jsonl').read_text(errors='replace')
                    stderr = (audit / 'stderr.log').read_text(errors='replace')
                    events = []
                    for line in stdout.splitlines():
                        try: events.append(json.loads(line))
                        except ValueError: events.append({'type':'raw_stdout','line':line})
                    accounting = {'type':'model_request_budget','maximum':maximum,
                                  'requests_sent':gateway.state['requests_sent'],
                                  'blocked_requests':gateway.state['blocked_requests']}
                    events += [accounting, {'type':'container_isolation','audit':str(audit/'isolation.json'),
                                           'network':'none','git_commits':1,'passed':True}]
                    tr = Trajectory(id='traj_'+uuid.uuid4().hex[:10], kind=kind, task_id=task_id, harness_id=harness_id,
                                    instructions=instructions, events=events, final_message=final, stdout=stdout, stderr=stderr,
                                    workspace_diff=changes, workspace_deletions=frozenset(deleted), exit_code=code,
                                    timed_out=timed_out, wall_time_s=time.monotonic()-started,
                                    model=self.model, reasoning_effort=self.reasoning_effort,
                                    sample_index=sample_index)
                    self.checkpoints.put(tr)
                    record.update(phase='completed', trajectory_id=tr.id, exit_code=code, timed_out=timed_out,
                                  request_count=gateway.state['requests_sent'], finished_at=time.time())
                    write_json(audit / 'isolation.json', record)
                    with self.event_lock:
                        with (self.audit_root/'completed_calls.jsonl').open('a') as f:
                            f.write(json.dumps(record)+'\n')
                    return tr
                except BaseException as exc:
                    record.update(phase='failed', error=type(exc).__name__+': '+str(exc), finished_at=time.time())
                    write_json(audit/'isolation.json',record)
                    raise
                finally:
                    subprocess.run(['docker','rm','-f',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
