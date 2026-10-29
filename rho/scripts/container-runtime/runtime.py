"""Trusted container bootstrap and loopback/Unix model transport (stdlib only)."""
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import select
import shlex
import signal
import shutil
import socket
import socketserver
import subprocess
import sys
import threading


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=*', '-c', 'core.hooksPath=/dev/null', '-C', '/app', *args])


def prepare(base):
    repo = Path('/app')
    if git('rev-parse', 'HEAD').decode().strip() != base:
        raise RuntimeError('Image HEAD does not match the task base commit')
    original_tracked = [p for p in git('ls-files', '-z').decode().split('\0') if p]
    def fingerprint(names):
        result = {}
        for name in names:
            path = repo / name
            if path.is_symlink(): result[name] = ['link', os.readlink(path)]
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
                result[name] = ['file', path.stat().st_mode & 0o777, digest.hexdigest()]
            elif path.is_dir(): result[name] = ['directory']
            else: result[name] = ['missing']
        return result
    image_sources = fingerprint(original_tracked)
    # Preserve the delivered image, including its build-generated tracked
    # assets/lockfile and untracked database/build state. HEAD is checked above.
    image_worktree_changes = [p for p in git('diff','--name-only','HEAD','-z').decode().split('\0') if p]
    image_untracked = [p for p in git('ls-files','--others','--exclude-standard','-z').decode().split('\0') if p]
    tracked = [name for name in git('ls-files', '-z').decode().split('\0')
               if name and ((repo / name).is_file() or (repo / name).is_symlink())]
    old_count = int(git('rev-list', '--count', 'HEAD'))
    # Remove ALL Git objects, refs and reflogs, including unreachable future commits.
    removed = []
    for root in ['/app', '/root', '/opt', '/workspace', '/home', '/go']:
        if not Path(root).exists():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            if '.git' in dirs:
                target = Path(directory) / '.git'
                if target.is_symlink(): target.unlink()
                else: shutil.rmtree(target)
                dirs.remove('.git')
                removed.append(str(target))
            if '.git' in files:
                (Path(directory) / '.git').unlink()
    # Build commands can disclose fixing commit IDs; caches under /tmp are hidden
    # by a fresh tmpfs. Preserve installed Python/Go/Node tools and dependencies.
    for name in ['/build.sh', '/preprocess.sh', '/root/.bash_history', '/root/.python_history',
                 '/root/.gitconfig', '/root/.git-credentials']:
        p = Path(name)
        if p.exists() or p.is_symlink(): p.unlink()
    # Go VCS caches can retain repository history. Installed module sources and
    # version-pinned dependency archives remain available for offline builds.
    for name in ['/go/pkg/mod/cache/vcs', '/root/go/pkg/mod/cache/vcs']:
        p = Path(name)
        if p.exists(): shutil.rmtree(p)
    suspicious = []
    for root in ['/app', '/root', '/opt', '/workspace', '/home']:
        for directory, dirs, files in os.walk(root, followlinks=False):
            # Package caches are legitimate; exact benchmark artifact names are not.
            for name in files:
                if name.lower() in {'test_patch.diff', 'gold.patch', 'gold_patch.diff',
                                    'ground_truth.json', 'preds.json', 'predictions.jsonl'}:
                    suspicious.append(str(Path(directory) / name))
    if suspicious:
        raise RuntimeError('Unexpected benchmark artifacts in image: ' + repr(suspicious))
    git('init', '--quiet')
    git('add', '-A')
    # Some base-tracked fixtures match .gitignore; preserve their tracked status.
    subprocess.run(['git', '-C', '/app', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'],
                   input=('\0'.join(tracked) + '\0').encode(), check=True)
    git('-c', 'user.name=Base snapshot', '-c', 'user.email=base@localhost',
        '-c', 'commit.gpgsign=false', 'commit', '--quiet', '--allow-empty', '-m', 'Base snapshot')
    assert int(git('rev-list', '--all', '--count')) == 1
    assert not git('remote').strip()
    link = Path('/workspace/task/repo')
    if link.exists(): shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to('/app', target_is_directory=True)
    Path('/rho-home').mkdir(exist_ok=True)
    Path('/rho-home/.codex').mkdir(exist_ok=True)
    # Login shells reset PATH in Debian's /etc/profile. Preserve the original
    # image toolchain PATH for Codex's bash -lc calls (Go, Node, Python, etc.).
    Path('/etc/profile.d/rho-image-path.sh').write_text('export PATH=' + shlex.quote(os.environ['PATH']) + '\n')
    projects, modules = set(), set()
    root_module = None
    if (repo / 'go.mod').exists():
        match = re.search(r'^module\s+([^\s]+)', (repo / 'go.mod').read_text(), re.M)
        if match: root_module = match[1]
    for name in tracked:
        if set(Path(name).parts) & {'vendor', 'node_modules', '.venv', 'venv', 'third_party', 'third-party', 'testdata', 'fixtures'}:
            continue
        if Path(name).name == 'package.json':
            try:
                package = json.loads((repo / name).read_text()).get('name')
                if package: projects.add(package)
            except (ValueError, OSError): pass
        elif Path(name).name == 'go.mod':
            match = re.search(r'^module\s+([^\s]+)', (repo / name).read_text(), re.M)
            if match and root_module and (match[1] == root_module or match[1].startswith(root_module + '/')):
                modules.add(match[1])
        elif Path(name).parent == Path('.') and Path(name).name in {'pyproject.toml', 'setup.cfg', 'setup.py'}:
            text = (repo / name).read_text(errors='replace')
            projects.update(re.findall(r'\bname\s*=\s*[\x22\x27]([\w.-]+)[\x22\x27]', text))
    prepared_sources = fingerprint(original_tracked)
    source_changes = [p for p in original_tracked if image_sources[p] != prepared_sources[p]]
    if source_changes:
        raise RuntimeError('Environment preparation changed base-image tracked source: ' + repr(source_changes[:30]))
    print(json.dumps({'base_matches': True, 'snapshot_commit': git('rev-parse', 'HEAD').decode().strip(), 'original_ancestor_count': old_count,
                      'git_commits_after': 1, 'git_dirs_removed': removed,
                      'project_packages': sorted(projects), 'project_modules': sorted(modules),
                      'image_untracked_files': image_untracked,
                      'source_comparison': {'tracked_paths':len(original_tracked), 'changed_paths':source_changes,
                          'image_worktree_changes_vs_base_commit':image_worktree_changes,
                          'verified_unchanged_paths':original_tracked,
                          'before_sha256':hashlib.sha256(json.dumps(image_sources,sort_keys=True).encode()).hexdigest(),
                          'after_sha256':hashlib.sha256(json.dumps(prepared_sources,sort_keys=True).encode()).hexdigest()},
                      'suspicious_artifacts': suspicious}))


def export_changes(base):
    changed = set(git('diff', '--name-only', '-z', base).decode().split('\0'))
    changed.update(git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'))
    changes, deleted = {}, []
    for name in sorted(changed - {''}):
        path = Path('/app') / name
        try:
            path.resolve().relative_to('/app')
        except ValueError:
            raise RuntimeError('Refusing to export a repository symlink outside /app')
        key = 'task/repo/' + name
        if not path.exists(): deleted.append(key)
        elif path.is_file():
            if path.stat().st_size > 16 * 1024 * 1024:
                raise RuntimeError('Unexpected oversized patch file: ' + name)
            changes[key] = base64.b64encode(path.read_bytes()).decode()
    print(json.dumps({'changes': changes, 'deleted': deleted}))


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect(self.server.socket_path)
            sockets = [self.request, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 60)
                for source in readable:
                    data = source.recv(65536)
                    if not data: return
                    (upstream if source is self.request else self.request).sendall(data)


def serve(port=18080, socket_path='/rho-model.sock'):
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = Server(('127.0.0.1', port), Relay)
    server.socket_path = socket_path
    server.serve_forever()


def dependency_setup():
    ca = Path('/rho-dependency-ca.crt').read_bytes()
    system = Path('/etc/ssl/certs/ca-certificates.crt')
    bundle = Path('/rho-dependency-bundle.crt')
    bundle.write_bytes((system.read_bytes() if system.exists() else b'') + b'\n' + ca)
    if system.exists(): system.write_bytes(bundle.read_bytes())
    env = {
        'HTTP_PROXY': 'http://127.0.0.1:18081', 'HTTPS_PROXY': 'http://127.0.0.1:18081',
        'http_proxy': 'http://127.0.0.1:18081', 'https_proxy': 'http://127.0.0.1:18081',
        'ALL_PROXY': '', 'all_proxy': '', 'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1',
        'SSL_CERT_FILE': str(bundle), 'REQUESTS_CA_BUNDLE': str(bundle), 'CURL_CA_BUNDLE': str(bundle),
        'PIP_CERT': str(bundle), 'NODE_EXTRA_CA_CERTS': str(bundle), 'npm_config_cafile': str(bundle),
        'PIP_INDEX_URL': 'https://pypi.org/simple', 'PIP_EXTRA_INDEX_URL': '',
        'npm_config_registry': 'https://registry.npmjs.org', 'npm_config_audit': 'false',
        'npm_config_fund': 'false', 'GOPROXY': 'https://proxy.golang.org',
        'GOSUMDB': 'sum.golang.org', 'GOPRIVATE': '', 'GONOPROXY': '', 'GONOSUMDB': '',
    }
    Path('/etc/profile.d/rho-dependencies.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()))
    Path('/rho-dependency-env.json').write_text(json.dumps(env))
    print(json.dumps(env))


def stop_tools():
    # Private PID namespace: leave init and its original sleep process alive,
    # stop Codex, relay and tool descendants before obtaining a stable patch.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        pid = int(entry.name)
        if pid in {1, os.getpid()}: continue
        try:
            if (entry / 'cmdline').read_bytes() == b'sleep\0infinity\0': continue
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError): pass


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare(sys.argv[2])
    elif sys.argv[1] == 'export': export_changes(sys.argv[2])
    elif sys.argv[1] == 'relay': serve()
    elif sys.argv[1] == 'dependency-relay': serve(18081, '/rho-dependency.sock')
    elif sys.argv[1] == 'dependency-setup': dependency_setup()
    elif sys.argv[1] == 'stop-tools': stop_tools()
