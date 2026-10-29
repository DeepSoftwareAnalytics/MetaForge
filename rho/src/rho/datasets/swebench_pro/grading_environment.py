"""Explicit, auditable grader repairs; no solver source or assertions change."""
import re
import shlex


def repair_scripts(run_script, entryscript, settings):
    if not settings:
        return run_script, entryscript
    workers = settings.get('workers', 2)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Invalid grader worker count')
    run_script = re.sub(r'\bnpx jest\b', f'npx jest --maxWorkers={workers}', run_script)
    prelude = (f'export GOMAXPROCS={workers}\n'
               f'export GOFLAGS="${{GOFLAGS:-}} -p={workers}"\n'
               f'export PYTEST_XDIST_AUTO_NUM_WORKERS={workers}\n')
    setup = []
    if settings.get('install_linux_headers'):
        setup.append('''if [ ! -f /usr/include/linux/hidraw.h ]; then
  if command -v apk >/dev/null; then
    apk add --no-cache linux-headers || exit 91
  elif command -v apt-get >/dev/null; then
    apt-get update && apt-get install -y --no-install-recommends linux-libc-dev || exit 91
  else
    echo 'No supported package manager for missing Linux headers' >&2; exit 91
  fi
fi
test -f /usr/include/linux/hidraw.h || exit 91''')
    fixtures = settings.get('test_fixtures', [])
    if fixtures:
        commit = settings['test_fixture_commit']
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Test fixture commit must be pinned')
        for path in fixtures:
            if '..' in path.split('/') or not re.fullmatch(r'internal/ext/testdata/[\w.-]+\.(json|yml)', path):
                raise ValueError('Only declared test data may be restored')
        setup.append('git checkout ' + commit + ' -- ' + ' '.join(map(shlex.quote, fixtures)) + ' || exit 92')
    # The source patch and official test checkout precede these repairs.
    needle = 'bash /workspace/run_script.sh '
    if entryscript.count(needle) != 1:
        raise ValueError('Unexpected official grader entry script')
    entryscript = prelude + entryscript.replace(needle, '\n'.join(setup) + '\n' + needle)
    return run_script, entryscript


ENVIRONMENT_FAILURE = re.compile(
    r'failed to create new OS thread|fatal error: newosproc|'
    r'fork/exec .*resource temporarily unavailable|spawn .* EAGAIN|'
    r'pthread_create.*(?:Resource temporarily unavailable|failed)|'
    r'fatal error: linux/hidraw\.h: No such file|'
    r'open testdata/[^\n]+: no such file or directory|'
    r'Tunnel connection failed: 403|No space left on device|'
    r'Cannot allocate memory|fatal error: out of memory', re.I)


def environment_failures(workspace):
    hits = []
    for name in ('stdout.log', 'stderr.log'):
        path = workspace / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if ENVIRONMENT_FAILURE.search(line):
                hits.append({'file': name, 'line': number, 'message': line[:500]})
                if len(hits) >= 20:
                    return hits
    return hits
