import json
import socket
import subprocess
from pathlib import Path

import pytest

from rho.agent.dependency_proxy import DependencyProxy
from rho.datasets.swebench_pro.grading_environment import repair_scripts, environment_failures


def test_repairs_keep_test_selection_and_source_patch_unchanged():
    run = 'npx jest --silent "$@"\ngo test -v ./...\n'
    entry = 'git apply /workspace/patch.diff\ngit checkout abc -- importer_test.go\nbash /workspace/run_script.sh TestImport\n'
    settings = {'workers': 2, 'install_linux_headers': True,
                'test_fixture_commit': 'a' * 40,
                'test_fixtures': ['internal/ext/testdata/import_flag_complex_metadata.json']}
    new_run, new_entry = repair_scripts(run, entry, settings)
    assert new_run == run.replace('npx jest', 'npx jest --maxWorkers=2')
    assert 'export GOFLAGS="${GOFLAGS:-} -p=2"' in new_entry
    assert new_entry.index('git apply') < new_entry.index('apk add')
    assert 'bash /workspace/run_script.sh TestImport' in new_entry
    assert 'git checkout ' + 'a'*40 + ' -- internal/ext/testdata/' in new_entry
    with pytest.raises(ValueError, match='test data'):
        repair_scripts(run, entry, {**settings, 'test_fixtures': ['internal/ext/importer.go']})


def test_resource_failure_detected_but_assertion_failure_is_not(tmp_path):
    (tmp_path / 'stdout.log').write_text('FAIL TestThing: expected 1, got 2\n')
    assert environment_failures(tmp_path) == []
    (tmp_path / 'stderr.log').write_text('runtime: failed to create new OS thread (errno=11)\n')
    assert environment_failures(tmp_path)[0]['file'] == 'stderr.log'


def test_grader_static_dependency_does_not_open_other_github_urls(tmp_path):
    url = 'https://raw.githubusercontent.com/example/dependency/main/schema.json'
    body = b'{"required": ["title"]}'
    sock = tmp_path / 'proxy.sock'
    with DependencyProxy(sock, tmp_path / 'logs', static_resources={url: body}) as proxy:
        proxy.set_projects(['task-project'], [])
        def request(target):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(sock))
                client.sendall(('GET ' + target + ' HTTP/1.1\r\nHost: raw.githubusercontent.com\r\nConnection: close\r\n\r\n').encode())
                chunks = []
                while data := client.recv(4096): chunks.append(data)
                return b''.join(chunks)
        good = request(url)
        assert b'200 OK' in good and good.endswith(body)
        for forbidden in [url + '?answer=1', url.replace('schema.json', 'solution.patch')]:
            assert b'403' in request(forbidden)
    assert not proxy.policy.check(url)  # Static exception did not change solver policy.


@pytest.mark.parametrize('passed,resource_error,expected_error', [
    (False, True, 'grading_environment_failure'), (False, False, None), (True, True, None)])
def test_evaluator_separates_environment_failure_from_wrong_patch(
        tmp_path, monkeypatch, passed, resource_error, expected_error):
    from tests.hermetic.test_swebench_pro_dataset import _make_eval_assets, _row
    from rho.datasets.swebench_pro import evaluator as module
    assets = _make_eval_assets(tmp_path)
    monkeypatch.setattr(module, 'ensure_official_assets', lambda root: assets)
    def run(command, **kwargs):
        if command[:2] == ['docker', 'run']:
            workspace = Path(command[command.index('-v') + 1].split(':')[0])
            (workspace / 'stderr.log').write_text(
                'runtime: failed to create new OS thread\n' if resource_error else '')
            (workspace / 'stdout.log').write_text('assertion failed\n' if not passed else '')
            (workspace / 'output.json').write_text(json.dumps({'tests': [
                {'name': name, 'status': 'PASSED' if passed else 'FAILED'}
                for name in ['test_fix', 'test_existing']]}))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    result = module.SWEbenchProDockerEvaluator(docker_pull='never').evaluate(
        row=_row(tmp_path, 'abc'), patch='diff --git a/a b/a\n', artifacts_dir=tmp_path / 'artifacts')
    assert result.passed is passed
    assert result.details.get('error') == expected_error
