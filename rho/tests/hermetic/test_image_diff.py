import json
from types import SimpleNamespace
from rho.agent.image_diff import inspect_image_diff
import pytest


def test_identical_tracked_content_is_distinguished_from_runtime_changes(tmp_path,monkeypatch):
    def run(args,**kwargs):
        return SimpleNamespace(stdout='C /app/code.py\nA /usr/sbin/docker-init\nA /rho-home\nC /etc\n' if args[1]=='diff' else '["/etc"]')
    monkeypatch.setattr('rho.agent.image_diff.subprocess.run',run)
    source={'verified_unchanged_paths':['code.py'],'changed_paths':[]}
    result=inspect_image_diff('test',tmp_path,source)
    assert result['passed']
    assert result['change_counts']['source_metadata_only_content_and_mode_verified']==1
    assert result['change_counts']['container_init_process']==1


def test_dependency_binary_change_fails_closed(tmp_path,monkeypatch):
    def run(args,**kwargs):
        return SimpleNamespace(stdout='C /usr/local/go/bin/go\n' if args[1]=='diff' else '[]')
    monkeypatch.setattr('rho.agent.image_diff.subprocess.run',run)
    with pytest.raises(RuntimeError,match='Unexpected base-image'):
        inspect_image_diff('test',tmp_path,{'verified_unchanged_paths':[],'changed_paths':[]})
    assert not json.loads((tmp_path/'base_image_comparison.json').read_text())['passed']
