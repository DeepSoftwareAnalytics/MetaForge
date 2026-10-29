import importlib.util
from pathlib import Path

import pytest

from rho.local_settings import codex_binary, codex_config, configure_network


def test_external_paths_do_not_depend_on_checkout_location(tmp_path, monkeypatch):
    binary = tmp_path / 'codex'
    config = tmp_path / 'private-config.toml'
    monkeypatch.setenv('RHO_CODEX_BINARY', str(binary))
    monkeypatch.setenv('RHO_CODEX_CONFIG', str(config))
    assert codex_binary('0.163.0') == binary
    assert codex_config(tmp_path / 'different-checkout/rho') == config


def test_proxy_is_not_replaced_without_explicit_override(monkeypatch):
    monkeypatch.delenv('RHO_HTTP_PROXY', raising=False)
    monkeypatch.setenv('HTTPS_PROXY', 'http://user-proxy:8080')
    monkeypatch.setenv('NO_PROXY', 'internal.example')
    configure_network()
    assert __import__('os').environ['HTTPS_PROXY'] == 'http://user-proxy:8080'
    assert 'internal.example' in __import__('os').environ['NO_PROXY']
    assert '127.0.0.1' in __import__('os').environ['NO_PROXY']


def test_explicit_proxy_override(monkeypatch):
    monkeypatch.setenv('RHO_HTTP_PROXY', 'http://chosen-proxy:8080')
    configure_network()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        assert __import__('os').environ[key] == 'http://chosen-proxy:8080'


def test_launcher_does_not_search_parent_directory_for_credentials(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('RHO_DASHSCOPE_KEY_FILE', raising=False)
    path = Path(__file__).resolve().parents[2] / 'scripts/rho-ds.py'
    spec = importlib.util.spec_from_file_location('portable_rho_ds', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match='Set DASHSCOPE_API_KEY'):
        module.configure_environment()
