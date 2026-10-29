from __future__ import annotations

import json
import time
from base64 import urlsafe_b64encode
from unittest.mock import patch

import pytest

from rho.selection.azure_auth import (
    AZURE_FOUNDRY_BASE_URL,
    AzureTokenProvider,
    azure_foundry_kwargs,
)


def _make_jwt(exp: int) -> str:
    header = urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = urlsafe_b64encode(json.dumps({"exp": exp}).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


def test_default_base_url_targets_foundry_openai_v1() -> None:
    assert AZURE_FOUNDRY_BASE_URL.endswith("/openai/v1")
    assert "services.ai.azure.com" in AZURE_FOUNDRY_BASE_URL
    assert "127.0.0.1" not in AZURE_FOUNDRY_BASE_URL


def test_token_provider_caches_within_ttl() -> None:
    tok = _make_jwt(int(time.time()) + 3600)
    calls: list[None] = []

    def fake_run(cmd, **kw):
        calls.append(None)

        class R:
            stdout = tok + "\n"
            returncode = 0
            stderr = ""

        return R()

    with patch("rho.selection.azure_auth.subprocess.run", side_effect=fake_run):
        provider = AzureTokenProvider()
        assert provider.get_token() == tok
        assert provider.get_token() == tok
    assert len(calls) == 1


def test_token_provider_refreshes_near_expiry() -> None:
    near = _make_jwt(int(time.time()) + 60)
    fresh = _make_jwt(int(time.time()) + 3600)
    outputs = iter([near, fresh])

    def fake_run(cmd, **kw):
        class R:
            stdout = next(outputs) + "\n"
            returncode = 0
            stderr = ""

        return R()

    with patch("rho.selection.azure_auth.subprocess.run", side_effect=fake_run):
        provider = AzureTokenProvider(refresh_skew_seconds=300)
        assert provider.get_token() == near
        assert provider.get_token() == fresh


def test_token_provider_raises_on_malformed_jwt() -> None:
    def fake_run(cmd, **kw):
        class R:
            stdout = "not-a-jwt\n"
            returncode = 0
            stderr = ""

        return R()

    with patch("rho.selection.azure_auth.subprocess.run", side_effect=fake_run):
        provider = AzureTokenProvider()
        with pytest.raises((IndexError, ValueError, KeyError)):
            provider.get_token()


def test_token_provider_raises_on_az_failure() -> None:
    def fake_run(cmd, **kw):
        class R:
            stdout = ""
            returncode = 1
            stderr = "ERROR: Please run 'az login'"

        return R()

    with patch("rho.selection.azure_auth.subprocess.run", side_effect=fake_run):
        provider = AzureTokenProvider()
        with pytest.raises(RuntimeError, match="az login"):
            provider.get_token()


def test_azure_foundry_kwargs_includes_base_and_key() -> None:
    tok = _make_jwt(int(time.time()) + 3600)

    def fake_run(cmd, **kw):
        class R:
            stdout = tok + "\n"
            returncode = 0
            stderr = ""

        return R()

    with patch("rho.selection.azure_auth.subprocess.run", side_effect=fake_run):
        # Reset module-level cached provider for test isolation.
        import rho.selection.azure_auth as az

        az._default_provider = None
        kw = azure_foundry_kwargs()
    assert kw["api_base"] == AZURE_FOUNDRY_BASE_URL
    assert kw["api_key"] == tok
