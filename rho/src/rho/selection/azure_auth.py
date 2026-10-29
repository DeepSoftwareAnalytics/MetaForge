from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time

AZURE_FOUNDRY_BASE_URL = os.environ.get(
    "RHO_AZURE_BASE_URL",
    "https://YOUR-RESOURCE.openai.azure.com/openai/v1",
)
_AZ_RESOURCE = os.environ.get(
    "RHO_AZURE_AAD_RESOURCE", "https://cognitiveservices.azure.com"
)
_DEFAULT_SKEW_S = 300


def _decode_jwt_exp(token: str) -> int:
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    return int(payload["exp"])


class AzureTokenProvider:
    """Thread-safe TTL cache around ``az account get-access-token``.

    Refreshes when expiry is within ``refresh_skew_seconds``. AAD tokens
    last ~60 min on this tenant; the az CLI itself caches the refresh
    token so cold-call cost is ~50ms after the first acquisition.

    Independent from Codex's own auth-command refresh (configured via
    ``refresh_interval_ms`` in ``configs/codex.azure-foundry.toml`` — 30 min).
    Codex spawns ``az`` inside its sandboxed subprocess; this provider runs
    in-process for the litellm-driven selection path. Both share the
    underlying ``az`` token cache via ``AZURE_CONFIG_DIR``, so cold-call
    cost amortizes across the two paths.
    """

    def __init__(self, refresh_skew_seconds: int = _DEFAULT_SKEW_S) -> None:
        self._token: str | None = None
        self._exp: int = 0
        self._skew = refresh_skew_seconds
        self._lock = threading.Lock()

    def get_token(self) -> str:
        with self._lock:
            now = int(time.time())
            if self._token is not None and (self._exp - now) > self._skew:
                return self._token
            self._token = self._fetch()
            self._exp = _decode_jwt_exp(self._token)
            return self._token

    @staticmethod
    def _fetch() -> str:
        proc = subprocess.run(
            [
                "az",
                "account",
                "get-access-token",
                "--resource",
                _AZ_RESOURCE,
                "--query",
                "accessToken",
                "-o",
                "tsv",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"az account get-access-token failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or 'no stderr'}"
            )
        token = proc.stdout.strip()
        if not token:
            raise RuntimeError("az account get-access-token returned an empty token")
        return token


_default_provider: AzureTokenProvider | None = None
_provider_lock = threading.Lock()


def get_azure_token() -> str:
    global _default_provider
    if _default_provider is None:
        with _provider_lock:
            if _default_provider is None:
                _default_provider = AzureTokenProvider()
    return _default_provider.get_token()


def azure_foundry_kwargs() -> dict:
    """Kwargs to pass to ``litellm.completion`` / ``litellm.embedding`` to
    route the call through Azure Azure OpenAI Foundry with an Entra Bearer."""
    return {"api_base": AZURE_FOUNDRY_BASE_URL, "api_key": get_azure_token()}


__all__ = [
    "AZURE_FOUNDRY_BASE_URL",
    "AzureTokenProvider",
    "azure_foundry_kwargs",
    "get_azure_token",
]
