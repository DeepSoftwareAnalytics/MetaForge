"""Skip probes for real-API selection tests under tests/codex/.

Kept out of conftest.py because pytest treats conftest as a plugin
module; a plain helper module is the idiomatic Python answer.
"""
from __future__ import annotations

import functools
import os
import subprocess


# Cached for the duration of a pytest process — `az login` mid-run is
# rare and we'd rather avoid invoking the az CLI per-test (slow + log-
# spammy). If you `az login` during a session, restart pytest.
@functools.lru_cache(maxsize=1)
def have_azure_foundry_token() -> bool:
    try:
        proc = subprocess.run(
            [
                "az", "account", "get-access-token",
                "--resource", "https://cognitiveservices.azure.com",
                "--query", "accessToken",
                "-o", "tsv",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def have_remote_embedder_credential() -> bool:
    """True iff a remote embedding-API credential is set. Used to gate the
    opt-in real-API embedder smoke test. The Azure OpenAI East US 2 resource
    has no embedding deployment, so we only accept OpenAI / OpenRouter."""
    return any(
        os.environ.get(name) for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")
    )
