"""Machine-local settings for the vendored experiment launchers.

Values come from the environment; credentials and experiment inputs are never
part of the source distribution.
"""
import os
from pathlib import Path
import shutil


def codex_config(root: Path) -> Path:
    return Path(os.environ.get('RHO_CODEX_CONFIG', str(root / 'configs/codex.dashscope.toml'))).expanduser().resolve()


def codex_binary(version: str | None = None) -> Path:
    explicit = os.environ.get('RHO_CODEX_BINARY')
    if explicit:
        return Path(explicit).expanduser().resolve()
    if version:
        return (Path.home() / '.codex/packages/standalone/releases' /
                f'{version}-x86_64-unknown-linux-musl/bin/codex').resolve()
    found = shutil.which('codex')
    return Path(found or str(Path.home() / '.local/bin/codex')).resolve()


def configure_network() -> None:
    """Keep the caller's proxy settings unless an explicit override is supplied."""
    if os.environ.get('RHO_HTTP_PROXY'):
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            os.environ[key] = os.environ['RHO_HTTP_PROXY']
    for key in ('NO_PROXY', 'no_proxy'):
        current = os.environ.get(key, '').split(',')
        os.environ[key] = ','.join(dict.fromkeys(x for x in current + ['localhost', '127.0.0.1', '127.0.0.2'] if x))
