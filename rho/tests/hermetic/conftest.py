from __future__ import annotations

import pytest

from tests.helpers import make_fake_agent


@pytest.fixture
def hermetic_short_solve_agent(monkeypatch):
    """Replace _build_agent with a fake codex agent for short-solve probes."""
    from rho import cli as cli_mod

    def _fake_build_agent(args, *, run_dir):
        return make_fake_agent("good")

    monkeypatch.setattr(cli_mod, "_build_agent", _fake_build_agent)
