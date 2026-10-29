from __future__ import annotations

from rho.datasets.gaia2.dispatcher import render_dispatcher
from rho.observability import is_runtime_scratch


def test_dispatcher_template_exposes_generic_are_commands() -> None:
    script = render_dispatcher()

    for command in ("list", "schema", "call", "state", "poll", "wait"):
        assert f'"{command}"' in script


def test_dispatcher_template_is_valid_python() -> None:
    compile(render_dispatcher(), "tools/are.py", "exec")


def test_gaia2_runtime_directories_are_scratch() -> None:
    assert is_runtime_scratch(".gaia2/handle.json")
    assert is_runtime_scratch(".gaia2_state/Emails.json")
    assert is_runtime_scratch("task/.gaia2/handle.json")
    assert is_runtime_scratch("task/.gaia2_state/Emails.json")
