from pathlib import Path

import pytest

from tests.helpers import make_fake_agent


@pytest.fixture
def toy_dataset_root() -> Path:
    return Path(__file__).parent / "fixtures" / "toy_dataset"


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def fake_agent_default_scripts():
    return make_fake_agent("good")
