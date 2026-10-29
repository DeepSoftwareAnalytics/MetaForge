from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rho.selection import RandomSelector, build_selector
from rho.protocols import Grade, Harness, Task, Trajectory


@dataclass(frozen=True)
class _StubTask:
    _id: str

    @property
    def id(self) -> str:
        return self._id

    @property
    def harness(self) -> Harness:
        raise NotImplementedError

    def materialize(self, dest: Path) -> None:
        raise NotImplementedError

    def query(self) -> str:
        return f"query for {self._id}"

    def grade(self, trajectory: Trajectory, *, artifacts_dir: Path | None = None) -> Grade:
        raise NotImplementedError


def _tasks(n: int) -> list[Task]:
    return [_StubTask(_id=f"t{i:03d}") for i in range(n)]


def test_random_selector_returns_exactly_k() -> None:
    picks = RandomSelector().select(_tasks(20), k=5, seed=123)
    assert len(picks) == 5
    assert len({t.id for t in picks}) == 5


def test_random_selector_deterministic_under_same_seed() -> None:
    a = RandomSelector().select(_tasks(20), k=5, seed=7)
    b = RandomSelector().select(_tasks(20), k=5, seed=7)
    assert [t.id for t in a] == [t.id for t in b]


def test_random_selector_different_seeds_differ() -> None:
    a = RandomSelector().select(_tasks(20), k=5, seed=1)
    b = RandomSelector().select(_tasks(20), k=5, seed=2)
    assert [t.id for t in a] != [t.id for t in b]


def test_random_selector_k_larger_than_pool_returns_all() -> None:
    picks = RandomSelector().select(_tasks(3), k=10, seed=1)
    assert len(picks) == 3


def test_random_selector_seed_none_preserves_order() -> None:
    # Matches the previous run_evolution semantics: no seed => stable order.
    picks = RandomSelector().select(_tasks(5), k=3, seed=None)
    assert [t.id for t in picks] == ["t000", "t001", "t002"]


def test_random_selector_does_not_need_trajectories(tmp_path: Path) -> None:
    sel = build_selector("random", workdir=tmp_path)
    # If random had needed trajectories, the line above would raise.
    assert sel is not None
