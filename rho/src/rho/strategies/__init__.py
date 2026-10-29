from __future__ import annotations

from rho.protocols import OptimizeStrategy
from rho.strategies.diagnose import DiagnoseStrategy
from rho.strategies.dynamic_cheatsheet import DynamicCheatsheetStrategy
from rho.strategies.letta_sleep import LettaSleepStrategy
from rho.strategies.query_only import QueryOnlyStrategy
from rho.strategies.trajectory import TrajectoryStrategy

OPTIMIZE_STRATEGY_CHOICES = (
    "query-only",
    "trajectory",
    "diagnosis",
    "diagnosis-no-consistency",
    "diagnosis-no-validation",
    "letta-sleep",
    "dynamic-cheatsheet",
)
DEFAULT_TRAJECTORIES_PER_TASK = 3


def build_optimize_strategy(
    name: str,
    *,
    trajectories_per_task: int = DEFAULT_TRAJECTORIES_PER_TASK,
) -> OptimizeStrategy:
    if name == "query-only":
        return QueryOnlyStrategy()
    if name == "trajectory":
        if not 1 <= trajectories_per_task <= 3:
            raise ValueError("trajectories_per_task must be in [1, 3]")
        return TrajectoryStrategy(trajectories_per_task=trajectories_per_task)
    if name == "diagnosis":
        return DiagnoseStrategy()
    if name == "diagnosis-no-consistency":
        return DiagnoseStrategy(include_consistency=False)
    if name == "diagnosis-no-validation":
        return DiagnoseStrategy(include_validation=False)
    if name == "letta-sleep":
        return LettaSleepStrategy()
    if name == "dynamic-cheatsheet":
        return DynamicCheatsheetStrategy()
    raise ValueError(f"unknown optimize strategy: {name}")


__all__ = [
    "DEFAULT_TRAJECTORIES_PER_TASK",
    "DiagnoseStrategy",
    "DynamicCheatsheetStrategy",
    "OPTIMIZE_STRATEGY_CHOICES",
    "LettaSleepStrategy",
    "QueryOnlyStrategy",
    "TrajectoryStrategy",
    "build_optimize_strategy",
]
