from __future__ import annotations

import dataclasses
from typing import Any

from rho.protocols import Grade


def normalize_validation_result(
    result: Any,
    *,
    scenario_id: str,
    config: str,
) -> Grade:
    passed = _passed(result)
    return Grade(
        passed=passed,
        score=1.0 if passed else 0.0,
        details={
            "scenario_id": scenario_id,
            "config": config,
            "validation": _serialize(result),
        },
    )


def _passed(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    for attr in ("passed", "success", "ok"):
        value = getattr(result, attr, None)
        if isinstance(value, bool):
            return value
    if isinstance(result, dict):
        for key in ("passed", "success", "ok"):
            value = result.get(key)
            if isinstance(value, bool):
                return value
    return bool(result)


def _serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialize(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())
    if hasattr(value, "__dict__"):
        return _serialize(vars(value))
    return repr(value)
