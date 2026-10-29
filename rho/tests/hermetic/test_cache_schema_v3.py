from __future__ import annotations

import pytest

from rho.selection.difficulty_selector import (
    _SCHEMA_VERSION, _result_from_cache_record,
)


def test_schema_version_is_3() -> None:
    assert _SCHEMA_VERSION == 3


def test_v2_record_rejected() -> None:
    v2_record = {
        "schema_version": 2,
        "parsed_difficulty": 5.0,
        "parsed_fingerprint": ("Failure mode partial propagation across modules with "
                               "scattered invariants requires contextual tracing of a "
                               "shared contract across multiple call sites making the "
                               "change non-local and mechanically subtle to verify."),
    }
    with pytest.raises(ValueError, match="schema_version"):
        _result_from_cache_record(v2_record, "task_x")


def test_v3_record_accepted() -> None:
    v3_record = {
        "schema_version": 3,
        "parsed_difficulty": 5.0,
        "parsed_fingerprint": ("Failure mode partial propagation across modules with "
                               "scattered invariants requires contextual tracing of a "
                               "shared contract across multiple call sites making the "
                               "change non-local and mechanically subtle to verify while "
                               "preserving ordering assumptions, boundary reconciliation, "
                               "and consistent state transitions across dependent branches."),
    }
    result = _result_from_cache_record(v3_record, "task_x")
    assert result.difficulty == 5.0


def test_judge_result_has_digest_token_estimate() -> None:
    from rho.selection.difficulty_selector import JudgeResult
    r = JudgeResult(
        task_id="t", difficulty=5.0,
        fingerprint=("Failure mode partial propagation across modules with "
                     "scattered invariants requires contextual tracing of a "
                     "shared contract across multiple call sites making the "
                     "change non-local and mechanically subtle to verify."),
        digest_token_estimate=4321,
    )
    assert r.digest_token_estimate == 4321


def test_v3_record_with_token_estimate_round_trips() -> None:
    v3_record = {
        "schema_version": 3,
        "parsed_difficulty": 4.0,
        "parsed_fingerprint": ("Failure mode partial propagation across modules with "
                               "scattered invariants requires contextual tracing of a "
                               "shared contract across multiple call sites making the "
                               "change non-local and mechanically subtle to verify while "
                               "preserving ordering assumptions, boundary reconciliation, "
                               "and consistent state transitions across dependent branches."),
        "digest_token_estimate": 5432,
    }
    result = _result_from_cache_record(v3_record, "task_y")
    assert result.digest_token_estimate == 5432
