from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rho.agent.codex_pool import (
    DEFAULT_CODEX_CONCURRENCY,
    CodexCliPool,
    _reset_global_codex_pool_for_tests,
    configure_global_codex_pool,
    global_codex_pool,
)


@pytest.fixture(autouse=True)
def _reset_pool():
    _reset_global_codex_pool_for_tests()
    yield
    _reset_global_codex_pool_for_tests()


def test_global_pool_defaults_to_thirty() -> None:
    assert global_codex_pool().max_concurrency == DEFAULT_CODEX_CONCURRENCY
    assert DEFAULT_CODEX_CONCURRENCY == 30


def test_configure_global_pool_before_use() -> None:
    configure_global_codex_pool(2)
    assert global_codex_pool().max_concurrency == 2


def test_pool_caps_concurrent_acquisitions() -> None:
    pool = CodexCliPool(max_concurrency=2)
    lock = threading.Lock()
    current = 0
    observed_max = 0

    def worker() -> None:
        nonlocal current, observed_max
        with pool.acquire():
            with lock:
                current += 1
                observed_max = max(observed_max, current)
            time.sleep(0.05)
            with lock:
                current -= 1

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(lambda _: worker(), range(5)))

    snapshot = pool.snapshot()
    assert observed_max == 2
    assert snapshot["submitted"] == 5
    assert snapshot["completed"] == 5
    assert snapshot["in_flight"] == 0
    assert snapshot["queued"] == 0
    assert snapshot["total_wait_time_s"] >= 0.0


def test_reconfigure_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        configure_global_codex_pool(0)
    with pytest.raises(ValueError, match="positive"):
        configure_global_codex_pool(-1)


def test_reconfigure_while_in_flight_raises() -> None:
    configure_global_codex_pool(1)
    pool = global_codex_pool()
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with pool.acquire():
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="in_flight=1"):
            configure_global_codex_pool(2)
    finally:
        release.set()
        thread.join(timeout=5)


def test_reconfigure_while_queued_raises() -> None:
    pool = CodexCliPool(max_concurrency=1)
    holder_entered = threading.Event()
    waiter_started = threading.Event()
    release_holder = threading.Event()
    waiter_finished = threading.Event()

    def holder() -> None:
        with pool.acquire():
            holder_entered.set()
            release_holder.wait(timeout=5)

    def waiter() -> None:
        waiter_started.set()
        with pool.acquire():
            pass
        waiter_finished.set()

    holder_thread = threading.Thread(target=holder)
    waiter_thread = threading.Thread(target=waiter)
    holder_thread.start()
    assert holder_entered.wait(timeout=5)
    waiter_thread.start()
    assert waiter_started.wait(timeout=5)

    try:
        deadline = time.monotonic() + 5
        while pool.snapshot()["queued"] != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.snapshot()["queued"] == 1
        with pytest.raises(RuntimeError, match="queued=1"):
            pool.configure(2)
    finally:
        release_holder.set()
        holder_thread.join(timeout=5)
        waiter_thread.join(timeout=5)
        assert waiter_finished.is_set()
