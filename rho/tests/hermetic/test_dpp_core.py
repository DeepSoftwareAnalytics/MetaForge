from __future__ import annotations

import numpy as np
import pytest

from rho.selection.dpp_selector import fast_greedy_map


def _verify_gains_match_slogdet(
    L: np.ndarray, picks: list[int], gains: list[float]
) -> None:
    """Each step's log-det gain must equal log det(L_R_t) - log det(L_R_{t-1})."""
    prev = 0.0
    for t, (pick, gain) in enumerate(zip(picks, gains)):
        R = picks[: t + 1]
        sign, logdet = np.linalg.slogdet(L[np.ix_(R, R)])
        assert sign > 0, f"L_R not PSD at step {t}, R={R}"
        expected_gain = logdet - prev
        assert gain == pytest.approx(expected_gain, abs=1e-6), (
            f"step {t}: gain={gain} vs expected={expected_gain}"
        )
        prev = logdet


def test_first_pick_is_argmax_diagonal() -> None:
    L = np.diag([1.0, 4.0, 2.0, 9.0, 3.0])
    picks, _ = fast_greedy_map(L, k=1)
    assert picks == [3]


def test_gains_equal_slogdet_differences_random_psd() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 5)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = np.array([0.5, 1.0, 0.2, 0.9, 0.7, 0.3, 0.8, 0.6])
    L = (r[:, None] * (X @ X.T)) * r[None, :]
    picks, gains = fast_greedy_map(L, k=5)
    assert len(picks) == 5
    assert len(set(picks)) == 5
    _verify_gains_match_slogdet(L, picks, gains)


def test_returns_early_when_marginal_gain_vanishes() -> None:
    L = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )
    picks, _ = fast_greedy_map(L, k=3)
    assert picks[0] in {0, 1}
    assert picks[1] in {2, 3}
    assert picks[2] in {2, 3}
    assert picks[1] != picks[2]


def test_k_larger_than_effective_rank_truncates() -> None:
    L = np.ones((3, 3))
    picks, gains = fast_greedy_map(L, k=3, eps=1e-9)
    assert len(picks) == 1
    assert gains == pytest.approx([0.0], abs=1e-9)


def test_gains_are_monotonically_non_increasing() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((12, 6)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = rng.uniform(0.3, 1.0, size=12)
    L = (r[:, None] * (X @ X.T)) * r[None, :]
    _, gains = fast_greedy_map(L, k=6)
    for a, b in zip(gains, gains[1:]):
        assert b <= a + 1e-9, f"greedy gain went up: {a} -> {b}"
