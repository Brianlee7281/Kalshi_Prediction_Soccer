"""Tests for LiveMatchModel (Task 3.1)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.common.types import Phase2Result
from src.engine.model import LiveMatchModel


def _make_params() -> dict:
    """Create a minimal production_params dict for testing."""
    return {
        "Q": [
            [-0.02, 0.01, 0.01, 0.00],
            [0.00, -0.01, 0.00, 0.01],
            [0.00, 0.00, -0.01, 0.01],
            [0.00, 0.00, 0.00, 0.00],
        ],
        "b": [0.1, 0.2, 0.15, 0.05, 0.1, -0.1, -0.05, -0.15],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }


def _make_phase2_result() -> Phase2Result:
    """Create a mock Phase2Result."""
    return Phase2Result(
        match_id="match_001",
        league_id=1,
        a_H=0.3,
        a_A=0.1,
        mu_H=1.5,
        mu_A=1.1,
        C_time=1.0,
        verdict="GO",
        skip_reason=None,
        param_version=42,
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={"home_win": "KX-EPL-ARS", "draw": "KX-EPL-DRAW"},
        market_implied=None,
        prediction_method="xgboost",
    )


def test_model_from_phase2() -> None:
    """Create LiveMatchModel from Phase2Result + params."""
    result = _make_phase2_result()
    params = _make_params()
    model = LiveMatchModel.from_phase2_result(result, params)

    # Identity
    assert model.match_id == "match_001"
    assert model.home_team == "Arsenal"
    assert model.away_team == "Chelsea"
    assert model.league_id == 1

    # Phase 2 inputs
    assert model.a_H == 0.3
    assert model.a_A == 0.1
    assert model.param_version == 42

    # Arrays loaded correctly
    assert model.b.shape == (8,)
    assert model.gamma_H.shape == (4,)
    assert model.gamma_A.shape == (4,)
    assert model.delta_H.shape == (5,)
    assert model.delta_A.shape == (5,)
    assert model.Q.shape == (4, 4)

    # basis_bounds with alpha_1=2.0: [0, 15, 30, 47, 62, 77, 87, 92, 93]
    expected_bounds = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0])
    np.testing.assert_array_almost_equal(model.basis_bounds, expected_bounds)

    # Precomputed grids populated
    assert len(model.P_grid) == 101  # 0..100
    assert len(model.P_fine_grid) == 31  # 0..30

    # P_grid[0] should be identity matrix
    np.testing.assert_array_almost_equal(model.P_grid[0], np.eye(4))

    # P_grid values should be valid probability matrices (rows sum to 1)
    for k in [1, 10, 50]:
        row_sums = model.P_grid[k].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4), decimal=10)

    # Kalshi tickers
    assert model.kalshi_tickers["home_win"] == "KX-EPL-ARS"

    # Default state
    assert model.engine_phase == "WAITING_FOR_KICKOFF"
    assert model.score == (0, 0)
    assert model.t == 0.0
    assert model.T_exp == 93.0


def test_update_time_excludes_halftime() -> None:
    """Verify model.t excludes halftime duration."""
    result = _make_phase2_result()
    params = _make_params()
    model = LiveMatchModel.from_phase2_result(result, params)

    # Simulate kickoff
    now = time.monotonic()
    model.kickoff_wall_clock = now - 2700.0  # 45 minutes ago
    model.engine_phase = "FIRST_HALF"
    model.update_time()
    assert abs(model.t - 45.0) < 0.1

    # Simulate halftime: 15 minutes of halftime accumulated
    model.engine_phase = "HALFTIME"
    model.halftime_accumulated = 900.0  # 15 min in seconds
    model.update_time()
    # During halftime, t should NOT update (stays at last value)
    assert abs(model.t - 45.0) < 0.1

    # Simulate second half: 60 minutes of wall clock elapsed, 15 min halftime
    model.kickoff_wall_clock = now - 3600.0  # 60 min wall clock
    model.halftime_accumulated = 900.0  # 15 min halftime
    model.engine_phase = "SECOND_HALF"
    model.update_time()
    # Effective time = (3600 - 900) / 60 = 45.0 minutes
    assert abs(model.t - 45.0) < 0.1

    # 75 minutes wall clock, still 15 min halftime
    model.kickoff_wall_clock = now - 4500.0  # 75 min wall clock
    model.update_time()
    # Effective time = (4500 - 900) / 60 = 60.0 minutes
    assert abs(model.t - 60.0) < 0.1


def test_order_allowed() -> None:
    """Verify order_allowed conditions."""
    result = _make_phase2_result()
    params = _make_params()
    model = LiveMatchModel.from_phase2_result(result, params)

    # Default: all clear → allowed
    assert model.order_allowed is True

    # Cooldown blocks
    model.cooldown = True
    assert model.order_allowed is False
    model.cooldown = False

    # OB freeze blocks
    model.ob_freeze = True
    assert model.order_allowed is False
    model.ob_freeze = False

    # Non-IDLE event state blocks
    model.event_state = "PRELIMINARY"
    assert model.order_allowed is False
    model.event_state = "CONFIRMED"
    assert model.order_allowed is False

    # Back to IDLE → allowed again
    model.event_state = "IDLE"
    assert model.order_allowed is True


def test_update_T_exp() -> None:
    """Dynamic T_exp: stoppage announcement updates T_exp and basis_bounds."""
    result = _make_phase2_result()
    params = _make_params()
    model = LiveMatchModel.from_phase2_result(result, params)

    assert model.T_exp == 93.0
    assert model.basis_bounds[-1] == 93.0

    # Announce 5 minutes stoppage → T_exp = 95
    model.update_T_exp(5)
    assert model.T_exp == 95.0
    assert model.basis_bounds[-1] == 95.0

    # Lower value doesn't downgrade (only increases)
    model.update_T_exp(3)
    assert model.T_exp == 95.0

    # Higher value does upgrade
    model.update_T_exp(8)
    assert model.T_exp == 98.0
    assert model.basis_bounds[-1] == 98.0


def test_T_exp_affects_mc_probabilities() -> None:
    """Leading team's win probability should DECREASE when T_exp increases."""
    from src.math.mc_core import mc_simulate_remaining

    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    gamma_H = np.array([0.0, -0.15, 0.10, -0.05])
    gamma_A = np.array([0.0, 0.10, -0.15, -0.05])
    delta_H = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    delta_A = np.array([0.10, 0.05, 0.0, -0.05, -0.10])
    Q = np.array([
        [-0.02, 0.01, 0.01, 0.0],
        [0.0, -0.01, 0.0, 0.01],
        [0.0, 0.0, -0.01, 0.01],
        [0.0, 0.0, 0.0, 0.0],
    ])
    Q_diag = -np.diag(Q).copy()
    Q_off = np.zeros_like(Q)
    for i in range(4):
        if Q_diag[i] > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = Q[i, j] / Q_diag[i]

    # Home leads 1-0 at minute 85
    def home_win_prob(T_exp: float) -> float:
        bounds = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, T_exp])
        scores = mc_simulate_remaining(
            t_now=85.0, T_end=T_exp, S_H=1, S_A=0, state=0, score_diff=1,
            a_H=-4.0, a_A=-4.0, b=b,
            gamma_H=gamma_H, gamma_A=gamma_A,
            delta_H=delta_H, delta_A=delta_A,
            Q_diag=Q_diag, Q_off=Q_off,
            basis_bounds=bounds, N=50_000, seed=42,
        )
        return float(np.sum(scores[:, 0] > scores[:, 1])) / len(scores)

    p_93 = home_win_prob(93.0)
    p_98 = home_win_prob(98.0)

    # More time → more chance for opponent to equalize → lower home win prob
    assert p_93 > p_98, (
        f"Home win prob should decrease with more stoppage time: "
        f"T=93 → {p_93:.4f}, T=98 → {p_98:.4f}"
    )
