"""Tests for Phase 2 pipeline — sanity check, backsolve, tier fallback, load_production_params."""

import numpy as np
import pytest

from src.common.types import MarketProbs
from src.prematch.phase2_pipeline import (
    _fetch_pinnacle_odds,
    backsolve_intensities,
    load_production_params,
    sanity_check,
)


# ── sanity_check ───────────────────────────────────────────


def test_sanity_check_pass():
    """Model and market probabilities within threshold → GO."""
    model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    market = MarketProbs(home_win=0.48, draw=0.28, away_win=0.24)
    verdict, reason = sanity_check(model, market)
    assert verdict == "GO"
    assert reason is None


def test_sanity_check_fail():
    """Large deviation → SKIP."""
    model = MarketProbs(home_win=0.70, draw=0.15, away_win=0.15)
    market = MarketProbs(home_win=0.40, draw=0.30, away_win=0.30)
    verdict, reason = sanity_check(model, market)
    assert verdict == "SKIP"  # max deviation = 0.30 > 0.15
    assert reason is not None
    assert "0.15" in reason


def test_sanity_check_no_market():
    """No market data available → proceed with model only."""
    model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    verdict, reason = sanity_check(model, None)
    assert verdict == "GO"
    assert reason is None


# ── backsolve_intensities ──────────────────────────────────


def test_backsolve_basic():
    """Verify backsolve produces reasonable a_H, a_A."""
    odds = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    b = np.zeros(6)
    Q = np.zeros((4, 4))
    a_H, a_A = backsolve_intensities(odds, b, Q)
    # a_H, a_A should be in reasonable range for soccer intensities
    assert -6.0 < a_H < -2.0, f"a_H={a_H} out of range"
    assert -6.0 < a_A < -2.0, f"a_A={a_A} out of range"

    # Verify backsolve actually reproduces the input probabilities
    from src.prematch.phase2_pipeline import _poisson_1x2
    from src.calibration.step_1_3_ml_prior import compute_C_time

    C_time = compute_C_time(b)
    mu_H = np.exp(a_H) * C_time
    mu_A = np.exp(a_A) * C_time
    p_h, p_d, p_a = _poisson_1x2(mu_H, mu_A)
    assert abs(p_h - 0.45) < 0.02, f"p_h={p_h:.3f} != 0.45"
    assert abs(p_d - 0.30) < 0.02, f"p_d={p_d:.3f} != 0.30"
    assert abs(p_a - 0.25) < 0.02, f"p_a={p_a:.3f} != 0.25"


def test_backsolve_strong_favourite():
    """Backsolve handles heavily favoured home team."""
    odds = MarketProbs(home_win=0.70, draw=0.18, away_win=0.12)
    b = np.zeros(6)
    Q = np.zeros((4, 4))
    a_H, a_A = backsolve_intensities(odds, b, Q)

    from src.prematch.phase2_pipeline import _poisson_1x2
    from src.calibration.step_1_3_ml_prior import compute_C_time

    C_time = compute_C_time(b)
    mu_H = np.exp(a_H) * C_time
    mu_A = np.exp(a_A) * C_time
    p_h, p_d, p_a = _poisson_1x2(mu_H, mu_A)
    assert abs(p_h - 0.70) < 0.03, f"p_h={p_h:.3f} != 0.70"
    # Home intensity should be higher than away
    assert a_H > a_A


def test_backsolve_even_match():
    """Backsolve handles nearly even odds."""
    odds = MarketProbs(home_win=0.36, draw=0.30, away_win=0.34)
    b = np.zeros(6)
    Q = np.zeros((4, 4))
    a_H, a_A = backsolve_intensities(odds, b, Q)

    from src.prematch.phase2_pipeline import _poisson_1x2
    from src.calibration.step_1_3_ml_prior import compute_C_time

    C_time = compute_C_time(b)
    mu_H = np.exp(a_H) * C_time
    mu_A = np.exp(a_A) * C_time
    p_h, p_d, p_a = _poisson_1x2(mu_H, mu_A)
    assert abs(p_h - 0.36) < 0.03, f"p_h={p_h:.3f} != 0.36"
    # Near-even: intensities should be close
    assert abs(a_H - a_A) < 0.3


# ── _fetch_pinnacle_odds ───────────────────────────────────


def test_fetch_pinnacle_no_data():
    """Pinnacle lookup returns None when no CSV data available."""
    result = _fetch_pinnacle_odds(9999, "FakeTeam", "NoTeam")
    assert result is None


# ── load_production_params (requires DB) ───────────────────


@pytest.mark.asyncio
async def test_load_production_params():
    """Verify we can load EPL params from DB (saved in Sprint 1)."""
    from src.common.config import Config

    config = Config.from_env()
    params = await load_production_params(config, 1204)
    # May be None if DB is not running or no params exist
    if params is not None:
        assert "Q" in params
        assert "b" in params
        assert "version" in params
        assert isinstance(params["Q"], list)
        assert isinstance(params["b"], list)
        assert len(params["b"]) == 6
    else:
        pytest.skip("DB not available or no params for league 1204")


def test_ekf_P0_by_prediction_method():
    """Phase2Result.ekf_P0 defaults to 0.25 and can be set explicitly."""
    from src.common.types import Phase2Result

    # Default (not specified)
    r = Phase2Result(
        match_id="m1", league_id=1, a_H=-4.0, a_A=-4.2,
        mu_H=1.5, mu_A=1.1, C_time=90.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="league_mle",
    )
    assert r.ekf_P0 == 0.25  # default

    # Explicit Tier 1 value
    r2 = Phase2Result(
        match_id="m2", league_id=1, a_H=-4.0, a_A=-4.2,
        mu_H=1.5, mu_A=1.1, C_time=90.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="backsolve_odds_api",
        ekf_P0=0.15,
    )
    assert r2.ekf_P0 == 0.15
