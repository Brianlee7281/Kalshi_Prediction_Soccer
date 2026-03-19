"""Tests for src/execution/kelly_sizer.py."""

from __future__ import annotations

import pytest

from src.common.types import MarketProbs, Signal, TickPayload
from src.execution.kelly_sizer import (
    apply_baker_mchale_shrinkage,
    apply_surprise_multiplier,
    compute_kelly_fraction,
    size_position,
)


def _make_payload(
    order_allowed: bool = True,
    home_win: float = 0.50,
    draw: float = 0.25,
    away_win: float = 0.25,
    over_25: float = 0.50,
    btts_yes: float = 0.40,
    mu_H: float = 1.0,
    mu_A: float = 0.8,
    ekf_P_H: float = 0.10,
    ekf_P_A: float = 0.10,
    surprise_score: float = 0.0,
) -> TickPayload:
    """Build a valid TickPayload with sensible defaults."""
    return TickPayload(
        match_id="test_match",
        t=45.0,
        engine_phase="SECOND_HALF",
        P_model=MarketProbs(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            over_25=over_25,
            btts_yes=btts_yes,
        ),
        sigma_MC=MarketProbs(
            home_win=0.003,
            draw=0.003,
            away_win=0.003,
            over_25=0.003,
            btts_yes=0.003,
        ),
        score=(1, 0),
        X=0,
        delta_S=1,
        mu_H=mu_H,
        mu_A=mu_A,
        a_H_current=0.3,
        a_A_current=-0.1,
        ekf_P_H=ekf_P_H,
        ekf_P_A=ekf_P_A,
        surprise_score=surprise_score,
        order_allowed=order_allowed,
        cooldown=False,
        ob_freeze=False,
        event_state="IDLE",
    )


def _make_signal(
    p_model: float = 0.62,
    p_kalshi: float = 0.55,
    market_type: str = "home_win",
    direction: str = "BUY_YES",
) -> Signal:
    """Build a minimal Signal for testing."""
    return Signal(
        match_id="test_match",
        ticker="TICKER-HOME",
        market_type=market_type,
        direction=direction,
        P_kalshi=p_kalshi,
        P_model=p_model,
        EV=abs(p_model - p_kalshi),
        kelly_fraction=0.0,
        kelly_amount=0.0,
        contracts=0,
    )


# ── compute_kelly_fraction ────────────────────────────────────


def test_kelly_fraction_basic():
    f = compute_kelly_fraction(0.62, 0.55)
    assert f == pytest.approx(0.155, abs=0.01)


def test_kelly_fraction_no_edge():
    f = compute_kelly_fraction(0.50, 0.55)
    assert f == 0.0


def test_kelly_fraction_degenerate():
    f = compute_kelly_fraction(0.62, 0.0)
    assert f == 0.0


# ── apply_baker_mchale_shrinkage ──────────────────────────────


def test_baker_mchale_full_shrink():
    # sigma_p == edge → shrinkage factor = 0
    result = apply_baker_mchale_shrinkage(0.15, 0.62, 0.55, 0.07)
    assert result == 0.0


def test_baker_mchale_partial_shrink():
    # sigma_p=0.04, edge=0.07 → edge_sq=0.0049, sigma_p_sq=0.0016
    # shrinkage = 1 - 0.0016/0.0049 ≈ 0.6735
    result = apply_baker_mchale_shrinkage(0.15, 0.62, 0.55, 0.04)
    expected_shrinkage = 1.0 - (0.04**2) / (0.07**2)
    assert result == pytest.approx(0.15 * expected_shrinkage, abs=0.001)


# ── apply_surprise_multiplier ─────────────────────────────────


def test_surprise_multiplier_neutral():
    result = apply_surprise_multiplier(1.0, 0.0)
    assert result == pytest.approx(0.10)


def test_surprise_multiplier_high():
    result = apply_surprise_multiplier(1.0, 0.70)
    assert result == pytest.approx(0.275)


# ── size_position ─────────────────────────────────────────────


def test_size_position_hard_caps():
    # Extreme edge → kelly wants more than $50
    signal = _make_signal(p_model=0.95, p_kalshi=0.10)
    payload = _make_payload(
        home_win=0.95, surprise_score=1.0, ekf_P_H=0.001, mu_H=0.1
    )
    sized = size_position(signal, payload, bankroll=1000.0)
    assert sized.kelly_amount <= 50.0


def test_size_position_per_match_cap():
    # bankroll=100 → per_match_cap = $10
    signal = _make_signal(p_model=0.95, p_kalshi=0.10)
    payload = _make_payload(
        home_win=0.95, surprise_score=1.0, ekf_P_H=0.001, mu_H=0.1
    )
    sized = size_position(signal, payload, bankroll=100.0)
    assert sized.kelly_amount <= 10.0
