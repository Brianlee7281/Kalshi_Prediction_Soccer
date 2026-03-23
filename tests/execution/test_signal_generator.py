"""Tests for src/execution/signal_generator.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.types import MarketProbs, TickPayload
from src.execution.signal_generator import (
    _get_market_mu,
    compute_dynamic_threshold,
    compute_edge,
    generate_signals,
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
    sigma_home_win: float = 0.003,
    sigma_draw: float = 0.003,
    sigma_away_win: float = 0.003,
    sigma_over_25: float = 0.003,
    sigma_btts_yes: float = 0.003,
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
            home_win=sigma_home_win,
            draw=sigma_draw,
            away_win=sigma_away_win,
            over_25=sigma_over_25,
            btts_yes=sigma_btts_yes,
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


# ── compute_edge ──────────────────────────────────────────────


def test_compute_edge_buy_yes():
    direction, ev = compute_edge(0.62, 0.55)
    assert direction == "BUY_YES"
    assert ev == pytest.approx(0.07, abs=1e-9)


def test_compute_edge_buy_no():
    direction, ev = compute_edge(0.30, 0.45)
    assert direction == "BUY_NO"
    assert ev == pytest.approx(0.15, abs=1e-9)


def test_compute_edge_hold():
    direction, ev = compute_edge(0.50, 0.50)
    assert direction == "HOLD"
    assert ev == 0.0


def test_compute_edge_near_zero():
    direction, _ev = compute_edge(0.001, 0.001)
    assert direction == "HOLD"


# ── compute_dynamic_threshold ─────────────────────────────────


def test_dynamic_threshold_degenerate_p():
    assert compute_dynamic_threshold(0.0, 0.003, 0.1, 1.0) == 1.0
    assert compute_dynamic_threshold(1.0, 0.003, 0.1, 1.0) == 1.0


def test_dynamic_threshold_decreases_late_match():
    theta_early = compute_dynamic_threshold(0.50, 0.003, 0.25, 1.5)
    theta_late = compute_dynamic_threshold(0.50, 0.003, 0.05, 0.3)
    assert theta_early > theta_late


# ── _get_market_mu ────────────────────────────────────────────


def test_get_market_mu_home_win():
    assert _get_market_mu("home_win", 1.2, 0.8) == 1.2


def test_get_market_mu_away_win():
    assert _get_market_mu("away_win", 1.2, 0.8) == 0.8


def test_get_market_mu_draw():
    assert _get_market_mu("draw", 1.2, 0.8) == 1.2


# ── generate_signals ─────────────────────────────────────────


def test_generate_signals_empty_kalshi():
    payload = _make_payload()
    assert generate_signals(payload, {}, {}) == []


def test_generate_signals_order_not_allowed():
    payload = _make_payload(order_allowed=False, home_win=0.90)
    p_kalshi = {"home_win": 0.50}
    tickers = {"home_win": "TICKER-HOME"}
    assert generate_signals(payload, p_kalshi, tickers) == []


def test_generate_signals_skips_existing_position():
    payload = _make_payload(home_win=0.70)
    p_kalshi = {"home_win": 0.50}
    tickers = {"home_win": "TICKER-HOME"}
    pos = SimpleNamespace(market_type="home_win")
    signals = generate_signals(payload, p_kalshi, tickers, open_positions={"pos1": pos})
    home_signals = [s for s in signals if s.market_type == "home_win"]
    assert len(home_signals) == 0


def test_generate_signals_filters_below_threshold():
    # Tiny edge (0.01) should be below threshold
    payload = _make_payload(home_win=0.51, ekf_P_H=0.20, mu_H=1.5)
    p_kalshi = {"home_win": 0.50}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    assert len(signals) == 0


def test_generate_signals_post_goal_edge_passes():
    """Post-goal scenario: high ekf_P + high mu, but 8% edge should still trade.

    Uses p_kalshi=0.38 (below MAX_ENTRY_PRICE=0.50) so the price cap doesn't block.
    """
    payload = _make_payload(home_win=0.46, ekf_P_H=0.25, mu_H=1.5)
    p_kalshi = {"home_win": 0.38}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    home_signals = [s for s in signals if s.market_type == "home_win"]
    assert len(home_signals) == 1
    assert home_signals[0].direction == "BUY_YES"
    assert home_signals[0].EV == pytest.approx(0.08, abs=1e-9)


def test_dynamic_threshold_capped_post_goal():
    """Sigma_model cap prevents threshold from exceeding ~4.5%."""
    # Without cap this would give theta ~0.17; with cap ~0.045
    theta = compute_dynamic_threshold(0.50, 0.003, 0.25, 1.5)
    assert theta < 0.06  # well below the uncapped ~0.17


def test_generate_signals_produces_signal():
    payload = _make_payload(home_win=0.70, ekf_P_H=0.05, mu_H=0.5)
    p_kalshi = {"home_win": 0.50}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    assert len(signals) >= 1
    assert signals[0].direction == "BUY_YES"


# ── entry filters (MIN_ENTRY_EDGE, MAX_ENTRY_PRICE) ─────────


def test_generate_signals_filters_below_min_edge():
    """3c edge passes dynamic theta but is below 4c MIN_ENTRY_EDGE floor."""
    payload = _make_payload(home_win=0.53, ekf_P_H=0.01, mu_H=0.2)
    p_kalshi = {"home_win": 0.50}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    home_signals = [s for s in signals if s.market_type == "home_win"]
    assert len(home_signals) == 0


def test_generate_signals_filters_high_entry_price():
    """Signal at 55c entry price is filtered by MAX_ENTRY_PRICE=0.50."""
    payload = _make_payload(home_win=0.65, ekf_P_H=0.05, mu_H=0.5)
    p_kalshi = {"home_win": 0.55}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    home_signals = [s for s in signals if s.market_type == "home_win"]
    assert len(home_signals) == 0


def test_generate_signals_passes_below_max_price_with_edge():
    """45c entry with 10c edge should pass both filters."""
    payload = _make_payload(home_win=0.55, ekf_P_H=0.05, mu_H=0.5)
    p_kalshi = {"home_win": 0.45}
    tickers = {"home_win": "TICKER-HOME"}
    signals = generate_signals(payload, p_kalshi, tickers)
    home_signals = [s for s in signals if s.market_type == "home_win"]
    assert len(home_signals) == 1
    assert home_signals[0].direction == "BUY_YES"
