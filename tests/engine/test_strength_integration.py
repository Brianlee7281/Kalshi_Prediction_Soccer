"""Integration tests for InPlayStrengthUpdater across the full pipeline."""

from __future__ import annotations

import asyncio

import numpy as np

from src.common.types import MarketProbs, Phase2Result, TickPayload
from src.engine.event_handlers import handle_goal
from src.engine.mc_pricing import compute_mc_prices
from src.engine.model import LiveMatchModel
from src.engine.strength_updater import InPlayStrengthUpdater


def _make_test_model() -> LiveMatchModel:
    """Create a minimal LiveMatchModel for testing (reused from test_event_handlers)."""
    result = Phase2Result(
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
        kalshi_tickers={"home_win": "KX-EPL-ARS"},
        market_implied=None,
        prediction_method="xgboost",
    )
    params = {
        "Q": [
            [-0.02, 0.01, 0.01, 0.00],
            [0.00, -0.01, 0.00, 0.01],
            [0.00, 0.00, -0.01, 0.01],
            [0.00, 0.00, 0.00, 0.00],
        ],
        "b": [0.1, 0.2, 0.15, 0.05, 0.1, -0.1, 0.08, -0.05],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }
    model = LiveMatchModel.from_phase2_result(result, params)
    model.engine_phase = "SECOND_HALF"
    model.t = 60.0
    model.tick_count = 100
    return model


def test_full_pipeline_goal_updater_mc() -> None:
    """Goal → updater runs → MC pricing still works end-to-end."""
    model = _make_test_model()
    model.t = 60.0
    model.mu_H_at_kickoff = 1.5
    model.mu_A_at_kickoff = 1.1
    model.mu_H_elapsed = 0.7
    model.mu_A_elapsed = 0.5

    handle_goal(model, "home", 60)

    assert model.a_H != 0.3  # was updated
    assert model.strength_updater is not None
    assert model.strength_updater.n_H == 1

    # MC pricing must not raise
    P_model, sigma_MC = asyncio.run(compute_mc_prices(model, N=10_000))

    assert 0.0 < P_model.home_win < 1.0
    assert abs(P_model.home_win + P_model.draw + P_model.away_win - 1.0) < 0.01


def test_mu_elapsed_tracks_correctly() -> None:
    """mu_H_elapsed = mu_H_at_kickoff - mu_H (computed by MC pricing)."""
    model = _make_test_model()
    model.mu_H_at_kickoff = 1.5
    model.mu_A_at_kickoff = 1.1

    # After MC pricing, mu_H is computed from model state.
    # We verify elapsed = kickoff - current.
    asyncio.run(compute_mc_prices(model, N=5_000))

    # mu_H was recomputed; elapsed should be kickoff minus the new value
    expected_elapsed = max(0.0, 1.5 - model.mu_H)
    assert abs(model.mu_H_elapsed - expected_elapsed) < 0.01

    # Force mu_H to 0 and recompute
    model.t = model.T_exp  # at match end, mu_H = 0
    asyncio.run(compute_mc_prices(model, N=5_000))
    assert abs(model.mu_H_elapsed - 1.5) < 0.01


def test_shrinkage_monotone_increasing() -> None:
    """Shrinkage factor grows monotonically with mu_elapsed."""
    updater = InPlayStrengthUpdater(
        a_H_init=0.3,
        a_A_init=0.1,
        sigma_a_sq=0.25,
        pre_match_home_prob=0.5,
    )

    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    shrinks = [updater._shrink_factor(mu) for mu in mu_values]

    # Strictly increasing
    for i in range(len(shrinks) - 1):
        assert shrinks[i] < shrinks[i + 1], (
            f"shrink({mu_values[i]})={shrinks[i]} >= shrink({mu_values[i+1]})={shrinks[i+1]}"
        )

    # Boundary checks
    assert shrinks[0] < 0.5   # shrink(0.1) with sigma_a_sq=0.25 → 0.1/0.35 ≈ 0.29
    assert shrinks[-1] > 0.9  # shrink(5.0) → 5.0/5.25 ≈ 0.95


def test_surprise_goal_flows_to_tick_payload() -> None:
    """Surprise away goal sets last_goal_type and appears in TickPayload."""
    model = _make_test_model()
    model.pre_match_home_prob = 0.70
    # Rebuild updater with the new home prob
    model.strength_updater = InPlayStrengthUpdater(
        a_H_init=model.a_H,
        a_A_init=model.a_A,
        sigma_a_sq=model.sigma_a ** 2,
        pre_match_home_prob=0.70,
    )
    model.mu_H_elapsed = 1.0
    model.mu_A_elapsed = 0.4

    handle_goal(model, "away", 65)

    assert model.last_goal_type == "SURPRISE"

    # Build TickPayload the same way tick_loop does
    dummy_probs = MarketProbs(home_win=0.5, draw=0.25, away_win=0.25)
    payload = TickPayload(
        match_id=model.match_id,
        t=model.t,
        engine_phase=model.engine_phase,
        odds_consensus=None,
        P_model=dummy_probs,
        sigma_MC=dummy_probs,
        P_reference=dummy_probs,
        reference_source="model",
        score=model.score,
        X=model.current_state_X,
        delta_S=model.delta_S,
        mu_H=model.mu_H,
        mu_A=model.mu_A,
        a_H_current=model.a_H,
        a_A_current=model.a_A,
        last_goal_type=model.last_goal_type,
        order_allowed=model.order_allowed,
        cooldown=model.cooldown,
        ob_freeze=model.ob_freeze,
        event_state=model.event_state,
    )

    assert payload.last_goal_type == "SURPRISE"
    assert payload.a_H_current == model.a_H
    assert payload.a_A_current == model.a_A


def test_no_strength_updater_backward_compat() -> None:
    """No updater → goal still registers, a_H unchanged."""
    model = _make_test_model()
    model.strength_updater = None
    original_a_H = model.a_H

    handle_goal(model, "home", 30)

    assert model.score == (1, 0)  # goal still registered
    assert model.a_H == original_a_H  # unchanged, no updater
