"""Tests for event handlers (Task 3.4)."""

from __future__ import annotations

import time

import numpy as np

from src.common.types import Phase2Result
from src.engine.event_handlers import (
    detect_events_from_poll,
    handle_goal,
    handle_period_change,
    handle_red_card,
)
from src.engine.model import LiveMatchModel


def _make_test_model() -> LiveMatchModel:
    """Create a minimal LiveMatchModel for testing."""
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
        "b": [0.1, 0.2, 0.15, 0.05, 0.1, -0.1],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }
    model = LiveMatchModel.from_phase2_result(result, params)
    model.engine_phase = "FIRST_HALF"
    model.t = 35.0
    model.tick_count = 100
    return model


def test_handle_goal_updates_score() -> None:
    """Goal updates score, delta_S, and event_state."""
    model = _make_test_model()
    handle_goal(model, "home", 35)
    assert model.score == (1, 0)
    assert model.delta_S == 1
    assert model.event_state == "CONFIRMED"
    # No cooldown — post-goal is when edges appear
    assert model.cooldown is False

    # Second goal (away)
    handle_goal(model, "away", 40)
    assert model.score == (1, 1)
    assert model.delta_S == 0


def test_handle_red_card_state_transition() -> None:
    """Red card triggers correct Markov state transition."""
    model = _make_test_model()
    assert model.current_state_X == 0

    # Home red: 0 → 1
    handle_red_card(model, "home", 60)
    assert model.current_state_X == 1
    assert model.cooldown is True
    assert model.cooldown_until_t == 35.5  # t=35.0 + 0.5 minute

    # Away red from state 1: 1 → 3
    model.tick_count = 200
    handle_red_card(model, "away", 70)
    assert model.current_state_X == 3

    # Test away red from state 0: 0 → 2
    model2 = _make_test_model()
    handle_red_card(model2, "away", 55)
    assert model2.current_state_X == 2

    # Home red from state 2: 2 → 3
    handle_red_card(model2, "home", 60)
    assert model2.current_state_X == 3


def test_handle_period_change_halftime() -> None:
    """Period change to HALFTIME records halftime_start."""
    model = _make_test_model()
    model.engine_phase = "FIRST_HALF"
    handle_period_change(model, "HALFTIME")
    assert model.engine_phase == "HALFTIME"
    assert model.halftime_start > 0
    assert model._last_period == "HALFTIME"

    # HALFTIME → SECOND_HALF computes halftime_accumulated
    ht_start = model.halftime_start
    # Simulate small time passage
    time.sleep(0.01)
    handle_period_change(model, "SECOND_HALF")
    assert model.engine_phase == "SECOND_HALF"
    assert model.halftime_accumulated > 0
    assert model._last_period == "SECOND_HALF"


def test_period_change_spam_prevention() -> None:
    """Same period re-emitted should be ignored."""
    model = _make_test_model()
    model.engine_phase = "FIRST_HALF"
    model._last_period = "FIRST_HALF"
    handle_period_change(model, "FIRST_HALF")
    # Should NOT change anything — same period
    assert model.engine_phase == "FIRST_HALF"
    assert model.halftime_start == 0.0  # not touched


def test_multi_goal_detection() -> None:
    """Score jumping by 2+ in one poll emits sequential goal events."""
    model = _make_test_model()
    model._last_score = (0, 0)
    model.t = 35.0
    poll_data = {
        "localteam": {"@goals": "2"},
        "visitorteam": {"@goals": "0"},
        "@status": "35",
    }
    events = detect_events_from_poll(model, poll_data)

    # Should have 2 goal events (both home)
    goal_events = [e for e in events if e["type"] == "goal"]
    assert len(goal_events) == 2
    assert all(e["team"] == "home" for e in goal_events)

    # _last_score should be updated
    assert model._last_score == (2, 0)

    # Mixed multi-goal: 0-0 → 1-2
    model2 = _make_test_model()
    model2._last_score = (0, 0)
    model2.t = 60.0
    poll_data2 = {
        "localteam": {"@goals": "1"},
        "visitorteam": {"@goals": "2"},
        "@status": "60",
    }
    events2 = detect_events_from_poll(model2, poll_data2)
    goal_events2 = [e for e in events2 if e["type"] == "goal"]
    assert len(goal_events2) == 3  # 1 home + 2 away
    home_goals = [e for e in goal_events2 if e["team"] == "home"]
    away_goals = [e for e in goal_events2 if e["team"] == "away"]
    assert len(home_goals) == 1
    assert len(away_goals) == 2


def test_handle_penalty() -> None:
    """Penalty sets PENALTY_PENDING + ob_freeze."""
    from src.engine.event_handlers import handle_penalty
    model = _make_test_model()
    handle_penalty(model, "home", 75)
    assert model.event_state == "PENALTY_PENDING"
    assert model.ob_freeze is True
    assert model.order_allowed is False


def test_handle_var_review() -> None:
    """VAR review sets VAR_REVIEW + ob_freeze."""
    from src.engine.event_handlers import handle_var_review
    model = _make_test_model()
    handle_var_review(model, 80)
    assert model.event_state == "VAR_REVIEW"
    assert model.ob_freeze is True
    assert model.order_allowed is False
