"""Tests for kalshi_live_poller pure functions.

Tests _detect_events_from_state and _kalshi_half_to_phase only.
The async poller loop is not tested here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.clients.kalshi_live_data import MatchState
from src.engine.kalshi_live_poller import (
    _detect_events_from_state,
    _kalshi_half_to_phase,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model(
    last_score: tuple[int, int] = (0, 0),
    t: float = 30.0,
    last_period: str = "FIRST_HALF",
) -> SimpleNamespace:
    """Minimal LiveMatchModel stub for pure-function tests."""
    return SimpleNamespace(
        _last_score=last_score,
        t=t,
        _last_period=last_period,
        # _processed_red_cards intentionally absent — created on first use
    )


def _make_state(
    home_score: int = 0,
    away_score: int = 0,
    half: str = "1st",
    minute: int = 30,
    stoppage: int = 0,
    status: str = "live",
    significant_events: list[dict] | None = None,
    last_play_ts: int | None = None,
    last_play_desc: str | None = None,
) -> MatchState:
    return MatchState(
        status=status,
        half=half,
        minute=minute,
        stoppage=stoppage,
        home_score=home_score,
        away_score=away_score,
        last_play_ts=last_play_ts,
        last_play_desc=last_play_desc,
        significant_events=significant_events or [],
    )


# ── _kalshi_half_to_phase ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "half, expected",
    [
        ("1st", "FIRST_HALF"),
        ("2nd", "SECOND_HALF"),
        ("HT", "HALFTIME"),
        ("FT", "FINISHED"),
        ("", None),
        ("unknown", None),
    ],
)
def test_kalshi_half_to_phase(half: str, expected: str | None) -> None:
    assert _kalshi_half_to_phase(half) == expected


# ── Goal detection ────────────────────────────────────────────────────────────


def test_score_0_to_1_home_emits_one_goal() -> None:
    model = _make_model(last_score=(0, 0))
    state = _make_state(home_score=1, away_score=0, half="1st")

    events = _detect_events_from_state(model, state)

    goal_events = [e for e in events if e["type"] == "goal"]
    assert len(goal_events) == 1
    assert goal_events[0]["team"] == "home"
    assert goal_events[0]["minute"] == state.minute
    # Model score tracking updated
    assert model._last_score == (1, 0)


def test_score_0_to_2_home_emits_two_goals() -> None:
    model = _make_model(last_score=(0, 0))
    state = _make_state(home_score=2, away_score=0, half="1st")

    events = _detect_events_from_state(model, state)

    goal_events = [e for e in events if e["type"] == "goal" and e["team"] == "home"]
    assert len(goal_events) == 2
    # Sequential t offsets
    assert goal_events[1]["t"] > goal_events[0]["t"]
    assert model._last_score == (2, 0)


def test_away_goal_emits_correct_team() -> None:
    model = _make_model(last_score=(1, 0))
    state = _make_state(home_score=1, away_score=1, half="2nd")

    events = _detect_events_from_state(model, state)

    goal_events = [e for e in events if e["type"] == "goal"]
    assert len(goal_events) == 1
    assert goal_events[0]["team"] == "away"


def test_no_score_change_no_goal_events() -> None:
    model = _make_model(last_score=(1, 0))
    state = _make_state(home_score=1, away_score=0, half="1st")

    events = _detect_events_from_state(model, state)

    assert not any(e["type"] == "goal" for e in events)


# ── Period change detection ───────────────────────────────────────────────────


def test_half_change_1st_to_2nd_emits_period_change() -> None:
    model = _make_model(last_period="FIRST_HALF")
    state = _make_state(half="2nd")

    events = _detect_events_from_state(model, state)

    pc_events = [e for e in events if e["type"] == "period_change"]
    assert len(pc_events) == 1
    assert pc_events[0]["new_phase"] == "SECOND_HALF"


def test_duplicate_period_change_not_emitted() -> None:
    model = _make_model(last_period="SECOND_HALF")
    state = _make_state(half="2nd")

    events = _detect_events_from_state(model, state)

    assert not any(e["type"] == "period_change" for e in events)


def test_ht_emits_halftime_phase() -> None:
    model = _make_model(last_period="FIRST_HALF")
    state = _make_state(half="HT", status="live", minute=45)

    events = _detect_events_from_state(model, state)

    pc_events = [e for e in events if e["type"] == "period_change"]
    assert len(pc_events) == 1
    assert pc_events[0]["new_phase"] == "HALFTIME"


def test_ft_emits_finished_phase() -> None:
    model = _make_model(last_period="SECOND_HALF")
    state = _make_state(half="FT", status="finished", minute=90)

    events = _detect_events_from_state(model, state)

    pc_events = [e for e in events if e["type"] == "period_change"]
    assert len(pc_events) == 1
    assert pc_events[0]["new_phase"] == "FINISHED"


# ── Red card detection ────────────────────────────────────────────────────────


def test_red_card_in_significant_events_emitted_once() -> None:
    rc_event = {"event_type": "red_card", "team": "home", "player": "J. Doe", "time": "55'"}
    model = _make_model()
    state = _make_state(significant_events=[rc_event])

    events = _detect_events_from_state(model, state)

    rc_events = [e for e in events if e["type"] == "red_card"]
    assert len(rc_events) == 1
    assert rc_events[0]["team"] == "home"
    assert rc_events[0]["minute"] == 55


def test_red_card_not_emitted_twice_on_repoll() -> None:
    rc_event = {"event_type": "red_card", "team": "home", "player": "J. Doe", "time": "55'"}
    model = _make_model()
    state = _make_state(significant_events=[rc_event])

    # First poll — should emit
    events_first = _detect_events_from_state(model, state)
    assert any(e["type"] == "red_card" for e in events_first)

    # Second poll with same event — should NOT emit again
    events_second = _detect_events_from_state(model, state)
    assert not any(e["type"] == "red_card" for e in events_second)


def test_non_red_card_significant_event_ignored() -> None:
    model = _make_model()
    state = _make_state(
        significant_events=[
            {"event_type": "yellow_card", "team": "away", "player": "A. Smith", "time": "22'"}
        ]
    )

    events = _detect_events_from_state(model, state)

    assert not any(e["type"] == "red_card" for e in events)


def test_red_card_minute_parsed_from_time_field() -> None:
    rc_event = {"event_type": "red_card", "team": "away", "player": "B. Jones", "time": "78'"}
    model = _make_model(t=78.5)
    state = _make_state(significant_events=[rc_event], minute=78)

    events = _detect_events_from_state(model, state)

    rc_events = [e for e in events if e["type"] == "red_card"]
    assert rc_events[0]["minute"] == 78
