"""Tests for Odds-API WS listener parsing functions (Task 3.3)."""

from __future__ import annotations

from src.engine.odds_api_listener import _odds_to_implied, _parse_odds_update


def test_parse_odds_update() -> None:
    """Parse a real Odds-API WS message."""
    msg = {
        "type": "updated",
        "bookie": "Bet365",
        "markets": [
            {
                "name": "ML",
                "odds": [
                    {"name": "home", "price": 2.10},
                    {"name": "draw", "price": 3.40},
                    {"name": "away", "price": 3.20},
                ],
            }
        ],
    }
    result = _parse_odds_update(msg)
    assert result is not None
    bookie, probs = result
    assert bookie == "Bet365"
    assert 0.99 < probs.home_win + probs.draw + probs.away_win < 1.01

    # Verify vig removed: raw implied sum would be >1.0, normalized = 1.0
    raw_sum = 1 / 2.10 + 1 / 3.40 + 1 / 3.20
    assert raw_sum > 1.0  # confirms there was vig

    # Non-update messages return None
    assert _parse_odds_update({"type": "welcome"}) is None
    assert _parse_odds_update({"type": "updated"}) is None  # no bookie
    assert _parse_odds_update({"type": "updated", "bookie": "Bet365"}) is None  # no markets

    # Missing ML market returns None
    msg_no_ml = {
        "type": "updated",
        "bookie": "Bet365",
        "markets": [{"name": "Spread", "odds": []}],
    }
    assert _parse_odds_update(msg_no_ml) is None


def test_odds_to_implied() -> None:
    """Convert decimal odds to vig-removed implied probabilities."""
    probs = _odds_to_implied(2.10, 3.40, 3.20)
    total = probs.home_win + probs.draw + probs.away_win
    assert abs(total - 1.0) < 0.01

    # Home is the favourite (lowest odds) → highest probability
    assert probs.home_win > probs.draw
    assert probs.home_win > probs.away_win

    # Even odds → equal probabilities
    probs_even = _odds_to_implied(3.0, 3.0, 3.0)
    assert abs(probs_even.home_win - 1 / 3) < 0.01
    assert abs(probs_even.draw - 1 / 3) < 0.01
    assert abs(probs_even.away_win - 1 / 3) < 0.01
