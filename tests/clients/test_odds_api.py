"""Tests for OddsApiClient — events, odds, league slugs, WebSocket."""

import json
import os
from pathlib import Path

import pytest

from src.clients.odds_api import LEAGUE_SLUGS, OddsApiClient

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _have_odds_api_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


# ── Unit tests (no API calls) ─────────────────────────────


def test_league_slugs_has_8_leagues():
    """Verify all 8 leagues are mapped."""
    assert len(LEAGUE_SLUGS) == 8
    assert LEAGUE_SLUGS["1204"] == "england-premier-league"
    assert LEAGUE_SLUGS["1440"] == "usa-mls"


def test_fixture_events_structure():
    """Verify saved events fixture has expected fields."""
    with open(FIXTURE_DIR / "odds_api_events.json") as f:
        events = json.load(f)
    assert isinstance(events, list)
    assert len(events) > 0
    evt = events[0]
    assert "id" in evt
    assert "home" in evt
    assert "away" in evt
    assert "date" in evt
    assert "league" in evt


def test_fixture_odds_structure():
    """Verify saved odds fixture has bookmakers data."""
    with open(FIXTURE_DIR / "odds_api_odds.json") as f:
        odds = json.load(f)
    assert isinstance(odds, dict)
    assert "bookmakers" in odds
    assert "id" in odds


# ── Live API tests (require ODDS_API_KEY) ─────────────────


@pytest.mark.asyncio
async def test_odds_api_events():
    """Fetch upcoming EPL events."""
    if not _have_odds_api_key():
        pytest.skip("ODDS_API_KEY not set")

    from src.common.config import Config
    config = Config.from_env()
    client = OddsApiClient(api_key=config.odds_api_key)
    try:
        events = await client.get_events("england-premier-league")
        assert isinstance(events, list)
        # May be empty if no upcoming matches, but should not crash
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_odds_api_odds():
    """Fetch odds for an EPL event (if available)."""
    if not _have_odds_api_key():
        pytest.skip("ODDS_API_KEY not set")

    from src.common.config import Config
    config = Config.from_env()
    client = OddsApiClient(api_key=config.odds_api_key)
    try:
        events = await client.get_events("england-premier-league")
        if events:
            odds = await client.get_odds(
                events[0]["id"], bookmakers="Bet365,Betfair Exchange",
            )
            assert isinstance(odds, dict)
    finally:
        await client.close()
