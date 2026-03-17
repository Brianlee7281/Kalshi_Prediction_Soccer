"""Tests for GoalserveClient — live scores, find_match, upcoming fixtures."""

import json
from pathlib import Path

import pytest

from src.clients.goalserve import GoalserveClient, _is_time_string

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "goalserve_live.json"


@pytest.fixture
def live_data() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def stub_client() -> GoalserveClient:
    """Client instance without HTTP (for sync helper tests)."""
    client = GoalserveClient.__new__(GoalserveClient)
    return client


# ── find_match_in_live: must search @id, @fix_id, @static_id ──


def test_find_match_by_id(stub_client: GoalserveClient, live_data: dict) -> None:
    """Find match using @id field."""
    result = stub_client.find_match_in_live("6760767", live_data)
    assert result is not None
    assert result["localteam"]["@name"] == "Tigre"


def test_find_match_by_fix_id(stub_client: GoalserveClient, live_data: dict) -> None:
    """Find match using @fix_id — critical anti-pattern fix."""
    result = stub_client.find_match_in_live("4310960", live_data)
    assert result is not None
    assert result["@id"] == "6760767"


def test_find_match_by_static_id(stub_client: GoalserveClient, live_data: dict) -> None:
    """Find match using @static_id — critical anti-pattern fix."""
    result = stub_client.find_match_in_live("3803546", live_data)
    assert result is not None
    assert result["@id"] == "6760767"


def test_find_match_not_found(stub_client: GoalserveClient, live_data: dict) -> None:
    """Return None for unknown match ID."""
    result = stub_client.find_match_in_live("9999999", live_data)
    assert result is None


# ── _is_time_string ──


def test_is_time_string() -> None:
    assert _is_time_string("15:00") is True
    assert _is_time_string("9:30") is True
    assert _is_time_string("FT") is False
    assert _is_time_string("HT") is False
    assert _is_time_string("45") is False
    assert _is_time_string("") is False


# ── Single-match (dict) vs multi-match (list) handling ──


def test_find_match_single_match_category(stub_client: GoalserveClient) -> None:
    """Handle category where 'match' is a dict (single match), not a list."""
    data = {
        "scores": {
            "category": {
                "@name": "Test",
                "@gid": "1",
                "matches": {
                    "match": {
                        "@id": "111",
                        "@fix_id": "222",
                        "@static_id": "333",
                        "localteam": {"@name": "TeamA"},
                        "visitorteam": {"@name": "TeamB"},
                    }
                },
            }
        }
    }
    result = stub_client.find_match_in_live("222", data)
    assert result is not None
    assert result["@id"] == "111"


# ── Live API test (requires GOALSERVE_API_KEY) ──


@pytest.mark.asyncio
async def test_goalserve_live_scores() -> None:
    """Fetch live scores — should return valid JSON with 'scores' key."""
    from src.common.config import Config

    config = Config.from_env()
    if not config.goalserve_api_key:
        pytest.skip("GOALSERVE_API_KEY not set")

    client = GoalserveClient(api_key=config.goalserve_api_key)
    try:
        data = await client.get_live_scores()
        assert "scores" in data
        assert "category" in data["scores"]
    finally:
        await client.close()
