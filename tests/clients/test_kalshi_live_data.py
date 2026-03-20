"""Tests for KalshiLiveDataClient — _parse_time_field and get_live_data parsing."""

from unittest.mock import MagicMock, patch

import pytest

from src.clients.kalshi_live_data import KalshiLiveDataClient, MatchState

# Real API response shape (no network calls made in tests)
_LIVE_DATA_FIXTURE = {
    "live_data": {
        "details": {
            "status": "live",
            "status_text": "2nd - 62'",
            "half": "2nd",
            "time": "62'",
            "home_same_game_score": 1,
            "away_same_game_score": 0,
            "home_significant_events": [{"type": "goal", "minute": 30}],
            "away_significant_events": [],
            "last_play": {
                "description": "Free kick Corinthians.",
                "occurence_ts": 1773971711,
            },
            "winner": "",
        },
        "milestone_id": "0855ee5b-1234-5678-abcd-000000000000",
        "type": "soccer_tournament_multi_leg",
    }
}

_FINISHED_FIXTURE = {
    "live_data": {
        "details": {
            "status": "finished",
            "status_text": "FT",
            "half": "2nd",
            "time": "90'",
            "home_same_game_score": 2,
            "away_same_game_score": 1,
            "home_significant_events": [],
            "away_significant_events": [],
            "last_play": None,
            "winner": "home",
        },
        "milestone_id": "0855ee5b-ft",
        "type": "soccer_tournament_multi_leg",
    }
}

_HALFTIME_FIXTURE = {
    "live_data": {
        "details": {
            "status": "live",
            "status_text": "HT",
            "half": "HT",
            "time": "45'",
            "home_same_game_score": 0,
            "away_same_game_score": 0,
            "home_significant_events": [],
            "away_significant_events": [],
            "last_play": {
                "description": "Half time.",
                "occurence_ts": 1773971800,
            },
            "winner": "",
        },
        "milestone_id": "0855ee5b-ht",
        "type": "soccer_tournament_multi_leg",
    }
}


@pytest.fixture
def client() -> KalshiLiveDataClient:
    mock_key = MagicMock()
    mock_key.sign.return_value = b"fakesignaturebytes"
    with patch.object(KalshiLiveDataClient, "_load_private_key", return_value=mock_key):
        c = KalshiLiveDataClient(api_key="test-key", private_key_path="fake/path.pem")
    return c


# ── _parse_time_field ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "time_str, expected",
    [
        ("62'", (62, 0)),
        ("90+3'", (90, 3)),
        ("45+1'", (45, 1)),
        ("", (0, 0)),
        ("0'", (0, 0)),
    ],
)
def test_parse_time_field(
    client: KalshiLiveDataClient,
    time_str: str,
    expected: tuple[int, int],
) -> None:
    assert client._parse_time_field(time_str) == expected


# ── get_live_data parsing ──────────────────────────────────────────────────────


def _make_mock_get(fixture: dict):
    """Return an async callable that yields the given fixture as a response."""
    async def _mock_get(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = fixture
        resp.raise_for_status = MagicMock()
        return resp

    return _mock_get


@pytest.mark.asyncio
async def test_get_live_data_live(
    client: KalshiLiveDataClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client._client, "get", _make_mock_get(_LIVE_DATA_FIXTURE))

    state = await client.get_live_data("0855ee5b-1234-5678-abcd-000000000000")

    assert isinstance(state, MatchState)
    assert state.status == "live"
    assert state.half == "2nd"
    assert state.minute == 62
    assert state.stoppage == 0
    assert state.home_score == 1
    assert state.away_score == 0
    assert state.last_play_ts == 1773971711
    assert state.last_play_desc == "Free kick Corinthians."
    assert len(state.significant_events) == 1
    assert state.significant_events[0]["type"] == "goal"


@pytest.mark.asyncio
async def test_get_live_data_finished(
    client: KalshiLiveDataClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client._client, "get", _make_mock_get(_FINISHED_FIXTURE))

    state = await client.get_live_data("0855ee5b-ft")

    assert state.half == "FT"
    assert state.status == "finished"
    assert state.minute == 90
    assert state.home_score == 2
    assert state.away_score == 1
    assert state.last_play_ts is None
    assert state.last_play_desc is None


@pytest.mark.asyncio
async def test_get_live_data_halftime(
    client: KalshiLiveDataClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client._client, "get", _make_mock_get(_HALFTIME_FIXTURE))

    state = await client.get_live_data("0855ee5b-ht")

    assert state.half == "HT"
    assert state.status == "live"
    assert state.minute == 45
    assert state.stoppage == 0
    assert state.last_play_ts == 1773971800
    assert state.last_play_desc == "Half time."
