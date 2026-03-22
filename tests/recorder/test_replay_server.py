"""Tests for ReplayServer (data/recordings/ format)."""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest

from src.recorder.replay_server import ReplayServer, _load_jsonl


def _create_recording(tmp_path: Path) -> Path:
    """Create a minimal recording directory matching data/recordings/ format."""
    rec_dir = tmp_path / "KXEPLGAME-26MAR20TEST"
    rec_dir.mkdir()

    # kalshi_live_data.jsonl — 3 sequential poll responses (MatchState shape)
    live_records = [
        {
            "_ts": 0.5,
            "status": "live", "half": "1st", "minute": 0, "stoppage": 0,
            "home_score": 0, "away_score": 0,
            "last_play_ts": None, "last_play_desc": None,
            "significant_events": [],
        },
        {
            "_ts": 1.5,
            "status": "live", "half": "1st", "minute": 35, "stoppage": 0,
            "home_score": 1, "away_score": 0,
            "last_play_ts": 1000001, "last_play_desc": "Goal",
            "significant_events": [],
        },
        {
            "_ts": 2.5,
            "status": "live", "half": "HT", "minute": 45, "stoppage": 0,
            "home_score": 1, "away_score": 0,
            "last_play_ts": None, "last_play_desc": None,
            "significant_events": [],
        },
    ]
    with open(rec_dir / "kalshi_live_data.jsonl", "w", encoding="utf-8") as f:
        for r in live_records:
            f.write(json.dumps(r) + "\n")

    # kalshi_ob.jsonl — 2 orderbook messages
    ob_records = [
        {
            "_ts": 0.5,
            "type": "orderbook_snapshot", "sid": 1, "seq": 1,
            "msg": {"market_ticker": "KXEPLGAME-26MAR20TEST-HOME", "yes": [[45, 100]]},
        },
        {
            "_ts": 1.0,
            "type": "orderbook_delta", "sid": 1, "seq": 2,
            "msg": {"market_ticker": "KXEPLGAME-26MAR20TEST-HOME", "yes": [[47, 50]]},
        },
    ]
    with open(rec_dir / "kalshi_ob.jsonl", "w", encoding="utf-8") as f:
        for r in ob_records:
            f.write(json.dumps(r) + "\n")

    # odds_api.jsonl — 1 update
    odds_records = [
        {
            "_ts": 1.0,
            "type": "updated", "bookie": "Bet365",
            "markets": [{"name": "ML", "odds": [
                {"name": "home", "price": 2.10},
                {"name": "draw", "price": 3.40},
                {"name": "away", "price": 3.20},
            ]}],
        },
    ]
    with open(rec_dir / "odds_api.jsonl", "w", encoding="utf-8") as f:
        for r in odds_records:
            f.write(json.dumps(r) + "\n")

    # metadata.json
    with open(rec_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "event_ticker": "KXEPLGAME-26MAR20TEST",
            "home_team": "TestHome",
            "away_team": "TestAway",
            "final_score": [1, 0],
        }, f)

    return rec_dir


def test_replay_server_loads_recording(tmp_path: Path) -> None:
    """ReplayServer loads all JSONL streams from recording directory."""
    rec_dir = _create_recording(tmp_path)
    server = ReplayServer(rec_dir, speed=100.0)

    assert len(server.kalshi_live_records) == 3
    assert len(server.kalshi_ob_records) == 2
    assert len(server.odds_api_records) == 1

    # Records sorted by _ts
    assert server.kalshi_live_records[0]["_ts"] == 0.5
    assert server.kalshi_live_records[2]["_ts"] == 2.5

    # Missing files handled gracefully
    missing = _load_jsonl(tmp_path / "nonexistent.jsonl")
    assert missing == []


@pytest.mark.asyncio
async def test_replay_milestones_endpoint(tmp_path: Path) -> None:
    """ReplayServer returns a dummy milestone UUID."""
    rec_dir = _create_recording(tmp_path)
    server = ReplayServer(rec_dir, speed=100.0)
    await server.start(kalshi_live_port=18555, odds_ws_port=18556, kalshi_ws_port=18557)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:18555/trade-api/v2/milestones") as resp:
                data = await resp.json()
                assert len(data["milestones"]) == 1
                assert data["milestones"][0]["id"] == "replay-milestone-uuid"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_replay_live_data_endpoint(tmp_path: Path) -> None:
    """ReplayServer serves kalshi live data in Kalshi API response format."""
    rec_dir = _create_recording(tmp_path)
    server = ReplayServer(rec_dir, speed=100.0)
    await server.start(kalshi_live_port=18555, odds_ws_port=18556, kalshi_ws_port=18557)

    try:
        url = "http://127.0.0.1:18555/trade-api/v2/live_data/soccer/milestone/replay-uuid"
        async with aiohttp.ClientSession() as session:
            # First poll → minute 0, score 0-0
            async with session.get(url) as resp:
                data = await resp.json()
                details = data["live_data"]["details"]
                assert details["status"] == "live"
                assert details["half"] == "1st"
                assert details["time"] == "0'"
                assert details["home_same_game_score"] == 0
                # No _ts in response
                assert "_ts" not in json.dumps(data)

            # Second poll → minute 35, goal scored
            async with session.get(url) as resp:
                data = await resp.json()
                details = data["live_data"]["details"]
                assert details["time"] == "35'"
                assert details["home_same_game_score"] == 1
                assert details["last_play"]["occurence_ts"] == 1000001

            # Third poll → halftime
            async with session.get(url) as resp:
                data = await resp.json()
                details = data["live_data"]["details"]
                assert details["half"] == "HT"

            # Fourth poll → replays last record (exhausted)
            async with session.get(url) as resp:
                data = await resp.json()
                details = data["live_data"]["details"]
                assert details["half"] == "HT"
    finally:
        await server.stop()
