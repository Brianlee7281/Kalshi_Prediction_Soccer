"""Tests for ReplayServer (Task 3.8)."""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest

from src.recorder.replay_server import ReplayServer, _load_jsonl


def _create_recording(tmp_path: Path) -> Path:
    """Create a minimal recording directory with test data."""
    rec_dir = tmp_path / "test_match"
    rec_dir.mkdir()

    # Goalserve JSONL — 3 sequential poll responses
    gs_records = [
        {"_ts": 0.0, "@status": "1", "localteam": {"@goals": "0"}, "visitorteam": {"@goals": "0"}},
        {"_ts": 3.0, "@status": "15", "localteam": {"@goals": "0"}, "visitorteam": {"@goals": "0"}},
        {"_ts": 6.0, "@status": "35", "localteam": {"@goals": "1"}, "visitorteam": {"@goals": "0"}},
    ]
    with open(rec_dir / "goalserve.jsonl", "w") as f:
        for r in gs_records:
            f.write(json.dumps(r) + "\n")

    # Odds-API JSONL — 2 updates
    odds_records = [
        {"_ts": 1.0, "type": "updated", "bookie": "Bet365", "markets": [{"name": "ML", "odds": [
            {"name": "home", "price": 2.10}, {"name": "draw", "price": 3.40}, {"name": "away", "price": 3.20},
        ]}]},
        {"_ts": 4.0, "type": "updated", "bookie": "Betfair Exchange", "markets": [{"name": "ML", "odds": [
            {"name": "home", "price": 1.90}, {"name": "draw", "price": 3.60}, {"name": "away", "price": 3.50},
        ]}]},
    ]
    with open(rec_dir / "odds_api.jsonl", "w") as f:
        for r in odds_records:
            f.write(json.dumps(r) + "\n")

    # Metadata
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump({"match_id": "test_match", "duration_s": 10.0, "record_counts": {"goalserve": 3, "odds_api": 2}}, f)

    return rec_dir


def test_replay_server_loads_recording(tmp_path: Path) -> None:
    """Create a minimal recording, verify ReplayServer loads it."""
    rec_dir = _create_recording(tmp_path)
    server = ReplayServer(rec_dir, speed=100.0)

    assert len(server.goalserve_records) == 3
    assert len(server.odds_api_records) == 2

    # Records sorted by _ts
    assert server.goalserve_records[0]["_ts"] == 0.0
    assert server.goalserve_records[2]["_ts"] == 6.0

    # Verify _load_jsonl handles missing files
    missing = _load_jsonl(tmp_path / "nonexistent.jsonl")
    assert missing == []


@pytest.mark.asyncio
async def test_replay_goalserve_endpoint(tmp_path: Path) -> None:
    """ReplayServer serves Goalserve responses in sequence."""
    rec_dir = _create_recording(tmp_path)
    server = ReplayServer(rec_dir, speed=100.0)
    await server.start(goalserve_port=18555, odds_ws_port=18556)

    try:
        async with aiohttp.ClientSession() as session:
            # First request → first record
            async with session.get("http://127.0.0.1:18555/soccernew/home") as resp:
                data = await resp.json()
                assert data["@status"] == "1"
                assert "_ts" not in data  # _ts stripped

            # Second request → second record
            async with session.get("http://127.0.0.1:18555/soccernew/home") as resp:
                data = await resp.json()
                assert data["@status"] == "15"

            # Third request → third record (goal scored)
            async with session.get("http://127.0.0.1:18555/soccernew/home") as resp:
                data = await resp.json()
                assert data["@status"] == "35"
                assert data["localteam"]["@goals"] == "1"

            # Fourth request → replays last record (exhausted)
            async with session.get("http://127.0.0.1:18555/soccernew/home") as resp:
                data = await resp.json()
                assert data["@status"] == "35"
    finally:
        await server.stop()
