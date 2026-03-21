"""Tests for MatchRecorder (Task 3.7)."""

from __future__ import annotations

import json
from pathlib import Path

from src.recorder.recorder import MatchRecorder


def test_recorder_creates_files(tmp_path: Path) -> None:
    """Recording creates JSONL files and metadata.json on finalize."""
    rec = MatchRecorder("test_match", base_dir=tmp_path)
    rec.record_odds_api({"type": "updated", "bookie": "Bet365"})
    rec.record_kalshi_live_data({"status": "live", "half": "1st", "minute": 10})
    rec.record_event({"type": "goal", "team": "home", "minute": 35})
    rec.finalize()

    match_dir = tmp_path / "test_match"
    assert (match_dir / "odds_api.jsonl").exists()
    assert (match_dir / "kalshi_live_data.jsonl").exists()
    assert (match_dir / "events.jsonl").exists()
    assert (match_dir / "metadata.json").exists()

    # Verify metadata contents
    with open(match_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["match_id"] == "test_match"
    assert meta["record_counts"]["odds_api"] == 1
    assert meta["record_counts"]["kalshi_live_data"] == 1
    assert meta["record_counts"]["events"] == 1
    assert meta["duration_s"] >= 0


def test_recorder_ts_field(tmp_path: Path) -> None:
    """Every record has a _ts field (float, monotonic relative time)."""
    rec = MatchRecorder("test_match", base_dir=tmp_path)
    rec.record_odds_api({"test": "data"})
    rec.record_odds_api({"test": "data2"})
    rec.finalize()

    with open(tmp_path / "test_match" / "odds_api.jsonl", encoding="utf-8") as f:
        line1 = json.loads(f.readline())
        line2 = json.loads(f.readline())

    assert "_ts" in line1
    assert isinstance(line1["_ts"], float)
    assert line1["_ts"] >= 0.0

    # Second record should have >= _ts of the first
    assert line2["_ts"] >= line1["_ts"]

    # Original data preserved
    assert line1["test"] == "data"
    assert line2["test"] == "data2"
