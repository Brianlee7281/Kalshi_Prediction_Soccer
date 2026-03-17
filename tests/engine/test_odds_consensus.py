"""Tests for OddsConsensus (Task 3.2)."""

from __future__ import annotations

import time
from unittest.mock import patch

from src.common.types import MarketProbs
from src.engine.odds_consensus import OddsConsensus


def test_consensus_high_confidence() -> None:
    """2+ fresh sources → HIGH confidence."""
    oc = OddsConsensus()
    oc.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    oc.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.48, draw=0.31, away_win=0.21),
    )
    result = oc.compute_reference()
    assert result.confidence == "HIGH"
    assert result.n_fresh_sources == 2


def test_consensus_none_all_stale() -> None:
    """All sources stale → NONE confidence."""
    oc = OddsConsensus()

    # Insert sources with timestamps far in the past (>10s ago)
    stale_time = time.monotonic() - 20.0
    oc.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    oc.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.48, draw=0.31, away_win=0.21),
    )
    # Force stale by backdating last_update
    for entry in oc.sources.values():
        entry.last_update = stale_time

    result = oc.compute_reference()
    assert result.confidence == "NONE"
    assert result.n_fresh_sources == 0


def test_consensus_betfair_weighted() -> None:
    """Betfair gets 2x weight in median."""
    oc = OddsConsensus()
    oc.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.60, draw=0.25, away_win=0.15),
    )
    oc.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.40, draw=0.35, away_win=0.25),
    )
    oc.update_bookmaker(
        "1xbet",
        MarketProbs(home_win=0.42, draw=0.33, away_win=0.25),
    )
    result = oc.compute_reference()

    # Weights: Betfair=2.0, Bet365=1.0, 1xbet=1.0 (total=4.0)
    # Sorted home_win: [0.40(1), 0.42(1), 0.60(2)] → cumulative: 1, 2, 4
    # Half = 2.0 → cumulative hits 2.0 at 0.42, exactly on boundary
    # → average of 0.42 and 0.60 = 0.51
    assert result.P_consensus.home_win > 0.45
    assert result.confidence == "HIGH"
    assert result.n_fresh_sources == 3


def test_event_detection() -> None:
    """2+ sources moving >3% in same direction → event_detected=True."""
    oc = OddsConsensus()

    # Initial odds
    oc.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    oc.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    oc.update_bookmaker(
        "1xbet",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )

    # Big move in same direction (home_win drops >3%) for 2+ sources
    oc.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.42, draw=0.35, away_win=0.23),
    )
    oc.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.43, draw=0.34, away_win=0.23),
    )

    result = oc.compute_reference()
    assert result.event_detected is True

    # Small move should NOT trigger event detection
    oc2 = OddsConsensus()
    oc2.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    oc2.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.50, draw=0.30, away_win=0.20),
    )
    # Move only 1% — below threshold
    oc2.update_bookmaker(
        "Betfair Exchange",
        MarketProbs(home_win=0.49, draw=0.305, away_win=0.205),
    )
    oc2.update_bookmaker(
        "Bet365",
        MarketProbs(home_win=0.49, draw=0.305, away_win=0.205),
    )

    result2 = oc2.compute_reference()
    assert result2.event_detected is False
