"""Tests for tick_loop and P_reference selection (Task 3.6)."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.common.types import MarketProbs, OddsConsensusResult
from src.engine.tick_loop import _sleep_until_next_tick, select_P_reference


def test_select_P_reference_high() -> None:
    """HIGH confidence → use consensus."""
    consensus = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.55, draw=0.25, away_win=0.20),
        confidence="HIGH",
        n_fresh_sources=3,
        bookmakers=[],
        event_detected=False,
    )
    P_model = MarketProbs(home_win=0.50, draw=0.28, away_win=0.22)
    P_ref, source = select_P_reference(consensus, P_model)
    assert source == "consensus"
    assert P_ref.home_win == 0.55


def test_select_P_reference_none() -> None:
    """No consensus → use model."""
    P_model = MarketProbs(home_win=0.50, draw=0.28, away_win=0.22)
    P_ref, source = select_P_reference(None, P_model)
    assert source == "model"
    assert P_ref.home_win == 0.50

    # NONE confidence also falls back to model
    consensus_none = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.0, draw=0.0, away_win=0.0),
        confidence="NONE",
        n_fresh_sources=0,
        bookmakers=[],
        event_detected=False,
    )
    P_ref2, source2 = select_P_reference(consensus_none, P_model)
    assert source2 == "model"
    assert P_ref2.home_win == 0.50


def test_select_P_reference_low_disagree() -> None:
    """LOW confidence + model disagrees by >10% → use model."""
    consensus = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.70, draw=0.15, away_win=0.15),
        confidence="LOW",
        n_fresh_sources=1,
        bookmakers=[],
        event_detected=False,
    )
    P_model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    P_ref, source = select_P_reference(consensus, P_model)
    assert source == "model"  # disagree by 0.25 > 0.10

    # LOW confidence + model agrees within 10% → use consensus
    consensus_agree = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.52, draw=0.27, away_win=0.21),
        confidence="LOW",
        n_fresh_sources=1,
        bookmakers=[],
        event_detected=False,
    )
    P_ref2, source2 = select_P_reference(consensus_agree, P_model)
    assert source2 == "consensus"
    assert P_ref2.home_win == 0.52


@pytest.mark.asyncio
async def test_sleep_until_next_tick() -> None:
    """Verify absolute time scheduling skips when behind."""
    start = time.monotonic()

    # Next tick is in the past → should return immediately
    await _sleep_until_next_tick(start - 5.0, tick_count=1, interval=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1  # returned nearly instantly

    # Next tick is slightly in the future → should sleep briefly
    now = time.monotonic()
    await _sleep_until_next_tick(now, tick_count=1, interval=0.05)
    elapsed2 = time.monotonic() - now
    assert elapsed2 >= 0.04  # slept approximately 50ms
    assert elapsed2 < 0.2  # but not excessively
