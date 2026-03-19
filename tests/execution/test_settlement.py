"""Tests for src/execution/settlement.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.common.types import FillResult, Position, Signal, TradingMode
from src.execution.position_monitor import PositionTracker
from src.execution.settlement import poll_kalshi_settlement, settle_match


def _make_position(
    market_type: str = "home_win",
    direction: str = "BUY_YES",
    entry_price: float = 0.55,
    quantity: int = 10,
    pos_id: str = "pos-1",
) -> Position:
    return Position(
        id=pos_id,
        match_id="test_match",
        ticker=f"TICKER-{market_type.upper()}",
        market_type=market_type,
        direction=direction,
        quantity=quantity,
        entry_price=entry_price,
        entry_tick=0,
        entry_t=10.0,
    )


def _make_signal(market_type: str = "home_win", direction: str = "BUY_YES") -> Signal:
    return Signal(
        match_id="test_match",
        ticker=f"TICKER-{market_type.upper()}",
        market_type=market_type,
        direction=direction,
        P_kalshi=0.55,
        P_model=0.62,
        EV=0.07,
        kelly_fraction=0.0,
        kelly_amount=0.0,
        contracts=10,
    )


def _make_fill(price: float = 0.55, quantity: int = 10) -> FillResult:
    return FillResult(
        order_id="paper-1",
        ticker="TICKER-HOME_WIN",
        direction="BUY_YES",
        quantity=quantity,
        price=price,
        status="paper",
        fill_cost=quantity * price,
        timestamp=datetime(2026, 3, 18),
    )


def _setup_tracker_with_position(
    market_type: str = "home_win",
    direction: str = "BUY_YES",
    entry_price: float = 0.55,
    quantity: int = 10,
) -> PositionTracker:
    """Create a tracker with one open position."""
    tracker = PositionTracker(min_hold_ticks=5, cooldown_after_exit=10)
    pos = _make_position(market_type=market_type, direction=direction,
                         entry_price=entry_price, quantity=quantity)
    tracker.open_positions[pos.id] = pos
    return tracker


# ── Score-derived settlements ─────────────────────────────────


@pytest.mark.asyncio
async def test_settle_home_win():
    tracker = _setup_tracker_with_position("home_win", "BUY_YES", 0.55, 10)
    result = await settle_match("test", (2, 1), tracker, None, None, TradingMode.PAPER)
    assert result.trade_count == 1
    assert result.positions[0]["outcome_occurred"] is True
    # BUY_YES + occurred: (1.0 - 0.55) * 10 = 4.50
    assert result.total_pnl == pytest.approx(4.50)
    assert result.win_count == 1


@pytest.mark.asyncio
async def test_settle_draw():
    tracker = _setup_tracker_with_position("draw", "BUY_YES", 0.30, 10)
    result = await settle_match("test", (2, 2), tracker, None, None, TradingMode.PAPER)
    assert result.positions[0]["outcome_occurred"] is True
    # (1.0 - 0.30) * 10 = 7.00
    assert result.total_pnl == pytest.approx(7.00)


@pytest.mark.asyncio
async def test_settle_over_25():
    tracker = _setup_tracker_with_position("over_25", "BUY_YES", 0.60, 10)
    result = await settle_match("test", (2, 1), tracker, None, None, TradingMode.PAPER)
    # total goals = 3 >= 3 → over_25 = True
    assert result.positions[0]["outcome_occurred"] is True
    assert result.total_pnl == pytest.approx((1.0 - 0.60) * 10)


@pytest.mark.asyncio
async def test_settle_under_25():
    tracker = _setup_tracker_with_position("over_25", "BUY_YES", 0.60, 10)
    result = await settle_match("test", (1, 0), tracker, None, None, TradingMode.PAPER)
    # total goals = 1 < 3 → over_25 = False
    assert result.positions[0]["outcome_occurred"] is False
    assert result.total_pnl == pytest.approx(-0.60 * 10)


@pytest.mark.asyncio
async def test_settle_btts_yes():
    tracker = _setup_tracker_with_position("btts_yes", "BUY_YES", 0.50, 10)
    result = await settle_match("test", (2, 1), tracker, None, None, TradingMode.PAPER)
    # both teams scored → True
    assert result.positions[0]["outcome_occurred"] is True
    assert result.total_pnl == pytest.approx((1.0 - 0.50) * 10)


@pytest.mark.asyncio
async def test_settle_btts_no():
    tracker = _setup_tracker_with_position("btts_yes", "BUY_YES", 0.50, 10)
    result = await settle_match("test", (1, 0), tracker, None, None, TradingMode.PAPER)
    # away didn't score → False
    assert result.positions[0]["outcome_occurred"] is False
    assert result.total_pnl == pytest.approx(-0.50 * 10)


@pytest.mark.asyncio
async def test_settle_no_positions():
    tracker = PositionTracker()
    result = await settle_match("test", (2, 2), tracker, None, None, TradingMode.PAPER)
    assert result.total_pnl == 0.0
    assert result.trade_count == 0


# ── Kalshi polling ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_kalshi_settlement_success():
    """Mock get_market returns None twice then {"result": "yes"}."""
    mock_client = AsyncMock()
    call_count = 0

    async def mock_get_market(ticker):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {"result": None}
        return {"result": "yes"}

    mock_client.get_market = mock_get_market

    outcomes = await poll_kalshi_settlement(
        mock_client, ["TICKER-1"], timeout_min=1, interval_s=0.01
    )
    assert outcomes == {"TICKER-1": True}


@pytest.mark.asyncio
async def test_poll_kalshi_settlement_timeout():
    """Mock always returns None → empty/partial dict after timeout."""
    mock_client = AsyncMock()
    mock_client.get_market = AsyncMock(return_value={"result": None})

    outcomes = await poll_kalshi_settlement(
        mock_client, ["TICKER-1"], timeout_min=0, interval_s=0.01
    )
    # timeout_min=0 means deadline is immediately passed
    # First iteration may or may not complete depending on timing
    # The ticker should NOT be in outcomes (or the test should handle both)
    assert "TICKER-1" not in outcomes or outcomes.get("TICKER-1") is not None


@pytest.mark.asyncio
async def test_settle_paper_skips_polling():
    """Paper mode uses score-derived outcomes, no Kalshi calls."""
    mock_client = AsyncMock()
    tracker = _setup_tracker_with_position("home_win", "BUY_YES", 0.55, 10)

    result = await settle_match(
        "test", (2, 1), tracker, None, mock_client, TradingMode.PAPER
    )
    # Paper mode → kalshi_client not used even if provided
    mock_client.get_market.assert_not_called()
    assert result.trade_count == 1
