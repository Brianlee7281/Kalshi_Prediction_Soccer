"""Tests for src/execution/order_manager.py."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.common.types import FillResult, Signal, TradingMode
from src.execution.order_manager import OrderManager


def _make_signal(
    p_model: float = 0.62,
    p_kalshi: float = 0.55,
    contracts: int = 10,
    direction: str = "BUY_YES",
    ticker: str = "TICKER-HOME",
    market_type: str = "home_win",
) -> Signal:
    return Signal(
        match_id="test_match",
        ticker=ticker,
        market_type=market_type,
        direction=direction,
        P_kalshi=p_kalshi,
        P_model=p_model,
        EV=abs(p_model - p_kalshi),
        kelly_fraction=0.10,
        kelly_amount=10.0,
        contracts=contracts,
    )


def _make_http_error(status_code: int, body: str) -> httpx.HTTPStatusError:
    """Create a mock HTTPStatusError with given status and body."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = body
    request = MagicMock(spec=httpx.Request)
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=request,
        response=response,
    )


# ── Paper mode ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_order_immediate_fill():
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    signal = _make_signal(contracts=18, p_kalshi=0.55)
    fill = await om.place_order(signal)
    assert fill is not None
    assert fill.status == "paper"
    assert fill.quantity == 18
    assert fill.price == pytest.approx(0.555)
    assert fill.fill_cost == pytest.approx(18 * 0.555)


@pytest.mark.asyncio
async def test_paper_order_generates_uuid():
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    signal = _make_signal()
    fill = await om.place_order(signal)
    assert fill is not None
    assert fill.order_id.startswith("paper-")


# ── Live mode ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_order_uses_p_model_price():
    """Verify limit order price is int(P_model * 100), NOT P_kalshi."""
    mock_client = AsyncMock()
    mock_client.submit_order = AsyncMock(
        return_value={
            "order": {
                "order_id": "live-123",
                "status": "pending",
                "count": 10,
            }
        }
    )

    om = OrderManager(
        kalshi_client=mock_client, trading_mode=TradingMode.LIVE, db_pool=None
    )
    signal = _make_signal(p_model=0.62, p_kalshi=0.55, contracts=10)
    await om.place_order(signal)

    mock_client.submit_order.assert_called_once()
    order_dict = mock_client.submit_order.call_args[0][0]
    # CRITICAL: yes_price = int(P_model * 100) = 62, NOT int(P_kalshi * 100) = 55
    assert order_dict["yes_price"] == int(0.62 * 100)
    assert order_dict["yes_price"] != int(0.55 * 100)


# ── manage_open_orders ────────────────────────────────────────


@pytest.mark.asyncio
async def test_manage_open_orders_cancels_stale():
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    signal = _make_signal()
    order_id = "stale-order-1"
    om.pending_orders[order_id] = {
        "signal": signal,
        "placed_at": time.time() - 35.0,  # 35s ago > 30s max
        "order_p_model": 0.62,
    }

    results = await om.manage_open_orders({"home_win": 0.62}, time.time())
    assert order_id not in om.pending_orders


@pytest.mark.asyncio
async def test_manage_open_orders_reprices_on_drift():
    """Order at P_model=0.45, current P_model=0.48 → drift=0.03 > 0.02 threshold."""
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    signal = _make_signal(p_model=0.45)
    order_id = "reprice-order-1"
    om.pending_orders[order_id] = {
        "signal": signal,
        "placed_at": time.time() - 5.0,  # recent, not stale
        "order_p_model": 0.45,
    }

    results = await om.manage_open_orders({"home_win": 0.48}, time.time())
    # Old order should be cancelled
    assert order_id not in om.pending_orders
    # A new fill should be returned (paper mode → immediate fill)
    assert len(results) == 1
    assert results[0].status == "paper"


@pytest.mark.asyncio
async def test_manage_open_orders_no_reprice_small_drift():
    """Drift=0.01 < 0.02 threshold → order unchanged."""
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    signal = _make_signal(p_model=0.45)
    order_id = "no-reprice-1"
    om.pending_orders[order_id] = {
        "signal": signal,
        "placed_at": time.time() - 5.0,
        "order_p_model": 0.45,
    }

    results = await om.manage_open_orders({"home_win": 0.46}, time.time())
    # Order should still be there
    assert order_id in om.pending_orders
    assert len(results) == 0


# ── Kalshi error handling ─────────────────────────────────────


@pytest.mark.asyncio
async def test_ticker_muted_on_market_closed():
    mock_client = AsyncMock()
    mock_client.submit_order = AsyncMock(
        side_effect=_make_http_error(400, "market_closed")
    )

    om = OrderManager(
        kalshi_client=mock_client, trading_mode=TradingMode.LIVE, db_pool=None
    )
    signal = _make_signal(ticker="CLOSED-TICKER")

    result = await om.place_order(signal)
    assert result is None
    assert om.ticker_muted["CLOSED-TICKER"] is True

    # Subsequent call returns None without hitting Kalshi
    result2 = await om.place_order(signal)
    assert result2 is None
    # submit_order should only have been called once (first call)
    assert mock_client.submit_order.call_count == 1


@pytest.mark.asyncio
async def test_entries_halted_on_insufficient_balance():
    mock_client = AsyncMock()
    mock_client.submit_order = AsyncMock(
        side_effect=_make_http_error(400, "insufficient_balance")
    )

    om = OrderManager(
        kalshi_client=mock_client, trading_mode=TradingMode.LIVE, db_pool=None
    )
    signal = _make_signal()

    result = await om.place_order(signal)
    assert result is None
    assert om.entries_halted is True

    # All subsequent calls return None
    result2 = await om.place_order(_make_signal(ticker="OTHER"))
    assert result2 is None


@pytest.mark.asyncio
async def test_price_out_of_range_returns_rejected():
    mock_client = AsyncMock()
    mock_client.submit_order = AsyncMock(
        side_effect=_make_http_error(400, "price_out_of_range")
    )

    om = OrderManager(
        kalshi_client=mock_client, trading_mode=TradingMode.LIVE, db_pool=None
    )
    signal = _make_signal()

    result = await om.place_order(signal)
    assert result is not None
    assert result.status == "rejected"
    assert result.quantity == 0


@pytest.mark.asyncio
async def test_place_order_returns_none_when_halted():
    om = OrderManager(
        kalshi_client=None, trading_mode=TradingMode.PAPER, db_pool=None
    )
    om.entries_halted = True

    signal = _make_signal()
    result = await om.place_order(signal)
    assert result is None
