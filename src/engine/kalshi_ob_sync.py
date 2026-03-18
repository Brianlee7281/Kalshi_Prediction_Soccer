"""Kalshi orderbook synchronization — maintains live P_kalshi.

Phase 3 coroutine that subscribes to Kalshi WebSocket for orderbook
and trade updates. Computes mid-price for each ticker and stores
in model.p_kalshi dict.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.kalshi_ob_sync")


async def kalshi_ob_sync(
    model: LiveMatchModel,
    ws_client: KalshiWSClient,
) -> None:
    """Coroutine: subscribe to Kalshi WS, maintain P_kalshi.

    Updates model.p_kalshi dict with latest mid-price for each ticker.
    Runs until model.engine_phase == "FINISHED".
    Records orderbook snapshots via recorder if attached.
    """
    tickers = list(model.kalshi_tickers.values())
    if not tickers:
        logger.warning("kalshi_ob_sync_no_tickers", match_id=model.match_id)
        return

    # Reverse map: ticker string → market type
    ticker_to_market: dict[str, str] = {
        v: k for k, v in model.kalshi_tickers.items()
    }

    async def on_orderbook(ticker: str, data: dict) -> None:
        """Process orderbook update → compute mid-price."""
        market_type = ticker_to_market.get(ticker)
        if market_type is None:
            return

        # Extract best bid/ask from orderbook
        yes_bids = data.get("yes", [])
        no_bids = data.get("no", [])

        best_bid = yes_bids[0][0] / 100.0 if yes_bids else None
        best_ask = (100 - no_bids[0][0]) / 100.0 if no_bids else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            mid = best_bid
        elif best_ask is not None:
            mid = best_ask
        else:
            return

        if not hasattr(model, "p_kalshi"):
            model.p_kalshi = {}
        model.p_kalshi[market_type] = mid

        # Record if recorder attached
        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_kalshi_ob({"ticker": ticker, "mid": mid, **data})

    async def on_trade(ticker: str, data: dict) -> None:
        """Log trade (no model update needed)."""
        logger.debug("kalshi_trade", ticker=ticker, data=data)

    try:
        await ws_client.connect(
            tickers=tickers,
            on_orderbook=on_orderbook,
            on_trade=on_trade,
        )
    except Exception as exc:
        logger.error("kalshi_ob_sync_error", error=str(exc))
