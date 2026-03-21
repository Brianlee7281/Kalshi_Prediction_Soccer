"""Kalshi orderbook synchronization — maintains live P_kalshi.

Phase 3 coroutine that subscribes to Kalshi WebSocket for orderbook
and trade updates. Maintains a local orderbook per ticker from
snapshots + deltas, computes mid-price, and stores in model.p_kalshi.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.kalshi_ob_sync")


# ── Local orderbook ──────────────────────────────────────────────

class _LocalBook:
    """Per-ticker orderbook built from snapshots + incremental deltas.

    Stores qty at each price level for YES and NO sides.
    Levels with qty <= 0 are removed.
    """

    __slots__ = ("yes", "no")

    def __init__(self) -> None:
        # price (float, dollars) → qty (float, dollars)
        self.yes: dict[float, float] = {}
        self.no: dict[float, float] = {}

    def apply_snapshot_fp(self, yes_fp: list, no_fp: list) -> None:
        """Replace book from yes_dollars_fp / no_dollars_fp arrays."""
        self.yes = {float(p): float(q) for p, q in yes_fp if float(q) > 0}
        self.no = {float(p): float(q) for p, q in no_fp if float(q) > 0}

    def apply_snapshot_cents(self, yes_levels: list, no_levels: list) -> None:
        """Replace book from live WS yes/no arrays ([price_cents, qty_cents])."""
        self.yes = {p / 100.0: q for p, q in yes_levels if q > 0}
        self.no = {p / 100.0: q for p, q in no_levels if q > 0}

    def apply_delta(self, side: str, price: float, delta_qty: float) -> None:
        """Apply an incremental qty change to one price level."""
        book = self.yes if side == "yes" else self.no
        new_qty = book.get(price, 0.0) + delta_qty
        if new_qty > 0:
            book[price] = new_qty
        else:
            book.pop(price, None)

    def best_bid(self) -> float | None:
        """Highest YES price with qty > 0."""
        return max(self.yes) if self.yes else None

    def best_ask(self) -> float | None:
        """1 - highest NO price (implied YES ask)."""
        return (1.0 - max(self.no)) if self.no else None

    def mid(self) -> float | None:
        """Mid-price from best bid/ask."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return bid if bid is not None else ask


# ── Main coroutine ───────────────────────────────────────────────

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

    # Local orderbook per ticker
    books: dict[str, _LocalBook] = defaultdict(_LocalBook)

    async def on_orderbook(ticker: str, data: dict) -> None:
        """Process orderbook snapshot or delta → update local book → compute mid."""
        market_type = ticker_to_market.get(ticker)
        if market_type is None:
            return

        book = books[ticker]

        # ── Snapshot (full book replacement) ──
        # Recorded format: yes_dollars_fp / no_dollars_fp
        yes_fp = data.get("yes_dollars_fp", [])
        no_fp = data.get("no_dollars_fp", [])
        # Live WS format: yes / no (cents)
        yes_cents = data.get("yes", [])
        no_cents = data.get("no", [])

        if yes_fp or no_fp:
            book.apply_snapshot_fp(yes_fp, no_fp)
        elif yes_cents or no_cents:
            book.apply_snapshot_cents(yes_cents, no_cents)
        else:
            # ── Delta (single level update) ──
            side = data.get("side")
            price_str = data.get("price_dollars")
            delta_str = data.get("delta_fp")
            if side and price_str is not None and delta_str is not None:
                book.apply_delta(side, float(price_str), float(delta_str))
            else:
                return

        # Compute mid and update model
        mid = book.mid()
        if mid is None:
            return

        if not hasattr(model, "p_kalshi"):
            model.p_kalshi = {}
        model.p_kalshi[market_type] = mid

        # Record if recorder attached
        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_kalshi_ob({"ticker": ticker, "mid": mid})

    async def on_trade(ticker: str, data: dict) -> None:
        """Log trade (no model update needed)."""
        logger.debug("kalshi_trade", ticker=ticker, data=data)

    async def _watch_finished() -> None:
        """Poll engine_phase and disconnect WS when match ends."""
        while model.engine_phase != "FINISHED":
            await asyncio.sleep(0.5)
        await ws_client.disconnect()

    watcher = asyncio.create_task(_watch_finished())
    try:
        await ws_client.connect(
            tickers=tickers,
            on_orderbook=on_orderbook,
            on_trade=on_trade,
        )
    except Exception as exc:
        logger.error("kalshi_ob_sync_error", error=str(exc))
    finally:
        watcher.cancel()
