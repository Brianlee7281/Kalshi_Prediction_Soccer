"""Phase 4 order management: paper fills + live Kalshi limit orders.

Paper mode: spread-adjusted fill with depth-limited partial fills.
Live mode: limit order at P_model (NOT P_kalshi), with repricing on drift.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import structlog

from src.clients.kalshi import KalshiClient
from src.common.types import FillResult, Signal, TradingMode
from src.execution.config import CONFIG
from src.execution.kelly_sizer import cost_per_contract

log = structlog.get_logger("order_manager")


def paper_fill_adjust(
    mid: float, direction: str, contracts: int
) -> tuple[float, int]:
    """Apply spread and depth adjustment for paper fills.

    Args:
        mid: Mid-price from orderbook.
        direction: "BUY_YES" or "BUY_NO".
        contracts: Requested number of contracts.

    Returns:
        (adjusted_price, filled_quantity)
    """
    half_spread = CONFIG.PAPER_HALF_SPREAD
    if direction == "BUY_YES":
        price = min(0.99, mid + half_spread)
    else:
        price = max(0.01, mid - half_spread)

    depth = CONFIG.PAPER_TYPICAL_DEPTH
    if contracts <= depth:
        filled = contracts
    else:
        fill_prob = depth / contracts
        excess = contracts - depth
        extra = sum(1 for _ in range(excess) if random.random() < fill_prob)
        filled = depth + extra

    return price, filled


class OrderManager:
    """Order placement and lifecycle management."""

    def __init__(
        self,
        kalshi_client: KalshiClient | None,
        trading_mode: TradingMode,
        db_pool: object,  # asyncpg.Pool — typed as object to avoid hard import
    ) -> None:
        self.kalshi_client = kalshi_client
        self.trading_mode = trading_mode
        self.db = db_pool
        self.pending_orders: dict[str, dict] = {}
        self.max_order_age_s: float = CONFIG.MAX_ORDER_LIFETIME_S
        self.reprice_threshold: float = CONFIG.REPRICE_THRESHOLD
        self.ticker_muted: dict[str, bool] = {}
        self.entries_halted: bool = False

    def is_ticker_muted(self, ticker: str) -> bool:
        """Check if a ticker is muted due to market_closed."""
        return self.ticker_muted.get(ticker, False)

    async def place_order(self, signal: Signal) -> FillResult | None:
        """Place an order for a signal. Returns None when muted or halted."""
        if self.entries_halted:
            log.warning("entries_halted", ticker=signal.ticker)
            return None

        if self.is_ticker_muted(signal.ticker):
            log.debug("ticker_muted", ticker=signal.ticker)
            return None

        if self.trading_mode == TradingMode.PAPER:
            fill_price, filled_qty = paper_fill_adjust(
                signal.P_kalshi, signal.direction, signal.contracts,
            )
            fill = FillResult(
                order_id=f"paper-{uuid4()}",
                ticker=signal.ticker,
                direction=signal.direction,
                quantity=filled_qty,
                price=fill_price,
                status="paper" if filled_qty > 0 else "rejected",
                fill_cost=filled_qty * cost_per_contract(fill_price, signal.direction),
                timestamp=datetime.now(timezone.utc),
            )
            log.info(
                "order_placed",
                mode="paper",
                ticker=signal.ticker,
                requested=signal.contracts,
                filled=filled_qty,
                mid=signal.P_kalshi,
                fill_price=round(fill_price, 4),
            )
            return fill

        # Live mode
        if self.kalshi_client is None:
            log.error("no_kalshi_client")
            return FillResult(
                order_id="none",
                ticker=signal.ticker,
                direction=signal.direction,
                quantity=0,
                price=0.0,
                status="rejected",
                fill_cost=0.0,
                timestamp=datetime.now(timezone.utc),
            )

        order = {
            "ticker": signal.ticker,
            "action": "buy",
            "side": "yes" if signal.direction == "BUY_YES" else "no",
            "type": "limit",
            "count": signal.contracts,
            "yes_price": int(signal.P_model * 100),
        }

        try:
            response = await self.kalshi_client.submit_order(order)
            order_id = response.get("order", {}).get("order_id", str(uuid4()))
            status = response.get("order", {}).get("status", "pending")
            filled_qty = response.get("order", {}).get("count", signal.contracts)
            fill_price = signal.P_model

            self.pending_orders[order_id] = {
                "signal": signal,
                "placed_at": time.time(),
                "order_p_model": signal.P_model,
            }

            log.info(
                "order_placed",
                mode="live",
                ticker=signal.ticker,
                contracts=signal.contracts,
                yes_price=order["yes_price"],
                order_id=order_id,
            )

            return FillResult(
                order_id=order_id,
                ticker=signal.ticker,
                direction=signal.direction,
                quantity=filled_qty if status == "filled" else 0,
                price=fill_price,
                status="full" if status == "filled" else "pending",
                fill_cost=filled_qty * cost_per_contract(fill_price, signal.direction) if status == "filled" else 0.0,
                timestamp=datetime.now(timezone.utc),
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response else ""

            if e.response and e.response.status_code == 429:
                log.warning("kalshi_rate_limited", ticker=signal.ticker)
                return FillResult(
                    order_id="none",
                    ticker=signal.ticker,
                    direction=signal.direction,
                    quantity=0,
                    price=0.0,
                    status="rejected",
                    fill_cost=0.0,
                    timestamp=datetime.now(timezone.utc),
                )

            if "market_closed" in body:
                self.ticker_muted[signal.ticker] = True
                log.warning("ticker_muted", ticker=signal.ticker)
                return None

            if "insufficient_balance" in body:
                self.entries_halted = True
                log.error("entries_halted", ticker=signal.ticker)
                return None

            if "price_out_of_range" in body:
                log.warning("price_out_of_range", ticker=signal.ticker)
                return FillResult(
                    order_id="none",
                    ticker=signal.ticker,
                    direction=signal.direction,
                    quantity=0,
                    price=0.0,
                    status="rejected",
                    fill_cost=0.0,
                    timestamp=datetime.now(timezone.utc),
                )

            log.error("kalshi_order_error", ticker=signal.ticker, error=str(e))
            return FillResult(
                order_id="none",
                ticker=signal.ticker,
                direction=signal.direction,
                quantity=0,
                price=0.0,
                status="rejected",
                fill_cost=0.0,
                timestamp=datetime.now(timezone.utc),
            )

        except (TimeoutError, asyncio.TimeoutError):
            log.warning("kalshi_order_timeout", ticker=signal.ticker)
            return FillResult(
                order_id="none",
                ticker=signal.ticker,
                direction=signal.direction,
                quantity=0,
                price=0.0,
                status="rejected",
                fill_cost=0.0,
                timestamp=datetime.now(timezone.utc),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.trading_mode == TradingMode.PAPER:
            self.pending_orders.pop(order_id, None)
            return True

        if self.kalshi_client is None:
            return False

        try:
            await self.kalshi_client.cancel_order(order_id)
            self.pending_orders.pop(order_id, None)
            return True
        except Exception as e:
            log.error("cancel_order_failed", order_id=order_id, error=str(e))
            return False

    async def manage_open_orders(
        self, current_p_model: dict[str, float], current_time: float
    ) -> list[FillResult]:
        """Manage pending orders: cancel stale, reprice on drift."""
        results: list[FillResult] = []
        orders_to_remove: list[str] = []

        for order_id, order in list(self.pending_orders.items()):
            age = current_time - order["placed_at"]
            signal = order["signal"]

            # Cancel stale orders
            if age > self.max_order_age_s:
                await self.cancel_order(order_id)
                orders_to_remove.append(order_id)
                log.info("order_expired", order_id=order_id, age_s=age)
                continue

            # Reprice on P_model drift
            new_p_model = current_p_model.get(
                signal.market_type, order["order_p_model"]
            )
            drift = abs(new_p_model - order["order_p_model"])

            if drift > self.reprice_threshold:
                await self.cancel_order(order_id)
                orders_to_remove.append(order_id)

                # Re-post at new P_model
                updated_signal = signal.model_copy(
                    update={"P_model": new_p_model}
                )
                fill = await self.place_order(updated_signal)
                log.info(
                    "order_repriced",
                    old_price=order["order_p_model"],
                    new_price=new_p_model,
                    drift=drift,
                )
                if fill is not None:
                    results.append(fill)

        # Clean up removed orders (already removed by cancel_order, but ensure)
        for oid in orders_to_remove:
            self.pending_orders.pop(oid, None)

        return results
