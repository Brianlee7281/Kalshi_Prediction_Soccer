"""Phase 4 execution loop — main coroutine wiring all execution modules.

Consumes TickPayloads from Phase 3's tick_loop via asyncio.Queue,
orchestrates signal generation, sizing, ordering, position monitoring,
exit decisions, and settlement.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from src.common.types import (
    ExitDecision,
    FillResult,
    MatchPnL,
    Position,
    Signal,
    TickPayload,
    TradingMode,
)
from src.execution.config import CONFIG
from src.execution.db_positions import close_position_db, save_position
from src.execution.exposure_manager import ExposureManager
from src.execution.kelly_sizer import cost_per_contract, size_position
from src.execution.order_manager import OrderManager, paper_fill_adjust
from src.execution.pnl_calculator import compute_unrealized_pnl
from src.execution.position_monitor import PositionTracker
from src.execution.redis_publisher import publish_position_update, publish_signal
from src.execution.settlement import settle_match
from src.execution.signal_generator import generate_signals

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

log = structlog.get_logger("execution_loop")

MARKET_TYPES = ["home_win", "draw", "away_win", "over_25", "btts_yes"]


def _build_exit_signal(pos: Position, exit_decision: ExitDecision) -> Signal:
    """Build a Signal for exiting a position."""
    return Signal(
        match_id=pos.match_id,
        ticker=pos.ticker,
        market_type=pos.market_type,
        direction="BUY_NO" if pos.direction == "BUY_YES" else "BUY_YES",
        P_kalshi=exit_decision.exit_price,
        P_model=pos.current_p_model,
        EV=0.0,
        kelly_fraction=0.0,
        kelly_amount=0.0,
        contracts=exit_decision.contracts_to_exit,
    )


def _paper_exit_fill(
    pos: Position, exit_price: float, contracts: int
) -> FillResult:
    """Simulate a paper exit fill with spread + depth adjustment.

    Exit spread direction is adverse (opposite of entry):
    - Exiting BUY_YES (selling YES): fill at mid - half_spread
    - Exiting BUY_NO (selling NO): fill at mid + half_spread
    """
    # Exit direction is opposite of position direction
    exit_direction = "BUY_NO" if pos.direction == "BUY_YES" else "BUY_YES"
    # Spread is adverse: selling YES = BUY_NO direction for pricing
    fill_price, filled_qty = paper_fill_adjust(exit_price, exit_direction, contracts)
    return FillResult(
        order_id=f"paper-exit-{uuid4()}",
        ticker=pos.ticker,
        direction=exit_direction,
        quantity=filled_qty,
        price=fill_price,
        status="paper" if filled_qty > 0 else "rejected",
        fill_cost=filled_qty * cost_per_contract(fill_price, exit_direction),
        timestamp=datetime.now(timezone.utc),
    )


def _compute_exit_pnl(pos: Position, fill: FillResult) -> float:
    """Compute realized PnL for an exit at the fill price."""
    if pos.direction == "BUY_YES":
        return (fill.price - pos.entry_price) * fill.quantity
    else:
        return ((1.0 - fill.price) - pos.entry_price) * fill.quantity


async def execution_loop(
    phase4_queue: asyncio.Queue,
    model: LiveMatchModel,
    db_pool: object,
    trading_mode: TradingMode,
    redis_client: object | None = None,
) -> MatchPnL:
    """Main Phase 4 coroutine — processes ticks and manages positions."""
    exposure = ExposureManager(db_pool, trading_mode)
    kalshi_client = getattr(model, "kalshi_client", None)
    orders = OrderManager(kalshi_client, trading_mode, db_pool)
    tracker = PositionTracker()
    bankroll = await exposure.get_bankroll()
    stale_check_interval = 300
    tick_counter = 0
    last_score = (0, 0)

    log.info(
        "execution_loop_started",
        match_id=model.match_id,
        mode=trading_mode.value,
        bankroll=bankroll,
    )

    while True:
        payload: TickPayload = await phase4_queue.get()
        tick_counter += 1

        if payload.engine_phase == "FINISHED":
            last_score = payload.score
            break

        last_score = payload.score

        # 1. Check exits (EVERY tick, even during cooldown)
        exits = tracker.check_exits(payload, model.p_kalshi)
        for exit_decision in exits:
            if exit_decision.position_id not in tracker.open_positions:
                continue
            pos = tracker.open_positions[exit_decision.position_id]

            if trading_mode == TradingMode.LIVE:
                exit_signal = _build_exit_signal(pos, exit_decision)
                fill = await orders.place_order(exit_signal)
            else:
                fill = _paper_exit_fill(
                    pos, exit_decision.exit_price, exit_decision.contracts_to_exit
                )

            if fill is not None and fill.quantity > 0:
                realized_pnl = _compute_exit_pnl(pos, fill)
                tracker.close_position(
                    pos.id,
                    exit_decision.trigger,
                    fill.quantity,
                    fill.price,
                    tick_counter,
                )
                db_id = getattr(pos, "db_id", None)
                if db_id is not None and db_pool is not None:
                    try:
                        await close_position_db(
                            db_pool, db_id, fill.price, tick_counter,
                            exit_decision.trigger.value, realized_pnl,
                        )
                    except Exception as exc:
                        log.error("exit_db_error", error=str(exc))
                # Return the exit proceeds to bankroll.  Entry cost was
                # deducted on open, so we credit back the exit revenue.
                # BUY_YES exit: sell YES → receive exit_price × qty
                # BUY_NO exit: sell NO → receive (1 - exit_price) × qty
                if pos.direction == "BUY_YES":
                    exit_revenue = fill.price * fill.quantity
                else:
                    exit_revenue = (1.0 - fill.price) * fill.quantity
                await exposure.update_bankroll(exit_revenue)
                bankroll += exit_revenue
                if pos.reservation_id is not None:
                    await exposure.release_exposure(pos.reservation_id)
                if redis_client:
                    await publish_position_update(redis_client, pos, "exit")

        # 2. Generate + execute new signals (only if order_allowed AND not halted)
        if payload.order_allowed and not orders.entries_halted:
            signals = generate_signals(
                payload, model.p_kalshi, model.kalshi_tickers,
                tracker.open_positions,
            )
            for signal in signals:
                if tracker.is_in_cooldown(signal.market_type, tick_counter):
                    continue
                if orders.is_ticker_muted(signal.ticker):
                    continue

                signal = size_position(signal, payload, bankroll)
                if signal.contracts <= 0:
                    continue

                cpc = cost_per_contract(signal.P_kalshi, signal.direction)
                amount = signal.contracts * cpc
                res_id = await exposure.reserve_exposure(
                    payload.match_id, signal.ticker, amount
                )
                if res_id is None:
                    continue

                fill = await orders.place_order(signal)
                if fill is not None and fill.quantity > 0:
                    await exposure.confirm_exposure(res_id, fill.fill_cost)
                    pos = tracker.add_position(
                        signal, fill, tick_counter, payload.t,
                        reservation_id=res_id,
                    )
                    try:
                        pos.db_id = await save_position(db_pool, pos)
                    except Exception as exc:
                        log.error("save_position_db_error", error=str(exc))
                    await exposure.update_bankroll(-fill.fill_cost)
                    bankroll -= fill.fill_cost
                    if redis_client:
                        await publish_position_update(redis_client, pos, "new_fill")
                        await publish_signal(redis_client, signal, fill)
                else:
                    await exposure.release_exposure(res_id)

        # 3. Stale reservation cleanup every 300 ticks
        if tick_counter % stale_check_interval == 0:
            await exposure.release_stale_reservations()

        # 4. Manage open orders (cancel stale + reprice on drift)
        current_p_model: dict[str, float] = {}
        for mt in MARKET_TYPES:
            val = getattr(payload.P_model, mt, None)
            if val is not None:
                current_p_model[mt] = val
        await orders.manage_open_orders(current_p_model, time.monotonic())

        if tick_counter % 60 == 0:
            log.info(
                "execution_tick",
                tick=tick_counter,
                open_positions=len(tracker.open_positions),
                bankroll=round(bankroll, 2),
            )

    # Settlement — collect reservation IDs before positions are cleared
    settled_res_ids = [
        pos.reservation_id
        for pos in tracker.open_positions.values()
        if pos.reservation_id is not None
    ]
    match_pnl = await settle_match(
        model.match_id, last_score, tracker, db_pool,
        kalshi_client, trading_mode,
    )
    for res_id in settled_res_ids:
        await exposure.release_exposure(res_id)

    log.info(
        "execution_loop_finished",
        match_id=model.match_id,
        total_pnl=match_pnl.total_pnl,
        trade_count=match_pnl.trade_count,
    )
    return match_pnl
