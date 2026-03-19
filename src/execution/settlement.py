"""Phase 4 settlement: derive outcomes from score, poll Kalshi in live mode.

Settlement computes final PnL for all open positions and closes them.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from src.common.types import MatchPnL, Position, TradingMode
from src.execution.db_positions import close_position_db
from src.execution.pnl_calculator import compute_settlement_pnl

if TYPE_CHECKING:
    from src.clients.kalshi import KalshiClient
    from src.execution.position_monitor import PositionTracker

log = structlog.get_logger("settlement")

# Market type → outcome derivation from final score
OUTCOME_MAP = {
    "home_win": lambda h, a: h > a,
    "draw": lambda h, a: h == a,
    "away_win": lambda h, a: h < a,
    "over_25": lambda h, a: (h + a) >= 3,
    "btts_yes": lambda h, a: h >= 1 and a >= 1,
}


async def poll_kalshi_settlement(
    kalshi_client: KalshiClient,
    tickers: list[str],
    timeout_min: int = 45,
    interval_s: int = 60,
) -> dict[str, bool]:
    """Poll Kalshi for settlement results on each ticker.

    Returns {ticker: outcome_occurred} — may be partial if timeout.
    """
    outcomes: dict[str, bool] = {}
    deadline = time.monotonic() + timeout_min * 60

    while time.monotonic() < deadline:
        for ticker in tickers:
            if ticker in outcomes:
                continue
            try:
                market = await kalshi_client.get_market(ticker)
                result = market.get("result")
                if result is not None:
                    outcomes[ticker] = (result == "yes")
                    log.info("ticker_settled", ticker=ticker, result=result)
            except Exception as exc:
                log.warning("settlement_poll_error", ticker=ticker, error=str(exc))

        if len(outcomes) == len(tickers):
            break
        await asyncio.sleep(interval_s)

    for ticker in tickers:
        if ticker not in outcomes:
            log.error("settlement_timeout", ticker=ticker)

    return outcomes


async def settle_match(
    match_id: str,
    final_score: tuple[int, int],
    tracker: PositionTracker,
    db_pool: object,
    kalshi_client: KalshiClient | None,
    trading_mode: TradingMode,
) -> MatchPnL:
    """Settle all open positions for a match.

    Paper mode uses score-derived outcomes.
    Live mode polls Kalshi and cross-checks against score.
    """
    h, a = final_score

    # Score-derived outcomes
    score_outcomes: dict[str, bool] = {
        mt: fn(h, a) for mt, fn in OUTCOME_MAP.items()
    }

    # In live mode, poll Kalshi and use their results as authoritative
    kalshi_outcomes: dict[str, bool] = {}
    if trading_mode == TradingMode.LIVE and kalshi_client is not None:
        tickers = list({pos.ticker for pos in tracker.open_positions.values()})
        if tickers:
            kalshi_outcomes = await poll_kalshi_settlement(kalshi_client, tickers)

    # Settle each position
    total_pnl = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    position_summaries: list[dict] = []

    for pos in list(tracker.open_positions.values()):
        # Determine outcome
        if trading_mode == TradingMode.LIVE and pos.ticker in kalshi_outcomes:
            outcome = kalshi_outcomes[pos.ticker]
            score_outcome = score_outcomes.get(pos.market_type)
            if score_outcome is not None and outcome != score_outcome:
                log.error(
                    "settlement_mismatch",
                    ticker=pos.ticker,
                    market_type=pos.market_type,
                    kalshi=outcome,
                    score=score_outcome,
                )
        else:
            outcome = score_outcomes.get(pos.market_type, False)

        realized_pnl = compute_settlement_pnl(pos, outcome)
        total_pnl += realized_pnl
        trade_count += 1
        if realized_pnl > 0:
            win_count += 1
        elif realized_pnl < 0:
            loss_count += 1

        # Close in DB if db_id exists
        db_id = getattr(pos, "db_id", None)
        if db_id is not None and db_pool is not None:
            try:
                await close_position_db(
                    db_pool, db_id, 1.0 if outcome else 0.0,
                    0, "settlement", realized_pnl,
                )
            except Exception as exc:
                log.error("settlement_db_close_error", position_id=pos.id, error=str(exc))

        position_summaries.append({
            "position_id": pos.id,
            "ticker": pos.ticker,
            "market_type": pos.market_type,
            "direction": pos.direction,
            "quantity": pos.quantity,
            "outcome_occurred": outcome,
            "realized_pnl": realized_pnl,
        })

    # Clear all open positions from tracker
    tracker.open_positions.clear()

    result = MatchPnL(
        match_id=match_id,
        total_pnl=total_pnl,
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        positions=position_summaries,
    )

    log.info(
        "match_settled",
        match_id=match_id,
        total_pnl=total_pnl,
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
    )
    return result
