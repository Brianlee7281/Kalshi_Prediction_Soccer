"""Tests for src/execution/execution_loop.py.

Smoke tests that verify the execution loop processes payloads
without crashing. Full integration tests require PostgreSQL + replay data.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.types import MarketProbs, MatchPnL, TickPayload, TradingMode
from src.execution.execution_loop import (
    _build_exit_signal,
    _compute_exit_pnl,
    _paper_exit_fill,
)
from src.common.types import Position


def _make_payload(
    engine_phase: str = "SECOND_HALF",
    order_allowed: bool = True,
    home_win: float = 0.50,
    t: float = 45.0,
    score: tuple[int, int] = (1, 0),
) -> TickPayload:
    return TickPayload(
        match_id="test_match",
        t=t,
        engine_phase=engine_phase,
        P_model=MarketProbs(
            home_win=home_win, draw=0.25, away_win=0.25,
            over_25=0.50, btts_yes=0.40,
        ),
        sigma_MC=MarketProbs(
            home_win=0.003, draw=0.003, away_win=0.003,
            over_25=0.003, btts_yes=0.003,
        ),
        score=score,
        X=0,
        delta_S=score[0] - score[1],
        mu_H=1.0,
        mu_A=0.8,
        a_H_current=0.3,
        a_A_current=-0.1,
        ekf_P_H=0.10,
        ekf_P_A=0.10,
        surprise_score=0.0,
        order_allowed=order_allowed,
        cooldown=False,
        ob_freeze=False,
        event_state="IDLE",
    )


def _make_position(
    direction: str = "BUY_YES",
    entry_price: float = 0.55,
    quantity: int = 10,
) -> Position:
    return Position(
        id="pos-1",
        match_id="test_match",
        ticker="TICKER-HOME",
        market_type="home_win",
        direction=direction,
        quantity=quantity,
        entry_price=entry_price,
        entry_tick=0,
        entry_t=10.0,
        current_p_model=0.50,
        current_p_kalshi=0.55,
    )


# ── Helper function tests ────────────────────────────────────


def test_build_exit_signal():
    pos = _make_position(direction="BUY_YES")
    from src.common.types import ExitDecision, ExitTrigger
    exit_dec = ExitDecision(
        position_id="pos-1",
        trigger=ExitTrigger.EDGE_REVERSAL,
        contracts_to_exit=10,
        exit_price=0.60,
        reason="test",
    )
    signal = _build_exit_signal(pos, exit_dec)
    assert signal.direction == "BUY_NO"  # opposite of BUY_YES
    assert signal.contracts == 10


def test_paper_exit_fill():
    pos = _make_position()
    fill = _paper_exit_fill(pos, exit_price=0.60, contracts=5)
    assert fill.quantity == 5
    assert fill.price == pytest.approx(0.595)
    assert fill.status == "paper"
    assert fill.order_id.startswith("paper-exit-")


def test_compute_exit_pnl_buy_yes():
    pos = _make_position(direction="BUY_YES", entry_price=0.55, quantity=10)
    fill = _paper_exit_fill(pos, exit_price=0.62, contracts=10)
    pnl = _compute_exit_pnl(pos, fill)
    # exit fill_price = 0.62 - 0.005 spread = 0.615
    # (0.615 - 0.55) * 10 = 0.65
    assert pnl == pytest.approx(0.65)


def test_compute_exit_pnl_buy_no():
    pos = _make_position(direction="BUY_NO", entry_price=0.45, quantity=10)
    fill = _paper_exit_fill(pos, exit_price=0.60, contracts=10)
    pnl = _compute_exit_pnl(pos, fill)
    # exit fill_price = 0.60 + 0.005 spread = 0.605
    # ((1.0 - 0.605) - 0.45) * 10 = -0.55
    assert pnl == pytest.approx(-0.55)


# ── Execution loop smoke tests ───────────────────────────────


@pytest.mark.asyncio
async def test_execution_loop_finished_immediately():
    """Feed a FINISHED payload → loop exits cleanly, returns MatchPnL."""
    from src.execution.execution_loop import execution_loop

    queue = asyncio.Queue(maxsize=1)
    finished_payload = _make_payload(engine_phase="FINISHED", score=(2, 2))
    await queue.put(finished_payload)

    # Mock model
    model = SimpleNamespace(
        match_id="test_match",
        p_kalshi={},
        kalshi_tickers={},
    )

    # Mock db_pool with async methods that exposure_manager needs
    mock_db = AsyncMock()

    async def _mock_fetchrow(query, *args):
        if "bankroll" in query.lower() or "balance" in query.lower():
            return {"balance": 10000.0}
        if "exposure" in query.lower() or "sum" in query.lower():
            return {"total": 0.0}
        return {"balance": 10000.0}

    mock_db.fetchrow = _mock_fetchrow
    mock_db.fetchval = AsyncMock(return_value=1)
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.execute = AsyncMock(return_value="UPDATE 0")

    result = await execution_loop(
        queue, model, mock_db, TradingMode.PAPER, redis_client=None
    )
    assert isinstance(result, MatchPnL)
    assert result.match_id == "test_match"


@pytest.mark.asyncio
async def test_execution_loop_processes_ticks():
    """Feed 5 ticks then FINISHED → no crash."""
    from src.execution.execution_loop import execution_loop

    queue = asyncio.Queue(maxsize=10)

    # 5 normal ticks — order_allowed=False to avoid DB calls
    for i in range(5):
        await queue.put(_make_payload(t=45.0 + i, order_allowed=False))
    # FINISHED
    await queue.put(_make_payload(engine_phase="FINISHED", score=(1, 0)))

    model = SimpleNamespace(
        match_id="test_match",
        p_kalshi={},
        kalshi_tickers={},
    )

    mock_db = AsyncMock()

    async def _mock_fetchrow(query, *args):
        if "bankroll" in query.lower() or "balance" in query.lower():
            return {"balance": 10000.0}
        if "exposure" in query.lower() or "sum" in query.lower():
            return {"total": 0.0}
        return {"balance": 10000.0}

    mock_db.fetchrow = _mock_fetchrow
    mock_db.fetchval = AsyncMock(return_value=1)
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.execute = AsyncMock(return_value="UPDATE 0")

    result = await asyncio.wait_for(
        execution_loop(queue, model, mock_db, TradingMode.PAPER, redis_client=None),
        timeout=10.0,
    )
    assert isinstance(result, MatchPnL)
    assert result.match_id == "test_match"
