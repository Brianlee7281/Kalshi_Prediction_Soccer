"""Tests for src/execution/pnl_calculator.py."""

from __future__ import annotations

import pytest

from src.common.types import Position
from src.execution.pnl_calculator import compute_settlement_pnl, compute_unrealized_pnl


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
    )


# ── compute_unrealized_pnl ───────────────────────────────────


def test_unrealized_buy_yes_profit():
    pos = _make_position(direction="BUY_YES", entry_price=0.55, quantity=10)
    result = compute_unrealized_pnl(pos, p_kalshi=0.62)
    assert result == pytest.approx(0.70)


def test_unrealized_buy_yes_loss():
    pos = _make_position(direction="BUY_YES", entry_price=0.55, quantity=10)
    result = compute_unrealized_pnl(pos, p_kalshi=0.48)
    assert result == pytest.approx(-0.70)


def test_unrealized_buy_no():
    pos = _make_position(direction="BUY_NO", entry_price=0.45, quantity=10)
    result = compute_unrealized_pnl(pos, p_kalshi=0.62)
    assert result == pytest.approx(-0.70)


# ── compute_settlement_pnl ───────────────────────────────────


def test_settlement_buy_yes_won():
    pos = _make_position(direction="BUY_YES", entry_price=0.55, quantity=10)
    result = compute_settlement_pnl(pos, outcome_occurred=True)
    assert result == pytest.approx(4.50)


def test_settlement_buy_yes_lost():
    pos = _make_position(direction="BUY_YES", entry_price=0.55, quantity=10)
    result = compute_settlement_pnl(pos, outcome_occurred=False)
    assert result == pytest.approx(-5.50)


def test_settlement_buy_no_won():
    pos = _make_position(direction="BUY_NO", entry_price=0.45, quantity=10)
    result = compute_settlement_pnl(pos, outcome_occurred=False)
    assert result == pytest.approx(5.50)


def test_settlement_buy_no_lost():
    pos = _make_position(direction="BUY_NO", entry_price=0.45, quantity=10)
    result = compute_settlement_pnl(pos, outcome_occurred=True)
    assert result == pytest.approx(-4.50)
