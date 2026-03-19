"""Phase 4 PnL computation: unrealized and settlement.

Pure functions — no database, no network, no API calls.
"""

from __future__ import annotations

from src.common.types import Position


def compute_unrealized_pnl(position: Position, p_kalshi: float) -> float:
    """Mark-to-market unrealized PnL based on current Kalshi price."""
    if position.direction == "BUY_YES":
        return (p_kalshi - position.entry_price) * position.quantity
    else:
        return ((1.0 - p_kalshi) - position.entry_price) * position.quantity


def compute_settlement_pnl(position: Position, outcome_occurred: bool) -> float:
    """Final PnL at contract settlement."""
    if position.direction == "BUY_YES":
        if outcome_occurred:
            return (1.0 - position.entry_price) * position.quantity
        else:
            return -position.entry_price * position.quantity
    else:
        if outcome_occurred:
            return -position.entry_price * position.quantity
        else:
            return (1.0 - position.entry_price) * position.quantity
