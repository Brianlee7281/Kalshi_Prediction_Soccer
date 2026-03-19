"""Phase 4 position persistence: save/close/query positions in PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.types import Position

if TYPE_CHECKING:
    import asyncpg


async def save_position(db: asyncpg.Pool, position: Position) -> int:
    """Insert a new position into the database. Returns the row id."""
    return await db.fetchval(
        "INSERT INTO positions "
        "(match_id, ticker, direction, quantity, entry_price, status, is_paper, entry_tick, entry_reason) "
        "VALUES ($1, $2, $3, $4, $5, 'OPEN', $6, $7, $8) "
        "RETURNING id",
        position.match_id,
        position.ticker,
        position.direction,
        position.quantity,
        position.entry_price,
        position.is_paper,
        position.entry_tick,
        position.market_type,
    )


async def close_position_db(
    db: asyncpg.Pool,
    position_id: int,
    exit_price: float,
    exit_tick: int,
    exit_reason: str,
    realized_pnl: float,
) -> None:
    """Close a position in the database."""
    await db.execute(
        "UPDATE positions "
        "SET status = 'CLOSED', exit_price = $2, exit_tick = $3, "
        "exit_reason = $4, realized_pnl = $5, closed_at = NOW() "
        "WHERE id = $1",
        position_id,
        exit_price,
        exit_tick,
        exit_reason,
        realized_pnl,
    )


async def get_open_positions(db: asyncpg.Pool, match_id: str) -> list[dict]:
    """Get all open positions for a match."""
    rows = await db.fetch(
        "SELECT * FROM positions WHERE match_id = $1 AND status = 'OPEN'",
        match_id,
    )
    return [dict(row) for row in rows]
