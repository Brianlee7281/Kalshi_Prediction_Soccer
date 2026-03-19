"""Tests for src/execution/db_positions.py.

These tests require a running PostgreSQL instance.
"""

from __future__ import annotations

import os

import pytest

try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from src.common.types import Position

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/quant_test"
)

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INT NOT NULL,
    entry_price DECIMAL(6,4) NOT NULL,
    exit_price DECIMAL(6,4),
    status TEXT NOT NULL DEFAULT 'OPEN',
    is_paper BOOLEAN NOT NULL DEFAULT TRUE,
    realized_pnl DECIMAL(10,2),
    entry_tick INT,
    exit_tick INT,
    entry_reason TEXT,
    exit_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);
"""

pytestmark = pytest.mark.skipif(
    not HAS_ASYNCPG, reason="asyncpg not installed"
)


@pytest.fixture
async def db_pool():
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
    except Exception:
        pytest.skip("PostgreSQL not available")
        return

    async with pool.acquire() as conn:
        await conn.execute(MIGRATION_SQL)

    yield pool

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM positions")

    await pool.close()


def _make_position() -> Position:
    return Position(
        id="pos-test-1",
        match_id="match-test-1",
        ticker="TICKER-HOME",
        market_type="home_win",
        direction="BUY_YES",
        quantity=10,
        entry_price=0.55,
        entry_tick=100,
        entry_t=35.0,
        is_paper=True,
    )


@pytest.mark.asyncio
async def test_save_and_retrieve(db_pool):
    from src.execution.db_positions import get_open_positions, save_position

    pos = _make_position()
    row_id = await save_position(db_pool, pos)
    assert row_id > 0

    open_positions = await get_open_positions(db_pool, "match-test-1")
    assert len(open_positions) >= 1
    assert open_positions[0]["ticker"] == "TICKER-HOME"
    assert open_positions[0]["status"] == "OPEN"


@pytest.mark.asyncio
async def test_close_position(db_pool):
    from src.execution.db_positions import (
        close_position_db,
        get_open_positions,
        save_position,
    )

    pos = _make_position()
    row_id = await save_position(db_pool, pos)

    await close_position_db(
        db_pool,
        position_id=row_id,
        exit_price=0.62,
        exit_tick=250,
        exit_reason="edge_decay",
        realized_pnl=0.70,
    )

    open_positions = await get_open_positions(db_pool, "match-test-1")
    assert len(open_positions) == 0
