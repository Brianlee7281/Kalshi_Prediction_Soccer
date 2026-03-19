"""Tests for src/execution/exposure_manager.py.

These tests require a running PostgreSQL instance. They are skipped if
asyncpg is not installed or the database is not available.
"""

from __future__ import annotations

import os

import pytest

try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from src.common.types import TradingMode

# Default test DB URL — override via TEST_DATABASE_URL env var
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
CREATE TABLE IF NOT EXISTS exposure_reservation (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    reserved_amount DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'RESERVED',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS bankroll (
    mode TEXT PRIMARY KEY,
    balance DECIMAL(12,2) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS bankroll_snapshot (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    balance DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO bankroll (mode, balance) VALUES ('paper', 10000.00) ON CONFLICT (mode) DO NOTHING;
INSERT INTO bankroll (mode, balance) VALUES ('live', 0.00) ON CONFLICT (mode) DO NOTHING;
"""

pytestmark = pytest.mark.skipif(
    not HAS_ASYNCPG, reason="asyncpg not installed"
)


@pytest.fixture
async def db_pool():
    """Create a test database pool, run migrations, and clean up after."""
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
    except Exception:
        pytest.skip("PostgreSQL not available")
        return

    # Run migration
    async with pool.acquire() as conn:
        await conn.execute(MIGRATION_SQL)

    yield pool

    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM exposure_reservation")
        await conn.execute("DELETE FROM bankroll_snapshot")
        await conn.execute("DELETE FROM positions")
        await conn.execute("UPDATE bankroll SET balance = 10000.00 WHERE mode = 'paper'")

    await pool.close()


@pytest.mark.asyncio
async def test_reserve_within_cap(db_pool):
    from src.execution.exposure_manager import ExposureManager

    em = ExposureManager(db_pool, TradingMode.PAPER)
    res_id = await em.reserve_exposure("match-1", "TICKER-1", 50.0)
    assert res_id is not None
    assert isinstance(res_id, int)


@pytest.mark.asyncio
async def test_reserve_exceeds_cap(db_pool):
    from src.execution.exposure_manager import ExposureManager

    # Set bankroll to 100 → cap = 20% = $20
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE bankroll SET balance = 100.00 WHERE mode = 'paper'")

    em = ExposureManager(db_pool, TradingMode.PAPER)
    res_id = await em.reserve_exposure("match-1", "TICKER-1", 50.0)
    assert res_id is None

    # Restore
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE bankroll SET balance = 10000.00 WHERE mode = 'paper'")


@pytest.mark.asyncio
async def test_confirm_updates_status(db_pool):
    from src.execution.exposure_manager import ExposureManager

    em = ExposureManager(db_pool, TradingMode.PAPER)
    res_id = await em.reserve_exposure("match-1", "TICKER-1", 50.0)
    assert res_id is not None

    await em.confirm_exposure(res_id, 45.0)

    row = await db_pool.fetchrow(
        "SELECT status, reserved_amount FROM exposure_reservation WHERE id = $1",
        res_id,
    )
    assert row["status"] == "CONFIRMED"
    assert float(row["reserved_amount"]) == 45.0


@pytest.mark.asyncio
async def test_release_updates_status(db_pool):
    from src.execution.exposure_manager import ExposureManager

    em = ExposureManager(db_pool, TradingMode.PAPER)
    res_id = await em.reserve_exposure("match-1", "TICKER-1", 50.0)
    assert res_id is not None

    await em.release_exposure(res_id)

    row = await db_pool.fetchrow(
        "SELECT status FROM exposure_reservation WHERE id = $1", res_id
    )
    assert row["status"] == "RELEASED"


@pytest.mark.asyncio
async def test_stale_release(db_pool):
    from src.execution.exposure_manager import ExposureManager

    # Insert a reservation that looks old
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO exposure_reservation (match_id, ticker, reserved_amount, status, created_at) "
            "VALUES ('match-old', 'TICKER-OLD', 25.0, 'RESERVED', NOW() - INTERVAL '120 seconds')"
        )

    em = ExposureManager(db_pool, TradingMode.PAPER)
    count = await em.release_stale_reservations(max_age_seconds=60)
    assert count >= 1


@pytest.mark.asyncio
async def test_bankroll_update(db_pool):
    from src.execution.exposure_manager import ExposureManager

    em = ExposureManager(db_pool, TradingMode.PAPER)
    before = await em.get_bankroll()

    await em.update_bankroll(-5.50)

    after = await em.get_bankroll()
    assert after == pytest.approx(before - 5.50)

    # Verify snapshot was created
    row = await db_pool.fetchrow(
        "SELECT balance FROM bankroll_snapshot WHERE mode = 'paper' ORDER BY id DESC LIMIT 1"
    )
    assert float(row["balance"]) == pytest.approx(after)
