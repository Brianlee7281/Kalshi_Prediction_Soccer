"""Phase 4 exposure management: reserve-confirm-release pattern.

Uses asyncpg for PostgreSQL access. Implements Pattern 4 from patterns.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.common.types import TradingMode
from src.execution.config import CONFIG

if TYPE_CHECKING:
    import asyncpg

log = structlog.get_logger("exposure_manager")


class ExposureManager:
    """Database-backed exposure tracking with cap enforcement."""

    def __init__(self, db_pool: asyncpg.Pool, trading_mode: TradingMode) -> None:
        self.db = db_pool
        self.trading_mode = trading_mode
        self.total_exposure_cap = CONFIG.TOTAL_EXPOSURE_CAP_FRAC
        self.per_match_cap = CONFIG.PER_MATCH_CAP_FRAC

    async def get_bankroll(self) -> float:
        """Get current bankroll balance for this trading mode."""
        row = await self.db.fetchrow(
            "SELECT balance FROM bankroll WHERE mode = $1",
            self.trading_mode.value,
        )
        return float(row["balance"]) if row else 0.0

    async def reserve_exposure(
        self, match_id: str, ticker: str, amount: float
    ) -> int | None:
        """Reserve exposure, respecting total cap. Returns reservation id or None."""
        row = await self.db.fetchrow(
            "SELECT COALESCE(SUM(reserved_amount), 0) AS total "
            "FROM exposure_reservation WHERE status IN ('RESERVED', 'CONFIRMED')"
        )
        current_total = float(row["total"])
        bankroll = await self.get_bankroll()

        if current_total + amount > bankroll * self.total_exposure_cap:
            log.warning(
                "exposure_cap_exceeded",
                current=current_total,
                requested=amount,
                cap=bankroll * self.total_exposure_cap,
            )
            return None

        res_id = await self.db.fetchval(
            "INSERT INTO exposure_reservation (match_id, ticker, reserved_amount, status) "
            "VALUES ($1, $2, $3, 'RESERVED') RETURNING id",
            match_id,
            ticker,
            amount,
        )
        log.info(
            "exposure_reserved",
            reservation_id=res_id,
            match_id=match_id,
            amount=amount,
        )
        return res_id

    async def confirm_exposure(
        self, reservation_id: int, actual_amount: float
    ) -> None:
        """Confirm a reservation after successful fill."""
        await self.db.execute(
            "UPDATE exposure_reservation "
            "SET status = 'CONFIRMED', reserved_amount = $2, resolved_at = NOW() "
            "WHERE id = $1",
            reservation_id,
            actual_amount,
        )

    async def release_exposure(self, reservation_id: int) -> None:
        """Release a reservation (order failed, cancelled, or position closed)."""
        await self.db.execute(
            "UPDATE exposure_reservation "
            "SET status = 'RELEASED', resolved_at = NOW() "
            "WHERE id = $1 AND status IN ('RESERVED', 'CONFIRMED')",
            reservation_id,
        )

    async def release_stale_reservations(self, max_age_seconds: int = 60) -> int:
        """Release RESERVED entries older than max_age_seconds."""
        result = await self.db.execute(
            "UPDATE exposure_reservation "
            "SET status = 'RELEASED', resolved_at = NOW() "
            "WHERE status = 'RESERVED' "
            f"AND created_at < NOW() - INTERVAL '{max_age_seconds} seconds'"
        )
        # asyncpg returns "UPDATE N"
        count = int(result.split()[-1]) if result else 0
        if count > 0:
            log.warning("stale_reservations_released", count=count)
        return count

    async def update_bankroll(self, delta: float) -> None:
        """Update bankroll balance and record snapshot."""
        await self.db.execute(
            "UPDATE bankroll SET balance = balance + $1, updated_at = NOW() "
            "WHERE mode = $2",
            delta,
            self.trading_mode.value,
        )
        await self.db.execute(
            "INSERT INTO bankroll_snapshot (mode, balance) "
            "SELECT mode, balance FROM bankroll WHERE mode = $1",
            self.trading_mode.value,
        )
