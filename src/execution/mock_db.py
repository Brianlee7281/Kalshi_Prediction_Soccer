"""In-memory mock for asyncpg.Pool, used in paper backtesting.

Routes the SQL queries used by ExposureManager and db_positions to
in-memory dict/list operations. Only the ~8 distinct queries actually
used by Phase 4 are handled — not a general SQL engine.
"""

from __future__ import annotations

import time
from typing import Any


class _MockRecord(dict):
    """Dict subclass that supports attribute access like asyncpg.Record."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class MockDBPool:
    """In-memory mock for asyncpg.Pool, used in paper backtesting.

    Supports the specific SQL patterns used by:
    - ExposureManager (bankroll, exposure_reservation)
    - db_positions (positions table)
    """

    def __init__(self, initial_bankroll: float = 10_000.0) -> None:
        self.bankroll: float = initial_bankroll
        self.mode: str = "paper"

        # exposure_reservation table
        self.reservations: list[dict] = []
        self._next_reservation_id: int = 1

        # positions table
        self.positions: list[dict] = []
        self._next_position_id: int = 1

        # bankroll_snapshot table
        self.snapshots: list[dict] = []

    # ── asyncpg.Pool interface ────────────────────────────────────

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute query and return a single value."""
        q = query.strip().upper()

        # SELECT balance FROM bankroll WHERE mode = $1
        if "BALANCE" in q and "BANKROLL" in q and "SELECT" in q:
            return self.bankroll

        # INSERT INTO exposure_reservation ... RETURNING id
        if "EXPOSURE_RESERVATION" in q and "INSERT" in q:
            return self._insert_reservation(*args)

        # INSERT INTO positions ... RETURNING id
        if "POSITIONS" in q and "INSERT" in q:
            return self._insert_position(*args)

        return None

    async def fetchrow(self, query: str, *args: Any) -> _MockRecord | None:
        """Execute query and return a single row."""
        q = query.strip().upper()

        # SELECT balance FROM bankroll WHERE mode = $1
        if "BALANCE" in q and "BANKROLL" in q:
            return _MockRecord({"balance": self.bankroll})

        # SELECT COALESCE(SUM(reserved_amount), 0) AS total
        # FROM exposure_reservation WHERE status IN ('RESERVED', 'CONFIRMED')
        if "EXPOSURE_RESERVATION" in q and "SUM" in q:
            total = sum(
                r["reserved_amount"]
                for r in self.reservations
                if r["status"] in ("RESERVED", "CONFIRMED")
            )
            return _MockRecord({"total": total})

        return None

    async def fetch(self, query: str, *args: Any) -> list[_MockRecord]:
        """Execute query and return multiple rows."""
        q = query.strip().upper()

        # SELECT * FROM positions WHERE match_id = $1 AND status = 'OPEN'
        if "POSITIONS" in q and "OPEN" in q:
            match_id = args[0] if args else None
            return [
                _MockRecord(p)
                for p in self.positions
                if p.get("match_id") == match_id and p.get("status") == "OPEN"
            ]

        return []

    async def execute(self, query: str, *args: Any) -> str:
        """Execute query and return status string."""
        q = query.strip().upper()

        # UPDATE exposure_reservation SET status = 'CONFIRMED' ...
        if "EXPOSURE_RESERVATION" in q and "CONFIRMED" in q:
            self._confirm_reservation(*args)
            return "UPDATE 1"

        # UPDATE exposure_reservation SET status = 'RELEASED' WHERE id = $1
        if "EXPOSURE_RESERVATION" in q and "RELEASED" in q and "WHERE ID" in q:
            self._release_reservation(*args)
            return "UPDATE 1"

        # UPDATE exposure_reservation SET status = 'RELEASED' WHERE status = 'RESERVED' AND created_at < ...
        if "EXPOSURE_RESERVATION" in q and "RELEASED" in q and "RESERVED" in q:
            count = self._release_stale_reservations()
            return f"UPDATE {count}"

        # UPDATE bankroll SET balance = balance + $1
        if "BANKROLL" in q and "BALANCE" in q and "UPDATE" in q:
            if args:
                self.bankroll += float(args[0])
            return "UPDATE 1"

        # INSERT INTO bankroll_snapshot
        if "BANKROLL_SNAPSHOT" in q and "INSERT" in q:
            self.snapshots.append({"balance": self.bankroll, "ts": time.time()})
            return "INSERT 0 1"

        # UPDATE positions SET status = 'CLOSED' ...
        if "POSITIONS" in q and "CLOSED" in q:
            self._close_position(*args)
            return "UPDATE 1"

        return "UPDATE 0"

    # ── Internal operations ───────────────────────────────────────

    def _insert_reservation(self, *args: Any) -> int:
        """INSERT INTO exposure_reservation (match_id, ticker, reserved_amount, status)."""
        rid = self._next_reservation_id
        self._next_reservation_id += 1
        self.reservations.append({
            "id": rid,
            "match_id": args[0] if len(args) > 0 else "",
            "ticker": args[1] if len(args) > 1 else "",
            "reserved_amount": float(args[2]) if len(args) > 2 else 0.0,
            "status": "RESERVED",
            "created_at": time.time(),
            "resolved_at": None,
        })
        return rid

    def _confirm_reservation(self, *args: Any) -> None:
        """UPDATE exposure_reservation SET status='CONFIRMED' WHERE id=$1."""
        if not args:
            return
        rid = args[0]
        actual_amount = float(args[1]) if len(args) > 1 else None
        for r in self.reservations:
            if r["id"] == rid:
                r["status"] = "CONFIRMED"
                r["resolved_at"] = time.time()
                if actual_amount is not None:
                    r["reserved_amount"] = actual_amount
                break

    def _release_reservation(self, *args: Any) -> None:
        """UPDATE exposure_reservation SET status='RELEASED' WHERE id=$1."""
        if not args:
            return
        rid = args[0]
        for r in self.reservations:
            if r["id"] == rid:
                r["status"] = "RELEASED"
                r["resolved_at"] = time.time()
                break

    def _release_stale_reservations(self, max_age_seconds: int = 60) -> int:
        """Release RESERVED entries older than max_age_seconds."""
        now = time.time()
        count = 0
        for r in self.reservations:
            if r["status"] == "RESERVED" and (now - r["created_at"]) > max_age_seconds:
                r["status"] = "RELEASED"
                r["resolved_at"] = now
                count += 1
        return count

    def _insert_position(self, *args: Any) -> int:
        """INSERT INTO positions (...) RETURNING id."""
        pid = self._next_position_id
        self._next_position_id += 1
        self.positions.append({
            "id": pid,
            "match_id": args[0] if len(args) > 0 else "",
            "ticker": args[1] if len(args) > 1 else "",
            "direction": args[2] if len(args) > 2 else "",
            "quantity": int(args[3]) if len(args) > 3 else 0,
            "entry_price": float(args[4]) if len(args) > 4 else 0.0,
            "status": "OPEN",
            "is_paper": bool(args[5]) if len(args) > 5 else True,
            "entry_tick": int(args[6]) if len(args) > 6 else 0,
            "entry_reason": args[7] if len(args) > 7 else "",
            "exit_price": None,
            "exit_tick": None,
            "exit_reason": None,
            "realized_pnl": None,
        })
        return pid

    def _close_position(self, *args: Any) -> None:
        """UPDATE positions SET status='CLOSED', exit_price=$2, ... WHERE id=$1."""
        if not args:
            return
        pid = args[0]
        for p in self.positions:
            if p["id"] == pid:
                p["status"] = "CLOSED"
                p["exit_price"] = float(args[1]) if len(args) > 1 else None
                p["exit_tick"] = int(args[2]) if len(args) > 2 else None
                p["exit_reason"] = args[3] if len(args) > 3 else None
                p["realized_pnl"] = float(args[4]) if len(args) > 4 else None
                break
