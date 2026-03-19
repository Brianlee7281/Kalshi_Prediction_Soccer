Implement Sprint 6a: REST API + P&L Service for the Phase 6 dashboard backend.

This sprint builds a read-only FastAPI backend that serves match state, positions, P&L analytics, and system status. No WebSocket yet, no frontend — pure REST API. It reads from the PostgreSQL tables populated by Phase 4 (positions, bankroll) and Phase 5 (match_schedule).

Read these files before writing any code:
- `src/common/types.py` — all existing types
- `docs/architecture.md` §5.1 — database schema (positions, match_schedule, tick_snapshots, bankroll, bankroll_snapshot tables)
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 6a section — this is your spec

Do NOT modify any `src/execution/` or `src/orchestrator/` files.

## Step 1: Create `src/dashboard/__init__.py`

Empty file (if it doesn't already exist — check first).

## Step 2: Create `src/dashboard/schemas.py`

Response models for the API. Use Pydantic BaseModel:

```python
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class MatchSummary(BaseModel):
    match_id: str
    league_id: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str
    trading_mode: str
    score: tuple[int, int] | None = None
    engine_phase: str | None = None
    open_positions: int = 0
    unrealized_pnl: float = 0.0


class PositionDetail(BaseModel):
    id: int
    match_id: str
    ticker: str
    direction: str
    quantity: int
    entry_price: float
    exit_price: float | None = None
    status: str
    is_paper: bool
    realized_pnl: float | None = None
    entry_reason: str | None = None
    exit_reason: str | None = None
    created_at: datetime


class PnLSummary(BaseModel):
    total_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl_per_trade: float
    max_drawdown: float
    by_market_type: dict[str, float]
    by_league: dict[str, float]


class SystemStatus(BaseModel):
    active_matches: int
    total_open_positions: int
    paper_bankroll: float
    live_bankroll: float
    last_phase1_run: datetime | None = None
    alerts: list[dict]
```

## Step 3: Create `src/dashboard/pnl_service.py`

**Function:** `async def compute_pnl_summary(db_pool: asyncpg.Pool, trading_mode: str = "paper", match_id: str | None = None) -> PnLSummary`

Logic:
1. Query closed/settled positions: `SELECT * FROM positions WHERE status IN ('CLOSED', 'SETTLED') AND is_paper = (trading_mode == 'paper')`
2. If match_id provided, add `AND match_id = $match_id` filter
3. Compute:
   - `total_pnl = sum(realized_pnl)` for all matching positions
   - `trade_count = len(positions)`
   - `win_count = count where realized_pnl > 0`
   - `loss_count = count where realized_pnl <= 0`
   - `win_rate = win_count / trade_count` if trade_count > 0 else 0.0
   - `avg_pnl_per_trade = total_pnl / trade_count` if trade_count > 0 else 0.0
4. Compute max_drawdown from bankroll_snapshot:
   - `SELECT balance, created_at FROM bankroll_snapshot WHERE mode = $trading_mode ORDER BY created_at`
   - Track peak: for each snapshot, update peak = max(peak, balance), drawdown = peak - balance, max_drawdown = max of all drawdowns
   - If no snapshots: max_drawdown = 0.0
5. Group by market_type: extract from ticker string or from a join with positions
6. Group by league: extract league_id from match_schedule join
7. Return PnLSummary

## Step 4: Create `src/dashboard/api.py`

FastAPI application with these endpoints:

```python
from fastapi import FastAPI, Query
import asyncpg

app = FastAPI(title="MMPP Dashboard API")
```

**`GET /api/matches`** → `list[MatchSummary]`
- Query match_schedule, optionally filtered by `status: str = None` and `league_id: int = None`
- For each match, join with `SELECT COUNT(*) FROM positions WHERE match_id=$1 AND status='OPEN'` for open_positions count
- Return list of MatchSummary

**`GET /api/matches/{match_id}`** → `MatchSummary` + tick history
- Single match from match_schedule
- Include last 100 ticks from tick_snapshots (if they exist)

**`GET /api/matches/{match_id}/ticks`** → `list[dict]`
- `SELECT * FROM tick_snapshots WHERE match_id = $1 ORDER BY t`
- Optional params: `from_t: float = None`, `to_t: float = None`
- Return as list of dicts

**`GET /api/positions`** → `list[PositionDetail]`
- `SELECT * FROM positions` with optional `match_id` and `status` filters

**`GET /api/positions/{position_id}`** → `PositionDetail`

**`GET /api/pnl`** → `PnLSummary`
- Call `compute_pnl_summary(db_pool, trading_mode)` with optional `trading_mode` and `match_id` query params

**`GET /api/pnl/history`** → `list[dict]`
- `SELECT balance, created_at FROM bankroll_snapshot WHERE mode = $1 ORDER BY created_at`

**`GET /api/system/status`** → `SystemStatus`
- Active matches: `SELECT COUNT(*) FROM match_schedule WHERE status = 'PHASE3_RUNNING'`
- Open positions: `SELECT COUNT(*) FROM positions WHERE status = 'OPEN'`
- Bankroll: `SELECT balance FROM bankroll WHERE mode = 'paper'` and `WHERE mode = 'live'`
- Alerts: `redis.lrange("system_alerts", 0, 19)` — parse JSON, return last 20. If Redis unavailable, return empty list.

**`GET /api/system/health`** → `{"status": "ok", "timestamp": "..."}`

The app needs a startup event that creates the asyncpg pool and optionally connects to Redis. Use FastAPI lifespan or on_event. Expose a `create_dashboard_app(db_pool, redis_client=None)` factory for testing.

## Step 5: Create tests

`tests/dashboard/__init__.py` — empty

`tests/dashboard/test_api.py` — use `httpx.AsyncClient` with FastAPI's test client:
```python
from httpx import AsyncClient, ASGITransport
# transport = ASGITransport(app=app)
# async with AsyncClient(transport=transport, base_url="http://test") as client:
```

Tests (all require seeded test DB):
- `test_get_matches_empty`: no rows → returns `[]`
- `test_get_matches_with_data`: insert 2 matches → returns 2 items
- `test_get_matches_filter_status`: insert 3 matches (2 SCHEDULED, 1 FINISHED) → filter by SCHEDULED → 2 items
- `test_get_positions`: insert 3 positions → returns 3
- `test_get_pnl_no_trades`: no closed positions → all zeros, win_rate=0
- `test_get_pnl_with_trades`: insert 2 CLOSED positions (realized_pnl=4.50, -2.00) → total_pnl=2.50, win_count=1, loss_count=1, win_rate=0.5
- `test_system_status`: insert 1 PHASE3_RUNNING match, 2 OPEN positions → active_matches=1, total_open_positions=2
- `test_health_endpoint`: returns 200 with `{"status": "ok"}`

`tests/dashboard/test_pnl_service.py`:
- `test_max_drawdown_calculation`: insert bankroll_snapshot series [100, 110, 95, 108, 90] → peak=110, max_drawdown=20 (110-90)
- `test_pnl_by_market_type`: 2 home_win trades (+3, +2), 1 draw trade (-1) → by_market_type={"home_win": 5.0, "draw": -1.0}
- `test_win_rate`: 3 wins, 2 losses → 0.6

## Step 6: Verify

1. `python -m pytest tests/dashboard/ -v` — all pass
2. Existing tests unaffected
3. `python -c "from src.dashboard.api import app; print(app.title)"` → `MMPP Dashboard API`
