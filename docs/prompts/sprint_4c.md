Implement Sprint 4c: Order Management + Exposure for the Phase 4 execution engine.

This sprint adds the database layer (PostgreSQL via asyncpg) and order execution. Paper mode fills immediately at P_kalshi. Live mode posts limit orders at P_model (NOT P_kalshi). Includes order repricing when P_model drifts, and Kalshi rejection handling.

Read these files before writing any code:
- `src/common/types.py` — all types including Signal, FillResult, Position, TradingMode
- `src/execution/config.py` — CONFIG constants
- `src/execution/position_monitor.py` — PositionTracker (you'll use it in integration test)
- `src/clients/kalshi.py` — KalshiClient with submit_order(), cancel_order() methods
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 4c section — this is your spec

Do NOT modify Sprint 4a/4b files. Do NOT modify `src/engine/` or `src/clients/kalshi.py`.

## Step 1: Add types to `src/common/types.py`

After the `ExitDecision` class, add:

```python
class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ExposureStatus(str, Enum):
    RESERVED = "reserved"
    CONFIRMED = "confirmed"
    RELEASED = "released"
```

## Step 2: Create `migrations/004_execution_tables.sql`

```sql
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
CREATE INDEX IF NOT EXISTS idx_positions_match ON positions(match_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

CREATE TABLE IF NOT EXISTS exposure_reservation (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    reserved_amount DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'RESERVED',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_reservation_status ON exposure_reservation(status);

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
```

## Step 3: Create `src/execution/order_manager.py`

### Class: `OrderManager`

**Constructor:** `__init__(self, kalshi_client: KalshiClient | None, trading_mode: TradingMode, db_pool: asyncpg.Pool)`
- `self.kalshi_client = kalshi_client`
- `self.trading_mode = trading_mode`
- `self.db = db_pool`
- `self.pending_orders: dict[str, dict] = {}` — order_id → `{"signal": Signal, "placed_at": float, "order_p_model": float}`
- `self.max_order_age_s = CONFIG.MAX_ORDER_LIFETIME_S` (30.0)
- `self.reprice_threshold = CONFIG.REPRICE_THRESHOLD` (0.02)
- `self.ticker_muted: dict[str, bool] = {}` — ticker → True if Kalshi returned "market_closed"
- `self.entries_halted: bool = False` — True if Kalshi returned "insufficient_balance"

**Method: `is_ticker_muted(self, ticker: str) -> bool`**
- Return `self.ticker_muted.get(ticker, False)`

**Method: `async def place_order(self, signal: Signal) -> FillResult | None`**

CRITICAL: Returns `None` (not FillResult) when muted or halted.

Logic:
1. If `self.entries_halted`: log warning, return `None`
2. If `self.is_ticker_muted(signal.ticker)`: log debug, return `None`
3. If `self.trading_mode == TradingMode.PAPER`:
   - Create FillResult with `order_id=f"paper-{uuid4()}"`, `status="paper"`, `quantity=signal.contracts`, `price=signal.P_kalshi`, `fill_cost=signal.contracts * signal.P_kalshi`, `timestamp=datetime.now(timezone.utc)`
   - Return it
4. If `self.trading_mode == TradingMode.LIVE`:
   - CRITICAL: limit order price is `int(signal.P_model * 100)` — this is P_model, NOT P_kalshi. We post at fair value and wait for fills. Posting at P_kalshi would be market-taking.
   - Build order dict: `{"ticker": signal.ticker, "action": "buy", "side": "yes" if signal.direction == "BUY_YES" else "no", "type": "limit", "count": signal.contracts, "yes_price": int(signal.P_model * 100)}`
   - Call `self.kalshi_client.submit_order(order)` inside try/except
   - On success: parse response, store in pending_orders, return FillResult
   - On `httpx.HTTPStatusError`:
     - Status 429: log "kalshi_rate_limited", return `FillResult(status="rejected", quantity=0, ...)`
     - Body contains "market_closed": `self.ticker_muted[signal.ticker] = True`, log "ticker_muted", return `None`
     - Body contains "insufficient_balance": `self.entries_halted = True`, log "entries_halted", return `None`
     - Body contains "price_out_of_range": log "price_out_of_range", return `FillResult(status="rejected", quantity=0, ...)`
     - Other: log error, return `FillResult(status="rejected", quantity=0, ...)`
   - On timeout: return `FillResult(status="rejected", quantity=0, ...)`

**Method: `async def cancel_order(self, order_id: str) -> bool`**
- Paper: remove from pending_orders, return True
- Live: call `self.kalshi_client.cancel_order(order_id)`, handle errors, return success

**Method: `async def manage_open_orders(self, current_p_model: dict[str, float], current_time: float) -> list[FillResult]`**

This is called every tick to manage pending limit orders. For each order in `self.pending_orders`:
1. If age > `self.max_order_age_s`: cancel it, log "order_expired"
2. Elif P_model drift > threshold: `abs(current_p_model.get(order["signal"].market_type, order["order_p_model"]) - order["order_p_model"]) > self.reprice_threshold` → cancel old order, re-post at new P_model, log "order_repriced"
3. Return list of any FillResults from repriced orders that got immediate fills

## Step 4: Create `src/execution/exposure_manager.py`

### Class: `ExposureManager`

**Constructor:** `__init__(self, db_pool: asyncpg.Pool, trading_mode: TradingMode)`

**`async def get_bankroll(self) -> float`**
- `SELECT balance FROM bankroll WHERE mode = $1`

**`async def reserve_exposure(self, match_id: str, ticker: str, amount: float) -> int | None`**
- Check total: `SELECT COALESCE(SUM(reserved_amount), 0) FROM exposure_reservation WHERE status IN ('RESERVED', 'CONFIRMED')`
- Get bankroll
- If total + amount > bankroll * CONFIG.TOTAL_EXPOSURE_CAP_FRAC: return None
- INSERT and return id

**`async def confirm_exposure(self, reservation_id: int, actual_amount: float) -> None`**
- UPDATE status='CONFIRMED', reserved_amount=actual_amount

**`async def release_exposure(self, reservation_id: int) -> None`**
- UPDATE status='RELEASED'

**`async def release_stale_reservations(self, max_age_seconds: int = 60) -> int`**
- UPDATE all RESERVED older than max_age_seconds → RELEASED, return count

**`async def update_bankroll(self, delta: float) -> None`**
- UPDATE bankroll balance, INSERT bankroll_snapshot

## Step 5: Create `src/execution/db_positions.py`

Three async functions using asyncpg:

**`async def save_position(db: asyncpg.Pool, position: Position) -> int`**
- INSERT into positions table, RETURNING id

**`async def close_position_db(db: asyncpg.Pool, position_id: int, exit_price: float, exit_tick: int, exit_reason: str, realized_pnl: float) -> None`**
- UPDATE positions SET status='CLOSED', exit_price, exit_tick, exit_reason, realized_pnl, closed_at=NOW()

**`async def get_open_positions(db: asyncpg.Pool, match_id: str) -> list[dict]`**
- SELECT * FROM positions WHERE match_id=$1 AND status='OPEN'

## Step 6: Create tests

`tests/execution/test_order_manager.py`:
- `test_paper_order_immediate_fill`: paper mode → FillResult status="paper", quantity matches signal.contracts
- `test_paper_order_generates_uuid`: order_id starts with "paper-"
- `test_live_order_uses_p_model_price`: Mock kalshi_client.submit_order, verify the order dict has `yes_price=int(signal.P_model * 100)` NOT `int(signal.P_kalshi * 100)`
- `test_manage_open_orders_cancels_stale`: insert order 35s ago → cancelled
- `test_manage_open_orders_reprices_on_drift`: order at P_model=0.45, current P_model=0.48 (drift=0.03 > 0.02) → order cancelled and re-posted
- `test_manage_open_orders_no_reprice_small_drift`: drift=0.01 < 0.02 → order unchanged
- `test_ticker_muted_on_market_closed`: mock Kalshi raising HTTPStatusError with "market_closed" body → ticker_muted=True, subsequent calls return None
- `test_entries_halted_on_insufficient_balance`: mock "insufficient_balance" → entries_halted=True, all calls return None
- `test_price_out_of_range_returns_rejected`: mock "price_out_of_range" → FillResult(status="rejected")
- `test_place_order_returns_none_when_halted`: entries_halted=True → returns None without calling Kalshi

For exposure_manager and db_positions tests: these need a real PostgreSQL test database. If the test infrastructure for asyncpg doesn't exist yet, create pytest fixtures that:
1. Create a test database (or use a test schema)
2. Run the migration SQL
3. Provide an asyncpg.Pool
4. Clean up after

`tests/execution/test_exposure_manager.py`:
- `test_reserve_within_cap`: bankroll=10000, reserve $50 → returns reservation id
- `test_reserve_exceeds_cap`: bankroll=100, cap=20%, reserve $50 → None
- `test_confirm_updates_status`: reserve → confirm → check DB status=CONFIRMED
- `test_release_updates_status`: reserve → release → check DB status=RELEASED
- `test_stale_release`: insert old RESERVED row → release_stale → count=1
- `test_bankroll_update`: delta=-5.50 → balance decremented, snapshot inserted

## Step 7: Verify

1. `python -m pytest tests/execution/test_order_manager.py -v`
2. `python -m pytest tests/execution/test_exposure_manager.py -v` (requires PostgreSQL)
3. `python -m pytest tests/execution/ -v` — all Sprint 4a + 4b + 4c tests pass
4. Existing tests unaffected
