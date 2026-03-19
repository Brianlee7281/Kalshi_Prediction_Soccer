Implement Sprint 4d: Execution Loop + Settlement — the final Phase 4 integration sprint.

This sprint wires all Sprint 4a/4b/4c modules into a single `execution_loop` coroutine that runs alongside Phase 3's tick_loop. It also adds settlement logic. The integration test replays a full recorded match (Brentford 2-2 Wolves, match 4190023) in paper mode and verifies P&L.

Read these files before writing any code:
- `src/execution/signal_generator.py` — `generate_signals()`
- `src/execution/kelly_sizer.py` — `size_position()`
- `src/execution/position_monitor.py` — `PositionTracker`
- `src/execution/order_manager.py` — `OrderManager`
- `src/execution/exposure_manager.py` — `ExposureManager`
- `src/execution/db_positions.py` — `save_position()`, `close_position_db()`
- `src/execution/pnl_calculator.py` — `compute_settlement_pnl()`
- `src/engine/tick_loop.py` — `tick_loop(model, phase4_queue, redis_client)`
- `src/engine/model.py` — `LiveMatchModel` (has `p_kalshi` dict, `kalshi_tickers` dict)
- `src/recorder/replay_server.py` — `ReplayServer`
- `src/clients/kalshi.py` — needs 2 new methods added
- `src/common/types.py` — all types
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 4d section — this is your spec

## Step 1: Add types to `src/common/types.py`

After `ExposureStatus`, add:

```python
class SettlementResult(BaseModel):
    position_id: int
    ticker: str
    market_type: str
    direction: str
    quantity: int
    outcome_occurred: bool
    settlement_price: float  # 1.0 or 0.0
    realized_pnl: float

class MatchPnL(BaseModel):
    match_id: str
    total_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    positions: list[dict]
```

## Step 2: Add 2 methods to `src/clients/kalshi.py`

Add these to the `KalshiClient` class, in the `# ─── Portfolio` section:

```python
async def get_order(self, order_id: str) -> dict:
    """GET /trade-api/v2/portfolio/orders/{order_id}"""
    return await self._get(f"/trade-api/v2/portfolio/orders/{order_id}")

async def get_fills(self, ticker: str, limit: int = 100) -> list[dict]:
    """GET /trade-api/v2/portfolio/fills?ticker={ticker}"""
    data = await self._get(
        "/trade-api/v2/portfolio/fills",
        params={"ticker": ticker, "limit": str(limit)},
    )
    return data.get("fills", [])
```

## Step 3: Create `src/execution/execution_loop.py`

This is the main Phase 4 coroutine. It consumes TickPayloads from an asyncio.Queue and orchestrates signal generation, sizing, ordering, position monitoring, and exits.

**Function:** `async def execution_loop(phase4_queue: asyncio.Queue, model: LiveMatchModel, db_pool: asyncpg.Pool, trading_mode: TradingMode, redis_client: object | None = None) -> MatchPnL`

Initialization:
```python
exposure = ExposureManager(db_pool, trading_mode)
kalshi_client = model.kalshi_client if hasattr(model, 'kalshi_client') else None
orders = OrderManager(kalshi_client, trading_mode, db_pool)
tracker = PositionTracker()  # uses CONFIG defaults: min_hold=150, cooldown=300
bankroll = await exposure.get_bankroll()
stale_check_interval = 300
tick_counter = 0
```

Main loop — execute this on every tick:
```
while True:
    payload = await phase4_queue.get()
    tick_counter += 1

    if payload.engine_phase == "FINISHED":
        break

    # 1. Check exits (EVERY tick, even during cooldown)
    exits = tracker.check_exits(payload, model.p_kalshi)
    for exit_decision in exits:
        pos = tracker.open_positions[exit_decision.position_id]
        if trading_mode == TradingMode.LIVE:
            exit_signal = _build_exit_signal(pos, exit_decision)
            fill = await orders.place_order(exit_signal)
        else:
            fill = _paper_exit_fill(pos, exit_decision.exit_price, exit_decision.contracts_to_exit)
        if fill is not None and fill.quantity > 0:
            realized_pnl = _compute_exit_pnl(pos, fill)
            tracker.close_position(pos.id, exit_decision.trigger, exit_decision.contracts_to_exit, fill.price, tick_counter)
            if hasattr(pos, 'db_id') and pos.db_id:
                await close_position_db(db_pool, pos.db_id, fill.price, tick_counter, exit_decision.trigger.value, realized_pnl)
            await exposure.update_bankroll(realized_pnl)
            bankroll += realized_pnl
            if redis_client:
                await _publish_position_update(redis_client, pos, "exit")

    # 2. Generate + execute new signals (only if order_allowed AND not halted)
    if payload.order_allowed and not orders.entries_halted:
        signals = generate_signals(payload, model.p_kalshi, model.kalshi_tickers, tracker.open_positions)
        for signal in signals:
            if tracker.is_in_cooldown(signal.market_type, tick_counter):
                continue
            if orders.is_ticker_muted(signal.ticker):
                continue
            signal = size_position(signal, payload, bankroll)
            if signal.contracts <= 0:
                continue
            amount = signal.contracts * signal.P_kalshi
            res_id = await exposure.reserve_exposure(payload.match_id, signal.ticker, amount)
            if res_id is None:
                continue
            fill = await orders.place_order(signal)
            if fill is not None and fill.quantity > 0:
                await exposure.confirm_exposure(res_id, fill.fill_cost)
                pos = tracker.add_position(signal, fill, tick_counter, payload.t)
                pos.db_id = await save_position(db_pool, pos)
                await exposure.update_bankroll(-fill.fill_cost)
                bankroll -= fill.fill_cost
                if redis_client:
                    await _publish_position_update(redis_client, pos, "new_fill")
                    await _publish_signal(redis_client, signal, fill)
            else:
                await exposure.release_exposure(res_id)

    # 3. Stale reservation cleanup every 300 ticks
    if tick_counter % stale_check_interval == 0:
        await exposure.release_stale_reservations()

    # 4. Manage open orders (cancel stale + reprice on drift)
    current_p_model = {}
    for mt in ["home_win", "draw", "away_win", "over_25", "btts_yes"]:
        val = getattr(payload.P_model, mt, None)
        if val is not None:
            current_p_model[mt] = val
    await orders.manage_open_orders(current_p_model, time.monotonic())
```

After loop breaks (FINISHED): call `settle_match()` and return MatchPnL.

Also implement these helper functions in the same file:
- `_build_exit_signal(pos, exit_decision) -> Signal` — builds a Signal for exiting
- `_paper_exit_fill(pos, exit_price, contracts) -> FillResult` — simulates paper exit fill
- `_compute_exit_pnl(pos, fill) -> float` — computes PnL for exit at fill price

## Step 4: Create `src/execution/settlement.py`

**Function:** `async def poll_kalshi_settlement(kalshi_client: KalshiClient, tickers: list[str], timeout_min: int = 45, interval_s: int = 60) -> dict[str, bool]`
- Polls `kalshi_client.get_market(ticker)` for each unsettled ticker
- Checks the `result` field — if non-null, the ticker is settled (`result == "yes"` means True)
- Loops every `interval_s` seconds until all settled or `timeout_min` minutes elapsed
- Returns `{ticker: outcome_occurred}` — may be partial if timeout

**Function:** `async def settle_match(match_id, final_score, tracker, db_pool, kalshi_client, trading_mode) -> MatchPnL`
- Derive outcomes from score:
  - `home_win = final_score[0] > final_score[1]`
  - `draw = final_score[0] == final_score[1]`
  - `away_win = final_score[0] < final_score[1]`
  - `over_25 = sum(final_score) >= 3`
  - `btts_yes = final_score[0] >= 1 and final_score[1] >= 1`
- In LIVE mode: call `poll_kalshi_settlement()`, cross-check against score outcomes, log any mismatch, use Kalshi result as authoritative
- In PAPER mode: use score-derived outcomes directly
- For each open position: map market_type to outcome, compute PnL via `compute_settlement_pnl()`, close in DB with status='SETTLED', update bankroll
- Return MatchPnL summary

## Step 5: Create `src/execution/redis_publisher.py`

Two helper functions:
- `async def _publish_position_update(redis_client, position, update_type)` — build PositionUpdateMessage, publish to "position_update" channel
- `async def _publish_signal(redis_client, signal, fill)` — build SignalMessage, publish to f"signal:{signal.match_id}"

Both should catch exceptions and log warnings (Redis failure should not crash execution).

## Step 6: Create tests

`tests/execution/test_settlement.py`:
- `test_settle_home_win`: score=(2,1) → home_win position profits
- `test_settle_draw`: score=(2,2) → draw position profits
- `test_settle_over_25`: score=(2,1) total=3 → over_25 profits
- `test_settle_under_25`: score=(1,0) total=1 → over_25 loses
- `test_settle_btts_yes`: score=(2,1) both scored → btts profits
- `test_settle_btts_no`: score=(1,0) away=0 → btts loses
- `test_settle_no_positions`: empty tracker → total_pnl=0
- `test_poll_kalshi_settlement_success`: mock get_market returns None twice then {"result": "yes"} → {ticker: True}
- `test_poll_kalshi_settlement_timeout`: mock always returns None → empty dict + logged error
- `test_settle_paper_skips_polling`: paper mode → no kalshi_client calls

`tests/execution/test_execution_loop.py`:
- Basic smoke test: feed 5 TickPayloads → no crash, returns MatchPnL
- Test with FINISHED payload → loop exits cleanly

The full match replay integration test (the critical one) — `test_sprint_4d_full_match_replay`:
- Setup: create test DB, run migration, start ReplayServer for match 4190023 at speed=100.0
- Run `tick_loop` and `execution_loop` concurrently via asyncio.gather
- Assert: `match_pnl.trade_count >= 1`
- Assert: draw positions have positive realized_pnl (it's a 2-2 draw)
- Assert: `abs(final_bankroll - (10000.0 + match_pnl.total_pnl)) < 0.01` (bankroll reconciliation)
- Assert: no orphaned RESERVED rows in exposure_reservation
- Assert: no OPEN positions remain after settlement

## Step 7: Verify

1. `python -m pytest tests/execution/test_settlement.py -v`
2. `python -m pytest tests/execution/test_execution_loop.py -v`
3. `python -m pytest tests/execution/ -v` — ALL Sprint 4a-4d tests pass
4. Existing tests unaffected
