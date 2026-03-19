Implement Sprint 4b: Position Tracking + Exit Logic for the Phase 4 execution engine.

This sprint builds in-memory position tracking and the 6-trigger exit decision system. No database, no network — pure state management. It depends on Sprint 4a being complete.

Read these files before writing any code:
- `src/common/types.py` — understand existing types including Signal, FillResult, TickPayload, TradingMode, ExitTrigger (added in Sprint 4a)
- `src/execution/config.py` — CONFIG with MIN_HOLD_TICKS=150, COOLDOWN_AFTER_EXIT=300, EKF_DIVERGENCE_THRESHOLD=1.5, EXPIRY_EVAL_MINUTE=85.0
- `src/execution/signal_generator.py` — `compute_edge()`, `compute_dynamic_threshold()`, `_get_market_mu()`, `_get_market_ekf_P()` (you will import these)
- `src/execution/kelly_sizer.py` — `compute_kelly_fraction()` (used for Trigger 3 POSITION_TRIM)
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 4b section — this is your spec

Do NOT modify any files in `src/engine/`, `src/clients/`, or Sprint 4a files (`config.py`, `signal_generator.py`, `kelly_sizer.py`).

## Step 1: Add types to `src/common/types.py`

After the `ExitTrigger` class (added in Sprint 4a), add:

```python
class Position(BaseModel):
    """In-memory position tracking for Phase 4.

    entry_price semantics:
    - For BUY_YES: entry_price = P_kalshi (the YES price paid)
    - For BUY_NO:  entry_price = 1 - P_kalshi (the NO price paid)
    In both cases, entry_price is the actual dollar cost per contract.
    PnL formulas in pnl_calculator.py depend on this convention.
    """
    id: str  # UUID
    match_id: str
    ticker: str
    market_type: str
    direction: str  # "BUY_YES" | "BUY_NO"
    quantity: int
    entry_price: float
    entry_tick: int
    entry_t: float  # match time at entry
    is_paper: bool = True

    # Tracking (updated each tick by check_exits)
    unrealized_pnl: float = 0.0
    current_p_model: float = 0.0
    current_p_kalshi: float = 0.0
    ticks_held: int = 0


class ExitDecision(BaseModel):
    """Decision to exit a position (full or partial)."""
    position_id: str
    trigger: ExitTrigger
    contracts_to_exit: int  # full position for most triggers; partial for POSITION_TRIM
    exit_price: float       # price to exit at (current P_kalshi)
    reason: str             # human-readable explanation
```

## Step 2: Create `src/execution/position_monitor.py`

This module contains `PositionTracker` — the in-memory position state manager that evaluates exit triggers every tick.

### Class: `PositionTracker`

**Constructor:** `__init__(self, min_hold_ticks: int = CONFIG.MIN_HOLD_TICKS, cooldown_after_exit: int = CONFIG.COOLDOWN_AFTER_EXIT)`
- `self.open_positions: dict[str, Position] = {}`
- `self.exit_cooldowns: dict[str, int] = {}` (market_type → tick when cooldown expires)
- `self.min_hold_ticks = min_hold_ticks` (default 150)
- `self.cooldown_after_exit = cooldown_after_exit` (default 300)

**Method: `add_position(self, signal: Signal, fill: FillResult, tick: int, t: float) -> Position`**
- Generate UUID for position id (`str(uuid.uuid4())`)
- CRITICAL: set `entry_price = fill.price` if `signal.direction == "BUY_YES"`, else `entry_price = 1.0 - fill.price` for BUY_NO
- Create Position with all fields from signal + fill
- Store in `self.open_positions[position.id]`
- Return position

**Method: `check_exits(self, payload: TickPayload, p_kalshi: dict[str, float]) -> list[ExitDecision]`**
- For each position in `self.open_positions.values()`:
  - If position's market_type not in p_kalshi: skip (no price data)
  - Increment `position.ticks_held += 1`
  - Update `position.current_p_model = getattr(payload.P_model, position.market_type)` (or skip if None)
  - Update `position.current_p_kalshi = p_kalshi[position.market_type]`
  - Compute current EV: for BUY_YES `ev = p_model - p_kalshi`, for BUY_NO `ev = p_kalshi - p_model`
  - Check 6 triggers IN ORDER — first match wins:

**Trigger 1 — EDGE_DECAY:** `ev < theta_exit` where theta_exit is computed via `compute_dynamic_threshold(p_model, sigma_mc, ekf_P, mu_market)` using current payload values.
- Skip if `ticks_held < min_hold_ticks`
- `contracts_to_exit = position.quantity` (full exit)
- reason: `f"edge_decay: ev={ev:.4f} < theta={theta:.4f}"`

**Trigger 2 — EDGE_REVERSAL:** Direction flipped. For BUY_YES: `p_model < p_kalshi`. For BUY_NO: `p_model > p_kalshi`.
- Ignores min_hold — fires immediately
- `contracts_to_exit = position.quantity` (full exit)
- reason: `f"edge_reversal: direction flipped"`

**Trigger 3 — POSITION_TRIM:** `position.quantity > 2 * kelly_optimal_contracts`
- Skip if `ticks_held < min_hold_ticks`
- Compute `kelly_frac = compute_kelly_fraction(p_model, p_kalshi)` (import from kelly_sizer)
- `kelly_optimal = max(1, int(kelly_frac * 10000.0 / p_kalshi))` (rough estimate using $10k reference bankroll)
- Only fire if `position.quantity > 2 * kelly_optimal`
- `contracts_to_exit = position.quantity - kelly_optimal` (PARTIAL exit)
- reason: `f"position_trim: {position.quantity} > 2x optimal {kelly_optimal}"`

**Trigger 4 — OPPORTUNITY_COST:** Opposite direction has EV exceeding entry threshold.
- Skip if `ticks_held < min_hold_ticks`
- For BUY_YES position: opposite EV = `p_kalshi - p_model`. For BUY_NO: opposite EV = `p_model - p_kalshi`.
- Compute theta_entry same as Trigger 1's theta.
- Fire if `opposite_ev > theta_entry`
- `contracts_to_exit = position.quantity` (full exit)
- reason: `f"opportunity_cost: opposite_ev={opposite_ev:.4f} > theta={theta:.4f}"`

**Trigger 5 — EXPIRY_EVAL:** `payload.t > CONFIG.EXPIRY_EVAL_MINUTE` (85.0) and holding to settlement is worse than exiting now.
- Skip if `ticks_held < min_hold_ticks`
- Simple heuristic: if current `ev < CONFIG.C_SPREAD` (edge doesn't cover spread cost), exit
- `contracts_to_exit = position.quantity` (full exit)
- reason: `f"expiry_eval: t={payload.t:.1f} ev={ev:.4f}"`

**Trigger 6 — EKF_DIVERGENCE:** `payload.ekf_P_H > CONFIG.EKF_DIVERGENCE_THRESHOLD` or `payload.ekf_P_A > CONFIG.EKF_DIVERGENCE_THRESHOLD` (1.5)
- Ignores min_hold — fires immediately (safety)
- `contracts_to_exit = position.quantity` (full exit)
- reason: `f"ekf_divergence: P_H={payload.ekf_P_H:.3f} P_A={payload.ekf_P_A:.3f}"`

Return list of all ExitDecisions generated.

**Method: `close_position(self, position_id: str, exit_trigger: ExitTrigger, contracts_exited: int, exit_price: float, current_tick: int) -> Position`**
- Get position from `self.open_positions[position_id]`
- If `contracts_exited >= position.quantity`: full exit — remove from `self.open_positions`, set cooldown `self.exit_cooldowns[position.market_type] = current_tick + self.cooldown_after_exit`
- If `contracts_exited < position.quantity`: partial exit (POSITION_TRIM) — reduce `position.quantity -= contracts_exited`, do NOT set cooldown, do NOT remove from open_positions
- Return the position

**Method: `is_in_cooldown(self, market_type: str, current_tick: int) -> bool`**
- Return `current_tick < self.exit_cooldowns.get(market_type, 0)`

**Method: `get_total_exposure(self) -> float`**
- Return `sum(pos.quantity * pos.entry_price for pos in self.open_positions.values())`

**Method: `get_match_exposure(self, match_id: str) -> float`**
- Same but filtered by match_id

## Step 3: Create `src/execution/pnl_calculator.py`

Two pure functions:

**`compute_unrealized_pnl(position: Position, p_kalshi: float) -> float`**
- BUY_YES: `(p_kalshi - position.entry_price) * position.quantity`
- BUY_NO: `((1.0 - p_kalshi) - position.entry_price) * position.quantity`

**`compute_settlement_pnl(position: Position, outcome_occurred: bool) -> float`**
Using the convention that entry_price = actual cost per contract:
- BUY_YES + occurred: `(1.0 - position.entry_price) * position.quantity`
- BUY_YES + not occurred: `-position.entry_price * position.quantity`
- BUY_NO + occurred: `-position.entry_price * position.quantity`
- BUY_NO + not occurred: `(1.0 - position.entry_price) * position.quantity`

## Step 4: Create tests

Create `tests/execution/test_position_monitor.py` with these tests:

- `test_add_position_stored`: add position → verify it's in tracker.open_positions
- `test_add_position_entry_price_buy_yes`: fill.price=0.55, direction="BUY_YES" → entry_price=0.55
- `test_add_position_entry_price_buy_no`: fill.price=0.55, direction="BUY_NO" → entry_price=0.45 (= 1.0 - 0.55)
- `test_min_hold_respected`: add at tick=0, check at tick=100 with small but positive edge → no exit (100 < 150)
- `test_edge_reversal_ignores_min_hold`: add at tick=0, at tick=5 set p_model < p_kalshi for BUY_YES → EDGE_REVERSAL fires with contracts_to_exit = position.quantity
- `test_edge_decay_exit`: add at tick=0, feed 160 ticks where edge slowly disappears → EDGE_DECAY fires after min_hold
- `test_position_trim_partial`: position.quantity=20, current kelly optimal ≈ 8 → contracts_to_exit=12, position still in open_positions with quantity=8 after close
- `test_position_trim_not_triggered`: position=8, kelly optimal=10 → no trigger (8 < 2*10)
- `test_ekf_divergence`: ekf_P_H=2.0 (> 1.5) → EKF_DIVERGENCE fires at tick=1 (ignores min_hold)
- `test_cooldown_after_full_exit`: close position at tick=100 with full exit → is_in_cooldown("home_win", 101) is True, is_in_cooldown("home_win", 401) is False (300 ticks later)
- `test_cooldown_not_set_on_partial_exit`: partial exit (POSITION_TRIM) → is_in_cooldown returns False
- `test_multiple_positions_independent`: 2 positions in different markets, only 1 triggers → only 1 ExitDecision
- `test_exits_evaluated_when_order_not_allowed`: order_allowed=False but position has reversed edge → still gets ExitDecision (exits are always checked)

Create `tests/execution/test_pnl_calculator.py` with these exact values:

- `test_unrealized_buy_yes_profit`: entry_price=0.55, p_kalshi=0.62, qty=10 → `(0.62 - 0.55) * 10 = 0.70`
- `test_unrealized_buy_yes_loss`: entry_price=0.55, p_kalshi=0.48, qty=10 → `(0.48 - 0.55) * 10 = -0.70`
- `test_unrealized_buy_no`: entry_price=0.45, p_kalshi=0.62, qty=10 → `((1-0.62) - 0.45) * 10 = -0.70`
- `test_settlement_buy_yes_won`: entry_price=0.55, occurred=True, qty=10 → `(1.0 - 0.55) * 10 = 4.50`
- `test_settlement_buy_yes_lost`: entry_price=0.55, occurred=False, qty=10 → `-0.55 * 10 = -5.50`
- `test_settlement_buy_no_won`: entry_price=0.45, occurred=False, qty=10 → `(1.0 - 0.45) * 10 = 5.50`
- `test_settlement_buy_no_lost`: entry_price=0.45, occurred=True, qty=10 → `-0.45 * 10 = -4.50`

For tests that need a TickPayload, create a helper that builds a valid one. For tests that need Signal/FillResult, build them with the minimum required fields.

Also create a production-constants integration test in the same file:

```python
async def test_production_params_no_early_exit():
    """With production MIN_HOLD_TICKS=150, no exit fires before tick 150
    (except EDGE_REVERSAL and EKF_DIVERGENCE which ignore min_hold)."""
    tracker = PositionTracker()  # uses CONFIG defaults: 150, 300
    # ... add position at tick=0
    # ... feed ticks 1-149 with small positive edge → assert no exits
    # ... at tick 150+ with edge reversed → assert EDGE_REVERSAL fires
```

## Step 5: Verify

1. `python -m pytest tests/execution/test_position_monitor.py -v` — all pass
2. `python -m pytest tests/execution/test_pnl_calculator.py -v` — all pass
3. `python -m pytest tests/execution/ -v` — all Sprint 4a + 4b tests pass
4. `python -m pytest tests/ -v --ignore=tests/execution` — existing tests unaffected

If any test fails, fix the implementation — do not change the test values.
