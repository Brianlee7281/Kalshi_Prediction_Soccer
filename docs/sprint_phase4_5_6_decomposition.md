# Sprint Decomposition: Phase 4 (Execution), Phase 5 (Orchestrator), Phase 6 (Dashboard)

## Changelog (from audit)
- [C1] ALPHA_SURPRISE 0.15 → 0.25 (Sprint 4a) — matches patterns.md `kelly_surprise_bonus = 0.25`
- [C2] Added order repricing logic with REPRICE_THRESHOLD=0.02 (Sprint 4c OrderManager)
- [C3] Fixed limit order price P_kalshi → P_model for live mode (Sprint 4c)
- [C4] μ_remaining → market-specific μ_H/μ_A in σ²_model formula (Sprint 4a)
- [C5] Added contracts_to_exit to ExitDecision; Trigger 3 is now partial exit (Sprint 4b)
- [C6] Added Kalshi rejection handling: market_closed, insufficient_balance, price_out_of_range (Sprint 4c)
- [C7] Added live settlement polling via poll_kalshi_settlement() (Sprint 4d)
- [NC1] Centralized all Phase 4 constants in ExecutionConfig dataclass (Sprint 4a)
- [NC2] min_hold_ticks=150, cooldown_after_exit=300 to match spec's intended durations at 1Hz (Sprint 4b)
- [NC3] Added open_positions param to generate_signals() to gate duplicates at source (Sprint 4a)
- [NC4] Documented entry_price semantics for BUY_YES vs BUY_NO in Position docstring (Sprint 4b)
- [NC5] Added production-constants integration test variant (Sprint 4b)

---

## Sprint 4a: Signal Generation + Kelly Sizing (Pure Math, No I/O)

### Goal
Given a TickPayload and Kalshi orderbook prices, produce a list of Signals with contract counts — no database, no API, no network.

### Prerequisites
- Sprint S3 complete (tick_loop emits TickPayload via asyncio.Queue)
- Recorded match data available for match 4190023 (Brentford 2-2 Wolves)

### Types to Add (this sprint only)

```python
# src/common/types.py — additions only

from enum import Enum

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"

class ExitTrigger(str, Enum):
    EDGE_DECAY = "edge_decay"
    EDGE_REVERSAL = "edge_reversal"
    POSITION_TRIM = "position_trim"
    OPPORTUNITY_COST = "opportunity_cost"
    EXPIRY_EVAL = "expiry_eval"
    EKF_DIVERGENCE = "ekf_divergence"

# Signal already exists in types.py (§2.5) — no changes needed
# FillResult already exists in types.py (§2.6) — no changes needed
```

### Constants [NC1 fix]

All Phase 4 constants are centralized in a single config object. Individual modules
import from here — no hardcoded magic numbers in function bodies.

```python
# src/execution/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionConfig:
    # Edge detection (§8.2)
    C_SPREAD: float = 0.01             # Kalshi effective spread
    C_SLIPPAGE: float = 0.005          # limit order execution delay cost
    Z_ALPHA: float = 1.645             # 95% one-tailed confidence
    N_MC: int = 50_000                 # Monte Carlo paths

    # Kelly sizing (§8.4, patterns.md Pattern 5)
    ALPHA_BASE: float = 0.10           # baseline Kelly multiplier
    ALPHA_SURPRISE: float = 0.25       # [C1 fix] surprise bonus — matches patterns.md kelly_surprise_bonus; TODO: recalibrate from 307-match backtest

    # Risk caps (§13.4)
    PER_ORDER_CAP: float = 50.0        # max dollars per order
    PER_MATCH_CAP_FRAC: float = 0.10   # max fraction of bankroll per match
    TOTAL_EXPOSURE_CAP_FRAC: float = 0.20  # max fraction across all positions

    # Order management (§8.5)
    MAX_ORDER_LIFETIME_S: float = 30.0 # cancel unfilled orders after this
    REPRICE_THRESHOLD: float = 0.02    # [C2 fix] cancel+repost if |P_model_now - P_order| > this

    # Position management (§13.4)
    MIN_HOLD_TICKS: int = 150          # [NC2 fix] ~150 seconds at 1Hz tick rate (spec says "~150 seconds")
    COOLDOWN_AFTER_EXIT: int = 300     # [NC2 fix] ~5 minutes at 1Hz tick rate (spec says "~5 minutes")
    EKF_DIVERGENCE_THRESHOLD: float = 1.5  # exit if P_H or P_A exceeds this
    EXPIRY_EVAL_MINUTE: float = 85.0   # begin expiry evaluation after this match minute

# Singleton — import this everywhere
CONFIG = ExecutionConfig()
```

### Modules to Build

#### `src/execution/signal_generator.py`

**Functions/classes to implement this sprint:**

- `compute_edge(p_model: float, p_kalshi: float) -> tuple[str, float]`
  - Compute EV for both directions:
    - `ev_yes = p_model - p_kalshi`
    - `ev_no = p_kalshi - p_model`
  - Return `("BUY_YES", ev_yes)` if `ev_yes > ev_no` and `ev_yes > 0`
  - Return `("BUY_NO", ev_no)` if `ev_no > ev_yes` and `ev_no > 0`
  - Return `("HOLD", 0.0)` otherwise
  - Log: `structlog.debug("edge_computed", market=..., ev=..., direction=...)`

- `compute_dynamic_threshold(p_hat: float, sigma_mc: float, ekf_P: float, mu_market: float) -> float` [C4 fix — renamed `mu_remaining` → `mu_market` to clarify it is market-specific]
  - `c_spread = CONFIG.C_SPREAD`
  - `c_slippage = CONFIG.C_SLIPPAGE`
  - `z_alpha = CONFIG.Z_ALPHA`
  - `sigma_mc_sq = p_hat * (1 - p_hat) / CONFIG.N_MC` (MC standard error)
  - `sigma_model_sq = ekf_P * (p_hat * (1 - p_hat) * mu_market) ** 2` [C4 fix — uses market-specific mu, not total] (Baker-McHale propagation via Delta method, §8.2)
  - `sigma_p = sqrt(sigma_mc_sq + sigma_model_sq)`
  - `theta = c_spread + c_slippage + z_alpha * sigma_p`
  - Edge cases: if `p_hat <= 0` or `p_hat >= 1`, return `1.0` (never trade degenerate probs)
  - Return `theta`

- `_get_market_mu(market_type: str, mu_H: float, mu_A: float) -> float` [C4 fix — helper to select market-specific mu]
  - `home_win` → `mu_H`
  - `away_win` → `mu_A`
  - `draw`, `over_25`, `btts_yes` → `max(mu_H, mu_A)` (conservative: use the larger remaining-goals estimate)

- `generate_signals(payload: TickPayload, p_kalshi: dict[str, float], kalshi_tickers: dict[str, str], open_positions: dict[str, "Position"] | None = None) -> list[Signal]` [NC3 fix — added open_positions parameter]
  - For each market_type in `["home_win", "draw", "away_win", "over_25", "btts_yes"]`:
    - Skip if `market_type` not in `kalshi_tickers` or not in `p_kalshi`
    - [NC3 fix] Skip if `open_positions` is not None and any position in `open_positions.values()` has `pos.market_type == market_type` (already have a position in this market)
    - Extract `p_model = getattr(payload.P_model, market_type)` — skip if `None`
    - Extract `p_k = p_kalshi[market_type]`
    - Compute `(direction, ev) = compute_edge(p_model, p_k)`
    - If `direction == "HOLD"`: continue
    - [C4 fix] Compute market-specific mu: `mu_market = _get_market_mu(market_type, payload.mu_H, payload.mu_A)`
    - Compute `ekf_P`:
      - `home_win` → `payload.ekf_P_H`
      - `away_win` → `payload.ekf_P_A`
      - `draw`, `over_25`, `btts_yes` → `max(payload.ekf_P_H, payload.ekf_P_A)` (conservative)
    - Compute `sigma_mc = getattr(payload.sigma_MC, market_type)` or `0.0` if None
    - Compute `theta = compute_dynamic_threshold(p_model, sigma_mc, ekf_P, mu_market)` [C4 fix — passes market-specific mu]
    - If `ev < theta`: continue
    - Append Signal with all fields populated (kelly_fraction and contracts filled by kelly_sizer, set to 0 here)
  - Log: `structlog.info("signals_generated", match_id=..., count=len(signals))`
  - Return list of Signals

**What this module does NOT do yet:**
- No exit signal generation (deferred to Sprint 4b)
- No orderbook depth check (deferred to Sprint 4c)

#### `src/execution/kelly_sizer.py`

**Functions/classes to implement this sprint:**

- `compute_kelly_fraction(p_model: float, p_kalshi: float) -> float`
  - `b = (1.0 / p_kalshi) - 1.0` (decimal odds minus 1)
  - `q = 1.0 - p_model`
  - `f_star = (b * p_model - q) / b`
  - Return `max(0.0, f_star)`
  - Edge case: if `p_kalshi <= 0` or `p_kalshi >= 1`, return `0.0`

- `apply_baker_mchale_shrinkage(f_star: float, p_model: float, p_kalshi: float, sigma_p: float) -> float`
  - `edge_sq = (p_model - p_kalshi) ** 2`
  - If `edge_sq <= 0`: return `0.0`
  - `shrinkage = max(0.0, 1.0 - (sigma_p ** 2) / edge_sq)`
  - Return `f_star * shrinkage`

- `apply_surprise_multiplier(f_shrunk: float, surprise_score: float) -> float`
  - `alpha_base = CONFIG.ALPHA_BASE`  # 0.10
  - `alpha_surprise = CONFIG.ALPHA_SURPRISE`  # [C1 fix] 0.25 (was 0.15; now matches patterns.md kelly_surprise_bonus; TODO: recalibrate from 307-match backtest)
  - `kelly_mult = alpha_base + alpha_surprise * surprise_score`
  - Return `f_shrunk * kelly_mult`

- `size_position(signal: Signal, payload: TickPayload, bankroll: float) -> Signal`
  - Compute `f_star = compute_kelly_fraction(signal.P_model, signal.P_kalshi)`
  - [C4 fix] Compute market-specific mu: `mu_market = _get_market_mu(signal.market_type, payload.mu_H, payload.mu_A)`
  - Compute `sigma_p` using `compute_dynamic_threshold` formula components (same as in signal_generator, using market-specific ekf_P and mu_market)
  - `f_shrunk = apply_baker_mchale_shrinkage(f_star, signal.P_model, signal.P_kalshi, sigma_p)`
  - `f_final = apply_surprise_multiplier(f_shrunk, payload.surprise_score)`
  - `dollar_amount = f_final * bankroll`
  - Apply hard caps:
    - `dollar_amount = min(dollar_amount, CONFIG.PER_ORDER_CAP)` ($50)
    - `dollar_amount = min(dollar_amount, CONFIG.PER_MATCH_CAP_FRAC * bankroll)` (10%)
  - `contracts = int(dollar_amount / signal.P_kalshi)` if `signal.P_kalshi > 0` else `0`
  - Update signal fields: `kelly_fraction`, `kelly_amount`, `contracts`, `surprise_score`
  - Log: `structlog.info("position_sized", ticker=..., kelly=..., contracts=...)`
  - Return updated signal

**What this module does NOT do yet:**
- No total exposure cap check (requires DB, deferred to Sprint 4c)
- No portfolio Kelly (requires multiple simultaneous matches, deferred post-backtest)
- No liquidity gate (requires live orderbook depth, deferred to Sprint 4c)

### DB Migrations (if any)
None — this sprint is pure math.

### Interface Contracts

- **tick_loop → signal_generator:** `tick_loop` puts `TickPayload` on `asyncio.Queue(maxsize=1)`. `generate_signals(payload, p_kalshi, kalshi_tickers, open_positions)` consumes it. [NC3 fix — open_positions added]
  - Caller provides: full `TickPayload` (all fields populated by tick_loop.py:79-102), `p_kalshi` dict from `model.p_kalshi` (maintained by `kalshi_ob_sync`), `kalshi_tickers` from `model.kalshi_tickers`, `open_positions` dict from `PositionTracker.open_positions`.
  - Callee returns: `list[Signal]` — may be empty.
  - Failure mode: If `p_kalshi` is empty (no Kalshi connection), returns empty list. Never raises.

- **signal_generator → kelly_sizer:** `size_position(signal, payload, bankroll)`.
  - Caller provides: `Signal` with `P_model`, `P_kalshi`, `direction`, `ticker`, `market_type` populated; `TickPayload`; `bankroll: float`.
  - Callee returns: same `Signal` object with `kelly_fraction`, `kelly_amount`, `contracts` filled.
  - Failure mode: If bankroll ≤ 0, returns signal with contracts=0. Never raises.

### Test Plan

**Unit tests** (per module):

`tests/execution/test_signal_generator.py`:
- `test_compute_edge_buy_yes`: p_model=0.62, p_kalshi=0.55 → ("BUY_YES", 0.07)
- `test_compute_edge_buy_no`: p_model=0.30, p_kalshi=0.45 → ("BUY_NO", 0.15)
- `test_compute_edge_hold`: p_model=0.50, p_kalshi=0.50 → ("HOLD", 0.0)
- `test_compute_edge_near_zero`: p_model=0.001, p_kalshi=0.001 → ("HOLD", 0.0)
- `test_dynamic_threshold_early_match`: t=5, high ekf_P → large threshold (~0.04)
- `test_dynamic_threshold_late_match`: t=85, low ekf_P → small threshold (~0.02)
- `test_dynamic_threshold_degenerate_p`: p_hat=0.0 → returns 1.0
- `test_get_market_mu_home_win`: market_type="home_win" → returns mu_H [C4 fix]
- `test_get_market_mu_away_win`: market_type="away_win" → returns mu_A [C4 fix]
- `test_get_market_mu_draw`: market_type="draw" → returns max(mu_H, mu_A) [C4 fix]
- `test_generate_signals_empty_kalshi`: empty p_kalshi → returns []
- `test_generate_signals_filters_below_threshold`: ev < theta → signal excluded
- `test_generate_signals_multiple_markets`: 2 markets with edge → 2 signals
- `test_generate_signals_skips_existing_position`: open_positions has home_win → no home_win signal emitted [NC3 fix]

`tests/execution/test_kelly_sizer.py`:
- `test_kelly_fraction_basic`: p_model=0.62, p_kalshi=0.55 → f*≈0.155
- `test_kelly_fraction_no_edge`: p_model=0.50, p_kalshi=0.55 → 0.0
- `test_kelly_fraction_degenerate`: p_kalshi=0.0 → 0.0
- `test_baker_mchale_full_shrink`: sigma_p equals edge → returns 0.0
- `test_baker_mchale_partial_shrink`: sigma_p=0.04, edge=0.07 → shrinkage≈0.67
- `test_surprise_multiplier_neutral`: surprise=0.0 → mult=0.10 [C1 fix — unchanged, alpha_base alone]
- `test_surprise_multiplier_high`: surprise=0.70 → mult=0.10 + 0.25*0.70 = 0.275 [C1 fix — was 0.205 with old 0.15]
- `test_size_position_hard_caps`: bankroll=1000, large kelly → capped at $50
- `test_size_position_per_match_cap`: bankroll=100, kelly wants $30 → capped at $10

**Integration test** (sprint-level):

```python
async def test_sprint_4a_integration():
    """Feed recorded TickPayloads from match 4190023, verify signals.

    Uses hardcoded Kalshi prices from recording.
    Verifies:
    1. Signals appear after goals (surprise goals produce larger edges)
    2. Kelly sizes are in reasonable range (1-15 contracts)
    3. No signals during cooldown (order_allowed=False)
    4. Dynamic threshold is higher early, lower late
    5. [NC3 fix] No duplicate signals for markets with open positions
    """
    # Load recorded TickPayloads from data/recordings/4190023/
    payloads = load_recorded_payloads("4190023")
    p_kalshi_series = load_recorded_kalshi_prices("4190023")
    kalshi_tickers = {"home_win": "KXEPLGAME-26MAR16BREWOL-BRE", ...}
    bankroll = 1000.0

    signals_over_time = []
    for payload, p_kalshi in zip(payloads, p_kalshi_series):
        signals = generate_signals(payload, p_kalshi, kalshi_tickers)
        for sig in signals:
            sig = size_position(sig, payload, bankroll)
        signals_over_time.append(signals)

    # Assert: no signals when order_allowed=False
    for i, payload in enumerate(payloads):
        if not payload.order_allowed:
            assert len(signals_over_time[i]) == 0

    # Assert: at least some signals exist after goals
    post_goal_signals = [s for t, sigs in ... if t > goal_minute for s in sigs]
    assert len(post_goal_signals) > 0

    # Assert: all contract counts are non-negative and ≤ 100
    for sigs in signals_over_time:
        for s in sigs:
            assert 0 <= s.contracts <= 100
            assert s.kelly_fraction >= 0
```

### Risks & Gotchas

- **TickPayload.P_model has `over_25` and `btts_yes` as Optional fields.** `generate_signals` must check for `None` before calling `compute_edge`. The `getattr(payload.P_model, market_type)` pattern returns `None` for these when MC didn't compute them.
- **Market type naming mismatch.** `MarketProbs` uses `over_25` but Kalshi ticker mapping may use different keys. Verify `kalshi_tickers` dict keys match `MarketProbs` field names exactly.
- **`sigma_MC` is a `MarketProbs` object**, not a single float. Each market has its own sigma — must extract per-market. Anti-pattern from post-mortem: "Single σ_MC float for all markets."
- **[C4 fix] `mu_market` is team-specific.** For home_win use mu_H; for away_win use mu_A; for draw/over/btts use max(mu_H, mu_A) as conservative estimate. Do NOT use mu_H + mu_A — that overestimates uncertainty for single-team markets.

---

## Sprint 4b: Position Tracking + Exit Logic (Pure State, No I/O)

### Goal
Given open positions and a stream of TickPayloads, produce exit decisions — all in-memory, no database, no API.

### Prerequisites
- Sprint 4a complete (signal generation produces sized Signals)

### Types to Add (this sprint only)

```python
# src/common/types.py — additions

class Position(BaseModel):
    """In-memory position tracking for Phase 4.

    [NC4 fix] entry_price semantics:
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

    # Tracking
    unrealized_pnl: float = 0.0
    current_p_model: float = 0.0
    current_p_kalshi: float = 0.0
    ticks_held: int = 0

class ExitDecision(BaseModel):
    """Decision to exit a position."""
    position_id: str
    trigger: ExitTrigger
    contracts_to_exit: int  # [C5 fix] how many contracts to exit. Full position for most triggers; partial for POSITION_TRIM.
    exit_price: float  # price to exit at (current P_kalshi)
    reason: str  # human-readable explanation
```

### Modules to Build

#### `src/execution/position_monitor.py`

**Functions/classes to implement this sprint:**

- `class PositionTracker`
  - `__init__(self, min_hold_ticks: int = CONFIG.MIN_HOLD_TICKS, cooldown_after_exit: int = CONFIG.COOLDOWN_AFTER_EXIT) -> None` [NC2 fix — defaults from CONFIG: 150, 300]
    - `self.open_positions: dict[str, Position] = {}`
    - `self.exit_cooldowns: dict[str, int] = {}` — market_type → tick when cooldown expires
    - `self.min_hold_ticks = min_hold_ticks`
    - `self.cooldown_after_exit = cooldown_after_exit`

  - `add_position(self, signal: Signal, fill: FillResult, tick: int, t: float) -> Position`
    - Create `Position` from `Signal` + `FillResult`
    - [NC4 fix] Set `entry_price = fill.price` for BUY_YES, `entry_price = 1.0 - fill.price` for BUY_NO (actual cost per contract)
    - Store in `self.open_positions[position.id]`
    - Log: `structlog.info("position_opened", ...)`
    - Return the position

  - `check_exits(self, payload: TickPayload, p_kalshi: dict[str, float]) -> list[ExitDecision]`
    - For each position in `self.open_positions`:
      - Update `ticks_held += 1`
      - Update `current_p_model` and `current_p_kalshi` from payload
      - Compute `unrealized_pnl`
      - Check 6 exit triggers in order (first match wins):
        1. **EDGE_DECAY**: `current_ev < theta_exit` (theta_exit computed same as entry threshold)
           - Skip if `ticks_held < min_hold_ticks`
           - [C5 fix] `contracts_to_exit = position.quantity` (full exit)
        2. **EDGE_REVERSAL**: direction flipped (was BUY_YES, now P_model < P_kalshi)
           - Allowed immediately (ignores min_hold)
           - [C5 fix] `contracts_to_exit = position.quantity` (full exit)
        3. **POSITION_TRIM**: position size > 2× current Kelly optimal
           - Skip if `ticks_held < min_hold_ticks`
           - [C5 fix] Compute `kelly_optimal_contracts = floor(compute_kelly_fraction(current_p_model, current_p_kalshi) * bankroll_est / current_p_kalshi)` where `bankroll_est` is a stored reference. `contracts_to_exit = position.quantity - kelly_optimal_contracts`. Only fire if `contracts_to_exit > 0`.
        4. **OPPORTUNITY_COST**: opposite direction has EV > theta_entry
           - Skip if `ticks_held < min_hold_ticks`
           - [C5 fix] `contracts_to_exit = position.quantity` (full exit)
        5. **EXPIRY_EVAL**: `payload.t > CONFIG.EXPIRY_EVAL_MINUTE` and expected P&L of hold < exit cost
           - Skip if `ticks_held < min_hold_ticks`
           - [C5 fix] `contracts_to_exit = position.quantity` (full exit)
        6. **EKF_DIVERGENCE**: `ekf_P_H > CONFIG.EKF_DIVERGENCE_THRESHOLD` or `ekf_P_A > CONFIG.EKF_DIVERGENCE_THRESHOLD`
           - Allowed immediately (safety trigger)
           - [C5 fix] `contracts_to_exit = position.quantity` (full exit)
    - Return list of `ExitDecision`

  - `close_position(self, position_id: str, exit_trigger: ExitTrigger, contracts_exited: int, exit_price: float, current_tick: int) -> Position` [C5 fix — added contracts_exited param]
    - If `contracts_exited >= position.quantity`: remove from `open_positions` (full exit)
    - If `contracts_exited < position.quantity`: reduce `position.quantity -= contracts_exited` (partial exit — POSITION_TRIM)
    - Set `exit_cooldowns[position.market_type] = current_tick + cooldown_after_exit` only on full exit
    - Log: `structlog.info("position_closed", trigger=..., pnl=..., partial=(contracts_exited < original_qty))`
    - Return the position

  - `is_in_cooldown(self, market_type: str, current_tick: int) -> bool`
    - Return `current_tick < self.exit_cooldowns.get(market_type, 0)`

  - `get_total_exposure(self) -> float`
    - Return sum of `pos.quantity * pos.entry_price` for all open positions

  - `get_match_exposure(self, match_id: str) -> float`
    - Return sum of exposure for positions in this match

**What this module does NOT do yet:**
- No database persistence (deferred to Sprint 4c)
- No actual order execution for exits (deferred to Sprint 4c)

#### `src/execution/pnl_calculator.py`

**Functions/classes to implement this sprint:**

- `compute_unrealized_pnl(position: Position, p_kalshi: float) -> float`
  - For BUY_YES: `(p_kalshi - position.entry_price) * position.quantity`
  - For BUY_NO: `((1 - p_kalshi) - position.entry_price) * position.quantity` [NC4 fix — entry_price already stores the NO cost]
  - Return the computed PnL

- `compute_settlement_pnl(position: Position, outcome_occurred: bool) -> float`
  - [NC4 fix] Using the convention that `entry_price` = cost per contract:
    - BUY_YES, occurred: `(1.0 - entry_price) * quantity` (paid entry_price, receive $1)
    - BUY_YES, not occurred: `-entry_price * quantity` (paid entry_price, receive $0)
    - BUY_NO, occurred: `-entry_price * quantity` (paid entry_price for NO, outcome occurred so NO loses)
    - BUY_NO, not occurred: `(1.0 - entry_price) * quantity` (paid entry_price for NO, outcome didn't occur so NO wins)

### DB Migrations (if any)
None — this sprint is pure in-memory state.

### Interface Contracts

- **signal_generator → position_tracker.add_position:** After a fill, `tracker.add_position(signal, fill_result, current_tick, payload.t)`
  - Caller provides: sized `Signal`, `FillResult` (from Sprint 4c, but interface defined now)
  - Callee returns: `Position` object

- **tick_loop → position_tracker.check_exits:** Every tick, `tracker.check_exits(payload, p_kalshi)`
  - Caller provides: `TickPayload`, current `p_kalshi` dict
  - Callee returns: `list[ExitDecision]` — may be empty. Each ExitDecision includes `contracts_to_exit`. [C5 fix]
  - Failure mode: never raises. If p_kalshi is missing for a position's market, skip that position.

- **position_tracker → kelly_sizer:** For trigger 3 (POSITION_TRIM), calls `compute_kelly_fraction` to get current optimal size
  - Reuses Sprint 4a's `kelly_sizer.compute_kelly_fraction`

### Test Plan

**Unit tests:**

`tests/execution/test_position_monitor.py`:
- `test_add_position_stored`: add position → verify in tracker.open_positions
- `test_add_position_entry_price_buy_yes`: BUY_YES fill at 0.55 → entry_price=0.55 [NC4 fix]
- `test_add_position_entry_price_buy_no`: BUY_NO fill at 0.55 → entry_price=0.45 [NC4 fix]
- `test_min_hold_respected`: position at tick 10, check at tick 50 → no exit (< 150 ticks) [NC2 fix — min_hold=150]
- `test_edge_reversal_ignores_min_hold`: reversal at tick 5 → exit allowed
- `test_edge_decay_exit`: EV drops below threshold after min_hold → EDGE_DECAY decision
- `test_edge_reversal_exit`: P_model < P_kalshi for BUY_YES position → EDGE_REVERSAL
- `test_position_trim_partial`: position=20 contracts, kelly optimal=8 → contracts_to_exit=12 [C5 fix]
- `test_position_trim_full_exit_not_triggered`: position=8 contracts, kelly optimal=10 → no trigger (position < 2× optimal)
- `test_expiry_eval_late_game`: t=87, hold expected PnL < exit cost → EXPIRY_EVAL
- `test_ekf_divergence`: ekf_P_H=2.0 → EKF_DIVERGENCE exit immediately
- `test_cooldown_after_exit`: close position → is_in_cooldown returns True for 300 ticks [NC2 fix]
- `test_cooldown_expired`: 301 ticks later → is_in_cooldown returns False [NC2 fix]
- `test_multiple_positions_independent`: 2 positions, only 1 triggers exit
- `test_no_exits_when_order_not_allowed`: payload.order_allowed=False → exits still evaluated (exits are always evaluated, only entries are gated)

`tests/execution/test_pnl_calculator.py`:
- `test_unrealized_buy_yes_profit`: entry=0.55, current=0.62, qty=10 → +0.70
- `test_unrealized_buy_yes_loss`: entry=0.55, current=0.48, qty=10 → -0.70
- `test_settlement_buy_yes_won`: entry=0.55, occurred=True, qty=10 → +4.50
- `test_settlement_buy_yes_lost`: entry=0.55, occurred=False, qty=10 → -5.50
- `test_settlement_buy_no_won`: entry=0.45, occurred=False, qty=10 → +5.50 [NC4 fix — entry_price=0.45 for BUY_NO when P_kalshi=0.55]
- `test_settlement_buy_no_lost`: entry=0.45, occurred=True, qty=10 → -4.50 [NC4 fix]

**Integration test:**

```python
async def test_sprint_4b_integration():
    """Simulate position lifecycle: entry → hold → exit trigger.

    Creates a position from a hardcoded signal, then feeds a stream
    of TickPayloads where P_model decays toward P_kalshi.
    Verifies the correct exit trigger fires at the right time.
    """
    tracker = PositionTracker(min_hold_ticks=5, cooldown_after_exit=10)  # reduced for fast test

    # Create a position (simulating post-fill)
    signal = Signal(match_id="4190023", ticker="KXEPLGAME-...", ...)
    fill = FillResult(order_id="paper-001", quantity=10, price=0.55, ...)
    pos = tracker.add_position(signal, fill, tick=0, t=35.0)

    # Feed payloads where edge gradually decays
    for tick in range(1, 100):
        # P_model starts at 0.62 and drops toward 0.55 over time
        p_model = 0.62 - (tick / 100) * 0.08
        payload = make_payload(t=35.0 + tick/60, p_model=p_model, ...)
        p_kalshi = {"home_win": 0.55}

        exits = tracker.check_exits(payload, p_kalshi)
        if exits:
            assert exits[0].trigger in (ExitTrigger.EDGE_DECAY, ExitTrigger.EDGE_REVERSAL)
            assert exits[0].contracts_to_exit == 10  # [C5 fix] full exit
            assert tick >= 5  # min_hold respected (unless reversal)
            break

    # Verify position was flagged for exit
    assert len(exits) > 0
    # Verify cooldown active after close
    tracker.close_position(pos.id, exits[0].trigger, exits[0].contracts_to_exit, 0.55, tick)  # [C5 fix]
    assert tracker.is_in_cooldown("home_win", tick + 1)
    assert not tracker.is_in_cooldown("home_win", tick + 11)
```

```python
# [NC5 fix] Production-constants integration test
async def test_sprint_4b_integration_production_params():
    """Same lifecycle test but with production MIN_HOLD_TICKS=150, COOLDOWN_AFTER_EXIT=300.

    Verifies that no exit fires before tick 150 (except EDGE_REVERSAL and EKF_DIVERGENCE).
    """
    tracker = PositionTracker()  # uses CONFIG defaults: min_hold=150, cooldown=300

    signal = Signal(match_id="4190023", ticker="KXEPLGAME-...", market_type="home_win", ...)
    fill = FillResult(order_id="paper-001", quantity=10, price=0.55, ...)
    pos = tracker.add_position(signal, fill, tick=0, t=35.0)

    # Edge decays slowly — should NOT trigger before tick 150
    for tick in range(1, 149):
        p_model = 0.60  # still above P_kalshi but small edge
        payload = make_payload(t=35.0 + tick/60, p_model=p_model, ...)
        p_kalshi = {"home_win": 0.55}
        exits = tracker.check_exits(payload, p_kalshi)
        assert len(exits) == 0, f"unexpected exit at tick {tick}"

    # At tick 160 with edge gone, should now trigger
    for tick in range(150, 200):
        p_model = 0.54  # edge reversed
        payload = make_payload(t=35.0 + tick/60, p_model=p_model, ...)
        p_kalshi = {"home_win": 0.55}
        exits = tracker.check_exits(payload, p_kalshi)
        if exits:
            assert exits[0].trigger == ExitTrigger.EDGE_REVERSAL
            break

    assert len(exits) > 0
```

### Risks & Gotchas

- **[NC4 fix] Position.entry_price semantics for BUY_NO:** `entry_price` = actual cost per contract. For BUY_YES, that's P_kalshi. For BUY_NO, that's `1 - P_kalshi`. The `add_position` method handles this mapping; downstream PnL formulas use `entry_price` uniformly.
- **Exit triggers can fire simultaneously.** First match wins — must process in order.
- **[NC2 fix] min_hold_ticks=150 and cooldown_after_exit=300 at 1Hz tick rate.** These match the spec's intended durations of "~150 seconds" and "~5 minutes" respectively. The §13.4 table says "50 ticks (~150 seconds)" which implies 3s/tick — that was a spec error since our tick_loop runs at 1Hz. We use the intended durations.
- **Anti-pattern from post-mortem: "No min_hold on positions → 1s churn."** This sprint implements min_hold from day one.

---

## Sprint 4c: Order Management + Exposure (DB + API)

### Goal
A Signal flows through reserve → place paper order → fill → confirm exposure → position created in DB. Paper mode only. Also: order repricing and Kalshi rejection handling.

### Prerequisites
- Sprint 4a complete (signal generation)
- Sprint 4b complete (position tracking)
- PostgreSQL running (docker compose up -d)

### Types to Add (this sprint only)

```python
# src/common/types.py — additions

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

### Modules to Build

#### `src/execution/order_manager.py`

**Functions/classes to implement this sprint:**

- `class OrderManager`
  - `__init__(self, kalshi_client: KalshiClient | None, trading_mode: TradingMode, db_pool: asyncpg.Pool) -> None`
    - `self.kalshi_client = kalshi_client`
    - `self.trading_mode = trading_mode`
    - `self.db = db_pool`
    - `self.pending_orders: dict[str, dict] = {}` — order_id → `{signal: Signal, placed_at: float, order_p_model: float}`
    - `self.max_order_age_s: float = CONFIG.MAX_ORDER_LIFETIME_S`
    - `self.reprice_threshold: float = CONFIG.REPRICE_THRESHOLD` [C2 fix]
    - `self.ticker_muted: dict[str, bool] = {}` [C6 fix] — ticker → True if market_closed
    - `self.entries_halted: bool = False` [C6 fix] — True if insufficient_balance received

  - `def is_ticker_muted(self, ticker: str) -> bool` [C6 fix]
    - Return `self.ticker_muted.get(ticker, False)`

  - `async def place_order(self, signal: Signal) -> FillResult | None` [C6 fix — return type now `FillResult | None`, None on mute/halt]
    - [C6 fix] If `self.entries_halted`: log warning, return None
    - [C6 fix] If `self.is_ticker_muted(signal.ticker)`: log debug, return None
    - If `trading_mode == PAPER`:
      - Simulate fill at `signal.P_kalshi` price, full quantity
      - Return `FillResult(order_id=f"paper-{uuid4()}", status="paper", quantity=signal.contracts, price=signal.P_kalshi, ...)`
    - If `trading_mode == LIVE`:
      - [C3 fix] Build Kalshi order dict: `{ticker, action: "buy", side: "yes"|"no", type: "limit", count: signal.contracts, yes_price: int(signal.P_model * 100)}` — limit order at P_model fair value, NOT P_kalshi
      - Call `self.kalshi_client.submit_order(order)`
      - Parse response: extract `order_id`, `status`
      - If status indicates immediate fill: return FillResult
      - Otherwise: store in `pending_orders` with `order_p_model = signal.P_model`, return partial FillResult
    - Log: `structlog.info("order_placed", mode=..., ticker=..., contracts=...)`
    - [C6 fix] On `httpx.HTTPStatusError`:
      - **status 429**: log warning "kalshi_rate_limited", return `FillResult(status="rejected", quantity=0, ...)`
      - **response body contains "market_closed"**: set `self.ticker_muted[signal.ticker] = True`, log warning "ticker_muted", return None
      - **response body contains "insufficient_balance"**: set `self.entries_halted = True`, log error "entries_halted", return None
      - **response body contains "price_out_of_range"**: log warning "price_out_of_range", return `FillResult(status="rejected", quantity=0, ...)`
      - **other**: log error, return `FillResult(status="rejected", quantity=0, ...)`
    - On timeout: log warning, return `FillResult(status="rejected", quantity=0, ...)`

  - `async def cancel_order(self, order_id: str) -> bool`
    - If paper mode: remove from pending_orders, return True
    - If live: call `self.kalshi_client.cancel_order(order_id)`, return success

  - `async def manage_open_orders(self, current_p_model: dict[str, float], current_time: float) -> list[FillResult]` [C2 fix — new method, replaces check_stale_orders]
    - For each order in `pending_orders`:
      - If age > `self.max_order_age_s`: cancel it, log "order_expired"
      - [C2 fix] Elif `abs(current_p_model.get(order["signal"].market_type, order["order_p_model"]) - order["order_p_model"]) > self.reprice_threshold`:
        - Cancel old order
        - Re-post at new P_model: update `order["signal"].P_model`, call `place_order(order["signal"])`
        - Log: `structlog.info("order_repriced", old_price=..., new_price=..., drift=...)`
      - [C2 fix] Elif edge has evaporated (EV at current P_model < theta_exit): cancel, log "order_edge_gone"
    - Return list of FillResults from any fills that occurred during repricing

**What this module does NOT do yet:**
- No partial fill handling for live mode (full fill or cancel only in this sprint)

#### `src/execution/exposure_manager.py`

**Functions/classes to implement this sprint:**

- `class ExposureManager`
  - `__init__(self, db_pool: asyncpg.Pool, trading_mode: TradingMode) -> None`
    - `self.db = db_pool`
    - `self.trading_mode = trading_mode`
    - `self.total_exposure_cap = CONFIG.TOTAL_EXPOSURE_CAP_FRAC`
    - `self.per_match_cap = CONFIG.PER_MATCH_CAP_FRAC`

  - `async def get_bankroll(self) -> float`
    - `SELECT balance FROM bankroll WHERE mode = $1` with `self.trading_mode.value`
    - Return balance as float

  - `async def reserve_exposure(self, match_id: str, ticker: str, amount: float) -> int | None`
    - Check current total exposure: `SELECT SUM(reserved_amount) FROM exposure_reservation WHERE status = 'RESERVED'`
    - Get bankroll
    - If `current_total + amount > bankroll * self.total_exposure_cap`: return None (rejected)
    - `INSERT INTO exposure_reservation (match_id, ticker, reserved_amount, status) VALUES ($1, $2, $3, 'RESERVED') RETURNING id`
    - Log: `structlog.info("exposure_reserved", ...)`
    - Return reservation_id

  - `async def confirm_exposure(self, reservation_id: int, actual_amount: float) -> None`
    - `UPDATE exposure_reservation SET status = 'CONFIRMED', reserved_amount = $2, resolved_at = NOW() WHERE id = $1`

  - `async def release_exposure(self, reservation_id: int) -> None`
    - `UPDATE exposure_reservation SET status = 'RELEASED', resolved_at = NOW() WHERE id = $1`

  - `async def release_stale_reservations(self, max_age_seconds: int = 60) -> int`
    - `UPDATE exposure_reservation SET status = 'RELEASED', resolved_at = NOW() WHERE status = 'RESERVED' AND created_at < NOW() - INTERVAL '{max_age_seconds} seconds'`
    - Return count of released reservations
    - Log: `structlog.warning("stale_reservations_released", count=...)`

  - `async def update_bankroll(self, delta: float) -> None`
    - `UPDATE bankroll SET balance = balance + $1, updated_at = NOW() WHERE mode = $2`
    - Also insert into `bankroll_snapshot`

#### `src/execution/db_positions.py`

**Functions/classes to implement this sprint:**

- `async def save_position(db: asyncpg.Pool, position: Position) -> int`
  - `INSERT INTO positions (match_id, ticker, direction, quantity, entry_price, status, is_paper, entry_tick, entry_reason) VALUES (...) RETURNING id`

- `async def close_position_db(db: asyncpg.Pool, position_id: int, exit_price: float, exit_tick: int, exit_reason: str, realized_pnl: float) -> None`
  - `UPDATE positions SET status = 'CLOSED', exit_price = $2, exit_tick = $3, exit_reason = $4, realized_pnl = $5, closed_at = NOW() WHERE id = $1`

- `async def get_open_positions(db: asyncpg.Pool, match_id: str) -> list[dict]`
  - `SELECT * FROM positions WHERE match_id = $1 AND status = 'OPEN'`

### DB Migrations (this sprint)

No new tables needed — all tables already defined in architecture.md §5.1 (`positions`, `exposure_reservation`, `bankroll`). But we need to ensure the tables exist.

```sql
-- migrations/004_execution_tables.sql
-- These tables were defined in architecture.md §5.1 but may not be created yet.
-- This migration creates them if they don't exist.

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

-- Seed paper bankroll
INSERT INTO bankroll (mode, balance) VALUES ('paper', 10000.00)
ON CONFLICT (mode) DO NOTHING;
INSERT INTO bankroll (mode, balance) VALUES ('live', 0.00)
ON CONFLICT (mode) DO NOTHING;
```

### Interface Contracts

- **signal_generator → exposure_manager.reserve_exposure:** Before placing an order.
  - Caller provides: `match_id`, `ticker`, `amount = signal.contracts * signal.P_kalshi`
  - Callee returns: `reservation_id: int | None` (None = rejected due to cap)
  - Failure mode: DB connection error → raises `asyncpg.PostgresError`. Caller must catch and skip.

- **exposure_manager → order_manager.place_order:** After successful reserve.
  - Caller provides: `Signal` with all fields populated
  - Callee returns: `FillResult | None` [C6 fix — None if ticker muted or entries halted]
  - Failure mode: Kalshi API error → returns FillResult with status="rejected" or None for muted/halted

- **order_manager → exposure_manager.confirm_exposure/release_exposure:** After order result.
  - If fill is not None and fill.status in ("full", "partial", "paper") and quantity > 0: `confirm_exposure(reservation_id, fill.fill_cost)`
  - If fill is None or fill.status == "rejected" or quantity == 0: `release_exposure(reservation_id)` [C6 fix — handle None return]
  - Failure mode: DB error → log and continue (worst case = orphaned reservation, cleaned by CRON)

- **order_manager → position_tracker.add_position:** After confirmed fill.
  - Caller provides: `Signal`, `FillResult`, tick, t
  - Callee returns: `Position`

### Test Plan

**Unit tests:**

`tests/execution/test_order_manager.py`:
- `test_paper_order_immediate_fill`: paper mode → FillResult with status="paper", full quantity
- `test_paper_order_generates_uuid`: paper order_id starts with "paper-"
- `test_live_order_rejected_no_client`: live mode, no kalshi_client → rejected
- `test_live_order_uses_p_model_price`: [C3 fix] live mode → yes_price = int(signal.P_model * 100), NOT P_kalshi
- `test_manage_open_orders_cancels_stale`: order age > 30s → cancelled
- `test_manage_open_orders_reprices_on_drift`: [C2 fix] order at P_model=0.45, current P_model=0.48 (drift=0.03 > 0.02) → order cancelled and re-posted at 0.48
- `test_manage_open_orders_no_reprice_small_drift`: [C2 fix] drift=0.01 < 0.02 → order unchanged
- `test_ticker_muted_on_market_closed`: [C6 fix] Kalshi returns "market_closed" → ticker_muted[ticker] = True, subsequent place_order returns None
- `test_entries_halted_on_insufficient_balance`: [C6 fix] Kalshi returns "insufficient_balance" → entries_halted = True, all subsequent place_order returns None
- `test_price_out_of_range_returns_rejected`: [C6 fix] Kalshi returns "price_out_of_range" → FillResult(status="rejected")

`tests/execution/test_exposure_manager.py` (requires test DB):
- `test_reserve_within_cap`: bankroll=10000, reserve $50 → success
- `test_reserve_exceeds_cap`: bankroll=100, total_cap=20%, reserve $50 → None
- `test_confirm_updates_status`: reserve → confirm → status=CONFIRMED
- `test_release_updates_status`: reserve → release → status=RELEASED
- `test_stale_release`: insert old RESERVED row → release_stale → count=1
- `test_bankroll_update`: delta=-5.50 → balance decremented

`tests/execution/test_db_positions.py` (requires test DB):
- `test_save_and_retrieve`: save → get_open → returns 1 position
- `test_close_position`: save → close → get_open returns empty

**Integration test:**

```python
async def test_sprint_4c_integration():
    """Full paper trade flow: signal → reserve → order → fill → confirm → position.

    Uses test database. Paper mode only.
    Verifies:
    1. Exposure reserved before order
    2. Paper fill at P_kalshi price
    3. Exposure confirmed after fill
    4. Position saved to DB
    5. Bankroll decremented
    """
    db = await create_test_db()
    await run_migration("004_execution_tables.sql")

    exposure = ExposureManager(db, TradingMode.PAPER)
    orders = OrderManager(None, TradingMode.PAPER, db)
    tracker = PositionTracker()

    # Build a signal
    signal = Signal(
        match_id="4190023", ticker="KXEPLGAME-26MAR16BREWOL-BRE",
        market_type="home_win", direction="BUY_YES",
        P_kalshi=0.55, P_model=0.62, EV=0.07,
        kelly_fraction=0.10, kelly_amount=10.0, contracts=18,
        surprise_score=0.0,
    )

    # Get initial bankroll
    bankroll_before = await exposure.get_bankroll()
    assert bankroll_before == 10000.0

    # Reserve
    amount = signal.contracts * signal.P_kalshi  # 18 * 0.55 = $9.90
    res_id = await exposure.reserve_exposure("4190023", signal.ticker, amount)
    assert res_id is not None

    # Place order
    fill = await orders.place_order(signal)
    assert fill is not None  # [C6 fix — check not None before accessing fields]
    assert fill.status == "paper"
    assert fill.quantity == 18

    # Confirm
    await exposure.confirm_exposure(res_id, fill.fill_cost)
    await exposure.update_bankroll(-fill.fill_cost)

    # Save position
    pos = tracker.add_position(signal, fill, tick=100, t=35.0)
    pos_id = await save_position(db, pos)
    assert pos_id > 0

    # Verify bankroll decreased
    bankroll_after = await exposure.get_bankroll()
    assert bankroll_after == bankroll_before - fill.fill_cost
```

### Risks & Gotchas

- **asyncpg connection pool.** Must be initialized before any execution code runs. The match container startup must create the pool.
- **Pattern 4 (Reserve-Confirm-Release):** The reserve→execute→confirm/release sequence is NOT atomic. If the process crashes between reserve and confirm, the stale reservation cleaner handles it — but we must make sure the cleaner actually runs. In Sprint 4d, the execution_loop schedules periodic stale cleanup.
- **[C3 fix] Limit order at P_model, not P_kalshi.** v5 §8.5 says we post a limit order at our model's fair value and wait for someone to trade against us. Posting at P_kalshi would be market-taking, which consumes the spread.
- **[C6 fix] Kalshi rejection state is per-container.** `ticker_muted` resets when the container restarts. `entries_halted` requires manual operator intervention or Kalshi balance top-up.
- **Kalshi 429 rate limiting.** With 5 markets × 1 signal check per tick = up to 5 orders per second in the worst case. Kalshi rate limit is ~10 req/s. Order placement should have a backoff. The current `_rate_limit_delay = 0.1` in KalshiClient handles this partially but may not be enough during bursts.
- **Anti-pattern from post-mortem: "`list()` on JSON string."** When loading `kalshi_tickers` from DB, always use `json.loads()`, never `list()`.

---

## Sprint 4d: Execution Loop + Settlement (Full Integration)

### Goal
Wire signal_generator + kelly_sizer + position_monitor + order_manager + exposure_manager into a single execution_loop coroutine. Add settlement logic including live Kalshi polling. Replay full match 4190023 in paper mode and verify P&L.

### Prerequisites
- Sprint 4a, 4b, 4c all complete
- Recorded match data for 4190023

### Types to Add (this sprint only)

```python
# src/common/types.py — additions

class SettlementResult(BaseModel):
    position_id: int         # [C7 fix — added: links back to position]
    ticker: str
    market_type: str
    direction: str           # [C7 fix — added]
    quantity: int            # [C7 fix — added]
    outcome_occurred: bool   # True if the outcome the contract tracks happened
    settlement_price: float  # 1.0 or 0.0
    realized_pnl: float     # [C7 fix — added: computed PnL for this position]

class MatchPnL(BaseModel):
    match_id: str
    total_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    positions: list[dict]  # summary per position
```

### Modules to Build

#### `src/execution/execution_loop.py`

**Functions/classes to implement this sprint:**

- `async def execution_loop(phase4_queue: asyncio.Queue, model: LiveMatchModel, db_pool: asyncpg.Pool, trading_mode: TradingMode, redis_client: object | None = None) -> MatchPnL`
  - Initialize:
    - `exposure = ExposureManager(db_pool, trading_mode)`
    - `orders = OrderManager(kalshi_client_or_none, trading_mode, db_pool)`
    - `tracker = PositionTracker()` [NC2 fix — uses CONFIG defaults: min_hold=150, cooldown=300]
    - `bankroll = await exposure.get_bankroll()`
    - `stale_check_interval = 300` (5 min in ticks)
    - `tick_counter = 0`
  - Main loop:
    ```python
    while True:
        payload: TickPayload = await phase4_queue.get()
        tick_counter += 1

        if payload.engine_phase == "FINISHED":
            break

        # 1. Check exits on all open positions (every tick, even during cooldown)
        exits = tracker.check_exits(payload, model.p_kalshi)
        for exit_decision in exits:
            pos = tracker.open_positions[exit_decision.position_id]
            if trading_mode == TradingMode.LIVE:
                # Place exit order
                exit_signal = _build_exit_signal(pos, exit_decision)
                fill = await orders.place_order(exit_signal)
            else:
                fill = _paper_exit_fill(pos, exit_decision.exit_price, exit_decision.contracts_to_exit)  # [C5 fix — uses contracts_to_exit]
            if fill is not None and fill.quantity > 0:  # [C6 fix — check not None]
                realized_pnl = compute_exit_pnl(pos, fill)
                tracker.close_position(pos.id, exit_decision.trigger, exit_decision.contracts_to_exit, fill.price, tick_counter)  # [C5 fix]
                await close_position_db(db_pool, pos.db_id, fill.price, tick_counter, exit_decision.trigger.value, realized_pnl)
                await exposure.update_bankroll(realized_pnl)
                bankroll += realized_pnl
                # Publish to Redis
                if redis_client:
                    await _publish_position_update(redis_client, pos, "exit")

        # 2. Generate new signals (only if order_allowed)
        if payload.order_allowed and not orders.entries_halted:  # [C6 fix — check entries_halted]
            signals = generate_signals(payload, model.p_kalshi, model.kalshi_tickers, tracker.open_positions)  # [NC3 fix — pass open_positions]
            for signal in signals:
                # Skip if in cooldown for this market
                if tracker.is_in_cooldown(signal.market_type, tick_counter):
                    continue
                # [NC3 fix — duplicate position check moved into generate_signals]
                # [C6 fix] Skip if ticker muted
                if orders.is_ticker_muted(signal.ticker):
                    continue
                # Size
                signal = size_position(signal, payload, bankroll)
                if signal.contracts <= 0:
                    continue
                # Reserve
                amount = signal.contracts * signal.P_kalshi
                res_id = await exposure.reserve_exposure(payload.match_id, signal.ticker, amount)
                if res_id is None:
                    continue
                # Execute
                fill = await orders.place_order(signal)
                if fill is not None and fill.quantity > 0:  # [C6 fix — check not None]
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

        # 3. Periodic stale reservation cleanup
        if tick_counter % stale_check_interval == 0:
            await exposure.release_stale_reservations()

        # 4. [C2 fix] Manage open orders: cancel stale + reprice on P_model drift
        current_p_model = {mt: getattr(payload.P_model, mt) for mt in ["home_win", "draw", "away_win", "over_25", "btts_yes"] if getattr(payload.P_model, mt) is not None}
        await orders.manage_open_orders(current_p_model, time.monotonic())
    ```
  - After loop exits (FINISHED):
    - Run settlement
    - Return MatchPnL

  - Log: `structlog.info("execution_tick", tick=..., open_positions=len(tracker.open_positions), bankroll=bankroll)`

#### `src/execution/settlement.py`

**Functions/classes to implement this sprint:**

- `async def poll_kalshi_settlement(kalshi_client: KalshiClient, tickers: list[str], timeout_min: int = 45, interval_s: int = 60) -> dict[str, bool]` [C7 fix — new function for live settlement polling]
  - Polls Kalshi `get_market(ticker)` every `interval_s` seconds for up to `timeout_min` minutes.
  - Returns `{ticker: outcome_occurred}` once all tickers have a non-null `result` field.
  - Logic:
    ```python
    outcomes: dict[str, bool] = {}
    deadline = time.monotonic() + timeout_min * 60
    while time.monotonic() < deadline:
        for ticker in tickers:
            if ticker in outcomes:
                continue
            market = await kalshi_client.get_market(ticker)
            result = market.get("result")
            if result is not None:
                outcomes[ticker] = (result == "yes")
                log.info("ticker_settled", ticker=ticker, result=result)
        if len(outcomes) == len(tickers):
            break
        await asyncio.sleep(interval_s)
    # Any unsettled tickers after timeout → log error, return partial
    for ticker in tickers:
        if ticker not in outcomes:
            log.error("settlement_timeout", ticker=ticker)
    return outcomes
    ```

- `async def settle_match(match_id: str, final_score: tuple[int, int], tracker: PositionTracker, db_pool: asyncpg.Pool, kalshi_client: KalshiClient | None, trading_mode: TradingMode) -> MatchPnL`
  - Determine outcomes from final score:
    - `home_win = final_score[0] > final_score[1]`
    - `draw = final_score[0] == final_score[1]`
    - `away_win = final_score[0] < final_score[1]`
    - `over_25 = sum(final_score) >= 3`
    - `btts_yes = final_score[0] >= 1 and final_score[1] >= 1`
  - [C7 fix] If `trading_mode == LIVE` and `kalshi_client`:
    - Collect all tickers from open positions
    - `kalshi_outcomes = await poll_kalshi_settlement(kalshi_client, tickers)`
    - Cross-check: for each position, verify `kalshi_outcomes[ticker]` matches our score-derived outcome
    - If discrepancy: log error "settlement_mismatch", use Kalshi's result (they are authoritative)
  - If `trading_mode == PAPER`: use score-derived outcomes directly (no polling)
  - For each open position:
    - Map `market_type` to outcome bool
    - Compute `realized_pnl = compute_settlement_pnl(position, outcome)`
    - Close position in DB with status='SETTLED'
    - Update bankroll
  - Return `MatchPnL` with totals
  - Log: `structlog.info("match_settled", match_id=..., pnl=..., trades=...)`

#### `src/execution/redis_publisher.py`

**Functions/classes to implement this sprint:**

- `async def _publish_position_update(redis_client, position: Position, update_type: str) -> None`
  - Build `PositionUpdateMessage`, publish to `"position_update"` channel

- `async def _publish_signal(redis_client, signal: Signal, fill: FillResult) -> None`
  - Build `SignalMessage`, publish to `f"signal:{signal.match_id}"` channel

### DB Migrations (if any)
None — using tables from Sprint 4c.

### Interface Contracts

- **Phase 3 tick_loop → execution_loop:** `asyncio.Queue(maxsize=1)`. tick_loop `put()`, execution_loop `get()`.
  - Critical: `maxsize=1` means if execution takes >1s, the queue drops old payloads. This is correct — we always want the latest state.
  - Actually, `Queue.put()` on a full queue blocks until space is available. So tick_loop's `put()` will block if execution is slow. Consider using `put_nowait()` with try/except and `get_nowait()` to always get latest, OR use a simple variable with a lock. **Decision: use `asyncio.Queue(maxsize=1)` with `put()` — if execution blocks, tick_loop waits, which is acceptable since both are in the same event loop.**

- **execution_loop → signal_generator:** Direct function call. `generate_signals(payload, p_kalshi, tickers, open_positions)` [NC3 fix]
- **execution_loop → kelly_sizer:** Direct function call. `size_position(signal, payload, bankroll)`
- **execution_loop → position_tracker:** Direct method calls. `check_exits()`, `add_position()`, `close_position(pos_id, trigger, contracts_to_exit, price, tick)` [C5 fix]
- **execution_loop → exposure_manager:** Direct async method calls. `reserve_exposure()`, etc.
- **execution_loop → order_manager:** Direct async method calls. `place_order()`, `manage_open_orders()` [C2 fix]

### Test Plan

**Unit tests:**

`tests/execution/test_settlement.py`:
- `test_settle_home_win`: score (2,1) → home_win positions profit, away_win positions lose
- `test_settle_draw`: score (2,2) → draw positions profit
- `test_settle_over_25`: score (2,1) → over_25 positions profit (total=3)
- `test_settle_under_25`: score (1,0) → over_25 positions lose (total=1)
- `test_settle_btts_yes`: score (2,1) → btts positions profit
- `test_settle_btts_no`: score (1,0) → btts positions lose
- `test_settle_no_positions`: empty tracker → pnl=0
- `test_poll_kalshi_settlement_success`: [C7 fix] mock get_market returns null twice then "yes" → returns {ticker: True}
- `test_poll_kalshi_settlement_timeout`: [C7 fix] mock never returns result → partial dict after timeout, logged error
- `test_settle_live_uses_kalshi_outcomes`: [C7 fix] live mode → calls poll_kalshi_settlement, uses Kalshi result
- `test_settle_paper_skips_polling`: [C7 fix] paper mode → no Kalshi API calls, uses score

**Integration test (THE KEY TEST):**

```python
async def test_sprint_4d_full_match_replay():
    """Replay match 4190023 (Brentford 2-2 Wolves) end-to-end in paper mode.

    This is the critical test that proves Phase 4 works.

    Setup:
    - ReplayServer serves recorded Goalserve + Kalshi data
    - Phase 3 tick_loop produces TickPayloads
    - Phase 4 execution_loop consumes them
    - Paper mode (no real Kalshi API)

    Verify:
    1. At least 1 trade is entered during the match
    2. All trades have positive EV at entry
    3. Settlement computes correct P&L for a 2-2 draw
    4. Draw market positions profit, home/away win positions lose (if held)
    5. Bankroll change matches sum of realized PnLs
    6. No orphaned reservations remain
    7. No positions left open after settlement
    """
    # Setup
    db = await create_test_db()
    await run_migration("004_execution_tables.sql")
    replay = ReplayServer("data/recordings/4190023", speed=100.0)
    await replay.start()

    # Build model from recorded Phase2Result
    model = build_test_model("4190023")
    phase4_queue = asyncio.Queue(maxsize=1)

    # Run Phase 3 tick_loop and Phase 4 execution_loop concurrently
    match_pnl = None
    async def run_phase4():
        nonlocal match_pnl
        match_pnl = await execution_loop(
            phase4_queue, model, db,
            TradingMode.PAPER, redis_client=None,
        )

    await asyncio.gather(
        tick_loop(model, phase4_queue=phase4_queue),
        run_phase4(),
    )

    # Assertions
    assert match_pnl is not None
    assert match_pnl.trade_count >= 1
    # 2-2 draw → draw positions should profit
    for pos in match_pnl.positions:
        if pos["market_type"] == "draw" and pos["direction"] == "BUY_YES":
            assert pos["realized_pnl"] > 0
    # Bankroll reconciliation
    final_bankroll = await ExposureManager(db, TradingMode.PAPER).get_bankroll()
    assert abs(final_bankroll - (10000.0 + match_pnl.total_pnl)) < 0.01
    # No orphaned reservations
    stale = await db.fetchval(
        "SELECT COUNT(*) FROM exposure_reservation WHERE status = 'RESERVED'"
    )
    assert stale == 0

    await replay.stop()
```

### Risks & Gotchas

- **Queue maxsize=1 semantics.** `asyncio.Queue.put()` blocks when full. Since tick_loop and execution_loop are both async in the same event loop, if execution takes >1 tick, tick_loop's `put()` awaits. This means execution_loop must process each tick faster than 1s on average. Paper mode is fast (no API calls). Live mode may need a non-blocking approach.
- **[C7 fix] Settlement polling in live mode.** Kalshi settles 10-30 minutes after final whistle. `poll_kalshi_settlement` polls every 60s for up to 45 minutes. The execution_loop holds open until settlement completes (match container stays alive). If settlement times out, positions are logged as unresolved — operator must manually reconcile.
- **Replay speed.** At `speed=100.0`, a 90-min match replays in ~54 seconds. Integration test should complete in <2 minutes.
- **Anti-pattern: "No cooldown after exit → instant re-entry loop."** Sprint 4b's PositionTracker implements cooldown. Sprint 4d's execution_loop checks `is_in_cooldown()` before entry.
- **P_kalshi availability.** `model.p_kalshi` is populated by `kalshi_ob_sync`. During replay, this is populated from recorded data. If no Kalshi data is available for some ticks, `p_kalshi` may be empty → no signals generated → no trades. This is correct behavior.
- **[C6 fix] Muted tickers and halted entries.** `orders.entries_halted` checked in main loop before signal generation. `orders.is_ticker_muted(ticker)` checked per-signal. Both states reset on container restart.

---

## Sprint 5a: Match Discovery + Lifecycle State Machine

### Goal
The orchestrator can discover upcoming matches across all 8 leagues, store them in `match_schedule`, and track their lifecycle state transitions.

### Prerequisites
- Sprint 4d complete (Phase 4 can run a match)
- PostgreSQL + Redis running

### Types to Add (this sprint only)

```python
# src/common/types.py — additions

class MatchStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    PHASE2_RUNNING = "PHASE2_RUNNING"
    PHASE2_DONE = "PHASE2_DONE"
    PHASE2_SKIPPED = "PHASE2_SKIPPED"
    PHASE3_RUNNING = "PHASE3_RUNNING"
    FINISHED = "FINISHED"
    ARCHIVED = "ARCHIVED"

class MatchScheduleRecord(BaseModel):
    match_id: str
    league_id: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: MatchStatus
    trading_mode: TradingMode
    param_version: int | None = None
    kalshi_tickers: dict[str, str] | None = None
    goalserve_fix_id: str | None = None
```

### Modules to Build

#### `src/orchestrator/match_discovery.py`

**Functions/classes to implement this sprint:**

- `async def discover_matches(db_pool: asyncpg.Pool) -> list[MatchScheduleRecord]`
  - Call `map_all_leagues()` from `cross_source_mapper` (already implemented)
  - Filter results: only matches with `match_status == "ALL_MATCHED"` and `kalshi_event_ticker is not None`
  - For each matched fixture:
    - Check if already in `match_schedule` DB table (by match_id)
    - If new: `INSERT INTO match_schedule (match_id, league_id, home_team, away_team, kickoff_utc, status, trading_mode, kalshi_tickers, goalserve_fix_id) VALUES (...)`
    - Set `trading_mode = 'paper'` (always paper until validation gates pass)
  - Log: `structlog.info("matches_discovered", new=..., existing=...)`
  - Return list of new records

- `async def get_actionable_matches(db_pool: asyncpg.Pool) -> dict[str, list[MatchScheduleRecord]]`
  - Query `match_schedule` for matches needing action:
    - `needs_phase2`: status=SCHEDULED AND kickoff_utc - NOW() <= 65 minutes
    - `needs_container`: status=PHASE2_DONE AND kickoff_utc - NOW() <= 2 minutes
    - `needs_cleanup`: status=FINISHED AND updated_at < NOW() - INTERVAL '1 hour'
  - Return `{"needs_phase2": [...], "needs_container": [...], "needs_cleanup": [...]}`

#### `src/orchestrator/lifecycle.py`

**Functions/classes to implement this sprint:**

- `class MatchLifecycle`
  - `__init__(self, db_pool: asyncpg.Pool) -> None`

  - `async def transition(self, match_id: str, from_status: MatchStatus, to_status: MatchStatus, **extra_fields) -> bool`
    - Validate transition is legal (define allowed transitions):
      ```
      SCHEDULED → PHASE2_RUNNING
      PHASE2_RUNNING → PHASE2_DONE | PHASE2_SKIPPED
      PHASE2_DONE → PHASE3_RUNNING
      PHASE3_RUNNING → FINISHED
      FINISHED → ARCHIVED
      ```
    - `UPDATE match_schedule SET status = $1, updated_at = NOW(), {extra_fields} WHERE match_id = $2 AND status = $3`
    - If row count = 0: log warning (concurrent modification or invalid transition), return False
    - Log: `structlog.info("match_transition", match_id=..., from=..., to=...)`
    - Return True

  - `async def get_status(self, match_id: str) -> MatchStatus | None`
    - `SELECT status FROM match_schedule WHERE match_id = $1`

  - `async def run_phase2(self, match_id: str) -> Phase2Result | None`
    - Transition SCHEDULED → PHASE2_RUNNING
    - Run Phase 2 backsolve (calls existing Phase 2 code)
    - If verdict == "GO":
      - Transition PHASE2_RUNNING → PHASE2_DONE with `param_version`, `kalshi_tickers`
      - Return Phase2Result
    - If verdict == "SKIP":
      - Transition PHASE2_RUNNING → PHASE2_SKIPPED
      - Return None
    - On exception: transition → PHASE2_SKIPPED, log error, return None

**What this module does NOT do yet:**
- No Docker container management (deferred to Sprint 5b)
- No heartbeat monitoring
- No recovery logic

### DB Migrations (if any)
None — `match_schedule` table already defined in architecture.md §5.1.

### Interface Contracts

- **discover_matches → cross_source_mapper.map_all_leagues:**
  - Caller provides: httpx.AsyncClient (optional, created internally if not provided)
  - Callee returns: `dict[str, list[dict]]` — league → matched fixtures
  - Failure mode: Individual league failures are caught internally; partial results returned.

- **lifecycle.transition → PostgreSQL:**
  - Uses conditional UPDATE (WHERE status = from_status) for optimistic concurrency
  - If concurrent modification: returns False, caller retries or logs

### Test Plan

**Unit tests:**

`tests/orchestrator/test_match_discovery.py` (requires test DB + mocked API responses):
- `test_discover_new_match`: mock map_all_sources → 1 ALL_MATCHED result → inserted into DB
- `test_discover_existing_match`: match already in DB → not duplicated
- `test_discover_missing_kalshi`: match without kalshi_event_ticker → skipped
- `test_actionable_phase2`: match SCHEDULED, kickoff in 60 min → in needs_phase2 list
- `test_actionable_container`: match PHASE2_DONE, kickoff in 1 min → in needs_container list
- `test_not_actionable_too_early`: match SCHEDULED, kickoff in 3 hours → not actionable

`tests/orchestrator/test_lifecycle.py` (requires test DB):
- `test_valid_transition`: SCHEDULED → PHASE2_RUNNING → True
- `test_invalid_transition`: SCHEDULED → FINISHED → False (not allowed)
- `test_concurrent_transition`: two transitions from SCHEDULED → only one succeeds
- `test_phase2_go`: mock phase2 returns GO → PHASE2_DONE
- `test_phase2_skip`: mock phase2 returns SKIP → PHASE2_SKIPPED

**Integration test:**

```python
async def test_sprint_5a_integration():
    """Discover matches and run lifecycle through Phase 2.

    Uses mocked API responses from cross_source_mapper.
    Verifies:
    1. Matches are discovered and stored in DB
    2. Actionable matches are identified at correct times
    3. Phase 2 transitions work correctly
    4. Status is accurate after each transition
    """
    db = await create_test_db()

    # Insert a mock match (simulating discovery)
    lifecycle = MatchLifecycle(db)
    await db.execute("""
        INSERT INTO match_schedule (match_id, league_id, home_team, away_team,
            kickoff_utc, status, trading_mode)
        VALUES ('test-001', 1204, 'Arsenal', 'Chelsea',
            NOW() + INTERVAL '60 minutes', 'SCHEDULED', 'paper')
    """)

    # Verify actionable
    actionable = await get_actionable_matches(db)
    assert len(actionable["needs_phase2"]) == 1

    # Run Phase 2 (mocked)
    with mock_phase2_returns_go():
        result = await lifecycle.run_phase2("test-001")
        assert result is not None
        assert result.verdict == "GO"

    status = await lifecycle.get_status("test-001")
    assert status == MatchStatus.PHASE2_DONE
```

### Risks & Gotchas

- **cross_source_mapper makes live API calls.** Integration tests must mock these. Use `unittest.mock.patch` or dependency injection.
- **Timezone handling.** `kickoff_utc` from Goalserve may be in a different timezone. All comparisons use UTC. The `_parse_kickoff` function in goalserve.py already returns UTC.
- **Anti-pattern from post-mortem: "Ticker JSON passing — always json.loads()."** When storing `kalshi_tickers` in DB (JSONB column), ensure it's a dict, not a JSON string.

---

## Sprint 5b: Container Management + Scheduler Loop + Recovery

### Goal
The orchestrator runs as a persistent service that discovers matches, triggers Phase 2, launches Docker containers for each match, monitors heartbeats, and recovers from restarts.

### Prerequisites
- Sprint 5a complete (discovery + lifecycle)
- Docker daemon accessible

### Types to Add (this sprint only)

```python
# No new Pydantic types needed — uses MatchScheduleRecord from 5a.
```

### Modules to Build

#### `src/orchestrator/container_manager.py`

**Functions/classes to implement this sprint:**

- `class ContainerManager`
  - `__init__(self, docker_client: docker.DockerClient | None = None) -> None`
    - If None: `self.client = docker.from_env()`
    - `self.network_name = "mmpp-net"`

  - `async def launch_match_container(self, match_id: str, phase2_result: Phase2Result, trading_mode: TradingMode) -> str`
    - Container name: `f"match-{match_id}"`
    - Check if container with this name already exists → remove if exists (stale container cleanup)
    - Environment variables: serialize Phase2Result fields
      - `MATCH_ID`, `LEAGUE_ID`, `A_H`, `A_A`, `PARAM_VERSION`, `KALSHI_TICKERS` (json.dumps!), `TRADING_MODE`, `EKF_P0`, etc.
    - Launch container on `mmpp-net` network
    - Log: `structlog.info("container_launched", match_id=..., container_name=...)`
    - Return container_id

  - `async def stop_container(self, match_id: str) -> bool`
    - Find container by name `f"match-{match_id}"`
    - Stop with timeout=30s, then remove
    - Return True if successful

  - `async def check_heartbeat(self, match_id: str) -> bool`
    - Check if container is running
    - Check Redis for recent tick messages: `GET tick:{match_id}:last_ts`
    - If no tick message in >60 seconds and container is running → unhealthy
    - Return True if healthy

  - `async def list_running_containers(self) -> list[str]`
    - List all containers with name prefix `match-` on `mmpp-net`
    - Return list of match_ids

#### `src/orchestrator/scheduler.py`

**Functions/classes to implement this sprint:**

- `async def orchestrator_main_loop(db_pool: asyncpg.Pool, redis_client: object) -> None`
  - `lifecycle = MatchLifecycle(db_pool)`
  - `containers = ContainerManager()`
  - `discovery_interval = 6 * 3600` (6 hours in seconds)
  - `check_interval = 60` (1 minute)
  - Main loop:
    ```python
    last_discovery = 0
    while True:
        now = time.time()

        # Periodic match discovery
        if now - last_discovery >= discovery_interval:
            await discover_matches(db_pool)
            last_discovery = now

        # Get actionable matches
        actionable = await get_actionable_matches(db_pool)

        # Trigger Phase 2
        for match in actionable["needs_phase2"]:
            result = await lifecycle.run_phase2(match.match_id)

        # Launch containers
        for match in actionable["needs_container"]:
            phase2 = await load_phase2_result(db_pool, match.match_id)
            if phase2:
                await containers.launch_match_container(
                    match.match_id, phase2, match.trading_mode
                )
                await lifecycle.transition(
                    match.match_id, MatchStatus.PHASE2_DONE, MatchStatus.PHASE3_RUNNING
                )

        # Cleanup finished matches
        for match in actionable["needs_cleanup"]:
            await containers.stop_container(match.match_id)
            await lifecycle.transition(
                match.match_id, MatchStatus.FINISHED, MatchStatus.ARCHIVED
            )

        # Heartbeat monitoring
        running = await containers.list_running_containers()
        for match_id in running:
            if not await containers.check_heartbeat(match_id):
                await _publish_alert(redis_client, match_id, "heartbeat_missing")

        await asyncio.sleep(check_interval)
    ```

- `async def recover_orchestrator_state(db_pool: asyncpg.Pool, containers: ContainerManager) -> None`
  - Query `match_schedule` for matches in intermediate states:
    - `PHASE2_RUNNING` with `updated_at < NOW() - INTERVAL '10 minutes'`: re-run Phase 2
    - `PHASE2_DONE` with kickoff past and no running container: launch container
    - `PHASE3_RUNNING` with no running container: mark as stale, alert
  - Log: `structlog.warning("orchestrator_recovery", ...)`

### DB Migrations (if any)
None.

### Interface Contracts

- **scheduler → lifecycle.run_phase2:** Direct call
  - Returns `Phase2Result | None`

- **scheduler → container_manager.launch_match_container:** Direct call
  - Caller provides: `match_id`, `Phase2Result`, `TradingMode`
  - Callee returns: container_id string
  - Failure mode: Docker API error → raises `docker.errors.APIError`. Caller catches, logs, alerts.

- **container_manager → Docker API:** Uses `docker` Python SDK
  - `client.containers.run(...)` for launch
  - `client.containers.get(...)` for status check
  - `client.networks.get("mmpp-net")` for network

### Test Plan

**Unit tests:**

`tests/orchestrator/test_container_manager.py` (mocked Docker):
- `test_launch_creates_container`: mock docker.containers.run → called with correct env vars
- `test_launch_removes_stale`: existing container with same name → removed first
- `test_stop_container`: mock stop and remove → called
- `test_heartbeat_healthy`: container running, recent tick → True
- `test_heartbeat_stale`: container running, no tick in 90s → False

`tests/orchestrator/test_scheduler.py` (mocked):
- `test_discovery_triggers_at_interval`: first loop → discovery runs; second loop within interval → skipped
- `test_phase2_triggered_at_65min`: match 60 min before kickoff → Phase 2 runs
- `test_container_launched_at_2min`: PHASE2_DONE, 1 min to kickoff → container launched
- `test_cleanup_after_1hour`: FINISHED match, 2 hours old → archived

`tests/orchestrator/test_recovery.py` (requires test DB):
- `test_recover_stale_phase2`: PHASE2_RUNNING for 15 min → re-triggered
- `test_recover_missed_launch`: PHASE2_DONE past kickoff → container launched
- `test_recover_dead_container`: PHASE3_RUNNING, no container → alert published

**Integration test:**

```python
async def test_sprint_5b_integration():
    """Orchestrator discovers match, runs Phase 2, launches container.

    Uses mocked Docker API and mocked data sources.
    Verifies the full lifecycle from SCHEDULED → PHASE3_RUNNING.
    """
    db = await create_test_db()

    # Insert a SCHEDULED match 60 min from now
    await db.execute("""
        INSERT INTO match_schedule (match_id, league_id, home_team, away_team,
            kickoff_utc, status, trading_mode)
        VALUES ('test-full-001', 1204, 'Arsenal', 'Chelsea',
            NOW() + INTERVAL '60 minutes', 'SCHEDULED', 'paper')
    """)

    lifecycle = MatchLifecycle(db)
    containers = MockContainerManager()  # doesn't actually launch Docker

    # Phase 2
    with mock_phase2_returns_go():
        result = await lifecycle.run_phase2("test-full-001")
        assert result is not None

    # Transition to PHASE2_DONE should have happened
    status = await lifecycle.get_status("test-full-001")
    assert status == MatchStatus.PHASE2_DONE

    # Simulate "kickoff approaching" — manually launch
    await containers.launch_match_container("test-full-001", result, TradingMode.PAPER)
    await lifecycle.transition("test-full-001", MatchStatus.PHASE2_DONE, MatchStatus.PHASE3_RUNNING)

    status = await lifecycle.get_status("test-full-001")
    assert status == MatchStatus.PHASE3_RUNNING
```

### Risks & Gotchas

- **Docker-in-Docker.** The orchestrator itself runs in a Docker container and needs to launch sibling containers. This requires mounting the Docker socket (`/var/run/docker.sock`). Security implications noted.
- **Anti-pattern from post-mortem: "Stale container cleanup."** Sprint 5b explicitly checks for and removes existing containers before launching new ones.
- **Environment variable serialization.** `KALSHI_TICKERS` must be `json.dumps()`, not `str()`. The container entrypoint must `json.loads()` it back.
- **Orchestrator SPOF mitigation.** The architecture specifies that running containers operate independently. If the orchestrator dies, running matches continue. Sprint 5b implements this by design — match containers have no runtime dependency on the orchestrator.

---

## Sprint 6a: REST API + P&L Service (Backend Only)

### Goal
A FastAPI backend serves match state, positions, P&L, and system status via REST endpoints. No frontend, no WebSocket yet.

### Prerequisites
- Sprint 4d complete (positions in DB)
- Sprint 5a complete (match_schedule in DB)

### Types to Add (this sprint only)

```python
# src/dashboard/schemas.py (API response models)

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
    exit_price: float | None
    status: str
    is_paper: bool
    realized_pnl: float | None
    entry_reason: str | None
    exit_reason: str | None
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
    last_phase1_run: datetime | None
    alerts: list[dict]
```

### Modules to Build

#### `src/dashboard/api.py`

**Functions/classes to implement this sprint:**

- FastAPI app with the following endpoints:

- `GET /api/matches` → `list[MatchSummary]`
  - Query `match_schedule` joined with latest tick snapshot and open position counts
  - Optional query param: `status` filter, `league_id` filter

- `GET /api/matches/{match_id}` → `MatchSummary` + full tick history
  - Return match details + last N tick snapshots from `tick_snapshots`

- `GET /api/matches/{match_id}/ticks` → `list[dict]`
  - Query `tick_snapshots WHERE match_id = $1 ORDER BY t`
  - Optional: `from_t`, `to_t` params for time range

- `GET /api/positions` → `list[PositionDetail]`
  - Query `positions` table
  - Optional: `match_id`, `status` filter

- `GET /api/positions/{position_id}` → `PositionDetail`

- `GET /api/pnl` → `PnLSummary`
  - Aggregate from `positions WHERE status IN ('CLOSED', 'SETTLED')`
  - Compute win rate, avg PnL, max drawdown from `bankroll_snapshot`
  - Break down by market_type (from ticker pattern) and by league_id

- `GET /api/pnl/history` → `list[dict]`
  - Return `bankroll_snapshot` time series for charts

- `GET /api/system/status` → `SystemStatus`
  - Active matches: count PHASE3_RUNNING
  - Open positions: count status=OPEN
  - Bankroll: from `bankroll` table
  - Alerts: from Redis LRANGE on `system_alerts` key (last 20)

- `GET /api/system/health` → `{"status": "ok", "timestamp": ...}`

#### `src/dashboard/pnl_service.py`

**Functions/classes to implement this sprint:**

- `async def compute_pnl_summary(db_pool: asyncpg.Pool, trading_mode: str = "paper", match_id: str | None = None) -> PnLSummary`
  - Query closed/settled positions
  - Compute: total_pnl, win_count, loss_count, win_rate
  - Compute max_drawdown from bankroll_snapshot series:
    - Peak tracking: `max_drawdown = max(peak - current for each snapshot)`
  - Group by market_type and league_id for breakdown
  - Return PnLSummary

### DB Migrations (if any)
None — all tables already exist.

### Interface Contracts

- **api.py → PostgreSQL:** All reads via `asyncpg.Pool` connection
  - Uses `db.fetch()` for list queries, `db.fetchrow()` for single records
  - All queries are read-only (no writes from dashboard)

- **api.py → Redis:** Read-only for alerts and latest tick state
  - `redis.lrange("system_alerts", 0, 19)` for recent alerts

### Test Plan

**Unit tests:**

`tests/dashboard/test_api.py` (using `httpx.AsyncClient` + test DB):
- `test_get_matches_empty`: no matches → empty list
- `test_get_matches_with_data`: insert 2 matches → returns 2
- `test_get_matches_filter_status`: filter by PHASE3_RUNNING → correct subset
- `test_get_positions`: insert positions → returns correct list
- `test_get_pnl_no_trades`: no positions → zeros
- `test_get_pnl_with_trades`: insert closed positions → correct summary
- `test_system_status`: returns valid SystemStatus
- `test_health_endpoint`: returns {"status": "ok"}

`tests/dashboard/test_pnl_service.py`:
- `test_max_drawdown_calculation`: known bankroll series → correct drawdown
- `test_pnl_by_market_type`: 2 home_win trades, 1 draw trade → correct breakdown
- `test_win_rate`: 3 wins, 2 losses → 60%

**Integration test:**

```python
async def test_sprint_6a_integration():
    """REST API returns correct data for a completed paper match.

    Insert match + positions + tick snapshots into DB.
    Query all endpoints and verify responses.
    """
    db = await create_test_db()
    # Insert test match
    await db.execute("INSERT INTO match_schedule ...")
    # Insert test positions (2 wins, 1 loss)
    await db.execute("INSERT INTO positions ...")
    # Insert bankroll snapshots
    await db.execute("INSERT INTO bankroll_snapshot ...")

    app = create_dashboard_app(db)
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # Matches
        r = await client.get("/api/matches")
        assert r.status_code == 200
        assert len(r.json()) == 1

        # Positions
        r = await client.get("/api/positions?match_id=test-001")
        assert r.status_code == 200
        assert len(r.json()) == 3

        # P&L
        r = await client.get("/api/pnl")
        assert r.status_code == 200
        pnl = r.json()
        assert pnl["trade_count"] == 3
        assert pnl["win_count"] == 2

        # System status
        r = await client.get("/api/system/status")
        assert r.status_code == 200
```

### Risks & Gotchas

- **Large tick_snapshots table.** A 90-minute match produces ~5400 rows. Queries with `ORDER BY t` need the index `idx_ticks_match`. Pagination may be needed for the ticks endpoint.
- **bankroll_snapshot for drawdown.** Must be populated during execution (Sprint 4d should insert snapshots after each fill). If missing, drawdown = 0.
- **JSON key naming convention:** `sigma_MC` everywhere, not Greek. Architecture §3.6 is explicit about this.

---

## Sprint 6b: WebSocket + Redis Subscriber + React Frontend

### Goal
Live match data streams from Redis → WebSocket → React UI. The dashboard shows live match cards, P_model charts, and position updates in real time.

### Prerequisites
- Sprint 6a complete (REST API working)
- Redis running with match containers publishing tick/event/signal/position messages

### Types to Add (this sprint only)

```python
# No new Python types — uses existing Redis message types from types.py
# React TypeScript types mirror the Python types
```

### Modules to Build

#### `src/dashboard/ws_handler.py`

**Functions/classes to implement this sprint:**

- `class DashboardWSManager`
  - `__init__(self, redis_client) -> None`
    - `self.redis = redis_client`
    - `self.active_connections: list[WebSocket] = []`
    - `self.subscribed_matches: set[str] = set()` — currently subscribed match IDs
    - `self._pubsub = None`

  - `async def connect(self, websocket: WebSocket) -> None`
    - Accept connection
    - Send initial state via REST (full snapshot)
    - Add to `active_connections`
    - Start listening for client messages (subscribe/unsubscribe match IDs)

  - `async def disconnect(self, websocket: WebSocket) -> None`
    - Remove from active_connections
    - Diff-based unsubscribe for match channels no longer needed

  - `async def start_redis_subscriber(self) -> None`
    - Subscribe to global channels: `position_update`, `system_alert`
    - Subscribe to match-specific channels based on `subscribed_matches`: `tick:{id}`, `event:{id}`, `signal:{id}`
    - On message: broadcast to all connected WebSocket clients
    - Log: `structlog.debug("ws_broadcast", channel=..., clients=len(self.active_connections))`

  - `async def update_match_subscriptions(self, match_ids: set[str]) -> None`
    - Compute diff: new_matches = match_ids - subscribed_matches, removed = subscribed_matches - match_ids
    - Subscribe to new channels, unsubscribe from removed
    - Update `self.subscribed_matches`

  - Reconnection: exponential backoff (1s base, 30s max, 10 retries) on Redis connection loss

- FastAPI WebSocket endpoint:
  ```python
  @app.websocket("/ws")
  async def websocket_endpoint(websocket: WebSocket):
      await manager.connect(websocket)
      try:
          while True:
              data = await websocket.receive_json()
              if data.get("type") == "subscribe":
                  await manager.update_match_subscriptions(set(data["match_ids"]))
      except WebSocketDisconnect:
          await manager.disconnect(websocket)
  ```

#### `src/dashboard/frontend/` (React)

**Components to build this sprint:**

- `CommandCenter.tsx` — Main view: grid of match cards
  - Each card: score, minute, engine_phase, P_model bars, order_allowed status
  - Cards appear/disappear as matches start/end
  - Click card → navigate to MatchDeepDive

- `MatchCard.tsx` — Single match summary card
  - Score display, match minute, engine_phase badge
  - Three probability bars (home_win, draw, away_win) from P_model
  - SurpriseScore indicator (orange when > 0.5)
  - EKF uncertainty indicator
  - Open positions count + unrealized P&L
  - order_allowed status light (green/red)

- `MatchDeepDive.tsx` — Detailed match view
  - P_model time series chart (updated via WebSocket)
  - Team strength trajectory (a_H, a_A over time)
  - Goal events annotated with SurpriseScore
  - Position entry/exit markers on chart
  - Current open positions table

- `PnLDashboard.tsx` — P&L analytics
  - Cumulative P&L line chart
  - Breakdown tables by market type, league
  - Win rate display

- `SystemStatus.tsx` — Operations view
  - Active container list
  - Phase 1 last run info
  - Alert feed
  - Bankroll display (paper + live)

- `hooks/useWebSocket.ts` — WebSocket connection management
  - Auto-reconnect with exponential backoff
  - Subscribe to match IDs based on current view
  - Parse incoming messages and update React state

### DB Migrations (if any)
None.

### Interface Contracts

- **WebSocket client → server:** JSON messages
  - `{"type": "subscribe", "match_ids": ["4190023", "4190024"]}`
  - `{"type": "unsubscribe", "match_ids": ["4190023"]}`

- **WebSocket server → client:** JSON messages from Redis channels
  - `{"channel": "tick:4190023", "data": {...TickMessage...}}`
  - `{"channel": "event:4190023", "data": {...EventMessage...}}`
  - `{"channel": "signal:4190023", "data": {...SignalMessage...}}`
  - `{"channel": "position_update", "data": {...PositionUpdateMessage...}}`
  - `{"channel": "system_alert", "data": {...SystemAlertMessage...}}`

- **React → REST API:** Standard fetch calls for initial state
  - On page load: `GET /api/matches`, `GET /api/system/status`
  - On match select: `GET /api/matches/{id}`, `GET /api/matches/{id}/ticks`
  - On P&L view: `GET /api/pnl`, `GET /api/pnl/history`

### Test Plan

**Unit tests:**

`tests/dashboard/test_ws_handler.py`:
- `test_connect_sends_initial_state`: new WS connection → receives snapshot
- `test_subscribe_adds_channels`: subscribe to match → Redis channels subscribed
- `test_unsubscribe_removes_channels`: unsubscribe → channels removed
- `test_broadcast_to_all_clients`: 2 connected clients → both receive message
- `test_disconnect_cleanup`: client disconnects → removed from active_connections
- `test_redis_reconnect`: simulate Redis connection drop → reconnects

**Integration test:**

```python
async def test_sprint_6b_integration():
    """WebSocket receives live tick updates from Redis.

    Publishes a tick message to Redis, verifies the connected
    WebSocket client receives it.
    """
    db = await create_test_db()
    redis = await create_test_redis()

    app = create_dashboard_app(db, redis)

    async with httpx.AsyncClient(app=app, base_url="http://test") as http:
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect("ws://localhost:8001/ws")

            # Subscribe to a match
            await ws.send_json({"type": "subscribe", "match_ids": ["4190023"]})

            # Publish a tick to Redis
            tick_msg = TickMessage(
                match_id="4190023", t=35.0, engine_phase="FIRST_HALF",
                P_model=MarketProbs(home_win=0.45, draw=0.30, away_win=0.25),
                ...
            )
            await redis.publish("tick:4190023", tick_msg.model_dump_json())

            # Receive on WebSocket
            msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            assert msg["channel"] == "tick:4190023"
            assert msg["data"]["t"] == 35.0

            await ws.close()
```

### Risks & Gotchas

- **Anti-pattern from architecture §3.6: "Redis subscribe once, not in loop."** The `start_redis_subscriber` creates the pubsub once, then diff-subscribes/unsubscribes as matches change.
- **WebSocket reconnection.** Clients must re-subscribe after reconnect. The `useWebSocket` hook should send the current match_ids on each reconnect.
- **JSON key naming: `sigma_MC`** everywhere, not `σ_MC`. TypeScript can't use Greek letters as property names.
- **Mid-match connect.** A new dashboard client connecting mid-match needs the current state. Sprint 6a's REST endpoints provide the full snapshot, then WebSocket provides incremental updates.

---

## Validation Questions

### 1. Type Completeness

| Type/Enum | Created in Sprint | First Used in Sprint |
|-----------|-------------------|---------------------|
| `TradingMode` | 4a | 4a (kelly_sizer import only), 4c (actual use) |
| `ExitTrigger` | 4a | 4b |
| `Signal` | Already exists (types.py) | 4a |
| `FillResult` | Already exists (types.py) | 4a (interface), 4c (actual use) |
| `Position` | 4b | 4b |
| `ExitDecision` | 4b | 4b |
| `OrderStatus` | 4c | 4c |
| `ExposureStatus` | 4c | 4c |
| `SettlementResult` | 4d | 4d |
| `MatchPnL` | 4d | 4d |
| `MatchStatus` | 5a | 5a |
| `MatchScheduleRecord` | 5a | 5a |
| `MatchSummary` | 6a | 6a |
| `PositionDetail` | 6a | 6a |
| `PnLSummary` | 6a | 6a |
| `SystemStatus` | 6a | 6a |

**No type is used before it is created.** `Signal` and `FillResult` already exist in `types.py`.

### 2. Interface Consistency

**Verified cross-module calls:**

| Caller → Callee | Match? | Notes |
|-----------------|--------|-------|
| `tick_loop.put(TickPayload)` → `execution_loop.get()` | OK | Both use `asyncio.Queue[TickPayload]` |
| `generate_signals(payload, p_kalshi, tickers, open_positions)` | OK | [NC3 fix] open_positions added |
| `size_position(signal, payload, bankroll)` | OK | Signal has P_model and P_kalshi; payload has surprise_score, ekf_P_H/A, mu_H/A |
| `tracker.add_position(signal, fill, tick, t)` | OK | Signal provides ticker, direction, market_type; fill provides quantity, price |
| `tracker.close_position(id, trigger, contracts_to_exit, price, tick)` | OK | [C5 fix] contracts_to_exit added |
| `exposure.reserve_exposure(match_id, ticker, amount)` | OK | All strings/floats from Signal |
| `orders.place_order(signal)` | OK | Returns `FillResult | None` [C6 fix] |
| `orders.manage_open_orders(current_p_model, current_time)` | OK | [C2 fix] new method |

**One mismatch found and resolved:** The old architecture.md `Signal` type has `P_reference` and `reference_source` fields that don't exist in the v5 `types.py` Signal. The v5 Signal correctly uses only `P_model` and `P_kalshi`. The signal_generator uses `P_model` directly (Pattern 1: P_model is Sole Authority).

### 3. Missing Kalshi Endpoints

| Method Phase 4 Calls | Exists in `kalshi.py`? | Notes |
|----------------------|----------------------|-------|
| `submit_order(order)` | YES | `POST /trade-api/v2/orders` |
| `cancel_order(order_id)` | YES | `DELETE /trade-api/v2/orders/{id}` |
| `get_balance()` | YES | `GET /trade-api/v2/portfolio/balance` |
| `get_positions()` | YES | `GET /trade-api/v2/portfolio/positions` |
| `get_orderbook(ticker)` | YES | `GET /trade-api/v2/markets/{ticker}/orderbook` |
| `get_market(ticker)` | YES | For settlement verification [C7 fix — used by poll_kalshi_settlement] |
| `get_order(order_id)` | **MISSING** | Needed to check fill status of pending orders |
| `get_fills(ticker)` | **MISSING** | Needed to confirm partial fills |

**Action required:** Add `get_order(order_id)` and `get_fills(ticker)` to `KalshiClient` in Sprint 4c or 4d:
- `GET /trade-api/v2/portfolio/orders/{order_id}` → order status
- `GET /trade-api/v2/portfolio/fills?ticker={ticker}` → fill confirmations

### 4. State Ownership

| State | Owner | Synchronized How |
|-------|-------|-----------------|
| `open_positions` | `PositionTracker` (in-memory, per container) | Single-threaded asyncio, no sync needed |
| `bankroll` | `bankroll` DB table (shared across containers) | DB reads/writes via `ExposureManager` |
| `pending_orders` | `OrderManager` (in-memory, per container) | Single-threaded asyncio |
| `p_kalshi` | `LiveMatchModel.p_kalshi` (per container) | Written by `kalshi_ob_sync`, read by `execution_loop` — asyncio safety |
| `exposure_reservation` | DB table (shared) | Atomic INSERT for reserve, conditional UPDATE for confirm/release |
| `ticker_muted` | `OrderManager` (in-memory, per container) | [C6 fix] Resets on container restart |
| `entries_halted` | `OrderManager` (in-memory, per container) | [C6 fix] Resets on container restart |

**Shared mutable state between containers:** Only `bankroll` and `exposure_reservation` tables. These are synchronized via PostgreSQL's ACID guarantees. The `reserve_exposure()` function uses a conditional INSERT that respects the total exposure cap.

**Risk:** Two containers could both read the same bankroll and both decide they have enough room, then both reserve. The total could exceed 20%. **Mitigation:** Use `SELECT ... FOR UPDATE` or check total reserved in the same transaction as the INSERT. Sprint 4c's `reserve_exposure` should use a single query:
```sql
INSERT INTO exposure_reservation (match_id, ticker, reserved_amount, status)
SELECT $1, $2, $3, 'RESERVED'
WHERE (SELECT COALESCE(SUM(reserved_amount), 0) FROM exposure_reservation WHERE status IN ('RESERVED', 'CONFIRMED')) + $3
  <= (SELECT balance * 0.20 FROM bankroll WHERE mode = $4)
RETURNING id
```

### 5. Error Propagation: Kalshi 429 (Rate Limit)

Trace through the system:

1. **`KalshiClient.submit_order()`** → `httpx.HTTPStatusError` with status 429
2. **`OrderManager.place_order()`** catches the exception → logs `structlog.warning("kalshi_rate_limited", ...)` → returns `FillResult(status="rejected", quantity=0)`
3. **`execution_loop`** receives `fill.quantity == 0` → calls `exposure.release_exposure(reservation_id)` → reservation released
4. **Position state:** No position was created. `PositionTracker` is unchanged.
5. **Bankroll:** Unchanged (no fill cost deducted).
6. **Next tick:** Signal may fire again. If rate limiting persists, orders continue to be rejected — no state corruption.

**Consistency verified:** The reserve→release pattern ensures no phantom exposure. The position is only created after a successful fill.

### 6. Replay Compatibility

| Sprint | Can integration test run on recorded data? | What's needed |
|--------|------------------------------------------|---------------|
| 4a | YES | Recorded TickPayloads + Kalshi prices from JSONL |
| 4b | YES | Hardcoded payloads (no external data needed) |
| 4c | YES | Test DB (local PostgreSQL), no live APIs |
| 4d | YES | ReplayServer for full match, test DB |
| 5a | NEEDS MOCKS | Needs mocked `map_all_sources` (live API calls in cross_source_mapper). Use fixture JSON responses. |
| 5b | NEEDS MOCKS | Needs mocked Docker API. Use `unittest.mock` for `docker.DockerClient`. |
| 6a | YES | Test DB with seeded data, no live services |
| 6b | NEEDS MOCKS | Needs test Redis instance. Can use `fakeredis` or local Redis. |

**All sprints can run without live APIs.** Sprints 5a, 5b, and 6b need mock/fake services but no real external API calls.
