Implement Sprint 4a: Signal Generation + Kelly Sizing for the Phase 4 execution engine.

This sprint builds the pure-math layer that takes a TickPayload (produced by Phase 3 every second) and Kalshi orderbook prices, and outputs a list of sized Signal objects. No database, no network, no API calls — pure functions only.

Read these files before writing any code:
- `src/common/types.py` — understand TickPayload, MarketProbs, Signal, FillResult
- `src/common/logging.py` — logging uses `get_logger(name)` returning structlog
- `src/engine/tick_loop.py` — understand how TickPayload is assembled (lines 79-102)
- `.claude/rules/patterns.md` — Pattern 1 (P_model sole authority), Pattern 2 (MarketProbs decomposition), Pattern 5 (SurpriseScore Kelly with kelly_surprise_bonus = 0.25)
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 4a section — this is your spec

Do NOT read or modify any files in `src/engine/`, `src/clients/`, `src/calibration/`, or `src/math/`.

## Step 1: Add types to `src/common/types.py`

At the end of the file (after the `SystemAlertMessage` class), add:

```python
# ─── Phase 4 Enums ────────────────────────────────────────────
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
```

## Step 2: Create `src/execution/config.py`

This is the single source of truth for all Phase 4 constants. Every execution module imports `CONFIG` from here. No hardcoded magic numbers anywhere else.

Write this file exactly:

```python
"""Phase 4 execution constants.

Import CONFIG from here — no magic numbers in function bodies.
All values from v5 architecture §8.2, §8.4, §8.5, §13.4 and patterns.md Pattern 5.
"""

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
    ALPHA_SURPRISE: float = 0.25       # surprise bonus — matches patterns.md kelly_surprise_bonus
                                       # TODO: recalibrate from 307-match backtest

    # Risk caps (§13.4)
    PER_ORDER_CAP: float = 50.0        # max dollars per order
    PER_MATCH_CAP_FRAC: float = 0.10   # max fraction of bankroll per match
    TOTAL_EXPOSURE_CAP_FRAC: float = 0.20  # max fraction across all positions

    # Order management (§8.5)
    MAX_ORDER_LIFETIME_S: float = 30.0 # cancel unfilled orders after this
    REPRICE_THRESHOLD: float = 0.02    # cancel+repost if |P_model_now - P_order| > this

    # Position management (§13.4)
    MIN_HOLD_TICKS: int = 150          # ~150 seconds at 1Hz tick rate
    COOLDOWN_AFTER_EXIT: int = 300     # ~5 minutes at 1Hz tick rate
    EKF_DIVERGENCE_THRESHOLD: float = 1.5  # exit if P_H or P_A exceeds this
    EXPIRY_EVAL_MINUTE: float = 85.0   # begin expiry evaluation after this match minute


CONFIG = ExecutionConfig()
```

## Step 3: Create `src/execution/signal_generator.py`

This module detects edges between P_model and P_kalshi and returns Signal objects for markets where the edge exceeds a dynamic threshold.

CRITICAL implementation details:
- `generate_signals` must return `[]` immediately if `payload.order_allowed` is False
- `MarketProbs` fields `over_25` and `btts_yes` can be `None` — check before using
- `sigma_MC` is a `MarketProbs` object, NOT a single float — extract per-market with `getattr`
- `mu_market` is team-specific: home_win uses mu_H, away_win uses mu_A, draw/over/btts uses max(mu_H, mu_A). DO NOT use `mu_H + mu_A`.
- `ekf_P` is team-specific: home_win uses ekf_P_H, away_win uses ekf_P_A, draw/over/btts uses max()
- Signal objects are Pydantic models — create new instances, don't mutate

Functions to implement (see sprint decomposition doc for full logic):

1. `compute_edge(p_model: float, p_kalshi: float) -> tuple[str, float]` — computes EV in both directions, returns best direction + EV or ("HOLD", 0.0)

2. `_get_market_mu(market_type: str, mu_H: float, mu_A: float) -> float` — maps market type to team-specific remaining goals

3. `_get_market_ekf_P(market_type: str, ekf_P_H: float, ekf_P_A: float) -> float` — maps market type to team-specific EKF uncertainty

4. `compute_dynamic_threshold(p_hat: float, sigma_mc: float, ekf_P: float, mu_market: float) -> float` — the formula from v5 §8.2:
   ```
   sigma_mc_sq = p_hat * (1 - p_hat) / CONFIG.N_MC
   sigma_model_sq = ekf_P * (p_hat * (1 - p_hat) * mu_market) ** 2
   sigma_p = sqrt(sigma_mc_sq + sigma_model_sq)
   theta = CONFIG.C_SPREAD + CONFIG.C_SLIPPAGE + CONFIG.Z_ALPHA * sigma_p
   ```
   Edge case: if p_hat <= 0 or p_hat >= 1, return 1.0

5. `generate_signals(payload: TickPayload, p_kalshi: dict[str, float], kalshi_tickers: dict[str, str], open_positions: dict[str, Any] | None = None) -> list[Signal]` — iterates the 5 market types, skips markets with existing positions, computes edge and threshold, returns Signals with kelly fields set to 0 (filled by kelly_sizer)

## Step 4: Create `src/execution/kelly_sizer.py`

This module takes a Signal and sizes the position using the Kelly criterion with Baker-McHale shrinkage and SurpriseScore adjustment.

Functions to implement:

1. `compute_kelly_fraction(p_model: float, p_kalshi: float) -> float`
   - `b = (1.0 / p_kalshi) - 1.0`
   - `f_star = (b * p_model - (1 - p_model)) / b`
   - Return `max(0.0, f_star)`. Return 0.0 if p_kalshi <= 0 or >= 1.

2. `compute_sigma_p(p_hat: float, ekf_P: float, mu_market: float) -> float` — same σ_p formula as in signal_generator's threshold

3. `apply_baker_mchale_shrinkage(f_star: float, p_model: float, p_kalshi: float, sigma_p: float) -> float`
   - `edge_sq = (p_model - p_kalshi) ** 2`
   - `shrinkage = max(0.0, 1.0 - sigma_p**2 / edge_sq)` if edge_sq > 0, else 0.0
   - Return `f_star * shrinkage`

4. `apply_surprise_multiplier(f_shrunk: float, surprise_score: float) -> float`
   - `kelly_mult = CONFIG.ALPHA_BASE + CONFIG.ALPHA_SURPRISE * surprise_score` (= 0.10 + 0.25 * surprise_score)
   - Return `f_shrunk * kelly_mult`

5. `size_position(signal: Signal, payload: TickPayload, bankroll: float) -> Signal`
   - Steps: compute f_star → compute sigma_p (using market-specific mu and ekf_P via `_get_market_mu`/`_get_market_ekf_P` imported from signal_generator) → shrinkage → surprise multiplier → dollar amount → apply caps (min of PER_ORDER_CAP and PER_MATCH_CAP_FRAC * bankroll) → contracts = int(dollar / P_kalshi)
   - Return new Signal via `signal.model_copy(update={...})` with kelly_fraction, kelly_amount, contracts, surprise_score filled

## Step 5: Create tests

Create `tests/execution/__init__.py` (empty file).

Create `tests/execution/test_signal_generator.py` with these exact tests:

- `test_compute_edge_buy_yes`: `compute_edge(0.62, 0.55)` → `("BUY_YES", 0.07)`
- `test_compute_edge_buy_no`: `compute_edge(0.30, 0.45)` → `("BUY_NO", 0.15)`
- `test_compute_edge_hold`: `compute_edge(0.50, 0.50)` → `("HOLD", 0.0)`
- `test_compute_edge_near_zero`: `compute_edge(0.001, 0.001)` → `("HOLD", ...)`
- `test_dynamic_threshold_degenerate_p`: `compute_dynamic_threshold(0.0, ...)` → `1.0` and `compute_dynamic_threshold(1.0, ...)` → `1.0`
- `test_dynamic_threshold_decreases_late_match`: early match (high ekf_P=0.25, high mu=1.5) produces higher theta than late match (low ekf_P=0.05, low mu=0.3)
- `test_get_market_mu_home_win`: `_get_market_mu("home_win", 1.2, 0.8)` → `1.2`
- `test_get_market_mu_away_win`: `_get_market_mu("away_win", 1.2, 0.8)` → `0.8`
- `test_get_market_mu_draw`: `_get_market_mu("draw", 1.2, 0.8)` → `1.2` (max)
- `test_generate_signals_empty_kalshi`: empty p_kalshi dict → `[]`
- `test_generate_signals_order_not_allowed`: order_allowed=False with large edge → `[]`
- `test_generate_signals_skips_existing_position`: open_positions has home_win → no home_win signal
- `test_generate_signals_filters_below_threshold`: EV below threshold → filtered out
- `test_generate_signals_produces_signal`: p_model=0.70, p_kalshi=0.50 → at least 1 signal with direction=BUY_YES

Create `tests/execution/test_kelly_sizer.py` with these exact tests:

- `test_kelly_fraction_basic`: `compute_kelly_fraction(0.62, 0.55)` → approximately 0.155 (within 0.01)
- `test_kelly_fraction_no_edge`: `compute_kelly_fraction(0.50, 0.55)` → `0.0`
- `test_kelly_fraction_degenerate`: `compute_kelly_fraction(0.62, 0.0)` → `0.0`
- `test_baker_mchale_full_shrink`: sigma_p = edge (0.07) → returns 0.0
- `test_baker_mchale_partial_shrink`: sigma_p=0.04, edge=0.07 → shrinkage ≈ 0.673, result ≈ 0.15 * 0.673
- `test_surprise_multiplier_neutral`: `apply_surprise_multiplier(1.0, 0.0)` → `0.10`
- `test_surprise_multiplier_high`: `apply_surprise_multiplier(1.0, 0.70)` → `0.275` (= 0.10 + 0.25*0.70)
- `test_size_position_hard_caps`: bankroll=1000, extreme edge → kelly_amount capped at $50 (PER_ORDER_CAP)
- `test_size_position_per_match_cap`: bankroll=100, extreme edge → kelly_amount capped at $10 (10% of bankroll)

For all tests that need a TickPayload, create a helper `_make_payload(order_allowed, home_win=0.50, ...)` that constructs a valid TickPayload with sensible defaults for all required fields.

## Step 6: Verify

Run these commands and confirm they pass:
1. `python -m pytest tests/execution/test_signal_generator.py -v`
2. `python -m pytest tests/execution/test_kelly_sizer.py -v`
3. `python -c "from src.execution.config import CONFIG; assert CONFIG.ALPHA_SURPRISE == 0.25"`
4. `python -c "from src.common.types import TradingMode, ExitTrigger; assert len(ExitTrigger) == 6"`
5. Existing tests still pass: `python -m pytest tests/ -v --ignore=tests/execution`

If any test fails, fix the implementation — do not change the test values. The test values are computed from the spec formulas and are correct.
