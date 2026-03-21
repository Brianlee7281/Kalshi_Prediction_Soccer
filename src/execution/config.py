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
    Z_ALPHA: float = 1.0               # ~84% one-tailed confidence
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
    MIN_HOLD_TICKS: int = 30           # ~30 seconds at 1Hz tick rate
    COOLDOWN_AFTER_EXIT: int = 0       # no cooldown — allow immediate re-entry after exit
    EKF_DIVERGENCE_THRESHOLD: float = 1.5  # exit if P_H or P_A exceeds this
    EXPIRY_EVAL_MINUTE: float = 85.0   # begin expiry evaluation after this match minute

    # Paper fill simulation
    PAPER_HALF_SPREAD: float = 0.005   # half of C_SPREAD; entry/exit fill price penalty
    PAPER_TYPICAL_DEPTH: int = 30      # contracts at best level; partial fills above this


CONFIG = ExecutionConfig()
