from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel


# Phase 1 calibration types (used by math core)
@dataclass
class RedCardTransition:
    minute: float
    from_state: int   # 0-3
    to_state: int     # 0-3
    team: str         # "home" | "away"


@dataclass
class IntervalRecord:
    match_id: str
    t_start: float              # interval start (minutes)
    t_end: float                # interval end (minutes)
    state_X: int                # Markov state {0,1,2,3}
    delta_S: int                # score diff (home - away)
    is_halftime: bool           # True for halftime break interval

    # Goal events within this interval
    home_goal_times: list[float] = field(default_factory=list)
    away_goal_times: list[float] = field(default_factory=list)
    goal_delta_before: list[int] = field(default_factory=list)  # ΔS before each goal

    # Red card events within this interval
    red_card_transitions: list[RedCardTransition] = field(default_factory=list)

    # Stoppage
    alpha_1: float = 0.0        # first half stoppage minutes


# §2.1 MarketProbs (shared across all phases)
class MarketProbs(BaseModel):
    home_win: float
    draw: float
    away_win: float
    over_25: float | None = None
    under_25: float | None = None
    btts_yes: float | None = None
    btts_no: float | None = None


# §2.2 ProductionParams (Phase 1 → Phase 2)
class ProductionParams(BaseModel):
    version: int
    league_id: int

    # Step 1.2: Q matrix (4×4 red card transitions)
    Q: list[list[float]]

    # Step 1.4: time basis coefficients (6 × 15-min bins)
    b: list[float]  # len=6

    # Step 1.4: score-differential effects
    gamma_H: float
    gamma_A: float
    delta_H: float
    delta_A: float

    # v5 asymmetric score-state effects (defaults = None → use symmetric delta_H/delta_A)
    delta_H_pos: list[float] | None = None   # shape [5], home when leading
    delta_H_neg: list[float] | None = None   # shape [5], home when trailing
    delta_A_pos: list[float] | None = None   # shape [5], away when trailing
    delta_A_neg: list[float] | None = None   # shape [5], away when leading
    # v5 stoppage time multipliers (default 0.0 = no stoppage adjustment)
    eta_H: float = 0.0     # 1st half stoppage, home
    eta_A: float = 0.0     # 1st half stoppage, away
    eta_H2: float = 0.0    # 2nd half stoppage, home
    eta_A2: float = 0.0    # 2nd half stoppage, away
    # v5 EKF process noise (default 0.01 = small drift)
    sigma_omega_sq: float = 0.01
    # v5 full gamma arrays (defaults = None → use scalar gamma_H/gamma_A)
    gamma_H_full: list[float] | None = None  # shape [4]
    gamma_A_full: list[float] | None = None  # shape [4]
    # v5 full delta arrays (defaults = None → use scalar delta_H/delta_A)
    delta_H_full: list[float] | None = None  # shape [5]
    delta_A_full: list[float] | None = None  # shape [5]

    # Step 1.4: optimization hyperparameter
    sigma_a: float

    # Step 1.3: XGBoost model (binary blob in DB)
    xgb_model_blob: bytes | None  # xgb.save_raw() → BYTEA, None = MLE fallback
    feature_mask: list[str] | None  # feature names used by this model

    # metadata
    trained_at: datetime
    match_count: int  # matches used in training
    brier_score: float  # validation Brier Score
    is_active: bool


# §2.3 Phase2Result (Phase 2 → Phase 3)
class Phase2Result(BaseModel):
    match_id: str
    league_id: int

    # backsolve results
    a_H: float  # home log-intensity parameter
    a_A: float  # away log-intensity parameter
    mu_H: float  # expected home goals = exp(a_H) * C_time
    mu_A: float

    # sanity check
    C_time: float  # time normalization constant
    verdict: str  # "GO" | "SKIP"
    skip_reason: str | None

    # version pinning — this match uses this param version for its entire lifetime
    param_version: int

    # match info
    home_team: str
    away_team: str
    kickoff_utc: datetime

    # Kalshi market tickers
    kalshi_tickers: dict[str, str]  # {"home_win": "KXEPLGAME-...", "draw": "...", ...}

    # pre-match odds (sanity check + Phase 4 reference)
    market_implied: MarketProbs | None  # Bet365/Betfair opening, vig-removed
    prediction_method: str  # "xgboost" | "form_mle" | "league_mle"

    # v5: initial EKF uncertainty — larger P0 = more aggressive early EKF updates
    ekf_P0: float = 0.25  # default = sigma_a² = 0.5² = 0.25



# §2.4 TickPayload (Phase 3 → Phase 4)
class TickPayload(BaseModel):
    match_id: str
    t: float  # effective play time (minutes), halftime excluded
    engine_phase: str  # FIRST_HALF | HALFTIME | SECOND_HALF | FINISHED

    # MMPP model pricing
    P_model: MarketProbs  # MMPP MC output
    sigma_MC: MarketProbs  # per-market MC standard error

    # match state
    score: tuple[int, int]  # (home, away)
    X: int  # Markov state: 0=11v11, 1=10v11, 2=11v10, 3=10v10
    delta_S: int  # score diff (home − away)
    mu_H: float  # remaining expected home goals
    mu_A: float

    # strength updater state
    a_H_current: float  # current (possibly updated) home log-intensity
    a_A_current: float  # current (possibly updated) away log-intensity
    last_goal_type: str = "NEUTRAL"  # SURPRISE | EXPECTED | NEUTRAL

    # v5 EKF state
    ekf_P_H: float = 0.0        # EKF uncertainty for home
    ekf_P_A: float = 0.0        # EKF uncertainty for away
    # v5 Layer 2 state
    hmm_state: int = 0           # HMM state: -1, 0, +1
    dom_index: float = 0.0       # DomIndex fallback value
    # v5 SurpriseScore
    surprise_score: float = 0.0  # continuous [0, 1], replaces categorical last_goal_type

    # trading permission (Phase 3 decides, Phase 4 respects)
    order_allowed: bool
    cooldown: bool  # post-event cooldown active
    ob_freeze: bool  # odds anomaly detected
    event_state: str  # IDLE | PRELIMINARY | CONFIRMED


# §2.5 Signal (Phase 4 internal)
class Signal(BaseModel):
    match_id: str
    ticker: str  # Kalshi ticker
    market_type: str  # "home_win", "draw", "away_win", etc.
    direction: str  # "BUY_YES" | "BUY_NO" | "HOLD"

    P_kalshi: float  # VWAP effective price from Kalshi orderbook
    P_model: float  # MMPP model probability

    EV: float  # expected value (cents)

    kelly_fraction: float  # raw Kelly fraction
    kelly_amount: float  # dollar amount after risk limits
    contracts: int  # final contract count
    surprise_score: float = 0.0


# §2.6 FillResult (Phase 4 internal)
class FillResult(BaseModel):
    order_id: str
    ticker: str
    direction: str
    quantity: int  # filled quantity
    price: float  # fill price
    status: str  # "full" | "partial" | "rejected" | "paper"
    fill_cost: float  # quantity × price
    timestamp: datetime


# §2.7 Redis Messages

# Channel: "tick:{match_id}"
class TickMessage(BaseModel):
    type: str = "tick"
    match_id: str
    t: float
    engine_phase: str
    P_model: MarketProbs  # MMPP output
    sigma_MC: MarketProbs
    order_allowed: bool
    cooldown: bool
    ob_freeze: bool
    event_state: str
    mu_H: float
    mu_A: float
    score: tuple[int, int]
    ekf_P_H: float = 0.0
    ekf_P_A: float = 0.0
    hmm_state: int = 0
    surprise_score: float = 0.0


# Channel: "event:{match_id}"
class EventMessage(BaseModel):
    type: str = "event"
    match_id: str
    event_type: str  # goal_confirmed, red_card, period_change, var_review, ...
    t: float
    payload: dict  # event-specific data


# Channel: "signal:{match_id}"
class SignalMessage(BaseModel):
    type: str = "signal"
    match_id: str
    ticker: str
    direction: str
    EV: float
    P_kalshi: float
    kelly_fraction: float
    fill_qty: int
    fill_price: float
    timestamp: float


# Channel: "position_update"
class PositionUpdateMessage(BaseModel):
    type: str  # "new_fill" | "exit" | "settled"
    match_id: str
    ticker: str
    direction: str
    quantity: int
    price: float


# Channel: "system_alert"
class SystemAlertMessage(BaseModel):
    type: str = "alert"
    severity: str  # "critical" | "warning" | "info"
    title: str
    details: dict[str, str]
    timestamp: float
