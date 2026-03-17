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


# §2.4 BookmakerState
class BookmakerState(BaseModel):
    name: str  # "Betfair Exchange", "Bet365", "1xbet", etc.
    implied: MarketProbs
    last_update: datetime
    is_stale: bool  # True if last_update > 10s ago


# §2.4 OddsConsensusResult
class OddsConsensusResult(BaseModel):
    P_consensus: MarketProbs  # weighted reference price (Betfair-heavy)
    confidence: str  # "HIGH" (2+ agree) | "LOW" (1 only) | "NONE" (no fresh data)
    n_fresh_sources: int  # how many bookmakers have fresh data
    bookmakers: list[BookmakerState]  # individual bookmaker states
    event_detected: bool  # True if 2+ sources moved >3% in same direction within 5s


# §2.4 TickPayload (Phase 3 → Phase 4)
class TickPayload(BaseModel):
    match_id: str
    t: float  # effective play time (minutes), halftime excluded
    engine_phase: str  # FIRST_HALF | HALFTIME | SECOND_HALF | FINISHED

    # TIER 1: Odds consensus (primary reference for trading)
    odds_consensus: OddsConsensusResult | None

    # TIER 2: MMPP model pricing (fallback when consensus unavailable)
    P_model: MarketProbs  # MMPP MC output
    sigma_MC: MarketProbs  # per-market MC standard error

    # Effective reference price (Phase 4 uses this for edge detection)
    # = odds_consensus.P_consensus if confidence HIGH, else P_model
    P_reference: MarketProbs
    reference_source: str  # "consensus" | "model"

    # match state
    score: tuple[int, int]  # (home, away)
    X: int  # Markov state: 0=11v11, 1=10v11, 2=11v10, 3=10v10
    delta_S: int  # score diff (home − away)
    mu_H: float  # remaining expected home goals
    mu_A: float

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

    P_reference: float  # reference probability for this market (consensus or model)
    reference_source: str  # "consensus" | "model"
    P_kalshi: float  # VWAP effective price from Kalshi orderbook
    P_model: float  # MMPP model probability (always computed, for logging)

    EV: float  # expected value (cents)
    consensus_confidence: str  # HIGH | LOW | NONE

    kelly_fraction: float  # raw Kelly fraction
    kelly_amount: float  # dollar amount after risk limits
    contracts: int  # final contract count


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
    P_reference: MarketProbs  # consensus or model (whatever Phase 3 chose)
    reference_source: str  # "consensus" | "model"
    P_model: MarketProbs  # MMPP output (always present)
    sigma_MC: MarketProbs
    consensus_confidence: str  # HIGH | LOW | NONE
    order_allowed: bool
    cooldown: bool
    ob_freeze: bool
    event_state: str
    mu_H: float
    mu_A: float
    score: tuple[int, int]


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
    P_reference: float
    P_kalshi: float
    reference_source: str
    consensus_confidence: str
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
