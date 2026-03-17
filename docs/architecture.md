# MMPP Soccer Live Trading System — Architecture

## 1. System Overview

Real-time soccer prediction market trading system. An MMPP (Marked Modulated Poisson Process) model computes true match probabilities every second during live games, detects edge against Kalshi prediction market prices, and executes trades automatically.

### Data Flow

```
Phase 1 (offline, weekly)
  Historical data → Train MMPP params (b, γ, δ, Q) → production_params DB
      ↓
Phase 2 (pre-match, kickoff −65min)
  Load params + team features + Bet365/Betfair odds → Backsolve a_H, a_A → Phase2Result
      ↓
Phase 3 (live, 90min, every 1s)
  ┌─ Tier 1: Odds-API WS → 5 bookmakers → OddsConsensus (PRIMARY reference)
  ├─ Tier 2: Goalserve poll → MMPP model → P_model (FALLBACK reference)
  └─ Tier 3: Goalserve poll → event context (goal/card/period for logging)
      ↓
  P_reference = consensus (if HIGH confidence) else P_model
      ↓                                                      asyncio.Queue
Phase 4 (live, 90min, every 1s)
  TickPayload → Edge = P_reference − P_kalshi → Kelly → Kalshi order → Settlement
```

### Stack

- **Language:** Python 3.11, TypeScript (dashboard)
- **Database:** PostgreSQL 16, Redis
- **Infra:** Docker (1 container per match), Docker Compose
- **Compute:** Numba JIT (MC simulation, ~7.6ms for 50K paths)
- **Execution:** Kalshi API (REST + WebSocket)
- **Live Data:** Goalserve (match events), Odds-API.io (bookmaker odds)
- **Dashboard:** FastAPI + React
- **Monitoring:** Prometheus, Grafana, Slack/SMS alerts

### Tradable Leagues (8)

| League | Goalserve ID | Odds-API.io Slug | Kalshi GAME Series |
|--------|:---:|---|---|
| EPL | 1204 | `england-premier-league` | `KXEPLGAME` |
| La Liga | 1399 | `spain-laliga` | `KXLALIGAGAME` |
| Serie A | 1269 | `italy-serie-a` | `KXSERIEAGAME` |
| Bundesliga | 1229 | `germany-bundesliga` | `KXBUNDESLIGAGAME` |
| Ligue 1 | 1221 | `france-ligue-1` | `KXLIGUE1GAME` |
| MLS | 1440 | `usa-mls` | `KXMLSGAME` |
| Brasileirão | 1141 | `brazil-brasileiro-serie-a` | `KXBRASILEIROGAME` |
| Argentina | 1081 | `argentina-liga-profesional` | `KXARGPREMDIVGAME` |

---

## 2. Type Contracts

All inter-phase data is defined as Pydantic models. These are the single source of truth —
code must use these exact field names and types.

### 2.1 MarketProbs (shared across all phases)

```python
class MarketProbs(BaseModel):
    home_win: float
    draw: float
    away_win: float
    over_25: float | None = None
    under_25: float | None = None
    btts_yes: float | None = None
    btts_no: float | None = None
```

### 2.2 ProductionParams (Phase 1 → Phase 2)

Stored in `production_params` table. One row per league per version.

```python
class ProductionParams(BaseModel):
    version: int                        # auto-increment, used for version pinning
    league_id: int

    # Step 1.2: Q matrix (4×4 red card transitions)
    Q: list[list[float]]

    # Step 1.4: time basis coefficients (6 × 15-min bins)
    b: list[float]                      # len=6

    # Step 1.4: score-differential effects
    gamma_H: float
    gamma_A: float
    delta_H: float
    delta_A: float

    # Step 1.4: optimization hyperparameter
    sigma_a: float

    # Step 1.3: XGBoost model (binary blob in DB)
    xgb_model_blob: bytes | None        # xgb.save_raw() → BYTEA, None = MLE fallback
    feature_mask: list[str] | None      # feature names used by this model

    # metadata
    trained_at: datetime
    match_count: int                    # matches used in training
    brier_score: float                  # validation Brier Score
    is_active: bool
```

### 2.3 Phase2Result (Phase 2 → Phase 3)

Passed to match container via environment variables.

```python
class Phase2Result(BaseModel):
    match_id: str
    league_id: int

    # backsolve results
    a_H: float                          # home log-intensity parameter
    a_A: float                          # away log-intensity parameter
    mu_H: float                         # expected home goals = exp(a_H) * C_time
    mu_A: float

    # sanity check
    C_time: float                       # time normalization constant
    verdict: str                        # "GO" | "SKIP"
    skip_reason: str | None

    # version pinning — this match uses this param version for its entire lifetime
    param_version: int

    # match info
    home_team: str
    away_team: str
    kickoff_utc: datetime

    # Kalshi market tickers
    kalshi_tickers: dict[str, str]      # {"home_win": "KXEPLGAME-...", "draw": "...", ...}

    # pre-match odds (sanity check + Phase 4 reference)
    market_implied: MarketProbs | None  # Bet365/Betfair opening, vig-removed
    prediction_method: str              # "xgboost" | "form_mle" | "league_mle"
```

### 2.4 TickPayload (Phase 3 → Phase 4)

Passed via `asyncio.Queue(maxsize=1)`. Also published to Redis for dashboard.

```python
class BookmakerState(BaseModel):
    name: str                           # "Betfair Exchange", "Bet365", "1xbet", etc.
    implied: MarketProbs
    last_update: datetime
    is_stale: bool                      # True if last_update > 10s ago

class OddsConsensusResult(BaseModel):
    P_consensus: MarketProbs            # weighted reference price (Betfair-heavy)
    confidence: str                     # "HIGH" (2+ agree) | "LOW" (1 only) | "NONE" (no fresh data)
    n_fresh_sources: int                # how many bookmakers have fresh data
    bookmakers: list[BookmakerState]    # individual bookmaker states
    event_detected: bool                # True if 2+ sources moved >3% in same direction within 5s

class TickPayload(BaseModel):
    match_id: str
    t: float                            # effective play time (minutes), halftime excluded
    engine_phase: str                   # FIRST_HALF | HALFTIME | SECOND_HALF | FINISHED

    # TIER 1: Odds consensus (primary reference for trading)
    odds_consensus: OddsConsensusResult | None

    # TIER 2: MMPP model pricing (fallback when consensus unavailable)
    P_model: MarketProbs                # MMPP MC output
    sigma_MC: MarketProbs               # per-market MC standard error

    # Effective reference price (Phase 4 uses this for edge detection)
    # = odds_consensus.P_consensus if confidence HIGH, else P_model
    P_reference: MarketProbs
    reference_source: str               # "consensus" | "model"

    # match state
    score: tuple[int, int]              # (home, away)
    X: int                              # Markov state: 0=11v11, 1=10v11, 2=11v10, 3=10v10
    delta_S: int                        # score diff (home − away)
    mu_H: float                         # remaining expected home goals
    mu_A: float

    # trading permission (Phase 3 decides, Phase 4 respects)
    order_allowed: bool
    cooldown: bool                      # post-event cooldown active
    ob_freeze: bool                     # odds anomaly detected
    event_state: str                    # IDLE | PRELIMINARY | CONFIRMED
```

### 2.5 Signal (Phase 4 internal)

```python
class Signal(BaseModel):
    match_id: str
    ticker: str                         # Kalshi ticker
    market_type: str                    # "home_win", "draw", "away_win", etc.
    direction: str                      # "BUY_YES" | "BUY_NO" | "HOLD"

    P_reference: float                  # reference probability for this market (consensus or model)
    reference_source: str               # "consensus" | "model"
    P_kalshi: float                     # VWAP effective price from Kalshi orderbook
    P_model: float                      # MMPP model probability (always computed, for logging)

    EV: float                           # expected value (cents)
    consensus_confidence: str           # HIGH | LOW | NONE

    kelly_fraction: float               # raw Kelly fraction
    kelly_amount: float                 # dollar amount after risk limits
    contracts: int                      # final contract count
```

### 2.6 FillResult (Phase 4 internal)

```python
class FillResult(BaseModel):
    order_id: str
    ticker: str
    direction: str
    quantity: int                        # filled quantity
    price: float                         # fill price
    status: str                          # "full" | "partial" | "rejected" | "paper"
    fill_cost: float                     # quantity × price
    timestamp: datetime
```

### 2.7 Redis Messages

All messages are JSON. Key convention: `sigma_MC` (not Greek σ_MC).

```python
# Channel: "tick:{match_id}"
class TickMessage(BaseModel):
    type: str = "tick"
    match_id: str
    t: float
    engine_phase: str
    P_reference: MarketProbs            # consensus or model (whatever Phase 3 chose)
    reference_source: str               # "consensus" | "model"
    P_model: MarketProbs                # MMPP output (always present)
    sigma_MC: MarketProbs
    consensus_confidence: str           # HIGH | LOW | NONE
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
    event_type: str                     # goal_confirmed, red_card, period_change, var_review, ...
    t: float
    payload: dict                       # event-specific data

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
    type: str                           # "new_fill" | "exit" | "settled"
    match_id: str
    ticker: str
    direction: str
    quantity: int
    price: float

# Channel: "system_alert"
class SystemAlertMessage(BaseModel):
    type: str = "alert"
    severity: str                       # "critical" | "warning" | "info"
    title: str
    details: dict[str, str]
    timestamp: float
```

---

## 3. Phase Design

### 3.1 Phase 1: Offline Calibration

**Role:** Train league-specific MMPP parameters. Store in `production_params` table. That's it.

**Does:** Load historical commentaries (Goalserve JSON, `data/commentaries/`) + odds (football-data.co.uk CSV, `data/odds_historical/`) → interval segmentation → Q matrix estimation → XGBoost prior training (with Pinnacle/Bet365 odds features) → joint NLL optimization → walk-forward CV validation → Go/No-Go → DB save.

**Does NOT:** Handle live data. Make per-match predictions. Touch execution or dashboard.

**Trigger:** Manual or weekly CRON. Publishes `PARAMS_UPDATED` to Redis. Running containers IGNORE this — they keep their pinned version until match ends. New params apply to the next match.

**Math core reuse (copy as-is):**
- `mc_core.py` — Numba JIT MC simulation
- `joint_nll.py` — Adam→L-BFGS NLL optimizer
- `markov_chain.py` — Q matrix estimation
- `mu_calculator.py` — remaining μ computation
- `stoppage.py` — stoppage time management (Phase A/C)

**Key design decisions:**
- XGBoost model stored as BYTEA blob in DB (container-independent, no file mount needed)
- football-data.co.uk odds features: Europe 5 leagues have opening (PSH) + closing (PSCH). Americas 3 leagues have closing (PSCH) only. Americas use closing as feature — acceptable for Phase 2 sanity check; true edge comes from in-play Phase 3.
- 3-tier fallback for a_H/a_A prediction: XGBoost (if model blob exists) → team form MLE (recent 5 matches) → league average MLE. XGBoost is the default, not the exception.

### 3.2 Phase 2: Pre-Match Initialization

**Role:** Produce `Phase2Result` for a specific match, 65 minutes before kickoff.

**Does:** Load production_params (version pinned) → collect team features (rolling stats from commentaries DB + current season odds from Odds-API.io Bet365/Betfair) → XGBoost predict a_H, a_A → compute mu_H, mu_A → sanity check (compare model P vs market implied, reject if too far) → Kalshi ticker matching → output Phase2Result as container env vars.

**Does NOT:** Live pricing. Order execution. Container management.

**Key design decisions:**
- Pre-match odds source: Odds-API.io (`Bet365` + `Betfair Exchange`), not Pinnacle (not available on odds-api.io). football-data.co.uk current season CSV as supplementary source (~10 day delay).
- Sanity check compares model 1x2 probs against market_implied. If max deviation > threshold → verdict = SKIP.
- Kalshi ticker matching: team name alias table + accent stripping + per-word matching. Time window: `close >= kickoff` (Kalshi close_time is weeks out, not same-day).

### 3.3 Phase 3: Live Pricing & Signal Aggregation

**Role:** Produce P_reference every second by combining odds consensus (primary) with MMPP model pricing (fallback). Emit TickPayload to Phase 4. Publish to Redis.

**Signal hierarchy (3 tiers):**

```
TIER 1 — ODDS CONSENSUS (primary, <2s latency)
  Odds-API.io WS → 5 bookmakers (Betfair Exchange, Bet365, 1xbet, Sbobet, DraftKings)
  → OddsConsensus aggregates: weighted median, Betfair-heavy
  → If 2+ bookmakers move >3% same direction within 5s → HIGH confidence
  → Also serves as event detection (faster than Goalserve)

TIER 2 — MMPP MODEL (fallback, 3-5s latency)
  Goalserve poll → model state update → MC pricing → P_model
  → Used when odds consensus is NONE (all sources stale)
  → Used as sanity check against consensus
  → Used for markets where bookmakers don't provide odds

TIER 3 — GOALSERVE CONTEXT (supplementary, 3s latency)
  → What happened: goal, red card, period change, VAR
  → Model state maintenance: score, X, delta_S, time
  → Dashboard logging and post-match analysis
```

**Why Betfair Exchange is the primary reference (academic basis):**
Croxson & Reade (2014) found Betfair EPL in-play markets to be semi-strong efficient — prices update swiftly and fully after goals. Angelini, De Angelis & Singleton (2021) confirmed this on 1,004 EPL matches but found systematic mispricing for 5+ minutes after surprise goals (favourite bias). Kalshi, as a newer prediction market with lower volume (~$500K vs Betfair's millions per match), is expected to be less efficient. The edge is the gap between these two markets.

**Coroutine structure:**
```python
async def run_engine(model):
    await asyncio.gather(
        tick_loop(model),               # every 1s: aggregate → TickPayload
        odds_api_listener(model),       # WS: 5 bookmaker live odds → OddsConsensus
        goalserve_poller(model),        # every 3s: match events → model state
    )
```

**OddsConsensus logic:**
```python
class OddsConsensus:
    sources: dict[str, BookmakerState]  # 5 bookmakers from Odds-API.io WS

    def compute_reference(self) -> OddsConsensusResult:
        fresh = {k: v for k, v in self.sources.items() if not v.is_stale}
        
        if len(fresh) == 0:
            return OddsConsensusResult(confidence="NONE", ...)
        
        # Weighted median — Betfair Exchange gets 2x weight
        weights = {"Betfair Exchange": 2.0, default: 1.0}
        P_consensus = weighted_median(fresh, weights)
        
        # Event detection: 2+ sources moved >3% in same direction within 5s
        movers = [s for s in fresh if abs(s.implied - s.prev_implied) > 0.03]
        event_detected = len(movers) >= 2

        confidence = "HIGH" if len(fresh) >= 2 else "LOW"
        return OddsConsensusResult(P_consensus, confidence, event_detected, ...)
```

**P_reference selection in tick_loop:**
```python
if odds_consensus.confidence == "HIGH":
    P_reference = odds_consensus.P_consensus
    reference_source = "consensus"
elif odds_consensus.confidence == "LOW":
    # Single source — cross-check with model
    if abs(odds_consensus.P_consensus - P_model) < 0.10:
        P_reference = odds_consensus.P_consensus  # source agrees with model
        reference_source = "consensus"
    else:
        P_reference = P_model  # big disagreement, trust model
        reference_source = "model"
else:  # NONE
    P_reference = P_model
    reference_source = "model"
```

**Key design decisions (from 44-issue post-mortem):**

1. **Time management (wall clock − halftime):**
   `model.t = (time.monotonic() - kickoff_wall_clock - halftime_accumulated) / 60`
   HALFTIME entry records `halftime_start`. SECOND_HALF entry adds elapsed to `halftime_accumulated`.
   This gives effective play time, not wall clock time. Prevents intensity model b_{i(t)} basis period misalignment.

2. **Tick scheduling (absolute time, not sleep(1)):**
   `_sleep_until_next_tick()` computes `next_tick = start + tick_count * interval` using monotonic clock.
   If MC takes >1s, the tick is skipped but model.t stays accurate on the next tick. No drift accumulation.

3. **Period change spam prevention:**
   Track `_last_period`. Only emit `period_change` event when period actually changes (e.g., FIRST_HALF → HALFTIME). Max 4 events per match.

4. **Multi-goal same poll:**
   If score diff from one poll is ≥2, create intermediate states and commit goals sequentially. Each goal gets its own state update and cooldown.

5. **Goalserve match_id field:**
   Search `@id`, `@fix_id`, AND `@static_id` when locating a match. Numeric status means "in progress".

6. **Late container join:**
   On first Goalserve poll, if status is numeric (e.g., "33"), set `model.t = 33` and `kickoff_wall_clock = now - (33 * 60)`.

7. **order_allowed conditions:**
   `not model.cooldown AND not model.ob_freeze AND model.event_state == IDLE`
   Phase 3 decides "can trade?", Phase 4 decides "should trade?".

### 3.4 Phase 4: Execution Engine

**Role:** Consume TickPayload. Detect edge between P_reference and Kalshi price. Size positions. Execute orders. Manage exits. Settle.

**Does:** Read TickPayload from asyncio.Queue → for each active market: extract P_reference (consensus or model) → compare vs Kalshi VWAP → if edge > threshold → Kelly sizing (with confidence-adjusted multiplier) → reserve-confirm-release → submit order (paper or live) → monitor exits (6 triggers) → poll Kalshi for settlement → compute P&L → Redis publish.

**Does NOT:** Pricing. Model state. Goalserve. Odds-API (that's Phase 3).

**signal_generator loop:**
```python
async def signal_generator(model, phase4_queue, ob_sync, executor):
    while True:
        payload: TickPayload = await phase4_queue.get()
        if payload.engine_phase == "FINISHED":
            break
        if not payload.order_allowed:
            continue
        for market_type in active_markets:
            p_ref = getattr(payload.P_reference, market_type)
            p_model = getattr(payload.P_model, market_type)
            ticker = model.kalshi_tickers[market_type]

            signal = generate_signal(
                p_ref, p_model, ticker, ob_sync, payload.reference_source,
                payload.odds_consensus.confidence if payload.odds_consensus else "NONE"
            )
            if signal.direction != "HOLD":
                fill = await execute_with_reservation(signal, ...)
                if fill and fill.quantity > 0:
                    model.bankroll -= fill.fill_cost
```

**Key design decisions:**

1. **Confidence-adjusted Kelly:**
   - `consensus_confidence == HIGH` → Kelly multiplier = 0.25 (quarter Kelly)
   - `consensus_confidence == LOW` → Kelly multiplier = 0.10 (tenth Kelly)
   - `reference_source == "model"` → Kelly multiplier = 0.10 + higher EV threshold (3¢ instead of 2¢)

2. **Edge detection uses P_reference, not P_model:**
   `EV = P_reference - P_kalshi` (for BUY_YES direction).
   The MMPP model is not the primary edge source — cross-market inefficiency is.

3. **Model as sanity check:**
   If `abs(P_reference - P_model) > 0.15`, log a warning. This means consensus and model strongly disagree. Don't block the trade (consensus is primary) but flag for review.

4. **Reserve-confirm-release (cross-container race condition fix):**
   `reserve_exposure()` — DB lock <10ms, write to `exposure_reservation` table (RESERVED).
   `execute_order()` — no lock, 1-5s fill wait.
   `confirm_exposure()` or `release_exposure()` — update reservation status.
   CRON: every 5min, release stale RESERVED entries older than 60s.

5. **Bankroll refresh after fill:**
   `model.bankroll -= fill.fill_cost` immediately after fill. Without this, subsequent Kelly uses stale startup bankroll.

6. **6 exit triggers:**
   edge_decay, edge_reversal, position_trim, opportunity_cost, expiry_eval, consensus_divergence.
   `min_hold_ticks = 50` (~150s) and `entry_cooldown_after_exit = 100` (~5min) built in from day one.
   Note: trigger 6 changed from "bet365_divergence" to "consensus_divergence" — exits when odds consensus moves >5% against our position.

7. **Kalshi rejection handling:**
   `market_closed` → mute ticker until reopened (halftime = 15min of suppressed retries without this).
   `insufficient_balance` → halt all new entries.
   `price_out_of_range` → skip this tick.

8. **Staleness gates:**
   Odds consensus: confidence == NONE → use model with higher threshold.
   Kalshi orderbook: >5s since last update → skip order.

9. **Paper/Live separation:**
   `ExecutionRouter` dispatches to `PaperExecutor` or `LiveExecutor` based on config. No if/else in core logic.

### 3.5 Orchestrator

**Role:** Discover matches → Phase 2 → launch containers → lifecycle management → cleanup.

**Does:** Scan Kalshi + Goalserve every 6h for upcoming matches → fixture↔market matching → save to match_schedule → trigger Phase 2 at kickoff −65min → if GO, launch Docker container at kickoff −2min → heartbeat monitoring → post-match cleanup → ARCHIVED.

**Key design decisions:**

1. **Fixture↔market matching:**
   Alias table + accent stripping + per-word matching. Time window: `close >= kickoff`.
   Kalshi ticker pattern: `KX{LEAGUE}{TYPE}-{season}{date}{teams}-{outcome}`.
   Example: `KXSERIEAGAME-26MAR15LAZACM-LAZ`.

2. **Ticker JSON passing:**
   Always `json.loads()`. Never `list()` on a JSON string. (Previous bug: `list('["KX..."]')` → `['[', '"', 'K', ...]`).

3. **Container networking:**
   Fixed network `mmpp-net`. All containers join this network to reach PostgreSQL and Redis.

4. **Stale container cleanup:**
   Check for existing container with same name before create. Remove if exists.

5. **Recovery on restart:**
   `recover_orchestrator_state()` catches SCHEDULED matches past their phase2_trigger time and runs them immediately.

6. **Orchestrator SPOF mitigation:**
   Running containers operate independently — if orchestrator dies for 30min, all running matches continue normally. Systemd auto-restart. Settlement polling is inside the match container, not the orchestrator.

### 3.6 Dashboard

**Role:** REST API + WebSocket to visualize system state.

**Does:** PostgreSQL reads (matches, positions, ticks) via REST → Redis pubsub subscribe for real-time push via WebSocket → React UI rendering.

**Key design decisions:**

1. **Redis subscribe once, not in loop:**
   Subscribe to global channels (`position_update`, `system_alert`) outside the while loop.
   Diff-based subscribe/unsubscribe for match-specific channels (`tick:{id}`, `event:{id}`, `signal:{id}`).

2. **JSON key naming:**
   `sigma_MC` everywhere (not Greek σ). TypeScript cannot use `data.σ_MC`.

3. **Bankroll API:**
   `SELECT * FROM bankroll WHERE mode = $current_trading_mode`. Not hardcoded to 'live'.

4. **Mid-match connect:**
   First connection gets full state via REST. Then WebSocket for incremental updates.

5. **WebSocket reconnection:**
   Exponential backoff (1s base, 30s max, 10 retries). Re-subscribe with current match_ids on reconnect.

### 3.7 Trading Logic & Thesis

#### 3.7.1 How This System Makes Money

The primary edge comes from **cross-market inefficiency between established bookmaker exchanges and Kalshi**, not from our model being smarter than the market.

Betfair Exchange is the world's largest betting exchange (~7M daily trades). Academic research (Croxson & Reade 2014, Angelini et al. 2021) shows it is semi-strong efficient for EPL in-play — prices update swiftly and fully after goals. Kalshi is a newer prediction market with ~100x less volume per match. The hypothesis: **Kalshi prices lag behind Betfair prices after in-game events**, and this lag is exploitable.

**Concrete example:**
```
Minute 34: Score 0-0. Betfair=0.45, Kalshi=0.45. No edge.
Minute 35: Home team scores. Score 1-0.
  t+0.5s: 1xbet moves to 0.60 (fastest bookmaker, received via Odds-API WS)
  t+1.0s: Betfair Exchange moves to 0.62 (efficient market, confirmation)
  t+1.5s: Odds consensus = 0.61 (HIGH confidence, 2+ sources agree)
  t+1.5s: Kalshi still at 0.48 (prediction market, slower participants)
  t+2.0s: System buys YES on Kalshi at 0.48. Edge = 13¢.
  t+15s:  Kalshi participants react. Price moves to 0.58.
  t+60s:  Kalshi settles at 0.60.
  
  If we exit at 0.58: profit = 10¢ per contract.
  If we hold to settlement (home wins): profit = 52¢ per contract.
```

**Key insight:** We don't need our own model to be faster or smarter than Betfair. We need Betfair to be faster than Kalshi. We're riding Betfair's efficiency, not competing with it.

**Preliminary empirical observation (2026-03-15, Verona 0-2 Genoa, Serie A):**
```
Kalshi ticker: KXSERIEAGAME-26MAR15VERGEN-VER (Verona win market)
Kickoff: 11:30 UTC. Goal 1: Vitinha 61' (minute-level, actual time unknown). 439 total trades.

Kalshi trade-level price reaction around estimated goal time (~12:48 UTC ±2min):
  12:47:54  yes=0.17
  12:48:18  yes=0.20
  12:49:35  yes=0.20
  12:51:04  yes=0.19   ← still near pre-goal price
  12:51:25  yes=0.11   ← first significant drop
  12:51:49  yes=0.06   ← cascade
  12:51:53  yes=0.05   ← new equilibrium

HONEST INTERPRETATION:
  - Goalserve says "61 minute" but actual goal time could be 12:46-12:52 UTC (±2-3min uncertainty)
  - If goal was at 12:48 → lag ≈ 3min (very promising)
  - If goal was at 12:51 → lag ≈ 20s (still interesting but much less)
  - We cannot distinguish these cases with minute-level data
  
WHAT WE CAN SAY:
  - Price moved from 0.19→0.05 (14¢ impact) — large enough to be interesting
  - The crash happened in a concentrated 28-second window (12:51:25-12:51:53)
  - 439 trades on this match — sufficient liquidity to trade
  
WHAT WE CANNOT SAY:
  - Exact lag in seconds — requires Sprint 3 recording infrastructure
```
Sprint -1 will measure liquidity and price impact at scale. The precise lag question is deferred to Sprint 3.

**Edge sources (ranked by conviction):**

**Primary — Cross-market lag (high conviction hypothesis):**
Betfair + bookmakers react to events within 1-3s. Kalshi, with fewer automated participants and no market suspension mechanism (unlike Betfair which suspends on goals), may have stale orders for 5-30s after events. The odds consensus from 5 bookmakers gives us a reliable reference price within 2s.

**Secondary — Model-based mispricing (medium conviction):**
Even Betfair shows systematic mispricing in specific conditions (Angelini et al. 2021):
- Favourite bias: expected winners underpriced on exchanges (reverse favourite-longshot bias)
- Surprise goals: longshot team scoring late → market underestimates their chances for 5+ minutes
- Red card overreaction: markets overadjust — MMPP's Markov state captures the true impact
- Stoppage time uncertainty: participants anchor to "90 minutes"

In these cases, MMPP can outperform even Betfair. The model is the edge source when consensus and model diverge AND the model has structural reasons to be right.

**Tertiary — Automation + discipline (low edge but consistent):**
Every second, across 8 leagues, all markets evaluated. No emotion, no fatigue, no missed opportunities.

#### 3.7.2 Entry Conditions (When to Buy)

ALL of the following must be true:

```
1. PERMISSION GATE
   order_allowed = True
   (cooldown = False AND ob_freeze = False AND event_state = IDLE)

2. REFERENCE PRICE AVAILABLE
   P_reference computed from odds consensus (preferred) or MMPP model (fallback)
   
3. EDGE DETECTION
   EV = P_reference - P_kalshi             (BUY_YES direction)
   EV = (1 - P_reference) - (1 - P_kalshi) (BUY_NO direction)
   Best direction = whichever has higher EV
   
   Threshold depends on confidence:
   - consensus HIGH: EV > 2¢ (THETA_ENTRY_HIGH)
   - consensus LOW or model-only: EV > 3¢ (THETA_ENTRY_LOW)

4. SIZING
   Kelly fraction = (P_reference * payout - 1) / (payout - 1)
   
   Confidence-adjusted multiplier:
   - consensus HIGH: Kelly * 0.25 (quarter Kelly)
   - consensus LOW: Kelly * 0.10 (tenth Kelly)
   - model-only: Kelly * 0.10 (tenth Kelly)
   
   Dollar amount = adjusted_kelly * bankroll
   Contracts = floor(dollar_amount / P_kalshi)
   
   Risk limits (ALL must pass):
   - Per-order cap: max $50
   - Per-match cap: max 10% of bankroll
   - Total exposure cap: max 20% of bankroll
   - Liquidity gate: contracts ≤ orderbook depth at target price

5. EXECUTION
   reserve_exposure() → submit limit order → wait fill (5s timeout)
   Full fill → confirm_exposure()
   Partial fill → confirm partial, cancel remainder
   No fill → release_exposure()
```

#### 3.7.3 Exit Conditions (When to Sell)

Positions are evaluated every tick. First trigger that fires wins. `min_hold_ticks = 50` (~150s) prevents churn.

```
TRIGGER 1 — EDGE DECAY
  Current EV (P_reference - P_kalshi) < THETA_EXIT (default: 0.5¢)
  Meaning: Kalshi has converged to reference price. Take profit.

TRIGGER 2 — EDGE REVERSAL
  Direction flipped: was BUY_YES but now P_reference < P_kalshi.
  Meaning: reference moved against us. Cut loss.

TRIGGER 3 — POSITION TRIM
  Current position size > 2× Kelly optimal for current P_reference.
  Meaning: reference moved against us, position is oversized. Reduce.

TRIGGER 4 — OPPORTUNITY COST
  Opposite direction on same market has EV > THETA_ENTRY.
  Meaning: better to flip than hold. Exit and re-enter opposite.

TRIGGER 5 — EXPIRY EVAL
  Match time > 85 minutes. Remaining theta value re-evaluated.
  If expected P&L of holding to settlement < cost of exiting now → exit.

TRIGGER 6 — CONSENSUS DIVERGENCE
  Odds consensus moved >5% against our position direction since entry.
  Meaning: the efficient market disagrees with our position. Respect it.
```

After any exit: `entry_cooldown_after_exit = 100 ticks` (~5 minutes) before re-entering the same market.

#### 3.7.4 Fundamental Risks & Assumptions

**ASSUMPTION 1: Kalshi is less efficient than Betfair/bookmakers.**
This is the core hypothesis. If Kalshi has sophisticated market makers matching Betfair's speed, the cross-market gap doesn't exist. Kalshi's lower volume ($500K vs Betfair's millions per EPL match) and lack of market suspension mechanism support this assumption, but it is unproven.

**ASSUMPTION 2: Odds-API.io delivers bookmaker prices with low enough latency.**
Betfair may react in 1s, but if Odds-API.io adds 5s of relay delay, we see Betfair's price after Kalshi has already moved. Must measure Odds-API.io WS latency during Sprint 7.

**ASSUMPTION 3: Cross-market correlation is high enough.**
Betfair and Kalshi trade on the same underlying event, but participant pools are completely different (UK/EU bettors vs US prediction market traders). Prices should correlate but the speed of convergence is unknown.

**Academic basis (what's proven vs what's assumed):**

Proven (Betfair Exchange, EPL):
- In-play markets are semi-strong efficient overall (Croxson & Reade 2014)
- Favourite bias exists: expected winners underpriced (Angelini et al. 2021)
- Surprise goals cause mispricing for 5+ minutes (Angelini et al. 2021)
- Twitter/social media contains information not yet in live prices after goals (Brown et al. 2017)
- Betfair voids unfairly matched bets after events; Kalshi does not

Assumed (not tested):
- Kalshi soccer market efficiency is lower than Betfair
- Odds-API.io relay latency is <3s for Betfair price changes
- Cross-market convergence window is >5s (enough to execute)

**If assumptions fail:**
- Assumption 1 fails → No cross-market edge. Pivot to model-only edge (MMPP vs Kalshi, lower conviction)
- Assumption 2 fails → Consider Betfair direct API (if US access possible) or faster data providers
- Assumption 3 fails → Markets are too independent; cross-reference is unreliable

**No live money until ALL assumptions are validated in Sprint 7.**

#### 3.7.5 Validation Metrics (Sprint 7 Graduation)

These metrics determine whether the edge exists. If they fail, the system does not go live regardless of code quality.

```
METRIC 1 — CROSS-MARKET LAG (most important)
  Requires Sprint 3 recorded data (second-level timestamps from Odds-API WS + Kalshi WS).
  Sprint -1 CANNOT measure this — minute-level goal times have ±2-3min uncertainty.
  For each goal/red card event, record:
    - t_consensus: when odds consensus first detects the event (2+ bookmakers move)
    - t_kalshi: when Kalshi best_ask moves >3¢ from pre-event price
    - lag = t_kalshi - t_consensus
  If median(lag) < 3s → Kalshi is too fast, cross-market edge doesn't exist → NO-GO
  If median(lag) > 10s → Strong edge → proceed with confidence

METRIC 2 — ODDS-API.IO RELAY LATENCY (new)
  For each bookmaker price change, record:
    - t_change: timestamp in Odds-API.io WS message
    - t_received: when our system receives it
    - relay_latency = t_received - t_change
  If median(relay_latency) > 3s → data source too slow → investigate alternatives

METRIC 3 — EDGE AT EXECUTION
  For each paper trade, record:
    - EV_signal: EV at the moment signal was generated
    - EV_fill: EV recalculated at actual fill price
    - EV_fill should be > 0 for majority of trades
  If mean(EV_fill) ≤ 0 → edge evaporates before execution → NO-GO

METRIC 4 — EDGE REALIZATION
  edge_realization = realized_pnl / theoretical_pnl
  theoretical_pnl = sum of EV_signal for all trades
  If edge_realization < 0.3 → too much slippage → NO-GO

METRIC 5 — CONSENSUS vs MODEL COMPARISON (new)
  Track: how often does consensus-triggered trade win vs model-triggered trade?
  This tells us which edge source actually works.

METRIC 6 — BASIC PROFITABILITY
  Paper P&L > 0 after 50+ trades
  Max drawdown < 15%
  If negative P&L → NO-GO regardless of other metrics
```

**Decision matrix after Sprint 7:**

| Metric 1 (lag) | Metric 3 (EV@fill) | Action |
|:---:|:---:|---|
| lag > 10s | EV > 0 | GO — strong edge, proceed to live |
| lag 3-10s | EV > 0 | CAUTIOUS GO — edge exists but thin, start with minimal capital |
| lag > 10s | EV ≤ 0 | FIX — lag exists but execution is too slow, optimize execution path |
| lag < 3s | any | PIVOT — cross-market edge doesn't exist, investigate model-only strategy |

---

## 4. External Services

All URLs, auth methods, and field names verified on 2026-03-15.

### 4.1 Goalserve

**Purpose:** Match event context (Phase 3 Tier 3) + MMPP model state maintenance + historical commentaries (Phase 1 training data). NOT the primary signal source — odds consensus is.

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /soccernew/home?json=1` | Live scores, all leagues | API key in URL path |
| `GET /commentaries/{league_id}?date={DD.MM.YYYY}&json=1` | Historical match details | API key in URL path |

**Base URL:** `https://www.goalserve.com/getfeed/{API_KEY}/`

**Live score response — key fields:**
- Match identification: search `@id`, `@fix_id`, AND `@static_id` (all three, not just one)
- Score: `match.localteam.@goals`, `match.visitorteam.@goals`
- Status: numeric string = current minute (e.g., "33"), "HT" = halftime, "FT" = finished
- Team names: `match.localteam.@name`, `match.visitorteam.@name`

**Commentaries response — key fields:**
- `summary` — goal scorers, red cards with minutes
- `matchinfo` — stadium, referee, weather
- `stats` — shots, possession, corners

**League coverage verified:**

| League | ID | Commentaries | Notes |
|--------|:---:|:---:|---|
| EPL | 1204 | ✅ 6 seasons | |
| La Liga | 1399 | ✅ 6 seasons | |
| Serie A | 1269 | ✅ 5 seasons | |
| Bundesliga | 1229 | ✅ 5 seasons | Some seasons spotty |
| Ligue 1 | 1221 | ✅ 4 seasons | |
| MLS | 1440 | ✅ 6 seasons | |
| Brasileirão | 1141 | ⚠️ 2024 OK, 2025 errors | Season starts April |
| Argentina | 1081 | ✅ 6 seasons | |

### 4.2 Kalshi

**Purpose:** Prediction market execution (Phase 4).

**Base URL:** `https://api.elections.kalshi.com`

**Auth:** RSA-PSS SHA-256. Three headers per request:
```
KALSHI-ACCESS-KEY: {api_key}
KALSHI-ACCESS-TIMESTAMP: {unix_ms}
KALSHI-ACCESS-SIGNATURE: base64(rsa_pss_sign(timestamp + METHOD + path))
```
Padding: `PSS(mgf=MGF1(SHA256), salt_length=MAX_LENGTH)`.

**REST endpoints:**
- `GET /trade-api/v2/markets?series_ticker={prefix}` — list markets by series
- `GET /trade-api/v2/markets/{ticker}` — single market detail (has `last_price_dollars`, `no_ask_dollars`, `open_interest_fp`)
- `GET /trade-api/v2/markets/{ticker}/orderbook` — full orderbook (`yes_dollars_fp`, `no_dollars_fp` arrays of [price, quantity])
- `POST /trade-api/v2/orders` — submit order
- `GET /trade-api/v2/portfolio/balance` — account balance

**IMPORTANT:** List endpoint returns `yes_ask=None, volume=None`. Must use single market endpoint or orderbook endpoint for actual prices.

**WebSocket:** `wss://api.elections.kalshi.com/trade-api/ws/v2`
- Auth via same RSA-PSS headers in WS handshake
- Subscribe: `{"cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": ["..."]}}`
- Receives: `orderbook_snapshot` (full book) then `orderbook_delta` (incremental updates)

**Market series mapping (verified):**

| League | GAME | 1H | BTTS | Total | Spread |
|--------|------|------|------|-------|--------|
| EPL | `KXEPLGAME` | `KXEPL1H` | `KXEPLBTTS` | `KXEPLTOTAL` | `KXEPLSPREAD` |
| La Liga | `KXLALIGAGAME` | `KXLALIGA1H` | `KXLALIGABTTS` | `KXLALIGATOTAL` | `KXLALIGASPREAD` |
| Serie A | `KXSERIEAGAME` | `KXSERIEA1H` | `KXSERIEABTTS` | `KXSERIEATOTAL` | `KXSERIEASPREAD` |
| Bundesliga | `KXBUNDESLIGAGAME` | `KXBUNDESLIGA1H` | `KXBUNDESLIGABTTS` | `KXBUNDESLIGATOTAL` | `KXBUNDESLIGASPREAD` |
| Ligue 1 | `KXLIGUE1GAME` | `KXLIGUE11H` | `KXLIGUE1BTTS` | `KXLIGUE1TOTAL` | `KXLIGUE1SPREAD` |
| MLS | `KXMLSGAME` | — | — | — | `KXMLSSPREAD` |
| Brasileirão | `KXBRASILEIROGAME` | — | — | `KXBRASILEIROTOTAL` | `KXBRASILEIROSPREAD` |
| Argentina | `KXARGPREMDIVGAME` | — | — | — | — |

**Ticker pattern:** `KX{LEAGUE}{TYPE}-{season}{date}{teams}-{outcome}`
Example: `KXSERIEAGAME-26MAR15LAZACM-LAZ` = Serie A fullmatch, season 26, March 15, Lazio vs AC Milan, outcome Lazio.
Each match has 3 outcome markets: `{teams}-HOME`, `{teams}-TIE`, `{teams}-AWAY`.

### 4.3 Odds-API.io

**Purpose:** PRIMARY signal source for live trading (Phase 3 Tier 1). Provides real-time odds from 5 bookmakers including Betfair Exchange. Also provides pre-match odds for Phase 2 features.

**Base URL:** `https://api.odds-api.io/v3`

**Auth:** API key as query parameter: `?apiKey={key}`

**REST endpoints:**
- `GET /sports` — list sports (returns `[{name, slug}]`)
- `GET /events?sport=football&league={slug}` — list events with scores and status
- `GET /odds?eventId={id}&bookmakers={names}` — odds for single event (bookmakers param is REQUIRED)
- `GET /bookmakers` — list available bookmakers

**Available bookmakers (verified):** Bet365, Betfair Exchange, Betfair Sportsbook, Sbobet, 1xbet, DraftKings, Sharp Exchange.
**NOT available:** Pinnacle.

**WebSocket:** `wss://api.odds-api.io/v3/ws?apiKey={key}`
- On connect: receives `welcome` message with bookmaker list
- Subscribe filter recommended: `?markets=ML,Spread,Totals` to reduce bandwidth (without filter: 100-200+ markets per event)
- Pushes live odds updates for all in-play events
- **5 bookmakers used for OddsConsensus:** Betfair Exchange (2x weight), Bet365, 1xbet, Sbobet, DraftKings
- **This is the PRIMARY signal source for Phase 3 Tier 1 — event detection and reference pricing**

**League slug mapping (verified):**

| League | Slug | Pending Events |
|--------|------|:---:|
| EPL | `england-premier-league` | 19 |
| La Liga | `spain-laliga` | 32 |
| Serie A | `italy-serie-a` | 31 |
| Bundesliga | `germany-bundesliga` | 27 |
| Ligue 1 | `france-ligue-1` | 28 |
| MLS | `usa-mls` | 46 |
| Brasileirão | `brazil-brasileiro-serie-a` | 56 |
| Argentina | `argentina-liga-profesional` | 56 |

**Rate limit:** Documentation says 5,000 req/hour. No rate limit headers returned in responses. Operate conservatively.

### 4.4 football-data.co.uk

**Purpose:** Historical odds for Phase 1 training. Free CSV download, no API key needed.

**European 5 leagues (opening + closing Pinnacle):**

| League | URL | Columns |
|--------|-----|---------|
| EPL | `/mmz4281/{season}/E0.csv` | PSH,PSD,PSA (opening) + PSCH,PSCD,PSCA (closing) + B365 + Betfair |
| La Liga | `/mmz4281/{season}/SP1.csv` | Same |
| Bundesliga | `/mmz4281/{season}/D1.csv` | Same |
| Serie A | `/mmz4281/{season}/I1.csv` | Same |
| Ligue 1 | `/mmz4281/{season}/F1.csv` | Same |

Season format: `2526` for 2025-26. Current season (2025-26): 291 matches, last date 2026-03-05.

**Americas 3 leagues (closing only):**

| League | URL | Columns |
|--------|-----|---------|
| MLS | `/new/USA.csv` | PSCH,PSCD,PSCA (closing) + B365 + Betfair. 15 seasons (2012-2026). |
| Brasileirão | `/new/BRA.csv` | Same. 15 seasons. |
| Argentina | `/new/ARG.csv` | Same. 15 seasons. |

**Update frequency:** ~10 day delay from match date. Sufficient for Phase 1 training and Phase 2 supplementary reference.

---

## 5. Data Model

### 5.1 PostgreSQL Schema

```sql
-- Phase 1 trained parameters
CREATE TABLE production_params (
    version SERIAL PRIMARY KEY,
    league_id INT NOT NULL,
    Q JSONB NOT NULL,
    b JSONB NOT NULL,
    gamma_H DECIMAL(8,6) NOT NULL,
    gamma_A DECIMAL(8,6) NOT NULL,
    delta_H DECIMAL(8,6) NOT NULL,
    delta_A DECIMAL(8,6) NOT NULL,
    sigma_a DECIMAL(8,6) NOT NULL,
    xgb_model_blob BYTEA,
    feature_mask JSONB,
    trained_at TIMESTAMPTZ NOT NULL,
    match_count INT NOT NULL,
    brier_score DECIMAL(6,4) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE
);

-- Match lifecycle
CREATE TABLE match_schedule (
    match_id TEXT PRIMARY KEY,
    league_id INT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_utc TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'SCHEDULED',
    -- SCHEDULED → PHASE2_RUNNING → PHASE2_DONE/SKIPPED → PHASE3_RUNNING → FINISHED → ARCHIVED
    trading_mode TEXT NOT NULL DEFAULT 'paper',
    param_version INT,
    kalshi_tickers JSONB,
    goalserve_fix_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Live tick data (Phase 3 output)
CREATE TABLE tick_snapshots (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    t DECIMAL(6,3) NOT NULL,
    engine_phase TEXT NOT NULL,
    p_home_win DECIMAL(6,4),
    p_draw DECIMAL(6,4),
    p_away_win DECIMAL(6,4),
    sigma_home_win DECIMAL(6,4),
    sigma_draw DECIMAL(6,4),
    sigma_away_win DECIMAL(6,4),
    score_home INT,
    score_away INT,
    mu_H DECIMAL(8,4),
    mu_A DECIMAL(8,4),
    order_allowed BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_ticks_match ON tick_snapshots(match_id, t);

-- Events (goals, red cards, period changes)
CREATE TABLE event_log (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    event_type TEXT NOT NULL,
    t DECIMAL(6,3) NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_events_match ON event_log(match_id);

-- Trading positions
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES match_schedule(match_id),
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity INT NOT NULL,
    entry_price DECIMAL(6,4) NOT NULL,
    exit_price DECIMAL(6,4),
    status TEXT NOT NULL DEFAULT 'OPEN',
    -- OPEN → CLOSED | SETTLED
    is_paper BOOLEAN NOT NULL DEFAULT TRUE,
    realized_pnl DECIMAL(10,2),
    entry_tick INT,
    exit_tick INT,
    entry_reason TEXT,
    exit_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);
CREATE INDEX idx_positions_match ON positions(match_id);
CREATE INDEX idx_positions_status ON positions(status);

-- Cross-container exposure reservation
CREATE TABLE exposure_reservation (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    reserved_amount DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'RESERVED',
    -- RESERVED → CONFIRMED | RELEASED
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX idx_reservation_status ON exposure_reservation(status);

-- Bankroll (paper and live separate)
CREATE TABLE bankroll (
    mode TEXT PRIMARY KEY,              -- 'paper' | 'live'
    balance DECIMAL(12,2) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bankroll history for drawdown tracking
CREATE TABLE bankroll_snapshot (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    balance DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ticker mapping cache
CREATE TABLE ticker_mapping (
    match_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    kalshi_ticker TEXT NOT NULL,
    PRIMARY KEY (match_id, market_type)
);
```

### 5.2 Redis Channels

| Channel | Publisher | Subscriber | Content |
|---------|-----------|------------|---------|
| `tick:{match_id}` | Phase 3 (match container) | Dashboard WS | TickMessage |
| `event:{match_id}` | Phase 3 (match container) | Dashboard WS | EventMessage |
| `signal:{match_id}` | Phase 4 (match container) | Dashboard WS | SignalMessage |
| `position_update` | Phase 4 (match container) | Dashboard WS | PositionUpdateMessage |
| `system_alert` | Orchestrator + containers | Dashboard WS + alert handler | SystemAlertMessage |
| `PARAMS_UPDATED` | Phase 1 worker | Orchestrator (informational only) | `{version, league_id}` |

---

## 6. Infrastructure

### 6.1 Docker Services

```yaml
services:
  postgres:        # PostgreSQL 16, port 5432, volume: pgdata
  redis:           # Redis 7, port 6379
  orchestrator:    # Match discovery, Phase 2, container lifecycle
  dashboard-api:   # FastAPI, port 8001
  dashboard-ui:    # React/Next.js, port 3001
  prometheus:      # Metrics, port 9090
  grafana:         # Dashboards, port 3000
  # match-{id}:   # Dynamic — one per live match, launched by orchestrator
```

### 6.2 Network

Fixed Docker network: `mmpp-net`. All services and dynamically launched match containers join this network.

### 6.3 Ports

| Service | Internal | External |
|---------|:---:|:---:|
| PostgreSQL | 5432 | 5432 |
| Redis | 6379 | 6379 |
| Dashboard API | 8001 | 8001 |
| Dashboard UI | 3001 | 3001 |
| Prometheus | 9090 | 9090 |
| Grafana | 3000 | 3000 |

---

## 7. Sprint Plan

Each sprint follows: design decomposition (chat) → implement (Claude Code) → review (chat).
Decomposition documents are created just-in-time, not upfront.

### Sprint -1: Feasibility Study (pre-build data analysis)

**Purpose:** Before writing any system code, answer three feasibility questions with data. This is NOT a GO/NO-GO gate for cross-market lag — minute-level goal times cannot measure second-level lag. Instead, this sprint validates whether the Kalshi soccer market is worth building for at all.

**Data sources (all available now):**
- Kalshi trades API: `GET /trade-api/v2/markets/trades?ticker={ticker}` — timestamped fills with `yes_price_dollars`, `count_fp`, `taker_side`
- Goalserve commentaries: `data/commentaries/` — 12,607 matches with goal times (minute-level)
- Kalshi settled markets: `GET /trade-api/v2/markets?status=settled&series_ticker=KX{LEAGUE}GAME` — ticker list for all completed matches

**Question 1: Is there enough liquidity to trade?**
For each league, measure:
- Trades per match (median, p25, p75)
- Dollar volume per match
- Bid-ask spread implied by consecutive trades
- % of matches with <20 trades (= untradeable)
If most matches have <50 trades, there's nothing to capture regardless of edge.

**Question 2: How much do Kalshi prices move on goals?**
For each goal event (Goalserve minute-level), measure:
- `pre_goal_price`: average Kalshi price in 2min window before goal minute
- `post_goal_price`: average price in 2min window after goal minute + 5min
- `price_impact = |post_goal_price - pre_goal_price|`
- Trade density in the 5min window after goal vs baseline
This tells us the magnitude of opportunity, even without knowing the exact timing.

**Question 3: Are there stale trades after goals?**
For each goal, look at trades in the 5min window after the goal minute:
- Are there trades executed near the pre-goal price AFTER the goal minute?
- How many contracts trade at "stale" prices (within 2¢ of pre-goal price)?
- What is the dollar value of these stale trades?
This is imprecise (±60s on goal timing) but if we see trades at pre-goal prices 2-3 minutes into the post-goal window, that's a signal even with the timing uncertainty.

**Important limitation:** Goalserve provides goal times at minute-level precision (e.g., "61'"), not second-level. The actual goal could be anywhere within that 60-second window. Combined with kickoff/halftime timing uncertainty (~±2min), all measurements have ±2-3min imprecision. This analysis CANNOT measure sub-minute lag. The real lag measurement happens in Sprint 3 with second-level recording.

**Output:**
- Per-league summary: liquidity, price impact, stale trade count
- Decision: which leagues are worth targeting (liquid enough + large price impacts)
- Rough estimate: if stale trades exist at scale, what's the theoretical dollar opportunity per match

**Decision criteria:**
- If most leagues have >100 trades/match AND price impact >5¢ on goals → **PROCEED** to Sprint 0
- If only 2-3 leagues are liquid → **PROCEED** with reduced scope (those leagues only)
- If <50 trades/match across all leagues → **PAUSE** — market too thin to trade

**Timeline:** 1-2 days. Just a Python script.
- **Done:** League-by-league feasibility report. Scope decision for Sprint 0+.

### Sprint 0: Project Skeleton
- Directory structure, pyproject.toml, Docker Compose (postgres + redis)
- PostgreSQL schema migration
- Copy math core 5 files + their tests → verify tests pass
- Copy `data/commentaries/`, `data/odds_historical/`, `keys/`
- **Done:** `make test` passes, `docker compose up` starts postgres + redis

### Sprint 1: Phase 1 Calibration Pipeline
- Commentaries parser (Goalserve JSON → intervals, goals, red cards)
- football-data.co.uk CSV loader (odds features)
- Q matrix, XGBoost, NLL optimizer, validation (using math core)
- production_params DB save (including xgb_model_blob BYTEA)
- **Done:** 8-league calibration → Go/No-Go per league, params in DB

### Sprint 2: Phase 2 + External Clients
- Goalserve REST client (live score + commentaries)
- Odds-API.io REST client (events, odds) + WebSocket client (live odds)
- Kalshi REST client (RSA-PSS auth, markets, orderbook, orders) + WebSocket client
- Phase 2 pipeline: features → XGBoost → backsolve → sanity check → Phase2Result
- **Done:** Phase2Result produced for a real upcoming match

### Sprint 3: Phase 3 Live Engine + Recording
- OddsConsensus (5 bookmaker aggregation, Betfair-heavy weighting, event detection)
- Odds-API.io WS listener → BookmakerState updates → OddsConsensus
- tick_loop (wall-clock, halftime exclusion, absolute time scheduling)
- P_reference selection logic (consensus HIGH → consensus, else → model)
- Event detection (Goalserve polling, period tracking, multi-goal handling)
- Event handlers (goal, red card, substitution, VAR)
- MC pricing → TickPayload production (with P_reference, P_model, odds_consensus)
- Redis publish (tick + event channels)
- **Recording infrastructure:** save all raw WS messages (Odds-API, Kalshi) + Goalserve polls to JSONL files with timestamps. One directory per match. This enables offline replay for all subsequent sprints.
- **ReplayServer:** reads recorded JSONL files and replays them as mock WS/HTTP endpoints with configurable speed (1x, 10x, 100x). All sprints after this use ReplayServer for development — no live games required.
- **Done:** Record 5+ live matches. ReplayServer can replay them. P_reference chart shows consensus reacting to goals before Goalserve.

### Sprint 4: Phase 4 Execution
- signal_generator (P_reference → per-market edge detection loop)
- Edge detection (P_reference vs P_kalshi VWAP, confidence-adjusted thresholds)
- Kelly sizing (incremental, confidence-adjusted multiplier, reserve-confirm-release)
- ExecutionRouter (paper/live), PaperExecutor, LiveExecutor
- Exit monitor (6 triggers, min_hold, cooldown)
- Settlement polling
- **Done:** ReplayServer with recorded matches → paper trades, positions in DB, P&L computed

### Sprint 5: Orchestrator + Docker
- Scheduler (Kalshi + Goalserve discovery, fixture↔market matching)
- Container manager (aiodocker, launch/stop/cleanup)
- Lifecycle manager (SCHEDULED → PHASE2 → PHASE3 → FINISHED → ARCHIVED)
- Heartbeat monitoring, recovery on restart
- Match engine Dockerfile + docker-compose integration
- **Done:** `docker compose up`, orchestrator auto-discovers and trades live matches

### Sprint 6: Dashboard
- Pydantic API models (15 models)
- FastAPI REST routes (matches, positions, ticks, P&L, system status)
- WebSocket handler (Redis pubsub → client push)
- React UI (Command Center, Match Deep Dive, P&L Analytics, System Ops)
- **Done:** localhost:3001 shows live match data in real time

### Sprint 7: Integration Testing + Paper Trading
- End-to-end integration tests using ReplayServer (recorded matches from Sprint 3)
- **Offline validation first:** replay 50+ recorded matches through full pipeline, measure all metrics
- Then 2-week live paper trading (no real money, but real-time data + real Kalshi orderbook)
- Measure all 6 validation metrics from §3.7.5:
  - Cross-market lag: median Kalshi lag behind consensus > 3s?
  - Odds-API.io relay latency: median < 3s?
  - Edge at execution: mean EV_fill > 0?
  - Edge realization > 0.3?
  - Consensus vs model comparison: which source generates profitable trades?
  - Basic profitability: P&L > 0, drawdown < 15%?
- **Done:** Metrics 1+3 pass → GO. Metric 1 fails (lag < 3s) → PIVOT to model-only strategy. ANY critical metric fails → diagnose before proceeding.

---

## 8. Existing Data Assets

Carried over from v1/v2 — do NOT re-collect.

| Path | Contents | Size |
|------|----------|------|
| `data/commentaries/` | Goalserve raw JSON, 8 leagues, 4-6 seasons each | ~44 files |
| `data/odds_historical/` | football-data.co.uk CSVs, Pinnacle + Bet365 + Betfair | ~33 files |
| `data/kalshi_football.db` | SQLite, v1 API-Football data (12,379 matches, 77K intervals) | 78 MB |
| `keys/kalshi_private.pem` | Kalshi RSA private key | — |
| `.env` | API keys: GOALSERVE, KALSHI, ODDS_API | — |

### Math Core Files (copy from v3 `src/` to v4 `src/math/`)

| v3 Path | v4 Name | Lines | Purpose | External deps |
|---------|---------|:---:|---------|---|
| `src/engine/mc_core.py` | `mc_core.py` | 357 | Numba JIT MC simulation | numba, numpy |
| `src/calibration/step_1_4_nll_optimize.py` | `step_1_4_nll_optimize.py` | ~700 | Adam→L-BFGS NLL optimizer | torch, scipy |
| `src/calibration/step_1_2_Q_estimation.py` | `step_1_2_Q_estimation.py` | ~380 | Q matrix estimation | numpy |
| `src/engine/compute_mu.py` | `compute_mu.py` | ~300 | Remaining μ computation | numpy |

Note: `stoppage.py` was planned in v1/v2 but never implemented. Stoppage time handling will be built into Phase 3 tick_loop in Sprint 3.