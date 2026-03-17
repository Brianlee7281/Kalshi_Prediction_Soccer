# Sprint 3: Phase 3 Live Engine + Recording — Decomposition

Reference: `docs/architecture.md` §3.3 (Phase 3), §2.4 (TickPayload), §5.2 (Redis)

## Overview

Phase 3 runs for 90 minutes during a live match. Three concurrent coroutines:

```
asyncio.gather(
    tick_loop(model),           # every 1s: aggregate → TickPayload → Phase 4 queue + Redis
    odds_api_listener(model),   # WS: 5 bookmaker live odds → OddsConsensus
    goalserve_poller(model),    # every 3s: match events → model state update
)
```

All three read/write a shared `LiveMatchModel` object. No locks needed — single-threaded asyncio.

---

## Task 3.1: LiveMatchModel (Shared State)

The central state object shared across all three coroutines.

**File:** `src/engine/model.py`

```python
@dataclass
class LiveMatchModel:
    """Mutable state for a single live match."""
    
    # Identity
    match_id: str
    league_id: int
    home_team: str
    away_team: str
    
    # Phase 2 inputs (immutable after init)
    a_H: float
    a_A: float
    param_version: int
    b: np.ndarray              # shape (6,) time profile
    gamma_H: np.ndarray        # shape (4,) home red card penalties
    gamma_A: np.ndarray        # shape (4,) away red card penalties
    delta_H: np.ndarray        # shape (5,) home score-diff effects
    delta_A: np.ndarray        # shape (5,) away score-diff effects
    Q: np.ndarray              # shape (4,4) generator matrix
    basis_bounds: np.ndarray   # shape (7,) basis period boundaries
    kalshi_tickers: dict[str, str]  # {"home_win": "KX...", "draw": "...", ...}
    
    # Time management (Pattern 3 from .claude/rules/patterns.md)
    kickoff_wall_clock: float = 0.0        # time.monotonic() at kickoff
    halftime_start: float = 0.0            # time.monotonic() when HT began
    halftime_accumulated: float = 0.0      # total halftime duration in seconds
    t: float = 0.0                         # EFFECTIVE play time in minutes
    T_exp: float = 93.0                    # expected match end (90 + avg stoppage)
    
    # Match state
    engine_phase: str = "WAITING_FOR_KICKOFF"  # WAITING | FIRST_HALF | HALFTIME | SECOND_HALF | FINISHED
    score: tuple[int, int] = (0, 0)        # (home, away)
    current_state_X: int = 0               # Markov state {0,1,2,3}
    delta_S: int = 0                       # score diff (home - away)
    mu_H: float = 0.0                      # remaining expected home goals
    mu_A: float = 0.0                      # remaining expected away goals
    
    # Event tracking
    event_state: str = "IDLE"              # IDLE | PRELIMINARY | CONFIRMED
    cooldown: bool = False
    cooldown_until_tick: int = 0
    ob_freeze: bool = False
    _last_period: str = ""                 # for period change spam prevention
    _last_score: tuple[int, int] = (0, 0)  # for multi-goal detection
    
    # Tick management
    tick_count: int = 0
    
    # Odds consensus state (updated by odds_api_listener)
    odds_consensus: OddsConsensus | None = None
    
    # Precomputed grids (for compute_mu)
    P_grid: dict[int, np.ndarray] = field(default_factory=dict)
    P_fine_grid: dict[int, np.ndarray] = field(default_factory=dict)
    
    @classmethod
    def from_phase2_result(cls, result: Phase2Result, params: dict) -> "LiveMatchModel":
        """Initialize from Phase2Result + production_params."""
        ...
    
    def update_time(self) -> None:
        """Update model.t from wall clock (Pattern 3).
        t = (monotonic() - kickoff_wall_clock - halftime_accumulated) / 60
        """
        ...
    
    @property
    def order_allowed(self) -> bool:
        """Phase 3 decides 'can trade?', Phase 4 decides 'should trade?'."""
        return (
            not self.cooldown
            and not self.ob_freeze
            and self.event_state == "IDLE"
        )
```

Requirements:
- Precompute P_grid (transition probability matrices) on init for compute_mu performance
- basis_bounds default: [0, 15, 30, 45+α₁, 60+α₁, 75+α₁, T_exp] where α₁ = first half stoppage
- from_phase2_result loads Q, b, gamma, delta arrays from production_params JSON

**Test:** `tests/engine/test_model.py`

```python
def test_model_from_phase2():
    """Create LiveMatchModel from Phase2Result + params."""
    # Use mock Phase2Result and params dict
    ...

def test_update_time_excludes_halftime():
    """Verify model.t excludes halftime duration."""
    # Set kickoff_wall_clock, simulate halftime, verify t
    ...

def test_order_allowed():
    """Verify order_allowed conditions."""
    ...
```

**Done:** LiveMatchModel creates from Phase2Result and tracks time correctly.

---

## Task 3.2: OddsConsensus

Aggregates 5 bookmaker odds into a single reference price.

**File:** `src/engine/odds_consensus.py`

```python
class OddsConsensus:
    """Aggregates live odds from 5 bookmakers into consensus reference price."""
    
    BOOKMAKERS = ["Betfair Exchange", "Bet365", "1xbet", "Sbobet", "DraftKings"]
    WEIGHTS = {"Betfair Exchange": 2.0}  # default weight = 1.0
    STALE_THRESHOLD_S = 10.0  # bookmaker data older than this = stale
    
    def __init__(self):
        self.sources: dict[str, BookmakerState] = {}
    
    def update_bookmaker(self, name: str, implied: MarketProbs) -> None:
        """Update a bookmaker's odds. Called by odds_api_listener on each WS message."""
        ...
    
    def compute_reference(self) -> OddsConsensusResult:
        """Compute consensus from all fresh (non-stale) sources.
        
        Logic (from architecture.md §3.3):
        - fresh = sources where last_update < STALE_THRESHOLD_S ago
        - If 0 fresh: confidence = NONE
        - If 1 fresh: confidence = LOW
        - If 2+ fresh: confidence = HIGH
        - P_consensus = weighted median (Betfair 2x weight)
        - event_detected = 2+ sources moved >3% same direction within 5s
        """
        ...
    
    def _weighted_median(
        self, values: list[float], weights: list[float]
    ) -> float:
        """Weighted median calculation."""
        ...
    
    def _detect_event(self) -> bool:
        """Check if 2+ bookmakers moved >3% in same direction within 5s."""
        ...
```

Requirements:
- BookmakerState tracks `implied`, `prev_implied`, `last_update`, `is_stale`
- Weighted median: expand each value by its weight, take median of expanded list
- Event detection: compare current vs previous implied for each fresh source
- Per-market consensus: compute separately for home_win, draw, away_win, over_25, etc.
- Thread-safe: not needed (single asyncio thread), but methods should be reentrant

**Test:** `tests/engine/test_odds_consensus.py`

```python
def test_consensus_high_confidence():
    """2+ fresh sources → HIGH confidence."""
    oc = OddsConsensus()
    oc.update_bookmaker("Betfair Exchange", MarketProbs(home_win=0.50, draw=0.30, away_win=0.20))
    oc.update_bookmaker("Bet365", MarketProbs(home_win=0.48, draw=0.31, away_win=0.21))
    result = oc.compute_reference()
    assert result.confidence == "HIGH"
    assert result.n_fresh_sources == 2

def test_consensus_none_all_stale():
    """All sources stale → NONE confidence."""
    oc = OddsConsensus()
    # Add sources with old timestamps, verify NONE
    ...

def test_consensus_betfair_weighted():
    """Betfair gets 2x weight in median."""
    oc = OddsConsensus()
    oc.update_bookmaker("Betfair Exchange", MarketProbs(home_win=0.60, draw=0.25, away_win=0.15))
    oc.update_bookmaker("Bet365", MarketProbs(home_win=0.40, draw=0.35, away_win=0.25))
    oc.update_bookmaker("1xbet", MarketProbs(home_win=0.42, draw=0.33, away_win=0.25))
    result = oc.compute_reference()
    # Betfair has 2x weight, so consensus should lean toward 0.60
    assert result.P_consensus.home_win > 0.45

def test_event_detection():
    """2+ sources moving >3% → event_detected=True."""
    ...
```

**Done:** OddsConsensus aggregates bookmaker odds with Betfair-heavy weighting.

---

## Task 3.3: Odds-API WebSocket Listener

Connects to Odds-API WS and feeds updates into OddsConsensus.

**File:** `src/engine/odds_api_listener.py`

```python
async def odds_api_listener(model: LiveMatchModel) -> None:
    """Coroutine: connect to Odds-API WS, update OddsConsensus on each message.
    
    WS URL: wss://api.odds-api.io/v3/ws?apiKey={key}&markets=ML,Spread,Totals&sport=football&status=live
    
    Message flow:
    1. Receive 'welcome' message
    2. Receive 'updated' messages: {type: "updated", bookie: "Bet365", markets: [...]}
    3. Parse ML odds → convert to implied probabilities → remove vig
    4. Call model.odds_consensus.update_bookmaker(bookie, implied_probs)
    
    Auto-reconnect with exponential backoff (1s base, 30s max).
    Runs until model.engine_phase == "FINISHED".
    
    Records all raw WS messages to JSONL if recorder is attached.
    """
    ...

def _parse_odds_update(message: dict) -> tuple[str, MarketProbs] | None:
    """Parse an Odds-API WS 'updated' message into (bookmaker_name, MarketProbs).
    Returns None if message is not relevant (wrong type, missing fields).
    """
    ...

def _odds_to_implied(home: float, draw: float, away: float) -> MarketProbs:
    """Convert decimal odds to implied probabilities with vig removal.
    p = 1/odds, then normalize so sum = 1.0.
    """
    ...
```

Requirements:
- Use `websockets` library
- Filter messages for relevant match (by event ID or team names)
- Reconnect on disconnect with exponential backoff
- Log every bookmaker update with structlog
- Pass raw messages to recorder (Task 3.7) if active

**Test:** `tests/engine/test_odds_api_listener.py`

```python
def test_parse_odds_update():
    """Parse a real Odds-API WS message."""
    from src.engine.odds_api_listener import _parse_odds_update
    msg = {
        "type": "updated",
        "bookie": "Bet365",
        "markets": [{"name": "ML", "odds": [
            {"name": "home", "price": 2.10},
            {"name": "draw", "price": 3.40},
            {"name": "away", "price": 3.20},
        ]}]
    }
    result = _parse_odds_update(msg)
    assert result is not None
    bookie, probs = result
    assert bookie == "Bet365"
    assert 0.99 < probs.home_win + probs.draw + probs.away_win < 1.01

def test_odds_to_implied():
    from src.engine.odds_api_listener import _odds_to_implied
    probs = _odds_to_implied(2.10, 3.40, 3.20)
    assert abs(probs.home_win + probs.draw + probs.away_win - 1.0) < 0.01
```

**Done:** WS listener parses odds updates and feeds OddsConsensus.

---

## Task 3.4: Goalserve Poller + Event Handlers

Polls Goalserve every 3s for match events and updates model state.

**File:** `src/engine/goalserve_poller.py`

```python
async def goalserve_poller(model: LiveMatchModel) -> None:
    """Coroutine: poll Goalserve every 3s, detect events, update model state.
    
    Detects: goals, red cards, period changes (FIRST_HALF→HALFTIME→SECOND_HALF→FINISHED).
    Updates: model.score, model.current_state_X, model.delta_S, model.engine_phase.
    
    Anti-patterns addressed:
    - Period change spam: track _last_period, emit only on actual change
    - Multi-goal same poll: if score diff ≥ 2, commit goals sequentially
    - Late container join: if first poll shows numeric status, sync time
    - Match ID search: @id, @fix_id, @static_id (all three)
    
    Records all poll responses to JSONL if recorder is attached.
    """
    ...
```

**File:** `src/engine/event_handlers.py`

```python
def handle_goal(
    model: LiveMatchModel,
    team: str,  # "home" | "away"
    minute: int,
) -> None:
    """Process a goal event.
    - Update score, delta_S
    - Recompute mu_H, mu_A
    - Set event_state = PRELIMINARY → CONFIRMED (after Goalserve confirms)
    - Set cooldown = True, cooldown_until_tick = tick_count + 50
    """
    ...

def handle_red_card(
    model: LiveMatchModel,
    team: str,
    minute: int,
) -> None:
    """Process a red card event.
    - Update current_state_X (Markov transition)
    - State transitions: home red → 0→1 or 2→3, away red → 0→2 or 1→3
    - Set cooldown
    """
    ...

def handle_period_change(
    model: LiveMatchModel,
    new_phase: str,
) -> None:
    """Process period change.
    - FIRST_HALF → HALFTIME: record halftime_start
    - HALFTIME → SECOND_HALF: compute halftime_accumulated
    - SECOND_HALF → FINISHED: set engine_phase
    - Spam prevention: only process if new_phase != _last_period
    """
    ...

def detect_events_from_poll(
    model: LiveMatchModel,
    poll_data: dict,
) -> list[dict]:
    """Compare poll data with model state, detect all events.
    Returns list of event dicts: [{type: "goal", team: "home", minute: 35}, ...]
    Handles multi-goal: if score jumped by 2+, emit sequential goals.
    """
    ...
```

Requirements:
- Goalserve status: numeric string = minute, "HT" = halftime, "FT" = finished
- Late join: first poll with numeric status → set model.t and kickoff_wall_clock
- Multi-goal: score 0-0 → 2-0 in one poll → emit goal at model.t, then goal at model.t+0.1
- Cooldown: 50 ticks after goal, 30 ticks after red card
- All events published to Redis `event:{match_id}` channel

**Test:** `tests/engine/test_event_handlers.py`

```python
def test_handle_goal_updates_score():
    model = _make_test_model()
    handle_goal(model, "home", 35)
    assert model.score == (1, 0)
    assert model.delta_S == 1
    assert model.cooldown == True

def test_handle_red_card_state_transition():
    model = _make_test_model()
    handle_red_card(model, "home", 60)
    assert model.current_state_X == 1  # 0→1

def test_handle_period_change_halftime():
    model = _make_test_model()
    model.engine_phase = "FIRST_HALF"
    handle_period_change(model, "HALFTIME")
    assert model.engine_phase == "HALFTIME"
    assert model.halftime_start > 0

def test_period_change_spam_prevention():
    model = _make_test_model()
    model.engine_phase = "FIRST_HALF"
    model._last_period = "FIRST_HALF"
    handle_period_change(model, "FIRST_HALF")
    # Should NOT change anything — same period
    assert model.engine_phase == "FIRST_HALF"

def test_multi_goal_detection():
    model = _make_test_model()
    model._last_score = (0, 0)
    poll_data = {"localteam": {"@goals": "2"}, "visitorteam": {"@goals": "0"}}
    events = detect_events_from_poll(model, poll_data)
    assert len(events) == 2  # two sequential goals
```

**Done:** Goalserve poller detects events and updates model state correctly.

---

## Task 3.5: MC Pricing Bridge

Connects the math core MC simulation to the live model.

**File:** `src/engine/mc_pricing.py`

```python
async def compute_mc_prices(model: LiveMatchModel, N: int = 50_000) -> tuple[MarketProbs, MarketProbs]:
    """Run MC simulation and return (P_model, sigma_MC).
    
    Uses src.math.mc_core.mc_simulate_remaining with current model state.
    Runs in executor (thread pool) to avoid blocking asyncio.
    
    Returns:
        P_model: MarketProbs with home_win, draw, away_win, over_25, btts_yes, etc.
        sigma_MC: MarketProbs with per-market standard errors (sqrt(p*(1-p)/N))
    """
    ...

def _results_to_market_probs(results: np.ndarray, S_H: int, S_A: int) -> MarketProbs:
    """Convert MC simulation results (N,2) array to MarketProbs.
    results[:,0] = final home goals, results[:,1] = final away goals.
    Count outcomes: home_win, draw, away_win, over_25, btts_yes, etc.
    """
    ...

def _compute_sigma(probs: MarketProbs, N: int) -> MarketProbs:
    """Per-market MC standard error: sqrt(p*(1-p)/N) for each market."""
    ...
```

Requirements:
- Run mc_simulate_remaining in `asyncio.get_event_loop().run_in_executor(None, ...)` to not block
- Compute ALL market types: home_win, draw, away_win, over_25, under_25, btts_yes, btts_no
- over_25: total goals > 2.5 → home_final + away_final >= 3
- btts_yes: both teams scored → home_final >= 1 AND away_final >= 1
- Use model's current a_H, a_A, b, gamma, delta, Q, state, score, t
- Also calls compute_remaining_mu from src/math/compute_mu.py to update model.mu_H, model.mu_A

**Test:** `tests/engine/test_mc_pricing.py`

```python
def test_results_to_market_probs():
    import numpy as np
    from src.engine.mc_pricing import _results_to_market_probs
    # Simulate: 60% home win, 20% draw, 20% away
    results = np.array([[2,1]]*600 + [[1,1]]*200 + [[0,1]]*200)
    probs = _results_to_market_probs(results, S_H=0, S_A=0)
    assert abs(probs.home_win - 0.60) < 0.02
    assert abs(probs.draw - 0.20) < 0.02

def test_sigma_computation():
    from src.engine.mc_pricing import _compute_sigma
    from src.common.types import MarketProbs
    probs = MarketProbs(home_win=0.50, draw=0.30, away_win=0.20)
    sigma = _compute_sigma(probs, 50000)
    # sigma for p=0.5: sqrt(0.25/50000) ≈ 0.00224
    assert abs(sigma.home_win - 0.00224) < 0.001
```

**Done:** MC pricing produces MarketProbs + sigma_MC from current model state.

---

## Task 3.6: tick_loop + P_reference + Redis Publish

The main 1-second loop that ties everything together.

**File:** `src/engine/tick_loop.py`

```python
async def tick_loop(
    model: LiveMatchModel,
    phase4_queue: asyncio.Queue | None = None,
) -> None:
    """Main tick loop. Every 1 second:
    1. Update model.t from wall clock
    2. Compute MC prices → P_model, sigma_MC
    3. Get OddsConsensus → P_consensus
    4. Select P_reference (Pattern 1 from .claude/rules/patterns.md)
    5. Build TickPayload
    6. Send to Phase 4 queue
    7. Publish to Redis
    8. Write to DB (tick_snapshots)
    
    Tick scheduling: absolute time, not sleep(1) (Pattern 3).
    next_tick = start + tick_count * 1.0
    If MC takes >1s, skip tick but model.t stays accurate.
    """
    ...

def select_P_reference(
    odds_consensus: OddsConsensusResult | None,
    P_model: MarketProbs,
) -> tuple[MarketProbs, str]:
    """Select P_reference from consensus or model.
    Returns (P_reference, reference_source).
    
    Logic (Pattern 1 from .claude/rules/patterns.md):
    - HIGH confidence: use consensus
    - LOW confidence: use consensus if agrees with model (±10%), else model
    - NONE: use model
    """
    ...

async def _publish_tick_to_redis(
    model: LiveMatchModel,
    payload: TickPayload,
) -> None:
    """Publish TickMessage to Redis channel 'tick:{match_id}'."""
    ...

async def _write_tick_snapshot(
    model: LiveMatchModel,
    payload: TickPayload,
) -> None:
    """INSERT into tick_snapshots table."""
    ...

async def _sleep_until_next_tick(
    start_time: float,
    tick_count: int,
    interval: float = 1.0,
) -> None:
    """Sleep until next absolute tick time. Skip if already past."""
    next_tick_time = start_time + tick_count * interval
    sleep_duration = next_tick_time - time.monotonic()
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)
```

Requirements:
- WAITING_FOR_KICKOFF: poll but don't price. Wait for Goalserve status to become numeric.
- HALFTIME: don't price, don't update model.t, just wait
- FIRST_HALF / SECOND_HALF: full pricing pipeline
- FINISHED: send final TickPayload with engine_phase="FINISHED", break loop
- TickPayload must include ALL fields from architecture.md §2.4
- Cooldown management: decrement cooldown_until_tick each tick

**Test:** `tests/engine/test_tick_loop.py`

```python
def test_select_P_reference_high():
    from src.engine.tick_loop import select_P_reference
    from src.common.types import MarketProbs, OddsConsensusResult
    consensus = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.55, draw=0.25, away_win=0.20),
        confidence="HIGH", n_fresh_sources=3, bookmakers=[], event_detected=False,
    )
    P_model = MarketProbs(home_win=0.50, draw=0.28, away_win=0.22)
    P_ref, source = select_P_reference(consensus, P_model)
    assert source == "consensus"
    assert P_ref.home_win == 0.55

def test_select_P_reference_none():
    from src.engine.tick_loop import select_P_reference
    from src.common.types import MarketProbs
    P_model = MarketProbs(home_win=0.50, draw=0.28, away_win=0.22)
    P_ref, source = select_P_reference(None, P_model)
    assert source == "model"
    assert P_ref.home_win == 0.50

def test_select_P_reference_low_disagree():
    """LOW confidence + model disagrees → use model."""
    from src.engine.tick_loop import select_P_reference
    from src.common.types import MarketProbs, OddsConsensusResult
    consensus = OddsConsensusResult(
        P_consensus=MarketProbs(home_win=0.70, draw=0.15, away_win=0.15),
        confidence="LOW", n_fresh_sources=1, bookmakers=[], event_detected=False,
    )
    P_model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    P_ref, source = select_P_reference(consensus, P_model)
    assert source == "model"  # disagree by 0.25 > 0.10

def test_sleep_until_next_tick():
    """Verify absolute time scheduling."""
    ...
```

**Done:** tick_loop produces TickPayload every second with correct P_reference selection.

---

## Task 3.7: Recording Infrastructure

Saves all raw data to JSONL for offline replay.

**File:** `src/recorder/recorder.py`

```python
class MatchRecorder:
    """Records all live data for a match to JSONL files.
    
    Directory structure (Pattern 7 from .claude/rules/patterns.md):
        data/recordings/{match_id}/
            odds_api.jsonl      # raw WS messages
            kalshi_ob.jsonl     # orderbook snapshots + deltas
            goalserve.jsonl     # poll responses
            ticks.jsonl         # TickPayload snapshots
            events.jsonl        # detected events
            metadata.json       # match info, start/end times
    
    Each line: {"_ts": 1234567890.123, ...original_data}
    _ts = time.monotonic() relative to recording start (our system clock)
    """
    
    def __init__(self, match_id: str, base_dir: Path = Path("data/recordings")):
        ...
    
    def record_odds_api(self, message: dict) -> None:
        """Append raw Odds-API WS message with _ts."""
        ...
    
    def record_kalshi_ob(self, data: dict) -> None:
        """Append Kalshi orderbook data with _ts."""
        ...
    
    def record_goalserve(self, response: dict) -> None:
        """Append Goalserve poll response with _ts."""
        ...
    
    def record_tick(self, payload: TickPayload) -> None:
        """Append tick snapshot with _ts."""
        ...
    
    def record_event(self, event: dict) -> None:
        """Append detected event with _ts."""
        ...
    
    def finalize(self) -> None:
        """Write metadata.json, close all file handles."""
        ...
```

Requirements:
- JSONL format: one JSON object per line, append mode
- `_ts` field on every record: `time.monotonic() - self.start_time`
- Create directory on init
- Buffered writing: flush every 10 records or every 5 seconds
- finalize() writes metadata.json with match_id, start_time, end_time, record_counts

**Test:** `tests/recorder/test_recorder.py`

```python
def test_recorder_creates_files(tmp_path):
    from src.recorder.recorder import MatchRecorder
    rec = MatchRecorder("test_match", base_dir=tmp_path)
    rec.record_odds_api({"type": "updated", "bookie": "Bet365"})
    rec.record_goalserve({"score": "1-0"})
    rec.finalize()
    
    assert (tmp_path / "test_match" / "odds_api.jsonl").exists()
    assert (tmp_path / "test_match" / "goalserve.jsonl").exists()
    assert (tmp_path / "test_match" / "metadata.json").exists()

def test_recorder_ts_field(tmp_path):
    import json
    from src.recorder.recorder import MatchRecorder
    rec = MatchRecorder("test_match", base_dir=tmp_path)
    rec.record_odds_api({"test": "data"})
    rec.finalize()
    
    with open(tmp_path / "test_match" / "odds_api.jsonl") as f:
        line = json.loads(f.readline())
        assert "_ts" in line
        assert isinstance(line["_ts"], float)
```

**Done:** Recording writes all data streams to JSONL files.

---

## Task 3.8: ReplayServer + Integration Test

Replays recorded data as mock WS/HTTP endpoints for offline development.

**File:** `src/recorder/replay_server.py`

```python
class ReplayServer:
    """Replays recorded match data as mock endpoints.
    
    Reads JSONL files from a recording directory and serves them at
    configurable speed (1x, 10x, 100x real-time).
    
    Provides:
    - Mock Goalserve HTTP endpoint (returns poll responses in sequence)
    - Mock Odds-API WS endpoint (sends odds updates with timing)
    - Mock Kalshi WS endpoint (sends orderbook snapshots/deltas)
    
    Usage:
        server = ReplayServer("data/recordings/match_123", speed=10.0)
        await server.start()  # starts mock endpoints
        # ... run engine against mock endpoints ...
        await server.stop()
    """
    
    def __init__(self, recording_dir: Path, speed: float = 1.0):
        ...
    
    async def start(self, goalserve_port: int = 8555, odds_ws_port: int = 8556) -> None:
        """Start mock HTTP + WS servers."""
        ...
    
    async def stop(self) -> None:
        """Stop all mock servers."""
        ...
    
    async def _serve_goalserve(self, request) -> web.Response:
        """Return next Goalserve poll response based on current replay time."""
        ...
    
    async def _serve_odds_ws(self, ws) -> None:
        """Send odds updates via WS with timing based on _ts and speed multiplier."""
        ...
```

**File:** `scripts/run_phase3.py` — CLI to run Phase 3 against live or replayed data

```python
"""
Usage:
  PYTHONPATH=. python scripts/run_phase3.py --match-id 12345 --league EPL          # live
  PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/match_12345   # replay
"""
```

**Test:** `tests/recorder/test_replay_server.py`

```python
def test_replay_server_loads_recording(tmp_path):
    """Create a minimal recording, verify ReplayServer loads it."""
    ...

@pytest.mark.asyncio
async def test_replay_goalserve_endpoint(tmp_path):
    """ReplayServer serves Goalserve responses in sequence."""
    ...
```

**Integration test:** After recording a real match (or using test data), run:
```bash
PYTHONPATH=. python scripts/run_phase3.py --replay data/recordings/{match_id}
```

Verify:
- tick_loop runs without crashing
- P_reference changes when odds update
- Events are detected from Goalserve polls
- TickPayloads are produced and logged

**Done:** ReplayServer replays recorded matches. Phase 3 engine runs end-to-end against replay.

---

## Execution Order

Task 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8

After each task, run `make test` and fix any issues before proceeding.
Git commit after each task with message `sprint3: {brief description}`.

Sprint 3 is DONE when:
- All tests pass
- Phase 3 engine runs against mock/replay data without crashing
- TickPayloads are produced with correct P_reference selection
- Recording writes all data streams to JSONL
- ReplayServer can replay a recorded match at 10x speed
- **BONUS (requires live match):** Record a real match during a weekend fixture, verify Odds-API WS updates arrive, Goalserve polls detect goals
