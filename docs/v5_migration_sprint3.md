# Sprint 3 Migration — Engine (Breaking Changes)

Reference: `docs/MMPP_v5_Complete.md` §7 (Phase 3), §8 (Phase 4 interface)

This is where v4 dies and v5 takes over. OddsConsensus is deleted, tick_loop is rewritten, EKF replaces Bayesian updater.

**Sub-sprint order:**
1. Create new modules first (ekf.py, hmm_estimator.py, dom_index.py, kalshi_ob_sync.py) — purely additive, no tests break.
2. Update model.py to support both old and new fields temporarily.
3. Rewrite strength_updater.py to use EKF (update its tests simultaneously).
4. Rewrite event_handlers.py (update its tests simultaneously).
5. Rewrite tick_loop.py — the new 7-step pipeline (update its tests simultaneously).
6. Delete odds_consensus.py and all references. Clean up types.py (remove deprecated fields). Update all remaining tests.

---

## Task 3.1: Create src/engine/ekf.py

**What:** Implement the Extended Kalman Filter for dynamic team strength estimation. This is the v5 replacement for `InPlayStrengthUpdater`'s Bayesian shrinkage.

**Files touched:**
- `src/engine/ekf.py` — new file

**Detailed steps:**
1. Create `EKFStrengthTracker` class:
```python
class EKFStrengthTracker:
    """Extended Kalman Filter for live team strength estimation.

    State: a_H(t), a_A(t) — current strength estimates
    Uncertainty: P_H(t), P_A(t) — current estimate variance
    """
    def __init__(self, a_H_init, a_A_init, P_0, sigma_omega_sq): ...

    def predict(self, dt: float) -> None:
        """Prediction step: P_i += σ²_ω × dt"""

    def update_goal(self, team: str, lambda_H: float, lambda_A: float, dt: float) -> None:
        """Update on goal: K = P/(P·λ + 1); a += K·(1 - λ·dt)"""

    def update_no_goal(self, lambda_H: float, lambda_A: float, dt: float) -> None:
        """Update on no-goal: weak negative evidence"""

    def compute_surprise_score(self, team: str, P_model_home_win: float) -> float:
        """SurpriseScore = 1 - P_model(scoring_team_wins | pre-goal state)"""

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Returns (a_H, a_A, P_H, P_A)"""
```
2. Implement EKF equations per v5 spec:
   - Prediction: `P_i(t|t-dt) = P_i(t-dt|t-dt) + σ²_ω × dt`
   - Goal update: `K = P_i / (P_i × λ_i + 1)`, `a_i += K × (1 - λ_i × dt)`, `P_i = (1 - K × λ_i) × P_i`
   - No-goal update: `K₀ = P_i × λ_i / (P_i × λ_i + 1)`, `a_i += K₀ × (0 - λ_i × dt)`
3. SurpriseScore: `1.0 - p_scoring_team_wins` (continuous float, not categorical)

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/engine/test_ekf.py` with:
  - `test_ekf_predict_increases_uncertainty` — P grows over time
  - `test_ekf_goal_update_increases_strength` — scoring team's a increases
  - `test_ekf_no_goal_decreases_strength` — prolonged no-goal slightly decreases a
  - `test_ekf_surprise_score_range` — always in [0, 1]
  - `test_ekf_surprise_score_underdog` — underdog goal → SurpriseScore > 0.5

**Verify:** `make test`

---

## Task 3.2: Create src/engine/dom_index.py

**What:** Implement DomIndex fallback for Layer 2 momentum estimation.

**Files touched:**
- `src/engine/dom_index.py` — new file

**Detailed steps:**
1. Create:
```python
class DomIndex:
    """Goal-based dominance index with exponential decay.

    DomIndex(t) = Σ_home_goals exp(-κ(t-t_g)) - Σ_away_goals exp(-κ(t-t_g))
    """
    DECAY_RATE = 0.1  # κ, goals half-life ~7 min

    def __init__(self): self._goal_times: list[tuple[float, str]] = []

    def record_goal(self, t: float, team: str) -> None: ...

    def compute(self, t: float) -> float:
        """Returns raw DomIndex value."""

    def momentum_state(self, t: float) -> float:
        """Returns tanh(DomIndex) ∈ (-1, +1)"""
```

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/engine/test_dom_index.py`
  - `test_dom_index_home_goal` — home goal → positive DomIndex
  - `test_dom_index_decay` — DomIndex decays toward 0 over time
  - `test_dom_index_momentum_state_range` — always in (-1, +1)

**Verify:** `make test`

---

## Task 3.3: Create src/engine/hmm_estimator.py (stub)

**What:** Create HMM Layer 2 module with graceful degradation to DomIndex. Stub implementation — HMM parameters require recorded match data which doesn't exist yet.

**Files touched:**
- `src/engine/hmm_estimator.py` — new file

**Detailed steps:**
1. Create `HMMEstimator` class with interface:
```python
class HMMEstimator:
    """Hidden Markov Model for tactical momentum.
    Gracefully degrades to DomIndex when HMM params unavailable.
    """
    def __init__(self, hmm_params: dict | None = None):
        self._dom_index = DomIndex()
        self._hmm_available = hmm_params is not None

    def update(self, live_stats: dict | None, t: float) -> None: ...

    def record_goal(self, t: float, team: str) -> None: ...

    @property
    def state(self) -> int:
        """Returns -1, 0, or +1 (HMM) or quantized DomIndex"""

    @property
    def dom_index_value(self) -> float: ...

    def adjust_intensity(self, lambda_H: float, lambda_A: float,
                         phi_H: float = 0.0, phi_A: float = 0.0,
    ) -> tuple[float, float]:
        """λ_adj = λ × exp(φ × Z_t)"""
```
2. When `hmm_params is None`, all methods delegate to `DomIndex`.
3. Add `logger.warning("hmm_not_trained")` on first call when in degraded mode.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests: `tests/engine/test_hmm_estimator.py` — test graceful degradation to DomIndex; test adjust_intensity with φ=0 returns unchanged intensities.

**Verify:** `make test`

---

## Task 3.4: Create src/engine/kalshi_ob_sync.py

**What:** Create the Kalshi orderbook synchronization coroutine for Phase 3. Maintains live P_kalshi for each market.

**Files touched:**
- `src/engine/kalshi_ob_sync.py` — new file

**Detailed steps:**
1. Create coroutine:
```python
async def kalshi_ob_sync(model: LiveMatchModel, ws_client: KalshiWSClient) -> None:
    """Phase 3 coroutine: subscribe to Kalshi WS, maintain P_kalshi.

    Updates model.p_kalshi dict with latest mid-price for each ticker.
    Runs until model.engine_phase == "FINISHED".
    Records orderbook snapshots via recorder if attached.
    """
```
2. On each orderbook update: compute mid-price = (best_bid + best_ask) / 2 / 100 (Kalshi uses cents).
3. Store in `model.p_kalshi[market_type] = mid_price`.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests: `tests/engine/test_kalshi_ob_sync.py` — mock WS messages, verify P_kalshi updates

**Verify:** `make test`

---

## Task 3.5: Add v5 fields to TickPayload (additive)

**What:** Add EKF, HMM, and SurpriseScore fields to TickPayload with defaults. Keep v4 fields (`P_reference`, `reference_source`, `odds_consensus`) — they'll be removed in Task 3.16.

**Files touched:**
- `src/common/types.py` — add fields to TickPayload

**Detailed steps:**
1. After `a_A_current` field (~line 156), add:
```python
# v5 EKF state
ekf_P_H: float = 0.0        # EKF uncertainty for home
ekf_P_A: float = 0.0        # EKF uncertainty for away
# v5 Layer 2 state
hmm_state: int = 0           # HMM state: -1, 0, +1
dom_index: float = 0.0       # DomIndex fallback value
# v5 SurpriseScore (replaces last_goal_type eventually)
surprise_score: float = 0.0  # continuous [0, 1]
```

**Breaking changes:** None — all have defaults.

**Test impact:**
- Existing tests that break: None
- New tests to add: None (tested as part of tick_loop rewrite)

**Verify:** `make test`

---

## Task 3.6: Add v5 fields to LiveMatchModel

**What:** Add EKF tracker, HMM estimator, asymmetric delta arrays, η params, and P_kalshi dict to LiveMatchModel. Keep v4 fields for now.

**Files touched:**
- `src/engine/model.py` — add fields + update `from_phase2_result`

**Detailed steps:**
1. Add new fields to `LiveMatchModel` dataclass:
```python
# v5 EKF
ekf_tracker: EKFStrengthTracker | None = None
sigma_omega_sq: float = 0.01
# v5 Layer 2
hmm_estimator: HMMEstimator | None = None
# v5 asymmetric delta
delta_H_pos: np.ndarray | None = None
delta_H_neg: np.ndarray | None = None
delta_A_pos: np.ndarray | None = None
delta_A_neg: np.ndarray | None = None
# v5 stoppage η
eta_H: float = 0.0
eta_A: float = 0.0
eta_H2: float = 0.0
eta_A2: float = 0.0
# v5 Kalshi live prices
p_kalshi: dict[str, float] = field(default_factory=dict)
# v5 SurpriseScore
surprise_score: float = 0.0
```
2. Update `from_phase2_result` to load new params from dict when available:
```python
# After existing array loading (~line 133):
delta_H_pos = np.array(params["delta_H_pos"]) if "delta_H_pos" in params else None
delta_H_neg = np.array(params["delta_H_neg"]) if "delta_H_neg" in params else None
# ... similar for delta_A_pos/neg, eta_*, sigma_omega_sq

# Create EKF tracker
ekf_P0 = result.ekf_P0 if hasattr(result, 'ekf_P0') else sigma_a ** 2
ekf_tracker = EKFStrengthTracker(
    a_H_init=result.a_H, a_A_init=result.a_A,
    P_0=ekf_P0,
    sigma_omega_sq=params.get("sigma_omega_sq", 0.01),
)

# Create HMM estimator (stub, degrades to DomIndex)
hmm_estimator = HMMEstimator(hmm_params=None)
```

**Breaking changes:** None — all new fields have defaults. `from_phase2_result` falls back gracefully when params don't have v5 keys.

**Test impact:**
- Existing tests that break: None — `_make_params()` in tests doesn't include v5 keys → defaults used.
- New tests to add: `tests/engine/test_model.py::test_model_v5_fields` — verify EKF tracker and HMM estimator created when v5 params provided.

**Verify:** `make test`

---

## Task 3.7: Switch mc_pricing to mc_simulate_remaining_v5

**What:** Update `compute_mc_prices` to use the v5 MC function with asymmetric delta and η parameters when available.

**Files touched:**
- `src/engine/mc_pricing.py` — update `compute_mc_prices`

**Detailed steps:**
1. Import `mc_simulate_remaining_v5` from `src.math.mc_core`.
2. In `compute_mc_prices` (~lines 61-85), check if model has asymmetric delta:
```python
if model.delta_H_pos is not None:
    scores = await loop.run_in_executor(None, functools.partial(
        mc_simulate_remaining_v5,
        t_now=model.t, T_end=model.T_exp, ...,
        delta_H_pos=model.delta_H_pos, delta_H_neg=model.delta_H_neg,
        delta_A_pos=model.delta_A_pos, delta_A_neg=model.delta_A_neg,
        ..., eta_H=model.eta_H, eta_A=model.eta_A,
        eta_H2=model.eta_H2, eta_A2=model.eta_A2,
        stoppage_1_start=45.0 + model.basis_bounds[3] - 45.0,
        stoppage_2_start=90.0,
    ))
else:
    # Fall back to old function (v4 compat)
    scores = await loop.run_in_executor(None, functools.partial(
        mc_simulate_remaining, ...  # existing call
    ))
```
3. Use EKF-adjusted `a_H`, `a_A` from model (which are already updated by EKF tracker via event_handlers).

**Breaking changes:** None — falls back to old function when model doesn't have v5 params.

**Test impact:**
- Existing tests that break: None — `test_mc_pricing.py` tests don't set v5 params on model → old path taken.
- New tests to add: `tests/engine/test_mc_pricing.py::test_mc_pricing_v5_params` — model with asymmetric delta → uses v5 function.

**Verify:** `make test`

---

## Task 3.8: Switch compute_mu usage

**What:** Update `compute_mc_prices` to call `compute_remaining_mu_v5` when model has asymmetric delta.

**Files touched:**
- `src/engine/mc_pricing.py` — update mu computation call

**Detailed steps:**
1. Import `compute_remaining_mu_v5` from `src.math.compute_mu`.
2. In `compute_mc_prices` (~line 39-41), branch:
```python
if model.delta_H_pos is not None:
    mu_H, mu_A = compute_remaining_mu_v5(model)
else:
    mu_H, mu_A = compute_remaining_mu(model)
```

**Breaking changes:** None — conditional branch.

**Test impact:**
- Existing tests that break: None
- New tests to add: None (covered by test_mc_pricing_v5_params from Task 3.7)

**Verify:** `make test`

---

## Task 3.9: Rewrite strength_updater.py → EKF-based + update tests

**What:** Replace Bayesian shrinkage with EKF delegation. Replace categorical `classify_goal` with continuous `SurpriseScore`. Update tests simultaneously.

**Files touched:**
- `src/engine/strength_updater.py` — rewrite
- `tests/engine/test_strength_updater.py` — rewrite

**Detailed steps:**
1. Rewrite `InPlayStrengthUpdater` to delegate to `EKFStrengthTracker`:
```python
class InPlayStrengthUpdater:
    """v5: Delegates to EKF for strength updates, computes SurpriseScore."""

    def __init__(self, a_H_init, a_A_init, sigma_a_sq, pre_match_home_prob,
                 ekf_tracker: EKFStrengthTracker | None = None):
        self.ekf = ekf_tracker or EKFStrengthTracker(
            a_H_init, a_A_init, P_0=sigma_a_sq, sigma_omega_sq=0.01
        )
        self.pre_match_home_prob = pre_match_home_prob
        self.n_H = 0
        self.n_A = 0

    def update_on_goal(self, team, lambda_H, lambda_A, dt=1.0):
        """EKF goal update."""
        if team == "home": self.n_H += 1
        else: self.n_A += 1
        self.ekf.update_goal(team, lambda_H, lambda_A, dt)
        return self.ekf.a_H, self.ekf.a_A

    def predict(self, dt): self.ekf.predict(dt)

    def update_no_goal(self, lambda_H, lambda_A, dt):
        self.ekf.update_no_goal(lambda_H, lambda_A, dt)

    def compute_surprise_score(self, team, P_model_home_win):
        return self.ekf.compute_surprise_score(team, P_model_home_win)

    def classify_goal(self, team):
        """Backward compat: convert SurpriseScore to categorical label."""
        score = self.compute_surprise_score(team, self.pre_match_home_prob)
        if score > 0.65: label = "SURPRISE"
        elif score < 0.40: label = "EXPECTED"
        else: label = "NEUTRAL"
        return GoalClassification(label=label, team=team,
                                  scoring_team_prob=1.0 - score)
```
2. Keep `GoalClassification`, `StrengthSnapshot` dataclasses for backward compat.
3. Keep `_shrink_factor` as a read-only method for backward compat (returns EKF-based value).
4. Rewrite tests:
   - `test_no_update_early_game` → `test_ekf_small_update_early` — small P → small K → small update
   - `test_strong_update_late_game` → `test_ekf_larger_update_late` — larger P after drift → bigger update
   - `test_classify_goal` — same thresholds (0.35, 0.60 based on scoring_team_prob) still work
   - Keep `test_zero_goals_penalized` logic but adjust expected magnitudes for EKF vs Bayesian

**Breaking changes:** `update_on_goal` signature changes (now takes `lambda_H, lambda_A, dt` instead of `mu_H_elapsed, mu_A_elapsed`). Fix included: YES — event_handlers updated in Task 3.10.

**Test impact:**
- Existing tests that break: `test_strength_updater.py` (all 3 tests) — rewritten in same task
- `test_strength_integration.py` breaks due to `update_on_goal` signature change — fixed in Task 3.10

**Verify:** `pytest tests/engine/test_strength_updater.py -v`

---

## Task 3.10: Update event_handlers for EKF + SurpriseScore

**What:** Update `handle_goal` to call EKF-based strength updater with new signature. Compute `surprise_score` (float) alongside `last_goal_type` (string, kept for backward compat). Add penalty/VAR handlers.

**Files touched:**
- `src/engine/event_handlers.py` — update `handle_goal`, add `handle_penalty`, `handle_var_review`
- `tests/engine/test_event_handlers.py` — update goal handler test

**Detailed steps:**
1. In `handle_goal` (~lines 61-67), change updater call:
```python
# Old:
# new_a_H, new_a_A = model.strength_updater.update_on_goal(
#     team, model.mu_H_elapsed, model.mu_A_elapsed)

# New: compute current intensities for EKF
lambda_H = math.exp(model.a_H + model.b[_current_basis(model.t)] +
                     model.gamma_H[model.current_state_X])
lambda_A = math.exp(model.a_A + model.b[_current_basis(model.t)] +
                     model.gamma_A[model.current_state_X])
new_a_H, new_a_A = model.strength_updater.update_on_goal(
    team, lambda_H, lambda_A, dt=1.0)

# Compute SurpriseScore
if hasattr(model.strength_updater, 'compute_surprise_score'):
    model.surprise_score = model.strength_updater.compute_surprise_score(
        team, _current_home_win_prob(model))
```
2. Keep `model.last_goal_type` assignment via `classify_goal()` for backward compat (classify_goal internally uses SurpriseScore thresholds now).
3. Add `handle_penalty`:
```python
def handle_penalty(model: LiveMatchModel, team: str, minute: int) -> None:
    """Process a penalty event — freeze orderbook until resolved."""
    model.event_state = "PENALTY_PENDING"
    model.ob_freeze = True
    logger.info("penalty_detected", team=team, minute=minute)
```
4. Add `handle_var_review`:
```python
def handle_var_review(model: LiveMatchModel, minute: int) -> None:
    """Process a VAR review event — freeze orderbook until resolved."""
    model.event_state = "VAR_REVIEW"
    model.ob_freeze = True
    logger.info("var_review_started", minute=minute)
```
5. Add helper `_current_basis(t)` to find basis index for time t using `model.basis_bounds`.
6. Update `test_event_handlers.py::test_handle_goal_updates_score` to verify `surprise_score` is set.

**Breaking changes:** `handle_goal` now computes intensities internally. Fix included: YES.

**Test impact:**
- Existing tests that break: `test_event_handlers.py::test_handle_goal_updates_score` — needs update to expect `surprise_score` field. Fix: add assertion `assert model.surprise_score >= 0.0`.
- `test_strength_integration.py` — fixed here (update `mu_H_elapsed`/`mu_A_elapsed` references to work with new EKF path)
- New tests: `test_handle_penalty`, `test_handle_var_review`

**Verify:** `pytest tests/engine/test_event_handlers.py tests/engine/test_strength_integration.py -v`

---

## Task 3.11: Update goalserve_poller for live_stats

**What:** Extract live statistics (shots, corners, dangerous attacks, possession) from Goalserve poll responses and feed to HMM estimator.

**Files touched:**
- `src/engine/goalserve_poller.py` — add live_stats extraction
- `src/engine/event_handlers.py` — add `detect_penalty_var_events`

**Detailed steps:**
1. After event dispatch (~line 98), add stats extraction:
```python
# Extract live_stats for Layer 2
live_stats = _extract_live_stats(match_data)
if live_stats and model.hmm_estimator:
    model.hmm_estimator.update(live_stats, model.t)
```
2. Add helper:
```python
def _extract_live_stats(match_data: dict) -> dict | None:
    stats = match_data.get("stats", {})
    if not stats: return None
    return {
        "shots_on_target_h": int(stats.get("shotsontarget", {}).get("localteam", 0)),
        "shots_on_target_a": int(stats.get("shotsontarget", {}).get("visitorteam", 0)),
        "corners_h": int(stats.get("corners", {}).get("localteam", 0)),
        "corners_a": int(stats.get("corners", {}).get("visitorteam", 0)),
        "possession_h": float(stats.get("possession", {}).get("localteam", 50)),
    }
```
3. Add penalty/VAR event detection in `detect_events_from_poll`:
```python
# Check for penalty events
if "penalty" in str(match_data.get("events", {})).lower():
    events.append({"type": "penalty", "team": ..., "minute": ...})
```

**Breaking changes:** None — additive behavior.

**Test impact:**
- Existing tests that break: None
- New tests to add: `test_live_stats_extraction` — verify stats parsing from mock Goalserve response

**Verify:** `make test`

---

## Task 3.12: Rewrite tick_loop — new 7-step pipeline + update tests

**What:** Remove signal hierarchy (`select_P_reference`), implement v5 7-step pipeline: time → EKF predict → Layer 2 → adjust intensities → MC → σ²_p → TickPayload. This is the central breaking change.

**Files touched:**
- `src/engine/tick_loop.py` — rewrite main loop
- `tests/engine/test_tick_loop.py` — rewrite tests

**Detailed steps:**
1. **Delete** `select_P_reference()` function entirely (~lines 136-160).
2. **Delete** consensus reference call (~lines 77-82): remove `model.odds_consensus.compute_reference()` and the `select_P_reference` call.
3. Replace main loop pipeline (~lines 70-106) with v5 7-step pipeline:
```python
# Step 1: Update effective match time
model.update_time()

# Step 2: EKF prediction step
if model.ekf_tracker:
    model.ekf_tracker.predict(dt=1.0)
    model.a_H, model.a_A = model.ekf_tracker.a_H, model.ekf_tracker.a_A

# Step 3: No-goal EKF update (weak negative evidence every tick)
if model.ekf_tracker:
    lambda_H = _compute_lambda(model, "home")
    lambda_A = _compute_lambda(model, "away")
    model.strength_updater.update_no_goal(lambda_H, lambda_A, dt=1.0)
    model.a_H, model.a_A = model.ekf_tracker.a_H, model.ekf_tracker.a_A

# Step 4: Layer 2 — HMM/DomIndex already updated by goalserve_poller

# Step 5: MC simulation (uses current a_H, a_A from EKF)
P_model, sigma_MC = await compute_mc_prices(model)

# Step 6: Compute σ²_p (total probability uncertainty)
# σ²_p = σ²_MC + σ²_model (Baker-McHale propagation)
# stored in sigma_MC for now (Phase 4 will use this)

# Step 7: Assemble TickPayload
payload = TickPayload(
    match_id=model.match_id, t=model.t,
    engine_phase=model.engine_phase,
    P_model=P_model, sigma_MC=sigma_MC,
    score=model.score, X=model.current_state_X,
    delta_S=model.delta_S, mu_H=model.mu_H, mu_A=model.mu_A,
    a_H_current=model.a_H, a_A_current=model.a_A,
    ekf_P_H=model.ekf_tracker.P_H if model.ekf_tracker else 0.0,
    ekf_P_A=model.ekf_tracker.P_A if model.ekf_tracker else 0.0,
    hmm_state=model.hmm_estimator.state if model.hmm_estimator else 0,
    dom_index=model.hmm_estimator.dom_index_value if model.hmm_estimator else 0.0,
    surprise_score=model.surprise_score,
    last_goal_type=model.last_goal_type,
    order_allowed=model.order_allowed,
    cooldown=model.cooldown, ob_freeze=model.ob_freeze,
    event_state=model.event_state,
    # v4 compat (still required by TickPayload until Task 3.16):
    odds_consensus=None,
    P_reference=P_model,        # v5: P_model IS the reference
    reference_source="model",   # always "model" in v5
)
```
4. Add helper `_compute_lambda(model, team)` to compute current intensity.
5. Update `_publish_tick_to_redis` to include v5 fields in TickMessage.
6. **Rewrite `tests/engine/test_tick_loop.py`**:
   - Delete `test_select_P_reference_high`, `test_select_P_reference_none`, `test_select_P_reference_low_disagree` — signal hierarchy removed.
   - Add `test_tick_loop_v5_pipeline` — verify P_model is sole output, no P_reference/consensus selection.
   - Add `test_tick_loop_ekf_predict_called` — verify EKF predict step runs each tick.
   - Keep `test_sleep_until_next_tick` — absolute time scheduling unchanged.

**Breaking changes:** `select_P_reference` removed. `tick_loop` no longer reads `model.odds_consensus`. Fix included: YES — all references cleaned up.

**Test impact:**
- Existing tests that break: `test_tick_loop.py` (3 tests on signal hierarchy) — deleted and replaced.
- New tests: 2 new tests for v5 pipeline.

**Verify:** `pytest tests/engine/test_tick_loop.py -v`

---

## Task 3.13: Demote odds_api_listener to logging-only

**What:** Remove the `model.odds_consensus.update_bookmaker()` call. The listener now only records raw messages for post-match analysis.

**Files touched:**
- `src/engine/odds_api_listener.py` — remove consensus update

**Detailed steps:**
1. Remove import of `OddsConsensus` if present.
2. Remove line ~84: `model.odds_consensus.update_bookmaker(bookie, implied)`.
3. Keep the WS connection, parsing, and `recorder.record_odds_api(msg)` call — recording continues.
4. Add log statement: `logger.debug("odds_recorded", bookie=bookie)` instead of updating consensus.
5. Update `tests/engine/test_odds_api_listener.py` — tests only verify parsing, not consensus update, so they should pass unchanged. Verify.

**Breaking changes:** Odds data no longer flows into model state.

**Test impact:**
- Existing tests that break: None — `test_parse_odds_update` and `test_odds_to_implied` test pure functions, not the consensus update path.
- New tests to add: None

**Verify:** `pytest tests/engine/test_odds_api_listener.py -v`

---

## Task 3.14: Delete odds_consensus.py + its tests

**What:** Remove the obsolete OddsConsensus module and its dedicated test file.

**Files touched:**
- `src/engine/odds_consensus.py` — delete
- `tests/engine/test_odds_consensus.py` — delete

**Detailed steps:**
1. Delete `src/engine/odds_consensus.py`.
2. Delete `tests/engine/test_odds_consensus.py` (5 tests).
3. Remove any remaining imports of `OddsConsensus` from other engine files (grep for `from src.engine.odds_consensus`).
4. Remove `odds_consensus` field assignment in `model.py` `from_phase2_result` (should already be None by default, just remove any explicit initialization).

**Breaking changes:** OddsConsensus class no longer exists.

**Test impact:**
- Tests removed: 5 tests from `test_odds_consensus.py`
- Net test count: -5 (offset by new tests added in earlier tasks)

**Verify:** `make test`

---

## Task 3.15: Remove v4 fields from TickPayload + update integration tests

**What:** Remove deprecated `P_reference`, `reference_source`, `odds_consensus` fields from TickPayload. Update all test code that constructs TickPayload with these fields.

**Files touched:**
- `src/common/types.py` — remove fields from TickPayload
- `tests/engine/test_strength_integration.py` — update TickPayload construction

**Detailed steps:**
1. In `types.py` TickPayload, remove:
   - `odds_consensus: OddsConsensusResult | None`
   - `P_reference: MarketProbs`
   - `reference_source: str`
2. In `test_strength_integration.py::test_surprise_goal_flows_to_tick_payload` (~lines 142-163), remove:
   - `odds_consensus=None,`
   - `P_reference=dummy_probs,`
   - `reference_source="model",`
3. Update `tick_loop.py` TickPayload construction (already done in Task 3.12 — remove the v4 compat lines added there).

**Breaking changes:** Any code constructing TickPayload with these fields will fail. Fix included: YES — all such code updated.

**Test impact:**
- Existing tests that break: `test_strength_integration.py::test_surprise_goal_flows_to_tick_payload` — fixed here.
- Other tests: `test_types.py::test_tick_payload_reference_source` — fixed in Task 3.16.

**Verify:** `pytest tests/engine/test_strength_integration.py -v`

---

## Task 3.16: Clean up types.py — remove all v4 remnants

**What:** Remove `OddsConsensusResult`, `BookmakerState` from types.py. Update `Signal` and `TickMessage` to remove consensus references. Update `test_types.py`.

**Files touched:**
- `src/common/types.py` — remove classes, update Signal/TickMessage
- `tests/test_types.py` — update/remove affected tests

**Detailed steps:**
1. Delete `BookmakerState` class.
2. Delete `OddsConsensusResult` class.
3. Update `Signal`: remove `P_reference`, `reference_source`, `consensus_confidence`. Add `P_model: float`, `surprise_score: float`.
4. Update `TickMessage`: remove `P_reference`, `reference_source`, `consensus_confidence`. Add `ekf_P_H`, `ekf_P_A`, `hmm_state`, `surprise_score`.
5. Update `test_types.py`:
   - Remove `test_tick_payload_reference_source` — field no longer exists.
   - Remove `test_odds_consensus_result` — class deleted.
   - Update `test_signal_fields` — check for `P_model` instead of `P_reference`.
   - Add `test_tick_payload_v5_fields` — verify `ekf_P_H`, `hmm_state`, `surprise_score` exist.

**Breaking changes:** Multiple classes/fields removed. Fix included: YES.

**Test impact:**
- Tests removed: `test_tick_payload_reference_source`, `test_odds_consensus_result`
- Tests updated: `test_signal_fields`
- Tests added: `test_tick_payload_v5_fields`

**Verify:** `pytest tests/test_types.py -v`

---

## Task 3.17: Update replay_server for Kalshi WS

**What:** Add mock Kalshi WebSocket endpoint to ReplayServer for offline development.

**Files touched:**
- `src/recorder/replay_server.py` — add Kalshi WS route

**Detailed steps:**
1. In `__init__`, load `kalshi_ob.jsonl` records (already done — `_kalshi_ob` is loaded).
2. In `start()`, add third server: `kalshi_ws_port: int = 8557`.
3. Add `_serve_kalshi_ws` handler (same timed delivery pattern as `_serve_odds_ws`).
4. Update `stop()` to clean up third server.

**Breaking changes:** None — additive.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/recorder/test_replay_server.py::test_replay_kalshi_ws` — verify Kalshi WS replay delivers orderbook snapshots.

**Verify:** `make test`

---

## Task 3.18: Remove odds_consensus from model.py

**What:** Remove the `odds_consensus` field from `LiveMatchModel` and any initialization code.

**Files touched:**
- `src/engine/model.py` — remove field + imports

**Detailed steps:**
1. Remove `odds_consensus: OddsConsensusResult | None = None` from LiveMatchModel fields.
2. Remove import of `OddsConsensusResult` from types.
3. Remove any initialization of `odds_consensus` in `from_phase2_result`.

**Breaking changes:** Code accessing `model.odds_consensus` will fail. Fix: all such code already removed in Tasks 3.12-3.14.

**Test impact:**
- Existing tests that break: None (all consensus references already removed)

**Verify:** `make test`

---

## Task 3.19: Final verification + CLAUDE.md update

**What:** Run full test suite, verify no v4 remnants, update CLAUDE.md and patterns.md for v5.

**Files touched:**
- `CLAUDE.md` — update architecture description
- `.claude/rules/patterns.md` — remove Pattern 1 (signal hierarchy), update Pattern 5 (Kelly)

**Detailed steps:**
1. Run `make test` — all tests must pass.
2. Grep for v4 remnants:
```bash
grep -r "OddsConsensus\|P_reference\|reference_source\|select_P_reference\|consensus_confidence\|signal hierarchy" src/ tests/
```
   Expect zero matches (or only documentation/comments).
3. Update `CLAUDE.md`:
   - Change "4 phases" to "6 phases" in architecture description.
   - Update Sprint 3 description to reflect v5 engine.
   - Mark Sprint 0-3 as complete.
4. Update `.claude/rules/patterns.md`:
   - **Delete Pattern 1** (signal hierarchy) — replaced by "P_model is sole authority".
   - **Update Pattern 2** (MarketProbs): remove "Phase 4's signal_generator decomposes per-market" → "Phase 4 uses TickPayload.P_model directly".
   - **Update Pattern 5** (Kelly): replace confidence-adjusted multipliers with SurpriseScore + Baker-McHale shrinkage.
   - **Add Pattern 8**: EKF prediction + update cycle (every tick).
5. Verify test count: should be ≥ 102 (original) minus deleted tests + new tests.

**Breaking changes:** None — documentation only.

**Test impact:**
- All tests pass.

**Verify:** `make test && make lint`

---

## Sprint 3 Dependency Graph

```
3.1 ─┬──→ 3.6 ──→ 3.7 ──→ 3.12
3.2 ─┤        ├──→ 3.8
3.3 ─┘        └──→ 3.9 ──→ 3.10 ──→ 3.12
3.4 ──→ 3.17
3.5 (independent, early)
3.11 ──→ 3.12
3.12 ──→ 3.13 ──→ 3.14 ──→ 3.15 ──→ 3.16 ──→ 3.18 ──→ 3.19

Critical path: 3.1 → 3.6 → 3.9 → 3.10 → 3.12 → 3.14 → 3.15 → 3.16 → 3.19
```

---

## Risk Callouts

### High Risk

1. **Task 3.9 (strength_updater → EKF): Numerical divergence.** The EKF's no-goal update accumulates small negative adjustments every second. If `σ²_ω` is too large or λ is miscalibrated, `a_H`/`a_A` can drift unreasonably over 90 minutes. **Watch for:** team strength drifting below -10 or above 2 (log-intensity). **Mitigation:** Clamp `a_H/a_A` to `[a_init - 1.5, a_init + 1.5]` and add an assertion in tests.

2. **Task 3.12 (tick_loop rewrite): EKF predict + no-goal update ordering.** The v5 spec says "EKF prediction step" (Step 2) and then "no-goal update" is implicit in the no-observation path. Getting the order wrong (predict after update, or double-counting) will cause P_H to grow without bound. **Watch for:** `P_H > 10.0` after 45 minutes of no goals. **Mitigation:** Log P_H every 100 ticks and assert P_H < 5.0 in integration test.

3. **Task 3.10 (event_handlers EKF call): Intensity computation.** `handle_goal` now needs to compute `λ_H` and `λ_A` at the current time, which requires knowing the current basis index, Markov state, and delta_S bin. Getting any of these wrong produces incorrect EKF updates. **Watch for:** λ values that are negative or > 1.0 per minute (unrealistic goal rate). **Mitigation:** Add `assert 0 < lambda_H < 2.0` guard in `handle_goal`.

### Medium Risk

4. **Task 1.3 (mc_core refactor): Numba JIT cache invalidation.** Adding `_mc_simulate_core` and `mc_simulate_remaining_v5` alongside the old function creates 3 JIT-compiled functions. If the cache (`cache=True`) contains stale entries from the old single-function version, Numba may fail to recompile. **Mitigation:** Delete `__pycache__` and `.nbi`/`.nbc` cache files before testing. Add a note to CI.

5. **Task 2.3 (Shin vig removal): Backsolve result shift.** Shin method gives slightly different implied probabilities than proportional normalization, which shifts the `a_H`/`a_A` backsolve result. While test assertions use wide ranges (`-6.0 < a_H < -2.0`), edge cases with extreme odds could fall outside. **Mitigation:** Run `test_backsolve_basic` immediately after change; widen tolerance if needed.

6. **Task 3.15 (remove P_reference from TickPayload): Downstream consumers.** Any code that reads `TickPayload.P_reference` (dashboard, recording analysis scripts) will break. The main codebase is covered, but ad-hoc scripts or notebooks may not be. **Mitigation:** Grep the entire repo for `P_reference` before removing.

### Low Risk

7. **Task 1.8 (save_production_params schema change): DB migration.** Changing gamma/delta from float columns to JSON text requires either ALTER TABLE or DROP+CREATE. Since no production data exists yet, this is low risk. **Mitigation:** Add a `CREATE TABLE IF NOT EXISTS` or migration script.

8. **Task 3.3 (HMM stub): Deferred functionality.** The HMM is a stub that degrades to DomIndex. If someone later implements HMM without updating the interface contract, the graceful degradation may silently produce wrong results. **Mitigation:** Add a `logger.warning("hmm_not_trained")` on every call when in degraded mode.
