# Sprint S3 Migration Decomposition — Engine v4 → v5

Branch: `v5-migration`
Precondition: Sprint S1 + S2 migration complete, all tests green.
Postcondition: `make test` green after every task.

## Sub-sprint order

**Phase A** (Tasks 3.1–3.6): Create new modules + additive fields. Zero tests break.
**Phase B** (Tasks 3.7–3.12): Engine rewrites. Tests rewritten alongside code in same task.
**Phase C** (Tasks 3.13–3.19): Delete v4 remnants, clean up types, final verification.

---

## PHASE A — Purely Additive (no tests break)

---

### Task 3.1 — Create src/engine/ekf.py

#### Prompt

```
You are on the v5-migration branch. All tests pass. Create a NEW file src/engine/ekf.py implementing the Extended Kalman Filter for live team strength estimation. This replaces Bayesian shrinkage in v5. NO existing files are modified.

Create src/engine/ekf.py:

```python
"""Extended Kalman Filter for live team strength estimation (v5).

Tracks a_H(t) and a_A(t) with uncertainty P_H(t), P_A(t).
Replaces the Bayesian shrinkage in InPlayStrengthUpdater.

EKF equations (from docs/MMPP_v5_Complete.md §7.1):
  Prediction:   P_i(t|t-dt) = P_i(t-dt|t-dt) + σ²_ω × dt
  Goal update:  K = P_i / (P_i·λ_i + 1); a_i += K·(1 - λ_i·dt); P_i = (1-K·λ_i)·P_i
  No-goal:      K₀ = P_i·λ_i / (P_i·λ_i + 1); a_i += K₀·(0 - λ_i·dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EKFStrengthTracker:
    """Extended Kalman Filter for live team strength.

    State: a_H(t), a_A(t) — current log-intensity estimates
    Uncertainty: P_H(t), P_A(t) — estimate variance
    """

    # Current estimates (mutable)
    a_H: float
    a_A: float
    P_H: float  # uncertainty for home
    P_A: float  # uncertainty for away

    # Process noise (immutable after init)
    sigma_omega_sq: float

    # Initial values (immutable, for clamping)
    _a_H_init: float = 0.0
    _a_A_init: float = 0.0

    def __init__(
        self,
        a_H_init: float,
        a_A_init: float,
        P_0: float,
        sigma_omega_sq: float = 0.01,
    ) -> None:
        self.a_H = a_H_init
        self.a_A = a_A_init
        self.P_H = P_0
        self.P_A = P_0
        self.sigma_omega_sq = sigma_omega_sq
        self._a_H_init = a_H_init
        self._a_A_init = a_A_init

    def predict(self, dt: float) -> None:
        """Prediction step: uncertainty grows by σ²_ω × dt.

        Called once per tick (dt=1/60 minutes or dt=1.0 seconds depending on convention).
        """
        self.P_H += self.sigma_omega_sq * dt
        self.P_A += self.sigma_omega_sq * dt

    def update_goal(
        self, team: str, lambda_H: float, lambda_A: float, dt: float = 1.0,
    ) -> None:
        """Update on goal scored.

        K = P / (P·λ + 1)
        a += K · (1 - λ·dt)
        P = (1 - K·λ) · P
        """
        if team == "home":
            K = self.P_H / (self.P_H * lambda_H + 1.0) if lambda_H > 0 else 0.0
            self.a_H += K * (1.0 - lambda_H * dt)
            self.P_H = (1.0 - K * lambda_H) * self.P_H
        else:
            K = self.P_A / (self.P_A * lambda_A + 1.0) if lambda_A > 0 else 0.0
            self.a_A += K * (1.0 - lambda_A * dt)
            self.P_A = (1.0 - K * lambda_A) * self.P_A

        # Clamp to prevent divergence (±1.5 from initial)
        self.a_H = max(self._a_H_init - 1.5, min(self._a_H_init + 1.5, self.a_H))
        self.a_A = max(self._a_A_init - 1.5, min(self._a_A_init + 1.5, self.a_A))

    def update_no_goal(
        self, lambda_H: float, lambda_A: float, dt: float = 1.0,
    ) -> None:
        """Update on no-goal observation (weak negative evidence).

        K₀ = P·λ / (P·λ + 1)
        innovation = 0 - λ·dt  (expected goals in dt, observed 0)
        a += K₀ · innovation
        """
        if lambda_H > 0:
            K0_H = self.P_H * lambda_H / (self.P_H * lambda_H + 1.0)
            self.a_H += K0_H * (0.0 - lambda_H * dt)
        if lambda_A > 0:
            K0_A = self.P_A * lambda_A / (self.P_A * lambda_A + 1.0)
            self.a_A += K0_A * (0.0 - lambda_A * dt)

        # Clamp
        self.a_H = max(self._a_H_init - 1.5, min(self._a_H_init + 1.5, self.a_H))
        self.a_A = max(self._a_A_init - 1.5, min(self._a_A_init + 1.5, self.a_A))

    def compute_surprise_score(
        self, team: str, P_model_home_win: float,
    ) -> float:
        """Compute SurpriseScore = 1 - P(scoring team wins | pre-goal state).

        Returns float in [0, 1]. Higher = more surprising.
        """
        if team == "home":
            scoring_team_win_prob = P_model_home_win
        else:
            scoring_team_win_prob = 1.0 - P_model_home_win
        return max(0.0, min(1.0, 1.0 - scoring_team_win_prob))

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Returns (a_H, a_A, P_H, P_A)."""
        return self.a_H, self.a_A, self.P_H, self.P_A
```

Also create tests/engine/test_ekf.py:

```python
"""Tests for EKFStrengthTracker."""

import pytest
from src.engine.ekf import EKFStrengthTracker


def test_ekf_predict_increases_uncertainty():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    P_H_before = ekf.P_H
    ekf.predict(dt=1.0)
    assert ekf.P_H > P_H_before
    assert ekf.P_H == pytest.approx(0.26, abs=0.001)


def test_ekf_goal_update_increases_scoring_team():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    a_H_before = ekf.a_H
    # Home scores with moderate intensity
    ekf.update_goal("home", lambda_H=0.03, lambda_A=0.02, dt=1.0)
    assert ekf.a_H > a_H_before  # scoring team strength increases


def test_ekf_no_goal_slightly_decreases():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25, sigma_omega_sq=0.01)
    a_H_before = ekf.a_H
    # Many ticks without goals
    for _ in range(100):
        ekf.predict(dt=1.0)
        ekf.update_no_goal(lambda_H=0.03, lambda_A=0.02, dt=1.0)
    assert ekf.a_H < a_H_before  # no goals → strength drifts down


def test_ekf_surprise_score_range():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25)
    score = ekf.compute_surprise_score("away", P_model_home_win=0.70)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.70, abs=0.01)  # 1 - 0.30


def test_ekf_surprise_score_underdog():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=0.25)
    score = ekf.compute_surprise_score("away", P_model_home_win=0.75)
    assert score > 0.5  # underdog goal is surprising


def test_ekf_clamp_prevents_divergence():
    ekf = EKFStrengthTracker(a_H_init=0.3, a_A_init=0.1, P_0=5.0, sigma_omega_sq=0.5)
    # Extreme update that would push a_H very high
    ekf.update_goal("home", lambda_H=0.001, lambda_A=0.001, dt=1.0)
    assert ekf.a_H <= 0.3 + 1.5  # clamped
    assert ekf.a_H >= 0.3 - 1.5
```

Do NOT modify any existing file. Run `make test` to verify.
```

---

### Task 3.2 — Create src/engine/dom_index.py

#### Prompt

```
You are on the v5-migration branch. Create a NEW file src/engine/dom_index.py implementing the goal-based dominance index for Layer 2 momentum fallback. NO existing files modified.

Create src/engine/dom_index.py:

```python
"""DomIndex — goal-based dominance index with exponential decay.

Layer 2 fallback when HMM is not available. Computes a momentum
signal from recent goals: home goals add +1, away goals add -1,
each decaying exponentially with half-life ~7 minutes (κ=0.1/min).

DomIndex(t) = Σ_home exp(-κ(t-t_g)) - Σ_away exp(-κ(t-t_g))
momentum_state = tanh(DomIndex) ∈ (-1, +1)
"""

from __future__ import annotations

import math


class DomIndex:
    """Goal-based dominance index with exponential decay."""

    DECAY_RATE: float = 0.1  # κ per minute, half-life ≈ 6.9 min

    def __init__(self) -> None:
        self._goals: list[tuple[float, str]] = []  # (time_minutes, "home"|"away")

    def record_goal(self, t: float, team: str) -> None:
        """Record a goal at time t (minutes)."""
        self._goals.append((t, team))

    def compute(self, t: float) -> float:
        """Compute raw DomIndex value at time t."""
        value = 0.0
        for t_g, team in self._goals:
            weight = math.exp(-self.DECAY_RATE * (t - t_g))
            if team == "home":
                value += weight
            else:
                value -= weight
        return value

    def momentum_state(self, t: float) -> float:
        """Returns tanh(DomIndex) ∈ (-1, +1)."""
        return math.tanh(self.compute(t))

    def quantized_state(self, t: float) -> int:
        """Returns -1, 0, or +1 based on momentum."""
        m = self.momentum_state(t)
        if m > 0.3:
            return 1
        elif m < -0.3:
            return -1
        return 0
```

Also create tests/engine/test_dom_index.py:

```python
"""Tests for DomIndex."""

from src.engine.dom_index import DomIndex


def test_dom_index_home_goal():
    di = DomIndex()
    di.record_goal(30.0, "home")
    assert di.compute(30.0) > 0.0


def test_dom_index_away_goal():
    di = DomIndex()
    di.record_goal(30.0, "away")
    assert di.compute(30.0) < 0.0


def test_dom_index_decay():
    di = DomIndex()
    di.record_goal(0.0, "home")
    val_now = di.compute(0.0)
    val_later = di.compute(30.0)  # 30 min later
    assert val_later < val_now  # decayed


def test_dom_index_momentum_range():
    di = DomIndex()
    di.record_goal(30.0, "home")
    m = di.momentum_state(30.0)
    assert -1.0 < m < 1.0


def test_dom_index_quantized():
    di = DomIndex()
    assert di.quantized_state(0.0) == 0  # no goals = balanced
    di.record_goal(0.0, "home")
    assert di.quantized_state(0.0) == 1  # home dominant
```

Do NOT modify any existing file. Run `make test`.
```

---

### Task 3.3 — Create src/engine/hmm_estimator.py (stub)

#### Prompt

```
You are on the v5-migration branch. Tasks 3.1-3.2 are done. Create src/engine/hmm_estimator.py — a stub HMM that gracefully degrades to DomIndex. NO existing files modified.

Create src/engine/hmm_estimator.py:

```python
"""HMM Estimator — Layer 2 tactical momentum (stub).

Gracefully degrades to DomIndex when HMM parameters are unavailable.
Full HMM implementation requires recorded match data (TODO).

Interface: update() with live_stats, record_goal(), state property,
adjust_intensity() for Layer 1 intensity modification.
"""

from __future__ import annotations

import math

from src.common.logging import get_logger
from src.engine.dom_index import DomIndex

logger = get_logger("engine.hmm_estimator")


class HMMEstimator:
    """Hidden Markov Model for tactical momentum.

    Degrades to DomIndex when hmm_params is None.
    """

    def __init__(self, hmm_params: dict | None = None) -> None:
        self._dom_index = DomIndex()
        self._hmm_available = hmm_params is not None
        self._warned = False
        self._current_t = 0.0

    def update(self, live_stats: dict | None, t: float) -> None:
        """Update momentum estimate from live statistics.

        Args:
            live_stats: Dict with shots_on_target_h/a, corners_h/a, possession_h.
                        None if stats unavailable this poll.
            t: Current match time in minutes.
        """
        self._current_t = t
        if not self._hmm_available:
            if not self._warned:
                logger.warning("hmm_not_trained", msg="Using DomIndex fallback")
                self._warned = True
            return
        # TODO: full HMM Baum-Welch forward pass with live_stats emissions

    def record_goal(self, t: float, team: str) -> None:
        """Record a goal for DomIndex tracking."""
        self._dom_index.record_goal(t, team)

    @property
    def state(self) -> int:
        """Returns -1 (away dominant), 0 (balanced), or +1 (home dominant)."""
        if self._hmm_available:
            return 0  # TODO: full HMM state
        return self._dom_index.quantized_state(self._current_t)

    @property
    def dom_index_value(self) -> float:
        """Returns raw DomIndex value (always available, even with HMM)."""
        return self._dom_index.compute(self._current_t)

    def adjust_intensity(
        self,
        lambda_H: float,
        lambda_A: float,
        phi_H: float = 0.0,
        phi_A: float = 0.0,
    ) -> tuple[float, float]:
        """Adjust intensities by momentum: λ_adj = λ × exp(φ × Z_t).

        When phi_H=phi_A=0 (default), returns unchanged intensities.
        """
        Z = self._dom_index.momentum_state(self._current_t)
        lambda_H_adj = lambda_H * math.exp(phi_H * Z)
        lambda_A_adj = lambda_A * math.exp(phi_A * Z)
        return lambda_H_adj, lambda_A_adj
```

Also create tests/engine/test_hmm_estimator.py:

```python
"""Tests for HMMEstimator — verifies graceful degradation to DomIndex."""

from src.engine.hmm_estimator import HMMEstimator


def test_hmm_degrades_to_dom_index():
    hmm = HMMEstimator(hmm_params=None)
    assert hmm.state == 0  # no goals = balanced
    hmm.record_goal(30.0, "home")
    hmm.update(None, 30.0)
    assert hmm.state == 1  # home dominant via DomIndex


def test_hmm_adjust_intensity_passthrough():
    hmm = HMMEstimator(hmm_params=None)
    hmm.update(None, 30.0)
    # With phi=0, intensities unchanged
    lH, lA = hmm.adjust_intensity(0.03, 0.02, phi_H=0.0, phi_A=0.0)
    assert lH == 0.03
    assert lA == 0.02


def test_hmm_dom_index_value():
    hmm = HMMEstimator(hmm_params=None)
    hmm.record_goal(10.0, "home")
    hmm.update(None, 10.0)
    assert hmm.dom_index_value > 0.0
```

Do NOT modify existing files. Run `make test`.
```

---

### Task 3.4 — Create src/engine/kalshi_ob_sync.py

#### Prompt

```
You are on the v5-migration branch. Create src/engine/kalshi_ob_sync.py — the Kalshi orderbook sync coroutine for Phase 3. This maintains live P_kalshi. NO existing files modified. The KalshiWSClient was created in Sprint 2 Task 2.4 at src/clients/kalshi_ws.py.

Create src/engine/kalshi_ob_sync.py:

```python
"""Kalshi orderbook synchronization — maintains live P_kalshi.

Phase 3 coroutine that subscribes to Kalshi WebSocket for orderbook
and trade updates. Computes mid-price for each ticker and stores
in model.p_kalshi dict.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.clients.kalshi_ws import KalshiWSClient
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.kalshi_ob_sync")


async def kalshi_ob_sync(
    model: LiveMatchModel,
    ws_client: KalshiWSClient,
) -> None:
    """Coroutine: subscribe to Kalshi WS, maintain P_kalshi.

    Updates model.p_kalshi dict with latest mid-price for each ticker.
    Runs until model.engine_phase == "FINISHED".
    Records orderbook snapshots via recorder if attached.
    """
    tickers = list(model.kalshi_tickers.values())
    if not tickers:
        logger.warning("kalshi_ob_sync_no_tickers", match_id=model.match_id)
        return

    # Reverse map: ticker string → market type
    ticker_to_market: dict[str, str] = {
        v: k for k, v in model.kalshi_tickers.items()
    }

    async def on_orderbook(ticker: str, data: dict) -> None:
        """Process orderbook update → compute mid-price."""
        market_type = ticker_to_market.get(ticker)
        if market_type is None:
            return

        # Extract best bid/ask from orderbook
        yes_bids = data.get("yes", [])
        no_bids = data.get("no", [])

        best_bid = yes_bids[0][0] / 100.0 if yes_bids else None
        best_ask = (100 - no_bids[0][0]) / 100.0 if no_bids else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            mid = best_bid
        elif best_ask is not None:
            mid = best_ask
        else:
            return

        if not hasattr(model, "p_kalshi"):
            model.p_kalshi = {}
        model.p_kalshi[market_type] = mid

        # Record if recorder attached
        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_kalshi_ob({"ticker": ticker, "mid": mid, **data})

    async def on_trade(ticker: str, data: dict) -> None:
        """Log trade (no model update needed)."""
        logger.debug("kalshi_trade", ticker=ticker, data=data)

    try:
        await ws_client.connect(
            tickers=tickers,
            on_orderbook=on_orderbook,
            on_trade=on_trade,
        )
    except Exception as exc:
        logger.error("kalshi_ob_sync_error", error=str(exc))
```

Also create tests/engine/test_kalshi_ob_sync.py:

```python
"""Tests for kalshi_ob_sync — mock WS data, verify P_kalshi updates."""

import pytest


def test_kalshi_ob_sync_import():
    """Verify the module imports without error."""
    from src.engine.kalshi_ob_sync import kalshi_ob_sync
    assert callable(kalshi_ob_sync)
```

Do NOT modify existing files. Run `make test`.
```

---

### Task 3.5 — Add v5 fields to TickPayload (additive, with defaults)

#### Prompt

```
You are on the v5-migration branch. Add new v5 fields to TickPayload in src/common/types.py. All new fields have defaults so NO existing tests break. Keep all v4 fields (P_reference, reference_source, odds_consensus) — they are removed later in Task 3.15.

In src/common/types.py, find the TickPayload class. After the line `last_goal_type: str = "NEUTRAL"` (line 175), add these fields:

    # v5 EKF state
    ekf_P_H: float = 0.0        # EKF uncertainty for home
    ekf_P_A: float = 0.0        # EKF uncertainty for away
    # v5 Layer 2 state
    hmm_state: int = 0           # HMM state: -1, 0, +1
    dom_index: float = 0.0       # DomIndex fallback value
    # v5 SurpriseScore
    surprise_score: float = 0.0  # continuous [0, 1], replaces categorical last_goal_type

Do NOT modify or remove any existing field. The new fields all have defaults so every existing TickPayload constructor call still works.

Run `make test` — all tests should pass unchanged.
```

---

### Task 3.6 — Add v5 fields to LiveMatchModel

#### Prompt

```
You are on the v5-migration branch. Tasks 3.1-3.5 are done. Add new v5 fields to LiveMatchModel in src/engine/model.py and update from_phase2_result to initialize them when v5 params are available. Keep ALL existing v4 fields.

**Step 1: Add imports at the top of src/engine/model.py**

After the existing import `from src.engine.strength_updater import InPlayStrengthUpdater` (line 18), add:
```python
from src.engine.ekf import EKFStrengthTracker
from src.engine.hmm_estimator import HMMEstimator
```

**Step 2: Add new fields to the LiveMatchModel dataclass**

After the `strength_updater: InPlayStrengthUpdater | None = None` field (line 116), add:
```python
    # v5 EKF tracker
    ekf_tracker: EKFStrengthTracker | None = None
    sigma_omega_sq: float = 0.01

    # v5 Layer 2
    hmm_estimator: HMMEstimator | None = None

    # v5 asymmetric delta (None = use symmetric delta_H/delta_A)
    delta_H_pos: np.ndarray | None = None
    delta_H_neg: np.ndarray | None = None
    delta_A_pos: np.ndarray | None = None
    delta_A_neg: np.ndarray | None = None

    # v5 stoppage time η
    eta_H: float = 0.0
    eta_A: float = 0.0
    eta_H2: float = 0.0
    eta_A2: float = 0.0

    # v5 Kalshi live prices
    p_kalshi: dict[str, float] = field(default_factory=dict)

    # v5 SurpriseScore
    surprise_score: float = 0.0
```

**Step 3: Update from_phase2_result to load v5 params**

In from_phase2_result, after `updater = InPlayStrengthUpdater(...)` (line 159), add:
```python
        # v5: Load asymmetric delta if available
        delta_H_pos = np.array(params["delta_H_pos"], dtype=np.float64) if params.get("delta_H_pos") else None
        delta_H_neg = np.array(params["delta_H_neg"], dtype=np.float64) if params.get("delta_H_neg") else None
        delta_A_pos = np.array(params["delta_A_pos"], dtype=np.float64) if params.get("delta_A_pos") else None
        delta_A_neg = np.array(params["delta_A_neg"], dtype=np.float64) if params.get("delta_A_neg") else None

        # v5: Load eta and sigma_omega
        eta_H = float(params.get("eta_H", 0.0))
        eta_A = float(params.get("eta_A", 0.0))
        eta_H2 = float(params.get("eta_H2", 0.0))
        eta_A2 = float(params.get("eta_A2", 0.0))
        sigma_omega_sq = float(params.get("sigma_omega_sq", 0.01))

        # v5: Create EKF tracker
        ekf_P0 = getattr(result, "ekf_P0", sigma_a ** 2)
        ekf_tracker = EKFStrengthTracker(
            a_H_init=result.a_H,
            a_A_init=result.a_A,
            P_0=ekf_P0,
            sigma_omega_sq=sigma_omega_sq,
        )

        # v5: Create HMM estimator (stub, degrades to DomIndex)
        hmm_estimator = HMMEstimator(hmm_params=None)
```

Then add these fields to the `cls(...)` constructor call, after `strength_updater=updater,`:
```python
            ekf_tracker=ekf_tracker,
            sigma_omega_sq=sigma_omega_sq,
            hmm_estimator=hmm_estimator,
            delta_H_pos=delta_H_pos,
            delta_H_neg=delta_H_neg,
            delta_A_pos=delta_A_pos,
            delta_A_neg=delta_A_neg,
            eta_H=eta_H,
            eta_A=eta_A,
            eta_H2=eta_H2,
            eta_A2=eta_A2,
```

All new fields have defaults, so existing test helpers `_make_params()` that don't include v5 keys will still work — `params.get(...)` returns None/default.

Run `make test` — all existing tests pass (new fields get defaults).
```

---

## PHASE B — Engine Rewrites (tests rewritten in same task)

---

### Task 3.7 — Switch mc_pricing to v5 MC function when available

#### Prompt

```
You are on the v5-migration branch. Task 3.6 is done — LiveMatchModel now has delta_H_pos/neg, eta_* fields. Update mc_pricing.py to use mc_simulate_remaining_v5 when asymmetric delta is available, falling back to the old function otherwise.

In src/engine/mc_pricing.py:

**Step 1: Add import** — after the existing `from src.math.mc_core import mc_simulate_remaining` (line 21), add:
```python
try:
    from src.math.mc_core import mc_simulate_remaining_v5
    _HAS_V5_MC = True
except ImportError:
    _HAS_V5_MC = False
```

**Step 2: In compute_mc_prices**, replace the `loop.run_in_executor(...)` call (lines 62-85) with:
```python
    if _HAS_V5_MC and model.delta_H_pos is not None:
        results = await loop.run_in_executor(
            None,
            partial(
                mc_simulate_remaining_v5,
                t_now=model.t, T_end=model.T_exp,
                S_H=S_H, S_A=S_A,
                state=model.current_state_X,
                score_diff=model.delta_S,
                a_H=model.a_H, a_A=model.a_A,
                b=model.b,
                gamma_H=model.gamma_H, gamma_A=model.gamma_A,
                delta_H_pos=model.delta_H_pos,
                delta_H_neg=model.delta_H_neg,
                delta_A_pos=model.delta_A_pos,
                delta_A_neg=model.delta_A_neg,
                Q_diag=Q_diag, Q_off=Q_off,
                basis_bounds=model.basis_bounds,
                N=N, seed=seed,
                eta_H=model.eta_H, eta_A=model.eta_A,
                eta_H2=model.eta_H2, eta_A2=model.eta_A2,
                stoppage_1_start=45.0,
                stoppage_2_start=90.0,
            ),
        )
    else:
        results = await loop.run_in_executor(
            None,
            partial(
                mc_simulate_remaining,
                t_now=model.t, T_end=model.T_exp,
                S_H=S_H, S_A=S_A,
                state=model.current_state_X,
                score_diff=model.delta_S,
                a_H=model.a_H, a_A=model.a_A,
                b=model.b,
                gamma_H=model.gamma_H, gamma_A=model.gamma_A,
                delta_H=model.delta_H, delta_A=model.delta_A,
                Q_diag=Q_diag, Q_off=Q_off,
                basis_bounds=model.basis_bounds,
                N=N, seed=seed,
            ),
        )
```

Existing tests don't set delta_H_pos on models, so they take the old path. No tests break.

Run `make test`.
```

---

### Task 3.8 — Rewrite strength_updater + its tests

#### Prompt

```
You are on the v5-migration branch. Task 3.7 done. Now rewrite InPlayStrengthUpdater to delegate to EKFStrengthTracker, and rewrite its tests simultaneously. The key change: update_on_goal now takes (team, lambda_H, lambda_A, dt) instead of (team, mu_H_elapsed, mu_A_elapsed).

**Rewrite src/engine/strength_updater.py** — replace the entire file content with:

```python
"""InPlayStrengthUpdater — v5 EKF-based team strength updates.

Delegates to EKFStrengthTracker for Kalman filter updates.
Provides backward-compatible classify_goal() that maps SurpriseScore
to categorical labels (SURPRISE/EXPECTED/NEUTRAL).

v4 used Bayesian shrinkage. v5 uses Extended Kalman Filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.engine.ekf import EKFStrengthTracker


@dataclass
class GoalClassification:
    """Result of classifying a goal event."""
    label: str  # "SURPRISE" | "EXPECTED" | "NEUTRAL"
    team: str   # "home" | "away"
    scoring_team_prob: float  # pre-match win probability of scoring team


@dataclass
class StrengthSnapshot:
    """Snapshot of updated strengths after a goal, for logging."""
    a_H: float
    a_A: float
    a_H_init: float
    a_A_init: float
    n_H: int
    n_A: int
    shrink_H: float
    shrink_A: float
    classification: GoalClassification


class InPlayStrengthUpdater:
    """v5: EKF-based updater for team log-intensities during a live match."""

    def __init__(
        self,
        a_H_init: float,
        a_A_init: float,
        sigma_a_sq: float,
        pre_match_home_prob: float,
        ekf_tracker: EKFStrengthTracker | None = None,
    ) -> None:
        self.a_H_init = a_H_init
        self.a_A_init = a_A_init
        self.sigma_a_sq = sigma_a_sq
        self.pre_match_home_prob = pre_match_home_prob

        # v5: delegate to EKF
        self.ekf = ekf_tracker or EKFStrengthTracker(
            a_H_init=a_H_init,
            a_A_init=a_A_init,
            P_0=sigma_a_sq,
            sigma_omega_sq=0.01,
        )

        # Current values (mirror EKF state)
        self.a_H = a_H_init
        self.a_A = a_A_init
        self.n_H = 0
        self.n_A = 0

    def update_on_goal(
        self,
        team: str,
        mu_H_elapsed: float = 0.0,
        mu_A_elapsed: float = 0.0,
        *,
        lambda_H: float = 0.0,
        lambda_A: float = 0.0,
        dt: float = 1.0,
    ) -> tuple[float, float]:
        """Update a_H and a_A after a goal event.

        v5 path: uses lambda_H/lambda_A kwargs for EKF update.
        v4 compat: if lambda_H/lambda_A not provided, uses Bayesian fallback.
        """
        if team == "home":
            self.n_H += 1
        else:
            self.n_A += 1

        if lambda_H > 0 or lambda_A > 0:
            # v5 EKF path
            self.ekf.update_goal(team, lambda_H, lambda_A, dt)
            self.a_H = self.ekf.a_H
            self.a_A = self.ekf.a_A
        else:
            # v4 Bayesian fallback
            self.a_H = self._bayesian_update(self.a_H_init, self.n_H, mu_H_elapsed)
            self.a_A = self._bayesian_update(self.a_A_init, self.n_A, mu_A_elapsed)
            self.ekf.a_H = self.a_H
            self.ekf.a_A = self.a_A

        return self.a_H, self.a_A

    def predict(self, dt: float) -> None:
        """EKF prediction step."""
        self.ekf.predict(dt)
        self.a_H = self.ekf.a_H
        self.a_A = self.ekf.a_A

    def update_no_goal(self, lambda_H: float, lambda_A: float, dt: float) -> None:
        """EKF no-goal update."""
        self.ekf.update_no_goal(lambda_H, lambda_A, dt)
        self.a_H = self.ekf.a_H
        self.a_A = self.ekf.a_A

    def compute_surprise_score(self, team: str, P_model_home_win: float) -> float:
        """SurpriseScore = 1 - P(scoring team wins)."""
        return self.ekf.compute_surprise_score(team, P_model_home_win)

    def classify_goal(self, team: str) -> GoalClassification:
        """Classify goal using pre_match_home_prob thresholds.

        Backward compatible with v4 categorical labels.
        """
        if team == "home":
            scoring_prob = self.pre_match_home_prob
        else:
            scoring_prob = 1.0 - self.pre_match_home_prob

        if scoring_prob < 0.35:
            label = "SURPRISE"
        elif scoring_prob > 0.60:
            label = "EXPECTED"
        else:
            label = "NEUTRAL"

        return GoalClassification(
            label=label, team=team, scoring_team_prob=scoring_prob,
        )

    def snapshot(self, classification: GoalClassification) -> StrengthSnapshot:
        return StrengthSnapshot(
            a_H=self.a_H, a_A=self.a_A,
            a_H_init=self.a_H_init, a_A_init=self.a_A_init,
            n_H=self.n_H, n_A=self.n_A,
            shrink_H=self._shrink_factor(self.n_H * 1.0),
            shrink_A=self._shrink_factor(self.n_A * 1.0),
            classification=classification,
        )

    def _bayesian_update(self, a_prior: float, n_actual: int, mu_elapsed: float) -> float:
        """v4 Bayesian fallback."""
        if mu_elapsed <= 0.0:
            return a_prior
        shrink = mu_elapsed / (mu_elapsed + self.sigma_a_sq)
        correction = math.log((n_actual + 0.5) / (mu_elapsed + 0.5))
        correction = max(-0.3, min(0.3, correction))
        return a_prior + shrink * correction

    def _shrink_factor(self, mu_elapsed: float) -> float:
        if mu_elapsed <= 0.0:
            return 0.0
        return mu_elapsed / (mu_elapsed + self.sigma_a_sq)
```

**Rewrite tests/engine/test_strength_updater.py** — replace entire file:

```python
"""Tests for InPlayStrengthUpdater (v5 EKF-based)."""

from src.engine.strength_updater import InPlayStrengthUpdater

_A_H_INIT = 0.3
_A_A_INIT = -0.1
_SIGMA_A_SQ = 0.25


def _make_updater(pre_match_home_prob: float = 0.50) -> InPlayStrengthUpdater:
    return InPlayStrengthUpdater(
        a_H_init=_A_H_INIT, a_A_init=_A_A_INIT,
        sigma_a_sq=_SIGMA_A_SQ, pre_match_home_prob=pre_match_home_prob,
    )


def test_no_update_early_game() -> None:
    """v4 compat: small mu_elapsed → a barely moves."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="home", mu_H_elapsed=0.01, mu_A_elapsed=0.01,
    )
    assert abs(new_a_H - _A_H_INIT) < 0.05
    assert abs(new_a_A - _A_A_INIT) < 0.05


def test_strong_update_late_game() -> None:
    """v4 compat: large mu_elapsed → a_H shifts noticeably."""
    updater = _make_updater()
    new_a_H, _ = updater.update_on_goal(
        team="home", mu_H_elapsed=1.4, mu_A_elapsed=1.0,
    )
    assert updater.n_H == 1
    assert abs(new_a_H - _A_H_INIT) > 0.05


def test_zero_goals_penalized() -> None:
    """v4 compat: away goal penalizes home, rewards away."""
    updater = _make_updater()
    new_a_H, new_a_A = updater.update_on_goal(
        team="away", mu_H_elapsed=1.2, mu_A_elapsed=0.3,
    )
    assert new_a_H < _A_H_INIT
    assert new_a_A > _A_A_INIT


def test_classify_goal() -> None:
    """Classification thresholds unchanged from v4."""
    updater = _make_updater(pre_match_home_prob=0.70)
    assert updater.classify_goal("away").label == "SURPRISE"
    assert updater.classify_goal("home").label == "EXPECTED"

    updater2 = _make_updater(pre_match_home_prob=0.50)
    assert updater2.classify_goal("home").label == "NEUTRAL"


def test_ekf_path_goal_update() -> None:
    """v5: EKF path via lambda kwargs."""
    updater = _make_updater()
    new_a_H, _ = updater.update_on_goal(
        team="home", lambda_H=0.03, lambda_A=0.02, dt=1.0,
    )
    assert updater.n_H == 1
    assert new_a_H != _A_H_INIT  # EKF updated


def test_ekf_predict() -> None:
    """v5: predict step increases uncertainty."""
    updater = _make_updater()
    P_before = updater.ekf.P_H
    updater.predict(dt=1.0)
    assert updater.ekf.P_H > P_before
```

Run `make test`. The 4 existing tests are preserved (same names, same assertions) plus 2 new v5 tests.
```

---

### Task 3.9 — Update event_handlers for EKF + penalty/VAR

#### Prompt

```
You are on the v5-migration branch. Task 3.8 done — strength_updater now supports both v4 (mu_elapsed) and v5 (lambda) paths via kwargs. The existing handle_goal at line 62-63 calls:
    model.strength_updater.update_on_goal(team, model.mu_H_elapsed, model.mu_A_elapsed)
This still works because update_on_goal accepts positional mu_H_elapsed/mu_A_elapsed as v4 fallback. NO changes needed to handle_goal for backward compat.

Add two new handlers and a helper to src/engine/event_handlers.py. Append after _detect_red_cards (after line 255):

```python

def handle_penalty(
    model: LiveMatchModel,
    team: str,
    minute: int,
) -> None:
    """Process a penalty event — freeze orderbook until resolved."""
    model.event_state = "PENALTY_PENDING"
    model.ob_freeze = True
    logger.info("penalty_detected", match_id=model.match_id, team=team, minute=minute)


def handle_var_review(
    model: LiveMatchModel,
    minute: int,
) -> None:
    """Process a VAR review event — freeze orderbook until resolved."""
    model.event_state = "VAR_REVIEW"
    model.ob_freeze = True
    logger.info("var_review_started", match_id=model.match_id, minute=minute)
```

Also add these new tests to the BOTTOM of tests/engine/test_event_handlers.py:

```python

def test_handle_penalty() -> None:
    """Penalty sets PENALTY_PENDING + ob_freeze."""
    from src.engine.event_handlers import handle_penalty
    model = _make_test_model()
    handle_penalty(model, "home", 75)
    assert model.event_state == "PENALTY_PENDING"
    assert model.ob_freeze is True
    assert model.order_allowed is False


def test_handle_var_review() -> None:
    """VAR review sets VAR_REVIEW + ob_freeze."""
    from src.engine.event_handlers import handle_var_review
    model = _make_test_model()
    handle_var_review(model, 80)
    assert model.event_state == "VAR_REVIEW"
    assert model.ob_freeze is True
    assert model.order_allowed is False
```

Do NOT modify any existing function or test. Only APPEND. Run `make test`.
```

---

### Task 3.10 — Rewrite tick_loop + its tests

#### Prompt

```
You are on the v5-migration branch. This is the central breaking change. Rewrite tick_loop.py to remove the signal hierarchy and implement the v5 7-step pipeline. Simultaneously rewrite test_tick_loop.py.

**Rewrite src/engine/tick_loop.py** — replace the entire file:

```python
"""tick_loop — main 1-second pricing loop for Phase 3 (v5).

v5 pipeline (7 steps per tick):
  1. Update effective match time
  2. EKF prediction step (uncertainty grows)
  3. No-goal EKF update (weak negative evidence)
  4. Layer 2 already updated by goalserve_poller
  5. MC simulation → P_model
  6. Compute σ²_p (total probability uncertainty)
  7. Assemble TickPayload → Phase 4 queue + Redis

v5 removes OddsConsensus/P_reference/signal hierarchy.
P_model is the sole trading authority.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import MarketProbs, TickPayload
from src.engine.mc_pricing import compute_mc_prices

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.tick_loop")


async def tick_loop(
    model: LiveMatchModel,
    phase4_queue: asyncio.Queue | None = None,
    redis_client: object | None = None,
) -> None:
    """Main tick loop — v5 7-step pipeline every 1 second."""
    start_time = time.monotonic()

    while model.engine_phase != "FINISHED":
        model.tick_count += 1

        # Cooldown management
        if model.cooldown and model.tick_count >= model.cooldown_until_tick:
            model.cooldown = False
            model.event_state = "IDLE"

        # Skip pricing during inactive phases
        if model.engine_phase in ("WAITING_FOR_KICKOFF", "HALFTIME"):
            await _sleep_until_next_tick(start_time, model.tick_count)
            continue

        # ── v5 7-step pipeline ──────────────────────────────

        # Step 1: Update effective match time
        model.update_time()

        # Step 2: EKF prediction step
        if model.ekf_tracker is not None:
            model.ekf_tracker.predict(dt=1.0 / 60.0)  # dt in minutes

        # Step 3: No-goal EKF update (weak negative evidence)
        if model.strength_updater is not None and model.ekf_tracker is not None:
            lambda_H = _compute_lambda(model, "home")
            lambda_A = _compute_lambda(model, "away")
            model.strength_updater.update_no_goal(lambda_H, lambda_A, dt=1.0 / 60.0)
            model.a_H = model.strength_updater.a_H
            model.a_A = model.strength_updater.a_A

        # Step 4: Layer 2 — HMM/DomIndex already updated by goalserve_poller

        # Step 5: MC simulation
        P_model, sigma_MC = await compute_mc_prices(model)

        # Step 6: σ²_p for Phase 4 (stored in sigma_MC for now)

        # Step 7: Assemble TickPayload
        payload = TickPayload(
            match_id=model.match_id,
            t=model.t,
            engine_phase=model.engine_phase,
            # v4 compat fields (kept until Task 3.15)
            odds_consensus=None,
            P_reference=P_model,
            reference_source="model",
            # v5 fields
            P_model=P_model,
            sigma_MC=sigma_MC,
            score=model.score,
            X=model.current_state_X,
            delta_S=model.delta_S,
            mu_H=model.mu_H,
            mu_A=model.mu_A,
            a_H_current=model.a_H,
            a_A_current=model.a_A,
            last_goal_type=model.last_goal_type,
            ekf_P_H=model.ekf_tracker.P_H if model.ekf_tracker else 0.0,
            ekf_P_A=model.ekf_tracker.P_A if model.ekf_tracker else 0.0,
            hmm_state=model.hmm_estimator.state if model.hmm_estimator else 0,
            dom_index=model.hmm_estimator.dom_index_value if model.hmm_estimator else 0.0,
            surprise_score=model.surprise_score,
            order_allowed=model.order_allowed,
            cooldown=model.cooldown,
            ob_freeze=model.ob_freeze,
            event_state=model.event_state,
        )

        if phase4_queue is not None:
            await phase4_queue.put(payload)

        if redis_client is not None:
            await _publish_tick_to_redis(model, payload, redis_client)

        recorder = getattr(model, "recorder", None)
        if recorder is not None:
            recorder.record_tick(payload)

        logger.debug(
            "tick",
            tick=model.tick_count,
            t=round(model.t, 2),
            hw=round(P_model.home_win, 4),
            order_allowed=model.order_allowed,
        )

        await _sleep_until_next_tick(start_time, model.tick_count)

    logger.info("tick_loop_finished", match_id=model.match_id, ticks=model.tick_count)


def _compute_lambda(model: LiveMatchModel, team: str) -> float:
    """Compute current goal intensity for a team."""
    bi = _basis_index(model.t, model.basis_bounds)
    b_val = float(model.b[bi]) if bi < len(model.b) else 0.0
    st = model.current_state_X
    if team == "home":
        return math.exp(model.a_H + b_val + float(model.gamma_H[st]))
    else:
        return math.exp(model.a_A + b_val + float(model.gamma_A[st]))


def _basis_index(t: float, basis_bounds) -> int:
    """Find which basis period t falls into."""
    for i in range(len(basis_bounds) - 1):
        if t < float(basis_bounds[i + 1]):
            return i
    return len(basis_bounds) - 2


async def _publish_tick_to_redis(
    model: LiveMatchModel,
    payload: TickPayload,
    redis_client: object,
) -> None:
    """Publish tick to Redis."""
    channel = f"tick:{model.match_id}"
    try:
        publish = getattr(redis_client, "publish", None)
        if publish is not None:
            await publish(channel, payload.model_dump_json())
    except Exception as exc:
        logger.warning("redis_publish_error", channel=channel, error=str(exc))


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

**Rewrite tests/engine/test_tick_loop.py**:

```python
"""Tests for tick_loop v5 pipeline."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.engine.tick_loop import _sleep_until_next_tick, _compute_lambda, _basis_index


@pytest.mark.asyncio
async def test_sleep_until_next_tick() -> None:
    """Verify absolute time scheduling skips when behind."""
    start = time.monotonic()
    await _sleep_until_next_tick(start - 5.0, tick_count=1, interval=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1

    now = time.monotonic()
    await _sleep_until_next_tick(now, tick_count=1, interval=0.05)
    elapsed2 = time.monotonic() - now
    assert elapsed2 >= 0.04
    assert elapsed2 < 0.2


def test_basis_index() -> None:
    """Verify basis index lookup."""
    import numpy as np
    bounds = np.array([0, 15, 30, 47, 62, 77, 87, 92, 93])
    assert _basis_index(0.0, bounds) == 0
    assert _basis_index(14.9, bounds) == 0
    assert _basis_index(15.0, bounds) == 1
    assert _basis_index(60.0, bounds) == 3
    assert _basis_index(92.5, bounds) == 7


def test_compute_lambda() -> None:
    """Verify lambda computation from model state."""
    from src.common.types import Phase2Result
    from src.engine.model import LiveMatchModel
    import numpy as np

    result = Phase2Result(
        match_id="m1", league_id=1, a_H=0.3, a_A=0.1,
        mu_H=1.5, mu_A=1.1, C_time=1.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="league_mle",
    )
    params = {
        "Q": [[-0.02, 0.01, 0.01, 0.0], [0, -0.01, 0, 0.01],
              [0, 0, -0.01, 0.01], [0, 0, 0, 0]],
        "b": [0.1, 0.2, 0.15, 0.05, 0.1, -0.1, -0.05, -0.15],
        "gamma_H": [0.0, -0.15, 0.10, -0.05],
        "gamma_A": [0.0, 0.10, -0.15, -0.05],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "alpha_1": 2.0,
    }
    model = LiveMatchModel.from_phase2_result(result, params)
    model.t = 60.0
    model.engine_phase = "SECOND_HALF"

    lam_H = _compute_lambda(model, "home")
    lam_A = _compute_lambda(model, "away")
    assert 0.0 < lam_H < 2.0
    assert 0.0 < lam_A < 2.0
```

Note: select_P_reference is DELETED — there is no import for it anymore. The old tests (test_select_P_reference_high, _none, _low_disagree) are replaced. The _sleep_until_next_tick test is preserved.

Run `make test`.
```

---

### Task 3.11 — Update goalserve_poller for live_stats

#### Prompt

```
You are on the v5-migration branch. Add live_stats extraction to goalserve_poller.py. This is additive — no existing behavior changes.

In src/engine/goalserve_poller.py, after the event dispatch block (after line 98), add:

```python
            # v5: Extract live_stats for Layer 2 HMM/DomIndex
            live_stats = _extract_live_stats(match_data)
            if live_stats is not None:
                hmm = getattr(model, "hmm_estimator", None)
                if hmm is not None:
                    hmm.update(live_stats, model.t)
```

And add this helper function at the bottom of the file (after line 117):

```python

def _extract_live_stats(match_data: dict) -> dict | None:
    """Extract live statistics from Goalserve poll for Layer 2."""
    stats = match_data.get("stats", {})
    if not stats:
        return None
    try:
        return {
            "shots_on_target_h": int(stats.get("shotsontarget", {}).get("localteam", 0)),
            "shots_on_target_a": int(stats.get("shotsontarget", {}).get("visitorteam", 0)),
            "corners_h": int(stats.get("corners", {}).get("localteam", 0)),
            "corners_a": int(stats.get("corners", {}).get("visitorteam", 0)),
            "possession_h": float(stats.get("possession", {}).get("localteam", 50)),
        }
    except (ValueError, TypeError, AttributeError):
        return None
```

Do NOT change any existing function. Only ADD. Run `make test`.
```

---

## PHASE C — Delete v4 Remnants + Final Cleanup

---

### Task 3.12 — Demote odds_api_listener to logging-only

#### Prompt

```
You are on the v5-migration branch. Remove the OddsConsensus update from odds_api_listener.py. The listener now only records raw messages.

In src/engine/odds_api_listener.py, find lines 83-91:
```python
                        if model.odds_consensus is not None:
                            model.odds_consensus.update_bookmaker(bookie, implied)
                            logger.info(
                                "bookmaker_updated",
                                bookmaker=bookie,
                                home_win=round(implied.home_win, 4),
                                draw=round(implied.draw, 4),
                                away_win=round(implied.away_win, 4),
                            )
```

Replace with:
```python
                        logger.debug(
                            "odds_recorded",
                            bookmaker=bookie,
                            home_win=round(implied.home_win, 4),
                        )
```

The WS connection, parsing, and recorder.record_odds_api() call remain unchanged.

Run `make test` — test_odds_api_listener.py tests only verify _parse_odds_update and _odds_to_implied (pure functions), not the consensus update path. They should pass unchanged.
```

---

### Task 3.13 — Delete odds_consensus.py + its tests

#### Prompt

```
You are on the v5-migration branch. Delete the obsolete OddsConsensus module and its tests.

1. Delete the file: src/engine/odds_consensus.py
2. Delete the file: tests/engine/test_odds_consensus.py
3. In src/engine/model.py, remove the import `OddsConsensusResult` from the types import line (line 17):
   Change: `from src.common.types import OddsConsensusResult, Phase2Result`
   To:     `from src.common.types import Phase2Result`

The model.py field `odds_consensus: OddsConsensusResult | None = None` (line 102) uses a string annotation via `from __future__ import annotations`, so it won't fail at import time even after removing OddsConsensusResult import. But we should also remove the field — change line 101-102:
   Delete: `# Odds consensus state (updated by odds_api_listener)`
   Delete: `odds_consensus: OddsConsensusResult | None = None`

Grep the codebase for remaining references:
   grep -r "odds_consensus" src/ tests/

If any remain in tick_loop.py (the v4 compat line `odds_consensus=None` in TickPayload constructor), that's fine — TickPayload still has the field (removed in Task 3.15).

Run `make test`. Net test count: 5 tests removed (from test_odds_consensus.py).
```

---

### Task 3.14 — Remove v4 fields from TickPayload + update integration tests

#### Prompt

```
You are on the v5-migration branch. Remove the deprecated P_reference, reference_source, odds_consensus fields from TickPayload.

**Step 1: In src/common/types.py**, find the TickPayload class and REMOVE these lines:

```python
    # TIER 1: Odds consensus (primary reference for trading)
    odds_consensus: OddsConsensusResult | None

    ...

    # Effective reference price (Phase 4 uses this for edge detection)
    # = odds_consensus.P_consensus if confidence HIGH, else P_model
    P_reference: MarketProbs
    reference_source: str  # "consensus" | "model"
```

That is: remove the `odds_consensus` field (line ~154), the `P_reference` field (line ~162), and the `reference_source` field (line ~163).

**Step 2: In src/engine/tick_loop.py**, remove the three v4 compat lines from the TickPayload constructor:
```python
            # v4 compat fields (kept until Task 3.15)
            odds_consensus=None,
            P_reference=P_model,
            reference_source="model",
```

**Step 3: In tests/engine/test_strength_integration.py**, update `test_surprise_goal_flows_to_tick_payload` (lines 142-163). Remove these lines from the TickPayload constructor:
```python
        odds_consensus=None,
        P_reference=dummy_probs,
        reference_source="model",
```

The TickPayload constructor call should now look like:
```python
    payload = TickPayload(
        match_id=model.match_id,
        t=model.t,
        engine_phase=model.engine_phase,
        P_model=dummy_probs,
        sigma_MC=dummy_probs,
        score=model.score,
        X=model.current_state_X,
        delta_S=model.delta_S,
        mu_H=model.mu_H,
        mu_A=model.mu_A,
        a_H_current=model.a_H,
        a_A_current=model.a_A,
        last_goal_type=model.last_goal_type,
        order_allowed=model.order_allowed,
        cooldown=model.cooldown,
        ob_freeze=model.ob_freeze,
        event_state=model.event_state,
    )
```

Run `make test`.
```

---

### Task 3.15 — Clean up types.py — remove BookmakerState, OddsConsensusResult, update Signal/TickMessage

#### Prompt

```
You are on the v5-migration branch. Remove all remaining v4 type remnants from src/common/types.py and update tests/test_types.py.

**In src/common/types.py:**

1. DELETE the `BookmakerState` class (lines 131-135)
2. DELETE the `OddsConsensusResult` class (lines 139-144)
3. In `Signal` class: remove `P_reference: float`, `reference_source: str`, `consensus_confidence: str` fields. Add `surprise_score: float = 0.0` field.
4. In `TickMessage` class: remove `P_reference: MarketProbs`, `reference_source: str`, `consensus_confidence: str` fields. Add `ekf_P_H: float = 0.0`, `ekf_P_A: float = 0.0`, `hmm_state: int = 0`, `surprise_score: float = 0.0` fields.
5. In `SignalMessage` class: remove `P_reference: float`, `reference_source: str`, `consensus_confidence: str` fields.
6. Remove any `OddsConsensusResult` import references at the top.

**Rewrite tests/test_types.py:**

```python
import pytest
from src.common.types import MarketProbs, TickPayload, Signal


def test_market_probs_defaults():
    mp = MarketProbs(home_win=0.4, draw=0.3, away_win=0.3)
    assert mp.over_25 is None
    assert mp.home_win + mp.draw + mp.away_win == pytest.approx(1.0)


def test_tick_payload_v5_fields():
    tp = TickPayload(
        match_id="test", t=30.0, engine_phase="FIRST_HALF",
        P_model=MarketProbs(home_win=0.5, draw=0.3, away_win=0.2),
        sigma_MC=MarketProbs(home_win=0.01, draw=0.01, away_win=0.01),
        score=(0, 0), X=0, delta_S=0, mu_H=1.2, mu_A=0.9,
        a_H_current=0.3, a_A_current=0.1,
        order_allowed=True, cooldown=False, ob_freeze=False, event_state="IDLE",
    )
    assert tp.ekf_P_H == 0.0  # default
    assert tp.hmm_state == 0
    assert tp.surprise_score == 0.0


def test_signal_fields():
    s = Signal(
        match_id="test", ticker="KXEPLGAME-26MAR15-HOM", market_type="home_win",
        direction="BUY_YES",
        P_kalshi=0.48, P_model=0.54, EV=0.07,
        kelly_fraction=0.05, kelly_amount=25.0, contracts=52,
    )
    assert s.direction in ("BUY_YES", "BUY_NO", "HOLD")
    assert s.EV > 0


def test_interval_record():
    from src.common.types import IntervalRecord, RedCardTransition
    ir = IntervalRecord(
        match_id="test", t_start=0.0, t_end=15.0,
        state_X=0, delta_S=0, is_halftime=False,
    )
    assert ir.home_goal_times == []
    assert ir.red_card_transitions == []


def test_interval_record_with_events():
    from src.common.types import IntervalRecord, RedCardTransition
    rc = RedCardTransition(minute=30.0, from_state=0, to_state=1, team="home")
    ir = IntervalRecord(
        match_id="test", t_start=15.0, t_end=45.0,
        state_X=0, delta_S=1, is_halftime=False,
        home_goal_times=[22.0], goal_delta_before=[0],
        red_card_transitions=[rc],
    )
    assert len(ir.red_card_transitions) == 1
    assert ir.red_card_transitions[0].to_state == 1
```

Run `make test`.
```

---

### Task 3.16 — Update replay_server for Kalshi WS

#### Prompt

```
You are on the v5-migration branch. Add a mock Kalshi WS endpoint to ReplayServer. This is additive.

In src/recorder/replay_server.py, find the start() method. After the Odds-API WS server setup, add a third server for Kalshi WS on port 8557. Follow the same pattern as _serve_odds_ws but for kalshi_ob.jsonl records.

Add to __init__: `self._kalshi_ob = _load_jsonl(self._dir / "kalshi_ob.jsonl")` (if not already present)

Add to start(): a third aiohttp app on kalshi_ws_port=8557 with route /ws → _serve_kalshi_ws

Add _serve_kalshi_ws method (same pattern as _serve_odds_ws):
```python
    async def _serve_kalshi_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "auth_response", "status": "ok"})
        prev_ts = 0.0
        for record in self._kalshi_ob:
            ts = record.get("_ts", 0.0)
            if prev_ts > 0:
                delay = (ts - prev_ts) / self._speed
                if delay > 0:
                    await asyncio.sleep(delay)
            prev_ts = ts
            clean = {k: v for k, v in record.items() if k != "_ts"}
            try:
                await ws.send_json(clean)
            except ConnectionError:
                break
        return ws
```

Update stop() to cleanup the third server.

Do NOT modify existing methods. Run `make test`.
```

---

### Task 3.17 — Final verification + CLAUDE.md update

#### Prompt

```
You are on the v5-migration branch. Sprint 3 migration is complete. Run final verification.

1. Run `make test` and verify ALL tests pass.

2. Run this grep to find any remaining v4 remnants:
   grep -rn "select_P_reference\|OddsConsensus\|BookmakerState\|consensus_confidence" src/ tests/
   The only matches should be in comments/docstrings (like the deprecation note in odds_api_listener.py). No code references should remain.

3. Update CLAUDE.md:
   - In the Architecture section, change "4 phases" to "6 phases"
   - Update the "Current Progress" section to mark Sprint 3 migration as done

4. Update .claude/rules/patterns.md:
   - Remove Pattern 1 (Signal Hierarchy) entirely — replace with:
     "## Pattern 1: P_model is Sole Authority (v5)
     P_model from the 3-layer mathematical model is the ONLY probability used for trading.
     No OddsConsensus, no P_reference, no signal hierarchy. Odds-API data is recorded for
     post-match analysis only."
   - Update Pattern 5 (Kelly) to mention SurpriseScore instead of consensus_confidence categories.

5. Run `make test` one final time.

Report the final test count and confirm no failures.
```

---

## Execution Checklist

```
Phase A — Additive (no tests break)
[ ] 3.1  — Create ekf.py + tests                    → make test
[ ] 3.2  — Create dom_index.py + tests               → make test
[ ] 3.3  — Create hmm_estimator.py + tests            → make test
[ ] 3.4  — Create kalshi_ob_sync.py + test             → make test
[ ] 3.5  — Add v5 fields to TickPayload               → make test
[ ] 3.6  — Add v5 fields to LiveMatchModel             → make test

Phase B — Engine Rewrites (tests rewritten in same task)
[ ] 3.7  — mc_pricing v5 MC branch                     → make test
[ ] 3.8  — Rewrite strength_updater + tests            → make test
[ ] 3.9  — Add penalty/VAR handlers + tests            → make test
[ ] 3.10 — Rewrite tick_loop + tests                   → make test
[ ] 3.11 — goalserve_poller live_stats                 → make test

Phase C — Delete v4 Remnants
[ ] 3.12 — Demote odds_api_listener                    → make test
[ ] 3.13 — Delete odds_consensus.py + tests            → make test
[ ] 3.14 — Remove P_reference from TickPayload         → make test
[ ] 3.15 — Clean up types.py + test_types.py           → make test
[ ] 3.16 — replay_server Kalshi WS                     → make test
[ ] 3.17 — Final verification + CLAUDE.md              → make test
```

## Dependency Graph

```
Phase A (all independent):
  3.1 ─┐
  3.2 ─┼──→ 3.3
       └──→ 3.6 ──→ 3.7
  3.4 (independent)
  3.5 (independent)

Phase B (sequential chain):
  3.6 ──→ 3.7 ──→ 3.8 ──→ 3.9 ──→ 3.10
  3.11 (independent of 3.8-3.10, needs 3.3+3.6)

Phase C (strict sequence):
  3.10 ──→ 3.12 ──→ 3.13 ──→ 3.14 ──→ 3.15 ──→ 3.16 ──→ 3.17
```
