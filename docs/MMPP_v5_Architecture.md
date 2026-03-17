# MMPP v5 Architecture — In-Play Soccer Trading on Kalshi

**Version:** 5.0 (2026-03-17)  
**Status:** Active — Accuracy-based edge, pivoted from speed-based  
**Replaces:** v4 architecture (cross-market lag thesis)

---

## Table of Contents

1. [Trading Thesis](#1-trading-thesis)
2. [System Overview](#2-system-overview)
3. [Data Layer](#3-data-layer)
4. [Layer 1: Goal Intensity Model](#4-layer-1-goal-intensity-model)
5. [Layer 2: In-Play State Estimator](#5-layer-2-in-play-state-estimator)
6. [Layer 3: Probability → Edge → Sizing](#6-layer-3-probability--edge--sizing)
7. [Event Handling](#7-event-handling)
8. [Phase Architecture](#8-phase-architecture)
9. [Parameter Estimation Summary](#9-parameter-estimation-summary)
10. [Validation Plan](#10-validation-plan)
11. [Implementation Phases](#11-implementation-phases)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [What's Preserved from v4](#13-whats-preserved-from-v4)

---

## 1. Trading Thesis

### 1.1 What Changed from v4

v4 assumed Betfair/bookmakers react faster than Kalshi, creating a cross-market
speed gap. Live measurement on Brentford 2-2 Wolves (2026-03-16) disproved this:

| Source | Reaction to goal | Notes |
|--------|-----------------|-------|
| Kalshi participants | 0–30s | Watching TV/streams; no suspension |
| Goalserve API | 30–60s | Polling delay |
| Odds-API bookmakers | 1–3 min | Market suspended during events |

Kalshi is **faster** than our bookmaker feeds because it has no suspension mechanism.
The speed edge does not exist with current data infrastructure.

### 1.2 New Thesis: Accuracy Edge

Kalshi participants are fast but **behaviorally biased**. Academic research identifies
three structural biases that create systematic mispricing in specific windows:

**Bias 1 — Surprise Goal Overreaction** *(Choi & Hui, 2014)*

When an underdog scores, Kalshi participants overestimate the underdog's new win
probability. The bias is approximately:

```
Bias(t) = Bias(0) × exp(−ρ × t)
```

where ρ ≈ 0.4/min (bias decays ~40% per minute, reaching ~10% after 5 minutes).
The overreaction is largest immediately after the goal and strongest when the scoring
team's pre-match probability is lowest.

**Bias 2 — Red Card Overreaction** *(Vecer et al., 2009; Titman et al., 2015)*

Markets overestimate the impact of a red card. Empirically validated multipliers:
- Penalised team scoring rate: ×0.67 (not ×0.50 as many participants assume)
- Opponent scoring rate: ×1.25 (not ×1.50 as many participants assume)

Kalshi prices typically overshoot these bounds in the 1–3 minutes post red card.

**Bias 3 — Stoppage Time Anchoring** *(Dixon & Robinson, 1998)*

Participants anchor on "90 minutes = game over" and stop adjusting prices at
approximately minute 87. Actual stoppage time averages 5–7 minutes. The MMPP
time profile shows b[7] = +0.20, meaning the stoppage period has a 22% elevated
goal intensity above baseline — a window Kalshi underprices systematically.

**Our edge:** MMPP computes probabilities without these biases. When P_model
diverges from P_kalshi in the specific windows above, we trade the correction.

### 1.3 Supporting Evidence

Single-match validation on Brentford 2-2 Wolves (2026-03-16):

| Metric | v1 (6-period, fixed a) | v2 (8-period, fixed a) | v3 (8p + strength updater) |
|--------|----------------------|----------------------|--------------------------|
| Mean \|gap\| all minutes | 3.4¢ | 3.9¢ | 4.2¢ |
| Late-game \|gap\| (78–90 min) | 6.9¢ | 7.4¢ | **4.2¢** |
| Goal 4 gap (78 min, 2-2 equaliser) | N/A | −6.1¢ | **−1.7¢** |
| Kalshi effective spread | ~1¢ | ~1¢ | ~1¢ |

Edge of 3–5¢ against 1¢ spread is tradeable. The strength updater specifically
resolved the large late-game gap caused by fixed team strengths.

### 1.4 Key Assumptions (Unvalidated)

| Assumption | Consequence if wrong |
|-----------|---------------------|
| Kalshi bias magnitude is ≥ Betfair bias (Choi & Hui measured Betfair) | Edge smaller than projected; need higher EV threshold |
| live_stats (shots, corners) available for all 8 leagues | Layer 2 HMM falls back to DomIndex |
| 307-match Kalshi backtest is representative | Paper trading may show different results |
| limit orders fill at acceptable rate | Effective edge lower after fill-rate adjustment |

**No live capital until 307-match backtest passes.** See §10 for validation plan.

---

## 2. System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA LAYER                             │
│                                                              │
│  Goalserve Pkg 1 (3-5s): scores, events, live_stats         │
│    → shots_on_target, corners, dangerous_attacks, possession │
│  Goalserve Pkg 2 (30s):  xG, detailed shots, VAR, added time│
│  Kalshi WS:              orderbook_delta + trade feed        │
│  Odds-API WS:            bookmaker odds (reference/logging)  │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│              LAYER 1: Goal Intensity λ(t)                    │
│                                                              │
│  log λ_i(t) = a_i(t) + f(t) + γ_i[X_t] + δ_i(ΔS_t)        │
│               + η_i · 1[t ∈ AT]                              │
│                                                              │
│  a_i(t)  : EKF Continuous Kalman (replaces fixed + Bayesian) │
│  f(t)    : 8-period piecewise (b[7]=+0.20 stoppage spike)    │
│  γ_i[X]  : 4-state Q matrix (red card Markov chain)         │
│  δ_i(ΔS) : Asymmetric score-state (4 params per team)        │
│  η_i     : Stoppage time multiplier (1.4–1.8× baseline)     │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│           LAYER 2: In-Play State Estimator                   │
│                                                              │
│  HMM 3-state: Z_t ∈ {−1, 0, +1}                            │
│    Observations: shots_on_target, corners,                   │
│    dangerous_attacks, possession (live_stats)                │
│  DomIndex: exponential decay momentum (fallback)             │
│  λ_adj = λ × exp(φ × Z_t)                                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│        LAYER 3: Probability → Edge → Sizing                  │
│                                                              │
│  MC Simulation (50K paths) → P_model (all markets)          │
│  Edge: P_model vs P_kalshi (model is PRIMARY)                │
│  SurpriseScore (continuous) → Kelly multiplier               │
│  Baker-McHale Kelly with σ_p uncertainty propagation        │
│  Dynamic entry threshold (adapts per tick)                   │
│  Bias decay exit timing (Choi & Hui τ*)                     │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                     EXECUTION                                 │
│  Limit order strategy (maker, not taker)                    │
│  Paper / Live executor                                       │
│  Settlement polling + P&L tracking                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Data Layer

### 3.1 Goalserve Package 1 — Live Score (3–5 second polls)

**Endpoint:** `soccernew/home?json=1`

| Field | Goalserve Path | Model Use |
|-------|---------------|-----------|
| Score | `localteam.@goals`, `visitorteam.@goals` | ΔS state |
| Match minute | `@status` (numeric string or "HT"/"FT") | Effective time t |
| Goal event | `events.event[type="goal"]` | Score update, EKF update |
| Red card | `events.event[type="yellowred"/"redcard"]` | Q state transition |
| Substitution | `events.event[type="subst"]` | Logging |
| Shots on target | `live_stats IOnTarget=home:X,away:Y` | HMM observation |
| Corners | `live_stats ICorner=home:X,away:Y` | HMM observation |
| Dangerous attacks | `live_stats IDangerousAttacks=home:X,away:Y` | HMM observation |
| Possession | `live_stats IPosession=home:X,away:Y` | HMM observation |
| Red card count | `live_stats IRedCard=home:X,away:Y` | Q state verification |
| Yellow cards | `live_stats IYellowCard=home:X,away:Y` | 2nd yellow detection |
| Penalties | `live_stats IPenalty=home:X,away:Y` | ob_freeze trigger |
| Stoppage time | `@inj_minute` | T_exp dynamic update |

**live_stats parsing:** The `live_stats.value` field is pipe-delimited:
```
"ICorner=home:5,away:2|IYellowCard=home:1,away:0|IRedCard=home:0,away:0|..."
```
Parse with:
```python
stats = {}
for pair in value.split("|"):
    key, val = pair.split("=")
    home_val, away_val = val.split(",")
    stats[key] = {
        "home": int(home_val.split(":")[1]),
        "away": int(away_val.split(":")[1]),
    }
```

**Critical:** live_stats provides shots, corners, possession every 3–5 seconds.
This enables Layer 2 HMM **without any paid external data source**.

### 3.2 Goalserve Package 2 — Live Game Stats (30-second polls)

**Endpoint:** `soccernew/stats/{match_id}`

| Field | Path | Model Use |
|-------|------|-----------|
| xG | `stats.localteam.xg` | EKF future enhancement |
| Shots detailed | `stats.localteam.shots.ongoal/offgoal/blocked` | HMM refinement |
| Possession per half | `stats.localteam.possestiontime.total_h1/h2` | Momentum |
| VAR cancelled | `summary.goals.player[].var_cancelled` | Goal rollback trigger |
| Penalty flag | `summary.goals.player[].penalty` | Penalty classification |
| Added time | `matchinfo.time.addedTime_period1/period2` | T_exp (historical only) |

### 3.3 Kalshi WebSocket

**Messages consumed:**
- `orderbook_snapshot` → full book on subscribe
- `orderbook_delta` → incremental book updates
- `trade` → executed trades (price, quantity, timestamp)

**Derived quantities:**
- `P_kalshi`: VWAP from trades in rolling 60s window
- Orderbook depth at target price (liquidity gate check)
- Staleness flag: if no update in >5s, skip this tick
- Market closed detection: rejection message → mute ticker

### 3.4 Odds-API WebSocket

**Role in v5:** Demoted to reference/monitoring only. NOT used for edge detection.

**Used for:**
- Pre-match odds → Phase 2 backsolve a_H, a_A
- In-play: logged for post-match analysis only
- Suspension events: informational, not actionable

---

## 4. Layer 1: Goal Intensity Model

### 4.1 Intensity Function

The instantaneous goal scoring rate for team i at time t:

```
log λ_i(t) = a_i(t) + f(t) + γ_i[X_t] + δ_i(ΔS_t) + η_i · 1[t ∈ AT]
```

where i ∈ {H, A} (home, away), and:

| Term | Description | Parameters |
|------|-------------|------------|
| a_i(t) | Dynamic team strength (EKF-updated) | σ²_ω, P_0 |
| f(t) | Time profile (8 piecewise periods) | b[0..7] |
| γ_i[X_t] | Red card state effect | γ_H[4], γ_A[4] |
| δ_i(ΔS_t) | Asymmetric score-state effect | δ_H⁺, δ_H⁻, δ_A⁺, δ_A⁻ |
| η_i | Stoppage time multiplier | η_H, η_A |

### 4.2 Dynamic Team Strength: Extended Kalman Filter

**Motivation:** Fixed a_H/a_A (original model) and per-goal Bayesian update (v4
strength_updater) both have the same flaw: they only update at goal events. In
reality, "40 minutes have passed and team H has scored 0 goals vs 1.3 expected"
is negative evidence that accumulates continuously. The EKF captures this.

**State model:** Team strength follows a random walk between matches and within
matches:

```
a_i(t + dt) = a_i(t) + ω(t),   ω ~ N(0, σ²_ω · dt)
```

This means uncertainty about team strength grows linearly with time in the absence
of observations, and shrinks when goals occur.

**Prediction step (every tick, dt = 1/60 minutes):**

```
a_i(t | t−dt) = a_i(t−dt | t−dt)          # state prediction (no drift)
P_i(t | t−dt) = P_i(t−dt | t−dt) + σ²_ω · dt   # uncertainty grows
```

**Update step — goal occurs at time t:**

A Poisson observation (n=1) in interval dt updates the state estimate.
Using the EKF linearisation of the Poisson log-likelihood:

```
λ_i = exp(a_i(t|t−dt) + f(t) + γ_i[X_t] + δ_i(ΔS_t))

# Kalman gain for Poisson observation
# Observation variance = λ_i (Poisson property: Var = mean)
K = P_i(t|t−dt) / (P_i(t|t−dt) · λ_i + 1)

# State update: observed 1 goal, expected λ_i·dt goals
a_i(t|t) = a_i(t|t−dt) + K · (1 − λ_i · dt)

# Uncertainty update
P_i(t|t) = (1 − K · λ_i) · P_i(t|t−dt)
```

**Update step — no goal in this tick:**

The absence of a goal is a negative observation (n=0 vs expected λ_i·dt):

```
innovation = 0 − λ_i · dt       # expected goals minus observed goals

# Kalman gain for the "no goal" observation
K_0 = P_i(t|t−dt) · λ_i / (P_i(t|t−dt) · λ_i + 1)

a_i(t|t) = a_i(t|t−dt) + K_0 · innovation
P_i(t|t) = P_i(t|t−dt)         # uncertainty unchanged (weak observation)
```

**Why the Kalman gains differ:**

At a goal event, the observation is surprising (probability λ_i·dt per tick is small),
so the update is proportionally large. The absence of a goal is expected, so the
update is small but accumulates over many ticks. The asymmetry is natural.

**Parameter definitions:**

- `σ²_ω`: innovation variance — controls how much team strength can drift within
  a match. Estimated via walk-forward MLE on historical data (see §9).
- `P_0`: initial state uncertainty — inherited from Phase 2 backsolve confidence.
  If the backsolve used high-quality Betfair closing odds, P_0 is small. If it used
  league-average fallback, P_0 is larger.

**Numerical stability:**

- Clamp P_i to [1e-6, 2.0] to prevent degenerate gains
- Clamp K, K_0 to [0, 0.95] to prevent overshooting
- Smooth EKF output with 3-tick exponential moving average to prevent jitter:
  ```
  a_i_smooth(t) = 0.7 · a_i(t) + 0.3 · a_i_smooth(t−1)
  ```

**Toggle:** `model.use_ekf = True/False` for A/B testing vs legacy strength_updater.

**Relationship to v4 strength_updater:**

The v4 `InPlayStrengthUpdater` used:
```
shrink = mu_elapsed / (mu_elapsed + σ_a²)
a_new = a_prior + shrink · log((n + 0.5) / (mu_elapsed + 0.5))
```
This is a valid empirical Bayes approximation but only updates at goal events and
does not propagate uncertainty. The EKF subsumes and supersedes it. The
strength_updater is retained as fallback when use_ekf=False.

**σ²_ω estimation procedure:**

```
For each historical match in training set:
  1. Fit a_H_first via MLE on first-half goals only
  2. Fit a_H_second via MLE on second-half goals only
  3. Record Δa_H = a_H_second − a_H_first

σ²_ω ≈ Var(Δa_H) / 45   (normalised by half-match duration in minutes)

Walk-forward CV: estimate σ²_ω on seasons 1..n, validate on season n+1.
Select σ²_ω that minimises validation Brier Score.
```

### 4.3 Time Profile f(t)

8-period piecewise constant, calibrated per league via Phase 1 NLL optimisation:

| Period | Minutes | Typical b value | Interpretation |
|--------|---------|----------------|----------------|
| b[0] | 0–15 | −0.17 | Early game suppression |
| b[1] | 15–30 | −0.13 | Building phase |
| b[2] | 30–45 | −0.10 | Pre-halftime |
| b[3] | 45–60 | +0.06 | Second half start (restart energy) |
| b[4] | 60–75 | −0.00 | Mid second half |
| b[5] | 75–85 | −0.08 | Late game suppression |
| b[6] | 85–90 | −0.05 | Kalshi freeze zone |
| b[7] | 90–T_exp | +0.20 | **Stoppage spike — primary edge zone** |

b[7] is the most important single parameter for the stoppage time edge thesis.
Kalshi participants anchor at minute 87–88 and stop updating, while the model
correctly assigns 22% elevated goal intensity in this window.

**Basis bounds construction:**
```python
alpha_1 = params.get("alpha_1", 0.0)  # first half stoppage (historical average)
basis_bounds = [
    0.0, 15.0, 30.0,
    45.0 + alpha_1,     # adjust for first-half stoppage
    60.0 + alpha_1,
    75.0 + alpha_1,
    85.0 + alpha_1,
    90.0 + alpha_1,
    T_exp               # dynamically updated by @inj_minute
]
```

### 4.4 Red Card Effect γ_i[X_t]

The Markov state X_t ∈ {0, 1, 2, 3} encodes current player counts:

| State | Description |
|-------|-------------|
| 0 | 11v11 (normal play) |
| 1 | 10v11 (home team reduced) |
| 2 | 11v10 (away team reduced) |
| 3 | 10v10 (both reduced) |

The 4-state Q matrix governs red card transitions. Transitions:
- Home red card: state 0→1, state 2→3
- Away red card: state 0→2, state 1→3

**Effect on scoring rates:**

γ_i[X_t] enters log λ_i additively, so it acts as a multiplicative factor on λ.
Expected values from Vecer et al. (2009) and Titman et al. (2015):

| State | Home λ multiplier | Away λ multiplier |
|-------|------------------|------------------|
| 11v11 (X=0) | 1.00 | 1.00 |
| 10v11 (X=1, home red) | 0.67 | 1.25 |
| 11v10 (X=2, away red) | 1.25 | 0.67 |
| 10v10 (X=3) | 0.84 | 0.84 |

These are the γ initialization values. The NLL optimisation refines them
per-league from historical data.

**Red card detection via live_stats:**

Primary: compare `IRedCard` count between consecutive polls.
Secondary (2nd yellow): `IYellowCard` increases AND `IRedCard` increases for
the same team in the same poll.

**Additivity assumption for state 3 (sparse data):**
```
q(1→3) ≈ q(0→2)   # away dismissal rate unchanged by home player count
q(2→3) ≈ q(0→1)   # home dismissal rate unchanged by away player count
```

### 4.5 Asymmetric Score-State Effect δ_i(ΔS_t)

**v4 limitation:** The symmetric parametric delta treated ΔS = +1 and ΔS = −1
symmetrically from each team's perspective. Empirically (Heuer et al. 2012,
Dixon & Robinson 1998), the effects are strongly asymmetric:

- A team leading at home plays more defensively than a team trailing at home
- Away teams trailing adopt a qualitatively different style than away teams leading
- Home advantage interacts with score state (a trailing home team attacks
  more aggressively than a trailing away team)

**v5 structure: 4 independent parameters per team**

```
δ_H⁺(ΔS):  home scoring rate when ΔS > 0 (home leading)
δ_H⁻(ΔS):  home scoring rate when ΔS < 0 (home trailing)
δ_A⁺(ΔS):  away scoring rate when ΔS > 0 (home leading, away trailing)
δ_A⁻(ΔS):  away scoring rate when ΔS < 0 (home trailing, away leading)
```

Applied to the log-intensity:
```
log λ_H(t) = a_H(t) + f(t) + γ_H[X_t] + δ_H(ΔS_t) + η_H · 1[AT]

where δ_H(ΔS) = δ_H⁺(ΔS) if ΔS > 0, else δ_H⁻(ΔS) if ΔS < 0, else 0
```

**Expected values from literature:**

| Situation | Home λ multiplier | Away λ multiplier |
|-----------|------------------|------------------|
| Home +1 (ΔS=+1) | ×0.85 (defensive) | ×1.15 (chasing) |
| Home +2 (ΔS=+2) | ×0.70 | ×1.35 |
| Away +1 (ΔS=−1) | ×1.25 (home adv + chasing) | ×0.90 |
| Away +2 (ΔS=−2) | ×1.40 | ×0.75 |

**Calibration:** Per-ΔS-bin MLE on 11,531 historical matches, with shrinkage
toward league average for sparse bins (ΔS ≤ −3 or ΔS ≥ +3 are merged).

**Scope:** δ_i(ΔS) operates on the ΔS bins {≤−2, −1, 0, +1, ≥+2} as before.
The distinction between e.g. 0-0 and 1-1 (same ΔS, different absolute scores)
is **not** modelled at this stage — absolute score effects are a future enhancement
requiring substantially more granular data.

### 4.6 Stoppage Time Multiplier η

Dixon & Robinson (1998) documented elevated goal rates during stoppage time.
Titman et al. (2015) quantified the effect at 1.4–1.8× normal intensity.

The b[7] period already captures much of this (b[7] ≈ +0.20). A separate η
term provides additive precision for the stoppage period:

```
log λ_i(t) += η_i · 1[t ∈ (45, 45+α₁)] + η_i² · 1[t ∈ (90, 90+α₂)]
```

**Calibration:** MLE on stoppage segments in historical commentaries.
Use `@inj_minute` where available; fall back to T_exp for historical data.

**T_exp dynamic update:** When Goalserve reports `@inj_minute`, update:
```python
model.update_T_exp(inj_minute)  # T_exp = max(T_exp, 90 + inj_minute)
```
This also updates `basis_bounds[-1]` to keep the b[7] period boundary correct.

---

## 5. Layer 2: In-Play State Estimator

### 5.1 Purpose and Relationship to Layer 1

Layer 1 estimates **structural team strength** (how good is each team overall).
Layer 2 estimates **current match momentum** (who is dominating right now).

A strong team can be under pressure (low momentum state). A weaker team can
be dominating through an organised defensive press. Layer 2 captures
within-match momentum from live match flow data, independent of historical
team quality estimates.

Layer 2 output adjusts Layer 1's λ multiplicatively.

### 5.2 HMM 3-State Model

**Hidden states:**
```
Z_t ∈ {−1, 0, +1}   (away dominant, balanced, home dominant)
```

**Observable signals (from Goalserve live_stats, every 3–5s):**

```python
O_t = [
    shots_on_target_home(t) − shots_on_target_away(t),   # ΔShots
    corners_home(t) − corners_away(t),                    # ΔCorners
    dangerous_attacks_home(t) − dangerous_attacks_away(t), # ΔAttacks
    possession_home(t) − 50.0,                            # ΔPoss (centred)
]
```

All quantities are delta values (change since last poll), not cumulative totals.

**State-dependent emission distributions:**

For state Z_t = z, each observation O_k ~ N(μ_k(z), σ_k²):

| Signal | Z=−1 (away dom) | Z=0 (balanced) | Z=+1 (home dom) |
|--------|-----------------|----------------|-----------------|
| ΔShots | μ < 0 | μ ≈ 0 | μ > 0 |
| ΔCorners | μ < 0 | μ ≈ 0 | μ > 0 |
| ΔAttacks | μ < 0 | μ ≈ 0 | μ > 0 |
| ΔPoss | μ < 0 | μ ≈ 0 | μ > 0 |

**HMM likelihood (standard discrete-time):**

```
L = δ · ∏_{t=1}^{T} Γ_t · diag(f_1(O_t), f_2(O_t), f_3(O_t)) · 1'
```

where δ is the initial state distribution and Γ_t is the time-varying
transition matrix.

**Goal-forced transitions:**

Goal events override the smooth HMM transition with a hard push:

```python
def on_goal(team: str, current_Z: int) -> np.ndarray:
    """Returns posterior probability vector after goal."""
    if team == "home":
        # P(Z → +1) increases substantially
        return np.array([0.10, 0.25, 0.65])
    else:
        return np.array([0.65, 0.25, 0.10])
```

**λ adjustment from HMM state:**

```
λ_H_adj(t) = λ_H(t) × exp(φ_H × Z_t)
λ_A_adj(t) = λ_A(t) × exp(φ_A × Z_t)
```

φ_H > 0: when Z_t = +1 (home dominant), home scoring rate is elevated.
φ_A < 0: when Z_t = +1 (home dominant), away scoring rate is suppressed.

Symmetry: φ_A = −φ_H if we assume the effect is symmetric across states.

**Parameter estimation:** Baum-Welch EM on historical matches with live_stats
recordings. Requires collecting live_stats data during Sprints 3+.

### 5.3 DomIndex — Simple Momentum Proxy

When HMM is not trained or live_stats are unavailable (data outage, league
without live_stats coverage), use exponential-decay momentum:

```
DomIndex(t) = Σ_{home goals} exp(−κ(t − t_g)) − Σ_{away goals} exp(−κ(t − t_g))
```

κ: decay rate parameter (recommended κ ≈ 0.1/min, meaning a goal's influence
halves in ~7 minutes).

**DomIndex → λ adjustment:**

```python
dom_adj = np.tanh(DomIndex(t))  # squash to (−1, +1)
λ_H_adj = λ_H × exp(φ_H × dom_adj)
λ_A_adj = λ_A × exp(φ_A × dom_adj)
```

**Calibration:** MLE of κ and φ on historical matches, maximising predictive
log-likelihood of second-half goals given first-half DomIndex trajectory.

### 5.4 Graceful Degradation

```
live_stats available AND HMM trained    → use HMM state Z_t
live_stats available, HMM not trained   → use DomIndex
live_stats unavailable                  → Layer 1 only (no Layer 2 adjustment)
```

Layer 2 is an enhancement, not a requirement. The system must function
correctly with Layer 1 alone.

---

## 6. Layer 3: Probability → Edge → Sizing

### 6.1 MC Simulation

Unchanged from v4: Monte Carlo simulation of remaining match using λ_adj(t).

```
N_MC   = 50,000 paths (validated: ~7.6ms with Numba JIT)
Output = MarketProbs (home_win, draw, away_win, over_25, under_25, btts_yes, btts_no)
σ_MC   = per-market Monte Carlo standard error: sqrt(p(1−p) / N_MC)
```

Run in thread executor (asyncio.get_event_loop().run_in_executor) to avoid
blocking the 1-second tick loop.

### 6.2 Edge Detection — P_model is PRIMARY

v5 removes the OddsConsensus signal hierarchy entirely. The model is the
sole reference:

```
EV_YES = P_model(home_win) − P_kalshi(home_win)
EV_NO  = (1 − P_model(home_win)) − (1 − P_kalshi(home_win))

Best direction = argmax(|EV_YES|, |EV_NO|)
Enter if best EV > θ_entry(t)
```

### 6.3 Dynamic Entry Threshold

Replace fixed 2¢/3¢ threshold with per-tick computation that adapts to
current model uncertainty:

```
θ_entry(t) = c_spread + c_slippage + z_α × σ_p(t)

where:
  c_spread   ≈ 0.010  (Kalshi effective spread, measured from trade data)
  c_slippage ≈ 0.005  (limit order: minimal; market order: higher)
  z_α        = 1.645  (95% confidence, one-tailed)
  σ_p(t)     = total probability estimation uncertainty (see §6.5)
```

This means the threshold is higher when the model is uncertain (e.g. early match,
sparse data) and lower when the model is well-calibrated (e.g. late match,
several goals already observed).

### 6.4 SurpriseScore — Continuous Goal Classification

Replaces discrete SURPRISE/EXPECTED/NEUTRAL classification from v4.

**Definition:**

```
SurpriseScore = 1 − P_model(scoring_team_wins | state immediately before goal)
```

This is computed at the moment of the goal, using the model's probability
just before the goal event updates the state.

| SurpriseScore range | Interpretation | Expected Kalshi overreaction |
|--------------------|---------------|------------------------------|
| > 0.65 | Strong underdog scores | Large (>8¢ possible) |
| 0.40–0.65 | Moderate surprise | Medium (3–8¢) |
| < 0.40 | Favourite scores | Small or none |

**Advantages over discrete classification:**

1. Accounts for current game state (a "favourite" at 2-0 down is not favoured)
2. Continuous → allows continuous Kelly multiplier scaling
3. Directly interpretable as "how surprised should the market be"

**Relationship to Choi & Hui (2014):**
The paper found that bias magnitude is positively correlated with pre-match
surprise. SurpriseScore generalises this to in-play state, which is more
appropriate for late-game situations.

### 6.5 Baker-McHale Kelly with Parameter Uncertainty Propagation

**Standard Kelly fraction:**

```
f* = (b · p̂ − q) / b

where b = (1/P_kalshi) − 1  (decimal odds minus 1),  p̂ = P_model,  q = 1 − p̂
```

**Baker-McHale shrinkage (2013):**

When p̂ is estimated with uncertainty σ²_p, the optimal fraction shrinks:

```
f_optimal = f* × (1 − σ²_p / (p̂ − p_market)²)

where p_market = P_kalshi (market price)
```

This means when model uncertainty σ²_p is high relative to the perceived edge
(p̂ − p_market), the position size shrinks toward zero automatically — without
any manual multiplier adjustments.

**Full uncertainty propagation (σ²_p):**

The total probability estimation uncertainty has two components:

```
σ²_p = σ²_MC + σ²_model

where:
  σ²_MC    = p̂(1−p̂) / N_MC              (Monte Carlo sampling error)
  σ²_model = σ²_a × (∂p̂/∂a_H)²          (team strength estimation error)
```

The partial derivative term (Delta method approximation):

```
∂p̂/∂a_H ≈ p̂(1−p̂) × μ_H_remaining
```

where μ_H_remaining is the expected remaining goals for the home team.

**Intuition:** When μ_H_remaining is large (many goals expected), a_H has a
large influence on p̂, so EKF uncertainty P_H translates into large probability
uncertainty. When μ_H_remaining is small (match nearly over), a_H uncertainty
matters less.

**Practical σ²_p computation:**

```python
sigma_MC_sq = p_model * (1 - p_model) / N_MC
sigma_model_sq = ekf_P_H * (p_model * (1 - p_model) * mu_H_remaining) ** 2
sigma_p_sq = sigma_MC_sq + sigma_model_sq

shrinkage = 1 - sigma_p_sq / max(1e-6, (p_model - p_kalshi) ** 2)
shrinkage = max(0.0, shrinkage)  # clamp to non-negative

f_optimal = f_kelly * shrinkage
```

**SurpriseScore → Kelly multiplier (continuous):**

```python
kelly_mult = alpha_base + alpha_surprise × SurpriseScore

where:
  alpha_base    = 0.10  (base fraction, conservative)
  alpha_surprise = calibrated from 307-match backtest
                   (expected range: 0.10–0.25)
```

At SurpriseScore = 0: kelly_mult = 0.10 (tenth Kelly)
At SurpriseScore = 1: kelly_mult = 0.10 + alpha_surprise (maximum)

**Combined sizing:**

```python
dollar_amount = f_optimal × kelly_mult × bankroll
contracts = floor(dollar_amount / P_kalshi)
```

**Risk limits (hard caps, all must pass):**

```
Per-order cap:    max $50
Per-match cap:    max 10% of bankroll
Total exposure:   max 20% of bankroll
Liquidity gate:   contracts ≤ orderbook depth at target price
```

### 6.6 Limit Order Strategy

Kalshi's effective spread is ~1¢ (trade-to-trade) but the visible orderbook
spread can be 20–60¢. Placing market orders against the orderbook would consume
most of the edge.

**Strategy: post limit orders at model fair value**

```
If P_model = 0.55 and Kalshi best_ask = 0.60:
  → Post limit buy order at 0.55
  → Wait for counterparty to sell into us
  → Fill at 0.55, capturing the 5¢ edge minus ~0¢ in fees
  → vs market order at 0.60: captures only ~0¢ edge
```

This converts the system into a price-maker. Fill rate will be lower than
market orders, but when orders fill, the edge is captured cleanly.

**Order lifetime management:**

```python
MAX_ORDER_LIFETIME = 30   # seconds
# If not filled: cancel and re-evaluate on next tick
# If model price has moved: cancel and re-post at new price
```

### 6.7 Exit Timing — Bias Decay Model

**Choi & Hui (2014) exponential decay:**

```
Bias(t + Δt) = Bias(0) × exp(−ρ × Δt)
```

Given entry with EV = Bias(0) and exit transaction cost c_exit:

```
Optimal hold time:
  τ* = (1/ρ) × ln(Bias(0) / (ρ × c_exit))

Expected profit at τ*:
  E[P] = Bias(0) / ρ × (1 − ρ × c_exit / Bias(0)) − c_exit
```

**ρ estimation:** Fit exponential decay to Kalshi price convergence after
goal events in 307-match dataset. Measure time from goal detection to price
stabilisation.

**6 exit triggers (first to fire wins):**

```
TRIGGER 1 — EDGE_DECAY
  Current EV < θ_exit (dynamic, computed same way as θ_entry)
  Meaning: Kalshi has converged to P_model.

TRIGGER 2 — EDGE_REVERSAL
  EV direction flipped (was BUY_YES, now P_model < P_kalshi).
  Meaning: P_model moved against us after entry.

TRIGGER 3 — POSITION_TRIM
  Position size > 2× current Kelly optimal.
  Meaning: P_model moved, position is now oversized.

TRIGGER 4 — OPPORTUNITY_COST
  Opposite direction on same market has EV > θ_entry.
  Meaning: better to flip than hold.

TRIGGER 5 — EXPIRY_EVAL
  t > 85 min. Compare expected P&L of holding to settlement
  vs exit now. Exit if expected P&L < c_exit.

TRIGGER 6 — EKF_DIVERGENCE
  EKF uncertainty P_H or P_A grew above 1.5 (model lost confidence).
  Meaning: model is not reliable, exit defensively.
```

`min_hold_ticks = 50` (~150s). `cooldown_after_exit = 100 ticks` (~5 min).

---

## 7. Event Handling

### 7.1 Event Detection Pipeline

```
Goalserve poll response (every 3s)
         ↓
detect_events_from_poll(prev_state, curr_state)
         ↓
Returns list of events:
  ├─ goal           → home or away score increased
  ├─ var_cancel     → score decreased, or var_cancelled="True" in Pkg 2
  ├─ red_card       → IRedCard count increased by 1
  ├─ second_yellow  → IYellowCard AND IRedCard both increased, same team
  ├─ penalty_decl   → IPenalty count increased (before goal appears)
  ├─ substitution   → events[type="subst"] (logged, no model impact)
  ├─ period_change  → status changed (1H→HT, HT→2H, 2H→FT)
  └─ stoppage_ann   → @inj_minute appeared or changed
```

**Multi-goal detection:** If score diff ≥ 2 in one poll, emit sequential
goal events with slight time offsets (t + 0.1, t + 0.2 etc.) to allow each
event to update the model state independently.

### 7.2 Event Handlers

**handle_goal(model, team, minute):**

```python
1. Snapshot: push (a_H, a_A, P_H, P_A, score, n_H, n_A) to rollback stack
2. Update score: model.score, model.delta_S
3. EKF update: update_on_goal(team, current λ)
4. HMM push: force Z_t toward scoring team's dominant state
5. DomIndex update: add exp(0) to scoring team's momentum sum
6. SurpriseScore: compute from pre-goal P_model
7. State: event_state = CONFIRMED, cooldown = True, cooldown_until_tick = tick + 50
8. Log: structured log with a_H_before, a_H_after, SurpriseScore
```

**handle_var_cancel(model, team, minute):**

```python
1. Rollback: pop rollback stack, restore (a_H, a_A, P_H, P_A, score, n_H, n_A)
2. HMM: revert to pre-goal transition probabilities
3. DomIndex: subtract the cancelled goal's contribution
4. State: event_state = IDLE, cooldown = False
5. Phase 4: signal any open positions to exit (edge invalidated)
6. Log: "var_cancelled", include original goal minute and team
```

**handle_penalty_declared(model, team, minute):**

```python
1. Freeze: model.ob_freeze = True
2. State: event_state = "PENALTY_PENDING"
3. Phase 4: no new orders while ob_freeze is True
4. Log: "penalty_declared"
# Do NOT adjust λ or a_H/a_A — wait for resolution
```

**handle_penalty_result(model, team, scored, minute):**

```python
1. Unfreeze: model.ob_freeze = False
2. State: event_state = IDLE
3. If scored: call handle_goal(model, team, minute)
4. If missed: log "penalty_missed", resume normal processing
```

**handle_red_card(model, team, minute):**

```python
1. Transition: update current_state_X via Q matrix transitions
   # home red: 0→1, 2→3 | away red: 0→2, 1→3
2. EKF: the new state_X will change γ_i[X_t] in λ, updating implicitly
3. State: cooldown = True, cooldown_until_tick = tick + 30
4. Log: "red_card_handled", old_state, new_state
```

**handle_period_change(model, new_phase):**

```python
if new_phase == "HALFTIME":
    model.halftime_start = time.monotonic()

elif new_phase == "SECOND_HALF":
    model.halftime_accumulated = time.monotonic() - model.halftime_start

elif new_phase == "FINISHED":
    model.engine_phase = "FINISHED"

model._last_period = new_phase
# Spam prevention: only process if new_phase != model._last_period
```

**handle_stoppage_announced(model, inj_minute):**

```python
model.update_T_exp(inj_minute)
# T_exp = max(T_exp, 90 + inj_minute)
# basis_bounds[-1] updated to new T_exp
```

### 7.3 Snapshot Stack for VAR Rollback

```python
@dataclass
class EventSnapshot:
    # EKF state
    a_H: float
    a_A: float
    ekf_P_H: float     # EKF uncertainty
    ekf_P_A: float
    # Strength updater state (legacy fallback)
    n_H: int
    n_A: int
    # Match state
    score: tuple[int, int]
    delta_S: int
    current_state_X: int
    # Timestamp
    t: float
    wall_clock: float
```

Stack size: 3 snapshots (sufficient for any realistic VAR review window).
On VAR cancel: pop most recent snapshot and restore all fields.

---

## 8. Phase Architecture

### 8.1 Phase 1: Offline Calibration

**Trigger:** Manual or weekly CRON. Running containers ignore PARAMS_UPDATED — they
keep their pinned version until match ends.

**Input:** 11,531 historical match commentaries (8 leagues, 4–6 seasons)
**Output:** Per-league `production_params` row in DB

**Steps:**

```
Step 1.1: Parse commentaries → IntervalRecord lists
Step 1.2: Q matrix MLE → red card transition rates
Step 1.3: XGBoost prior → a_H_init, a_A_init from pre-match odds features
Step 1.4: Joint NLL optimisation → b[8], γ_H[4], γ_A[4], δ_H[5], δ_A[5]
Step 1.4b: [NEW] Asymmetric δ MLE → δ_H⁺, δ_H⁻, δ_A⁺, δ_A⁻
Step 1.4c: [NEW] Stoppage time η MLE → η_H, η_A
Step 1.4d: [NEW] σ²_ω MLE → walk-forward on historical a_H drift
Step 1.5: Walk-forward CV Brier Score validation
Step 1.6: Go/No-Go → DB save
```

**New production_params fields (v5 additions):**

```python
# Asymmetric score-state parameters
delta_H_pos: list[float]  # δ_H when ΔS > 0 (5 bins: ≤-2 to ≥+2)
delta_H_neg: list[float]  # δ_H when ΔS < 0
delta_A_pos: list[float]  # δ_A when ΔS > 0
delta_A_neg: list[float]  # δ_A when ΔS < 0

# Stoppage time multipliers
eta_H: float              # first half stoppage multiplier
eta_A: float
eta_H2: float             # second half stoppage multiplier
eta_A2: float

# EKF process noise
sigma_omega_sq: float     # σ²_ω, estimated from historical a_H drift
```

### 8.2 Phase 2: Pre-Match Setup (kickoff −65 min)

**Input:** Pre-match odds (Odds-API Tier 1, Pinnacle/football-data.co.uk Tier 2)
**Output:** Phase2Result → container env vars

**Steps:**

```
1. Load production_params for this league (version pinned)
2. Fetch pre-match odds via Odds-API (Bet365 + Betfair Exchange)
3. Remove vig → market_implied probabilities
4. Backsolve a_H, a_A via scipy.optimize.minimize (Nelder-Mead)
5. Compute P_0 = backsolve confidence from residual of sanity check
6. Match Kalshi tickers via alias table
7. Sanity check: if |P_model − P_market| > 0.15 → SKIP
8. Save Phase2Result to DB
```

**Phase2Result v5 additions:**

```python
sigma_a: float            # backsolve residual → P_0 for EKF initialisation
ekf_P0: float             # initial EKF uncertainty = sigma_a² (or from residual)
```

### 8.3 Phase 3: Live Engine (kickoff to FT, 1-second loop)

**Architecture:** Three asyncio coroutines sharing LiveMatchModel (no locks needed):

```python
async def run_engine(model):
    await asyncio.gather(
        tick_loop(model, phase4_queue, redis_client),
        goalserve_poller(model),
        kalshi_ob_sync(model),          # orderbook + trade WS
    )
```

**tick_loop every 1 second:**

```
1. Update model.t from wall clock (exclude halftime)
2. EKF prediction step (every tick): grow P_H, P_A by σ²_ω · dt
3. HMM update from live_stats (every 3–5s, when new poll arrives)
4. DomIndex update
5. compute_mc_prices(model) → P_model, σ_MC (in executor thread)
6. Compute σ_p² (full uncertainty propagation)
7. Build TickPayload
8. Send to Phase 4 queue (asyncio.Queue maxsize=1)
9. Publish to Redis
10. Record tick if recorder attached
```

**TickPayload v5:**

```python
class TickPayload(BaseModel):
    match_id: str
    t: float
    engine_phase: str

    # Layer 1
    P_model: MarketProbs
    sigma_MC: MarketProbs
    a_H_current: float          # current EKF-updated home strength
    a_A_current: float
    ekf_P_H: float              # EKF uncertainty for home
    ekf_P_A: float

    # Layer 2
    hmm_state: int              # −1, 0, +1
    dom_index: float
    live_stats: LiveStatsSnapshot

    # Match state
    score: tuple[int, int]
    X: int
    delta_S: int
    mu_H: float
    mu_A: float
    surprise_score: float       # most recent goal's SurpriseScore (0 if no goal yet)

    # Trading permission
    order_allowed: bool
    cooldown: bool
    ob_freeze: bool
    event_state: str            # IDLE | CONFIRMED | PENALTY_PENDING | VAR_REVIEW
```

### 8.4 Phase 4: Execution Engine

**Input:** TickPayload queue from Phase 3 + Kalshi orderbook
**Output:** Paper/live trades, positions in DB

**Signal generation (per-tick, per-market):**

```python
for market_type in active_markets:
    p_model = getattr(payload.P_model, market_type)
    p_kalshi = ob_sync.vwap(ticker, window=60)  # 60s VWAP

    if p_kalshi is None or ob_sync.is_stale(ticker):
        continue

    ev_yes = p_model - p_kalshi
    ev_no  = (1 - p_model) - (1 - p_kalshi)

    direction = "YES" if ev_yes > ev_no else "NO"
    ev = max(ev_yes, ev_no)

    sigma_p = compute_sigma_p(payload, market_type)
    theta_entry = C_SPREAD + C_SLIPPAGE + 1.645 * sigma_p

    if ev > theta_entry and payload.order_allowed:
        kelly_mult = ALPHA_BASE + ALPHA_SURPRISE * payload.surprise_score
        f_optimal  = kelly_fraction(p_model, p_kalshi) * baker_mchale_shrinkage(...)
        size       = f_optimal * kelly_mult * bankroll
        → submit limit order
```

### 8.5 Phases 5–6: Orchestrator + Dashboard

Unchanged from v4 design except:
- No OddsConsensus dependency in Phase 3 tick_loop
- Dashboard shows: a_H_current, ekf_P_H, hmm_state, dom_index, surprise_score
- Odds-API listener retained but demoted to logging-only coroutine

---

## 9. Parameter Estimation Summary

### 9.1 Complete Parameter Table

| Parameter | Symbol | Current Value | Estimation Method | Data Source |
|-----------|--------|--------------|-------------------|-------------|
| Time profile | b[0..7] | Calibrated | NLL optimisation | 11,531 commentaries |
| Home red card | γ_H[4] | Calibrated | NLL optimisation | 11,531 commentaries |
| Away red card | γ_A[4] | Calibrated | NLL optimisation | 11,531 commentaries |
| Q matrix | Q[4×4] | Calibrated | Red card freq MLE | 11,531 commentaries |
| Home delta (sym) | δ_H[5] | Calibrated | NLL optimisation | 11,531 commentaries |
| Away delta (sym) | δ_A[5] | Calibrated | NLL optimisation | 11,531 commentaries |
| **Asym home delta lead** | **δ_H⁺** | **TODO** | **Asymmetric MLE** | 11,531 commentaries |
| **Asym home delta trail** | **δ_H⁻** | **TODO** | **Asymmetric MLE** | 11,531 commentaries |
| **Asym away delta lead** | **δ_A⁺** | **TODO** | **Asymmetric MLE** | 11,531 commentaries |
| **Asym away delta trail** | **δ_A⁻** | **TODO** | **Asymmetric MLE** | 11,531 commentaries |
| **Stoppage multiplier** | **η_H, η_A** | **TODO** | **Stoppage MLE** | 11,531 commentaries |
| **EKF process noise** | **σ²_ω** | **TODO** | **Walk-forward MLE** | 11,531 commentaries |
| Initial EKF uncertainty | P_0 | sigma_a=0.5 | From Phase 2 backsolve | Per-match |
| Correction cap | cap | 0.3 | Grid search (Brier) | 11,531 commentaries |
| **HMM emission params** | **μ_k(z), σ_k** | **TODO** | **Baum-Welch EM** | live_stats recordings |
| **HMM transition rates** | **Γ** | **TODO** | **Baum-Welch EM** | live_stats recordings |
| **HMM λ adjustment** | **φ_H, φ_A** | **TODO** | **Baum-Welch EM** | live_stats recordings |
| **DomIndex decay** | **κ** | **TODO** | **MLE** | 11,531 commentaries |
| **Bias decay rate** | **ρ** | **TODO** | **Price decay fit** | 307 Kalshi EPL events |
| **Optimal hold time** | **τ\*** | 50 ticks | **Analytic: (1/ρ)ln(...)** | From ρ |
| **Kelly base multiplier** | **α_base** | 0.10 | **Sharpe optimisation** | 307 Kalshi backtest |
| **Kelly surprise scale** | **α_surprise** | **TODO** | **Sharpe optimisation** | 307 Kalshi backtest |
| Kalshi spread cost | c_spread | 0.010 | Measured | 307 Kalshi EPL events |
| Slippage cost | c_slippage | 0.005 | Measured | 307 Kalshi EPL events |

### 9.2 σ²_ω Estimation Procedure (Detail)

This is the most novel parameter and requires careful estimation:

```
For each historical match m in training set:

  1. Segment goals into first half (t < 45) and second half (t > 45)
  
  2. Estimate a_H_first via MLE on first-half goals:
     a_H_first = argmax_a Σ_{g in H1} log λ_H(t_g | a) − ∫_0^45 λ_H(t | a) dt
  
  3. Estimate a_H_second similarly on second-half goals:
     a_H_second = argmax_a Σ_{g in H2} log λ_H(t_g | a) − ∫_45^90 λ_H(t | a) dt
  
  4. Record Δa_H(m) = a_H_second(m) − a_H_first(m)

σ²_ω_raw = Var(Δa_H) across all matches M
σ²_ω = σ²_ω_raw / 45   (normalise: 45 minutes per half)

Walk-forward CV validation:
  For n = 1, 2, ..., N_seasons:
    Estimate σ²_ω on seasons 1..n
    Validate Brier Score on season n+1 using EKF with this σ²_ω
  Select σ²_ω minimising mean validation Brier Score
```

### 9.3 ρ Estimation Procedure (Bias Decay Rate)

Requires 307-match Kalshi trade data:

```
For each goal event in the 307 matches:

  1. Record Kalshi price at goal detection time: P_kalshi(t_goal)
  2. Record Kalshi price at t_goal + 1, 2, 3, 4, 5 minutes
  3. Record MMPP P_model at t_goal
  
  4. Define "initial bias": B_0 = P_kalshi(t_goal) − P_model(t_goal)
     (positive = Kalshi overshot vs model for the scoring team)
  
  5. Fit exponential decay: B(Δt) = B_0 × exp(−ρ × Δt)
     via NLS regression across all goals and time points

ρ_final = median(per-goal ρ estimates) to avoid outlier sensitivity
```

---

## 10. Validation Plan

### 10.1 307-Match Kalshi Historical Backtest

This is the single most important validation step. The system does NOT go
live until this backtest passes.

**Data:** `data/feasibility/` — 307 EPL matches, ~1.4M Kalshi trades (Sprint -1)

**Reconstruction method:**

```python
# Step 1: Reconstruct minute-by-minute Kalshi prices
for match in matches:
    for minute in range(1, 91):
        trades_in_window = trades[(t >= minute-1) & (t < minute+0.5)]
        P_kalshi[minute] = vwap(trades_in_window)  # 90-second VWAP centred on minute

# Step 2: Run MMPP model for each match
for match in matches:
    initialize_model(a_H, a_A, b, Q, delta, eta)
    for minute in range(1, 91):
        update_events(goals_before_this_minute, red_cards_before_this_minute)
        ekf_predict_step(dt=1)
        if event_at_this_minute:
            ekf_update_step(event)
        P_model[minute] = run_mc(model_state)

# Step 3: Simulate trades
for minute, goal_event in goal_events:
    # Entry: when P_model diverges from P_kalshi
    ev = P_model[minute] - P_kalshi[minute]
    if ev > theta_entry:
        enter(direction="YES", price=P_kalshi[minute], size=kelly_size)
    
    # Exit: fixed 3-minute hold (initial backtest) then bias decay τ*
    exit(at=minute+3, price=P_kalshi[minute+3])
```

**Entry timing options to test:**

```
A. Immediate (t + 0):  enter as soon as goal detected
B. Delayed (t + 1min): enter after initial Kalshi overshoot stabilises
C. Filtered:           enter only when SurpriseScore > 0.5
```

**Success criteria (ALL must pass):**

```
CRITERION 1: Total P&L > 0 over 307 matches
CRITERION 2: Sharpe ratio > 1.0 (annualised, based on match frequency)
CRITERION 3: Win rate on SurpriseScore > 0.5 trades ≥ 55%
CRITERION 4: Mean edge realisation ≥ 0.30
  (realised P&L / theoretical EV at entry)
CRITERION 5: Max drawdown < 20% of starting bankroll
CRITERION 6: Edge significant in late-game window (min 78–90)
  Mean |gap| in stoppage time > 3¢ for at least 30% of matches
```

### 10.2 Live Paper Trading (Post-Backtest)

2-week paper trading on live matches with real Kalshi orderbook (no real money).

**Metrics to measure:**

```
METRIC 1: Limit order fill rate (target: > 40%)
METRIC 2: EV at fill vs EV at signal (target: > 70% retention)
METRIC 3: Actual P_kalshi convergence time (target: matches ρ estimate)
METRIC 4: live_stats availability across 8 leagues
METRIC 5: EKF stability (P_H, P_A should stay in [0.01, 1.5])
```

---

## 11. Implementation Phases

### Phase A — Model Core (immediate, data available)

Priority order based on expected impact vs implementation effort:

```
A1. Asymmetric δ(ΔS) MLE calibration
    Files: src/calibration/step_1_4_nll_optimize.py
    Complexity: Medium (add 4 params, asymmetric lookup)
    Expected impact: HIGH (resolves Brentford equaliser gap)

A2. live_stats parser (Goalserve pipe-delimited)
    Files: src/engine/goalserve_poller.py (new: live_stats_parser.py)
    Complexity: Low (pure parsing)
    Expected impact: Enables Layer 2

A3. SurpriseScore continuous (replace discrete classification)
    Files: src/engine/strength_updater.py, src/common/types.py
    Complexity: Low (change one method signature)
    Expected impact: MEDIUM (better Kelly scaling)

A4. Stoppage time η calibration
    Files: src/calibration/step_1_4_nll_optimize.py
    Complexity: Low (add 4 params)
    Expected impact: MEDIUM

A5. Verify on Brentford recording data
    Run analyse_inplay_accuracy.py with asymmetric δ
    Target: Goal 4 gap < 1¢
```

### Phase B — EKF + Validation

```
B1. σ²_ω estimation pipeline
    Files: src/calibration/step_1_6_sigma_omega.py (new)
    Requires: All Phase A steps complete

B2. EKF implementation (replaces strength_updater)
    Files: src/engine/ekf_updater.py (new)
    Add toggle: model.use_ekf = True/False

B3. 307-match Kalshi historical backtest
    Files: scripts/backtest_307.py (new)
    Success criteria: all 6 criteria from §10.1

B4. Baker-McHale Kelly with σ_p propagation
    Files: src/execution/kelly_sizing.py
    Requires: B2 (σ²_a from EKF)

B5. Dynamic entry threshold θ_entry(t)
    Files: src/execution/signal_generator.py
```

### Phase C — Layer 2 + Advanced

```
C1. HMM momentum estimator with live_stats
    Requires: live_stats data from multiple recorded matches (Sprint 3+)

C2. ρ estimation from Kalshi trade history
    Requires: 307-match trade data analysis

C3. Optimal exit timing τ* (bias decay model)
    Requires: ρ from C2

C4. Multi-match portfolio Kelly
    Files: src/execution/portfolio_kelly.py (new)
    Requires: Multiple simultaneous matches in production
```

### Phase D — Production

```
D1. Sprint 4 execution (limit orders, adapted from v4)
D2. Sprint 5 orchestrator (remove OddsConsensus dependency)
D3. Sprint 6 dashboard (add EKF uncertainty, HMM state displays)
D4. 2-week paper trading on live matches
D5. Live trading with real capital (post-paper-trading validation)
```

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MMPP not accurate enough to beat Kalshi in practice | Medium | Fatal | 307-match backtest gates all live trading |
| Kalshi bias effects smaller than Betfair research | Medium | High | Measure actual ρ from 307 matches before tuning α_surprise |
| live_stats not available for all leagues | Medium | Medium | DomIndex fallback; test each league during Sprint 3 recording |
| EKF σ²_ω poorly estimated → unstable updates | Low | High | Toggle EKF off; fall back to v4 strength_updater |
| Limit orders don't fill at acceptable rate | Medium | Medium | Measure fill rate in paper trading; adjust limit price strategy |
| Goalserve live_stats drops during match | Low | Low | Graceful degradation to Layer 1 only; no crash |
| VAR cancel detection delayed beyond 90s window | Low | Low | Snapshot stack has 3 entries; 90s window covers realistic VAR review |
| EKF diverges (P_H > 2.0) | Low | Medium | Hard clamp P_H to [1e-6, 1.5]; log divergence events |
| Kalshi markets closed or illiquid during edge window | Medium | Medium | Liquidity gate check before every order; skip if depth < order size |

---

## 13. What's Preserved from v4

| Component | Status | Notes |
|-----------|--------|-------|
| MMPP math core (mc_core, compute_mu, NLL) | ✅ Keep | Do NOT modify |
| Phase 1 calibration pipeline | ✅ Keep + extend | Add asymmetric δ, η, σ²_ω steps |
| Team aliases (251 teams, 8 leagues) | ✅ Keep | src/calibration/team_aliases.py |
| Goalserve client | ✅ Keep | src/clients/goalserve.py |
| Kalshi client (RSA auth) | ✅ Keep | src/clients/kalshi.py |
| Odds-API client | ✅ Keep (demoted) | src/clients/odds_api.py |
| Recording infrastructure | ✅ Keep | src/recorder/ |
| ReplayServer | ✅ Keep | src/recorder/replay_server.py |
| Phase 2 backsolve pipeline | ✅ Keep + extend | Add ekf_P0 output |
| Strength updater (v4 Bayesian) | ✅ Keep as fallback | use_ekf=False toggle |
| 8-period basis (b[7] stoppage) | ✅ Keep | Already calibrated |
| T_exp dynamic update | ✅ Keep | src/engine/model.py |
| Brentford recording data | ✅ Keep | data/recordings/4190023/ |
| Sprint -1 data (307 EPL, 1.4M trades) | ✅ Keep | data/feasibility/ |
| 11,531 match commentaries | ✅ Keep | data/commentaries/ |

| Component | Status | Reason |
|-----------|--------|--------|
| OddsConsensus (Tier 1 primary) | ❌ Remove | No speed edge; bookmakers suspend |
| odds_api_listener (WS for trading) | ❌ Demote | Reference/logging only |
| Signal hierarchy (HIGH/LOW/NONE) | ❌ Remove | Model is primary |
| Confidence-adjusted Kelly (0.25/0.35) | ❌ Replace | Baker-McHale + SurpriseScore |
| Sprint 3 tick_loop (consensus-first) | ❌ Rewrite | Model-first architecture |
| Sprint 4 signal_generator (consensus) | ❌ Rewrite | Dynamic threshold + limit orders |
| Discrete SURPRISE/EXPECTED/NEUTRAL | ❌ Replace | Continuous SurpriseScore |

---

## Appendix A: Mathematical Notation Reference

| Symbol | Description | Where defined |
|--------|-------------|---------------|
| λ_i(t) | Goal intensity for team i at time t | §4.1 |
| a_i(t) | Dynamic team strength (log-scale) | §4.2 |
| f(t) = b[bin(t)] | Time profile | §4.3 |
| γ_i[X_t] | Red card effect (log-scale) | §4.4 |
| δ_i(ΔS_t) | Asymmetric score-state effect | §4.5 |
| η_i | Stoppage time multiplier (log-scale) | §4.6 |
| X_t | Markov red card state ∈ {0,1,2,3} | §4.4 |
| ΔS_t | Score differential (home − away) | §4.5 |
| AT | Stoppage time interval | §4.6 |
| P_i(t\|t) | EKF posterior uncertainty for a_i | §4.2 |
| σ²_ω | EKF process noise (strength drift) | §4.2, §9.2 |
| Z_t | HMM hidden state ∈ {−1,0,+1} | §5.2 |
| φ_H, φ_A | HMM λ adjustment coefficients | §5.2 |
| κ | DomIndex decay rate | §5.3 |
| P_model | MMPP MC output probabilities | §6.1 |
| P_kalshi | Kalshi VWAP price | §6.2 |
| σ_MC | MC standard error | §6.1 |
| σ²_p | Total probability uncertainty | §6.5 |
| SurpriseScore | Continuous goal surprise metric | §6.4 |
| θ_entry(t) | Dynamic entry threshold | §6.3 |
| f* | Kelly fraction | §6.5 |
| α_base, α_surprise | Kelly multiplier parameters | §6.5 |
| ρ | Bias decay rate (Choi & Hui) | §6.7, §9.3 |
| τ* | Optimal hold time | §6.7 |
| B_0 | Initial bias at entry | §6.7 |

---

## Appendix B: Key Academic References

| Paper | Key Finding | Applied in |
|-------|-------------|-----------|
| Dixon & Coles (1997) | Bivariate Poisson with τ correction for low-score outcomes | Phase 1 calibration baseline |
| Dixon & Robinson (1998) | Non-homogeneous Poisson; score-state effects; stoppage spike | §4.3, §4.5, §4.6 |
| Koopman & Lit (2015) | Dynamic team strength via state-space model | Motivates EKF approach (§4.2) |
| Vecer, Kopřiva & Ichiba (2009) | Red card effect: 0.67× / 1.25× empirically validated | §4.4 |
| Titman et al. (2015) | Joint goal/booking model; stoppage intensity 1.4–1.8× | §4.6, §5 |
| Heuer et al. (2012) | Score-state effects; asymmetric behavior when leading/trailing | §4.5 |
| Choi & Hui (2014) | Surprise goal overreaction; bias decays ~40%/min | §1.2, §6.4, §6.7 |
| Ötting et al. (2023) | HMM 3-state momentum model for football | §5.2 |
| Decroos et al. (2019) | VAEP: action value for momentum estimation | Motivates DomIndex (§5.3) |
| Baker & McHale (2013) | Kelly shrinkage under parameter uncertainty | §6.5 |
| Busseti, Ryu & Boyd (2016) | Risk-constrained Kelly via convex optimisation | §6.5 (future: portfolio Kelly) |
| Croxson & Reade (2014) | Betfair semi-strong efficiency; swift price updates | §1.1 (why speed edge failed) |
