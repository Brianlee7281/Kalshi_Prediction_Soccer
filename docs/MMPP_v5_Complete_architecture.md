# MMPP v5 — How This System Works

---

## 1. The Problem We Are Solving

### 1.1 The Basic Idea

Kalshi is a US-regulated prediction market where people trade on the outcome of
soccer matches. During a live game, the probability of each outcome — home win,
draw, away win — changes every second as goals are scored, players are sent off,
and time runs down.

Kalshi participants set prices by trading with each other. Most of them are
watching the game on TV and reacting emotionally to what they see. This creates
a predictable pattern: **in certain situations, the crowd systematically misprices
the true probability of each outcome.**

Our system — MMPP — calculates the mathematically correct probability every second
using a rigorous statistical model. When the model's probability diverges from
the Kalshi market price by enough to cover transaction costs, we place a trade.
The trade profits when the market corrects toward the true probability.

---

### 1.2 What We Tried First (and Why It Failed)

Our first approach assumed that established bookmakers like Betfair react to
in-game events faster than Kalshi. The plan was to use Betfair's price as the
"true" reference and trade the gap whenever Kalshi lagged behind.

We measured this live during Brentford vs. Wolves (2026-03-16):

| Data Source | Reaction time to a goal |
|-------------|------------------------|
| Kalshi participants | 0–30 seconds |
| Goalserve API (our event feed) | 30–60 seconds |
| Bookmakers via Odds-API | 1–3 minutes (suspended during events) |

Kalshi turned out to be **faster** than all our data sources, because it has no
suspension mechanism — participants are watching the game live and trading
immediately. There was no speed gap to exploit.

---

### 1.3 The Approach That Works: Accuracy Edge

Kalshi participants are fast, but they are also **behaviorally biased**. Three
specific biases have been documented in peer-reviewed research:

**Bias 1 — Surprise Goal Overreaction**
*(Choi & Hui, Journal of Economic Behavior & Organization, 2014)*

When an underdog scores, the market overshoots. Participants overestimate how
much the goal changes the game. The overreaction is largest immediately after
the goal and decays over the next 5 minutes. Choi and Hui (2014) found that
in Betfair's in-play market, the bias decreases by approximately 40% per
minute after the first goal, with roughly 90% corrected by minute 5.

This multiplicative decay (residual ≈ 0.6^t) can be approximated as an
exponential:

```
Market mispricing at time t after goal ≈ Initial mispricing × e^(−ρ × t)
```

where ρ ≈ 0.5/min fits the Betfair data (−ln(0.6) ≈ 0.51). However, Choi
and Hui did not report an explicit ρ parameter — this is our continuous
approximation of their discrete finding. Furthermore, their data is from
Betfair, which has higher liquidity and more market makers than Kalshi. The
actual Kalshi decay rate ρ_kalshi is estimated from our 307-match backtest
(see §8.6) and may differ substantially in either direction.

**Bias 2 — Red Card Overreaction**
*(Vecer et al., Journal of Quantitative Analysis in Sports, 2009)*
*(Titman et al., Journal of the Royal Statistical Society, 2015)*

When a player is sent off, the market overcorrects. The empirically validated
effect of a red card on scoring rates is:

- The team that lost a player scores at **roughly 67% of their previous rate**
- The opposing team scores at **roughly 120–125% of their previous rate**

Kalshi participants typically price a larger adjustment than this — treating a
red card as more catastrophic than the data supports.

**Bias 3 — Stoppage Time Anchoring**
*(Dixon & Robinson, Journal of the Royal Statistical Society, 1998)*

Participants mentally treat "90 minutes" as the end of the game. They stop
adjusting prices around minute 87, even though actual stoppage time averages
5–7 minutes and goal intensity during stoppage is **22% higher** than the
90-minute baseline. A team trailing by one goal with 5 minutes of stoppage
time remaining has a meaningfully higher probability of equalizing than most
Kalshi participants price.

**Our edge:** MMPP computes probabilities without these biases. In the specific
windows where bias is strongest, P_model diverges from P_kalshi by 3–8¢.
Against a ~1¢ effective spread, this is a profitable edge.

---

### 1.4 Empirical Validation So Far

We validated this on a single match (Brentford 2-2 Wolves, 2026-03-16):

| Metric | What it measures | Result |
|--------|-----------------|--------|
| Mean model–market gap, all minutes | Is there any divergence? | 4.2¢ |
| Mean gap in minutes 78–90 | Is stoppage bias real? | 4.2¢ |
| Gap after 2-2 equalizer (minute 78) | Does model handle surprise goals? | −1.7¢ (model was right, market overreacted) |
| Kalshi effective spread | What does a trade cost? | ~1¢ |

A 4.2¢ average gap against a 1¢ spread is a theoretical edge of 3.2¢ per
contract. This is a single match — not proof — but it is directionally consistent
with the academic predictions.

**Validation gate:** We have historical Kalshi trade data for 307 EPL matches.
No live capital is deployed until a backtest across all 307 matches passes
specific profitability criteria.

---

## 2. How the System Fits Together

### 2.1 The Core Loop

At the highest level, the system does one thing repeatedly during a live match:

```
Every second:
  1. Compute the true probability of each outcome using our model
  2. Compare it to the current Kalshi market price
  3. If the gap is large enough to be profitable after costs → place a trade
  4. Monitor open positions and exit when the gap closes
```

This loop runs for the full 90+ minutes of every match, across all 8 leagues
we cover simultaneously.

---

### 2.2 The Three Mathematical Layers

The probability computation in step 1 is not a single calculation — it is three
layers that build on each other:

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Goal Intensity Model                                    │
│                                                                  │
│ Question: "At this exact moment, how likely is each team        │
│           to score in the next minute?"                          │
│                                                                  │
│ Output: λ_H(t) and λ_A(t) — scoring rates for home and away    │
│                                                                  │
│ Inputs: team strength, current time, score state, red cards,    │
│         whether we're in stoppage time                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │ feeds into
┌──────────────────────────────▼──────────────────────────────────┐
│ LAYER 2: In-Play State Estimator                                 │
│                                                                  │
│ Question: "Who is dominating the game RIGHT NOW,                 │
│           independent of the scoreline?"                         │
│                                                                  │
│ Output: Z_t — momentum state {away dominant, balanced,          │
│         home dominant}                                           │
│                                                                  │
│ Inputs: shots on target, corners, dangerous attacks,            │
│         possession from live match data                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Layer 2 adjusts Layer 1 rates
┌──────────────────────────────▼──────────────────────────────────┐
│ LAYER 3: Probability → Edge → Trade Size                         │
│                                                                  │
│ Question: "Given these adjusted scoring rates, what is the      │
│           probability of each outcome for the rest of           │
│           the match? Is there a profitable trade?"              │
│                                                                  │
│ Output: P_model (win/draw/loss probabilities), trade size,      │
│         entry/exit timing                                        │
│                                                                  │
│ Inputs: λ_adj from Layers 1+2, remaining time, current score,   │
│         Kalshi market prices, model uncertainty                  │
└─────────────────────────────────────────────────────────────────┘
```

Each layer can be improved independently. Improving Layer 1's accuracy
automatically improves everything downstream.

---

### 2.3 The Six Operational Phases

The system runs in six operational phases. Some run once (offline), some run
once per match, and some run continuously during the match:

```
PHASE 1 — Calibration (runs offline, once per week)
  ─────────────────────────────────────────────────
  Input:  Historical match data (11,531 matches across 8 leagues)
  Does:   Learns all the model parameters from data
  Output: Calibrated parameters saved to database

PHASE 2 — Pre-Match Setup (runs 65 minutes before kickoff)
  ─────────────────────────────────────────────────────────
  Input:  Current pre-match betting odds, calibrated parameters
  Does:   Estimates this specific match's team strength values
          Finds the Kalshi market tickers for this match
  Output: Match-specific model initialization, ready to trade

PHASE 3 — Live Engine (runs every second, kickoff to final whistle)
  ──────────────────────────────────────────────────────────────────
  Input:  Live match events (goals, red cards, stats every 3–5s)
          Live Kalshi order book
  Does:   Runs all three mathematical layers
          Updates team strength estimates as events occur
          Computes P_model for all markets every second
  Output: TickPayload — a complete snapshot of model state every second

PHASE 4 — Execution (runs every second, consuming Phase 3 output)
  ──────────────────────────────────────────────────────────────────
  Input:  TickPayload from Phase 3, live Kalshi prices
  Does:   Detects edges (P_model vs P_kalshi)
          Calculates optimal trade size
          Places and monitors limit orders
          Exits positions when edge closes
  Output: Trades, positions, P&L

PHASE 5 — Orchestrator (runs continuously)
  ──────────────────────────────────────────
  Input:  Upcoming fixtures from Kalshi and Goalserve
  Does:   Discovers upcoming matches
          Triggers Phase 2 at the right time
          Launches a dedicated container for each match
          Cleans up after match ends
  Output: Running match containers

PHASE 6 — Dashboard (runs continuously)
  ────────────────────────────────────────
  Input:  All data from Redis (published by Phases 3 and 4)
  Does:   Displays live model state, positions, P&L
  Output: Web UI for monitoring
```

---

### 2.4 How Phases Connect to Layers

The three mathematical layers and six operational phases are not the same thing.
Here is how they connect:

| Mathematical Layer | Where it lives operationally |
|-------------------|------------------------------|
| Layer 1 — parameters learned | Phase 1 (calibration) |
| Layer 1 — runs live | Phase 3 (live engine) |
| Layer 2 — parameters learned | Phase 1 (calibration) |
| Layer 2 — runs live | Phase 3 (live engine) |
| Layer 3 — parameters learned | Phase 1 + Kalshi backtest |
| Layer 3 — runs live | Phase 4 (execution) |

Phases 1–2 are the **learning** and **initialization** stages.
Phases 3–4 are the **live trading** stages.
Phases 5–6 are the **operational infrastructure** stages.

---

### 2.5 Data Sources

Three live data sources feed the system during a match:

**Goalserve** — our primary event source
- Polls every 3–5 seconds
- Provides: goals, red cards, period changes, VAR decisions
- Also provides: live match statistics (shots on target, corners, dangerous
  attacks, possession) — these feed Layer 2
- 30-second delay from real world; too slow to be our price reference,
  but accurate enough for model state updates

**Kalshi WebSocket** — our price source and execution venue
- Real-time order book and trade feed
- We read prices from here and place orders here
- Five markets per match: home win, draw, away win, over 2.5 goals, both teams score

**Odds-API** — reference and logging only
- Pre-match: used to set initial team strength estimates (Phase 2)
- Live: logged for post-match analysis, not used for trading decisions
- Bookmakers suspend their markets during goals — too slow and unreliable
  as a live reference

---

### 2.6 What Makes This Different from Simple Prediction

A common approach to sports betting is to build a model that predicts match
outcomes and bet whenever the model disagrees with the market. This system
is different in several important ways:

**It trades the process, not the prediction.**
We are not trying to predict who wins. We are trying to identify the specific
moments when the market misprices the probability — and we have academically
documented reasons to believe those moments are predictable.

**It updates continuously.**
The model does not make a single pre-match prediction. It updates its probability
estimates every second, incorporating new information as the match unfolds. When
a goal is scored, the model recalculates the scoring rates, updates its estimate
of team strength, and produces new probabilities — all within milliseconds.

**It sizes positions mathematically.**
Position sizing uses the Kelly Criterion with Baker-McHale shrinkage for parameter
uncertainty. This means the system automatically bets less when it is less certain,
and more when it is more certain.

**It has a structured edge hypothesis.**
The three biases described in §1.3 are not speculation — they are replicated
findings from peer-reviewed research. We are not hoping the market is wrong;
we have specific, testable predictions about when and how it will be wrong.


---


---

## 3. Phase 1: Calibration

### 3.1 Purpose and When It Runs

Phase 1 is the learning phase. It runs offline — not during a live match — and
produces the set of statistical parameters that every subsequent phase depends on.

**When it runs:** Once per week, or manually when new historical data becomes
available. It takes several minutes per league.

**What it consumes:**
- Historical match event data: 11,531 matches across 8 leagues, 4–6 seasons each,
  stored as Goalserve JSON commentary files
- Historical Betfair pre-game closing odds: from football-data.co.uk CSVs,
  used to initialize per-match team strength values (a_H_init, a_A_init)

**What it produces:** One row per league in the `production_params` database table,
containing every learned parameter the live system will use.

**Key design rule:** Running containers in production are never interrupted by a
Phase 1 update. When calibration completes and new parameters are saved, all
currently-running match containers keep their original parameter version for
the lifetime of that match. The new parameters only apply to the next match.

---

### 3.2 The Calibration Pipeline

Phase 1 runs seven steps in sequence for each league:

```
Step 1: Parse commentaries
  Raw Goalserve JSON → structured match records
  Each match: list of goals (with minute), red cards (with minute),
  final score, home team, away team, date

Step 2: Segment intervals
  Each match → list of IntervalRecords
  An interval is a continuous stretch of time where
  the score (ΔS) and player counts (X) are constant
  Intervals split at: goal events, red card events,
  halftime, and 15-minute time boundaries

Step 3: Estimate Q matrix (red card transition rates)
  Count how often each red card transition occurs per minute of play
  Produces a 4×4 generator matrix for the Markov chain
  over player-count states {11v11, 10v11, 11v10, 10v10}

Step 4: Initialize team strengths from Betfair pre-game odds
  For each match, backsolve a_H_init and a_A_init from
  Betfair closing odds in the football-data.co.uk CSVs.
  This gives the NLL optimizer a warm start grounded in
  the most accurate pre-game probability available —
  Betfair's fully-traded closing price.
  Matches without available Betfair odds fall back to
  league-average goal rates.

Step 5: Joint NLL optimization
  The main learning step — described in detail in §4
  Learns: time profile b[8], red card effects γ,
  score-state effects δ, and match-level team strengths a_H, a_A

Step 6: Asymmetric δ MLE  [v5 addition]
  Estimates separate score-state effects for leading vs trailing
  Described in §4.5

Step 7: Stoppage time η MLE  [v5 addition]
  Estimates the intensity multiplier during stoppage periods
  Described in §4.6

Step 8: EKF process noise σ²_ω estimation  [v5 addition]
  Estimates how much team strength drifts within a match
  Described in §4.2

Step 9: Walk-forward cross-validation
  Validates using Brier Score on held-out seasons
  GO if model Brier Score < uniform (0.33, 0.33, 0.33) baseline
  NO-GO if not better than random — do not save parameters

Step 10: Save to database
  All parameters saved to production_params table with version number
  Previous version marked inactive
```

---

## 4. Layer 1: The Goal Intensity Model

### 4.1 The Core Idea

Goals in soccer are rare, random events. A useful way to model them is as a
**Poisson process** — a mathematical framework where events occur randomly at
some average rate, and the number of events in any time window follows a
Poisson distribution.

The key insight is that this rate is not constant. It changes every second
based on who is on the pitch, what the score is, and how much time is left.
We call this changing rate the **goal intensity**, denoted λ(t).

Layer 1 is entirely about computing λ(t) accurately for both teams at every
moment during the match.

---

### 4.2 The Intensity Function

The intensity function for team i (where i is home H or away A) is:

```
log λ_i(t) = a_i(t)          +  f(t)          +  γ_i[X_t]
             (team strength)     (time profile)    (red cards)

           +  δ_i(ΔS_t)      +  η_i · 1[AT]
              (score state)      (stoppage time)
```

The log is used so that all terms add together linearly but the resulting
intensity is always positive. Each term answers a specific question:

| Term | Symbol | Question it answers |
|------|--------|---------------------|
| Team strength | a_i(t) | How strong is this team in this match? |
| Time profile | f(t) | What time of the match is it? |
| Red card effect | γ_i[X_t] | How many players does each team have? |
| Score state | δ_i(ΔS_t) | Who is winning, and by how much? |
| Stoppage time | η_i | Are we in added time? |

We now examine each term in detail.

---

### 4.3 Team Strength: a_i(t) — The Dynamic Kalman Filter

#### What it represents

`a_i(t)` is the baseline log-intensity for team i at time t. It captures
everything about the team that is not already explained by the other terms —
their quality, form, tactical setup, and how these change throughout the match.

A value of `a_H = -3.8` means the home team scores at roughly
`exp(-3.8) ≈ 0.022` goals per minute in neutral conditions, or about 2 goals
per 90 minutes. A higher value means a stronger-attacking team.

#### The problem with a fixed value

The simplest approach is to set `a_H` once before the match and keep it fixed
for 90 minutes. This has an important flaw: it ignores what the match is
telling us about team strength in real time.

Consider a match where the home team is expected to score 1.4 goals (a_H set
accordingly), but after 60 minutes they have scored 0 goals. The fixed-a model
still treats them as a 1.4-goal team. A smarter model would revise its estimate
downward — the absence of goals is itself evidence that the team is weaker than
expected, or that today is not their day.

#### The Extended Kalman Filter

We use an **Extended Kalman Filter (EKF)** to update `a_i(t)` continuously
throughout the match. The EKF is a well-established algorithm for estimating
a hidden state that evolves over time, given noisy observations.

In our case, the hidden state is team strength `a_i(t)`, and the observations
are whether or not goals are scored each second.

**State equation** — team strength evolves as a random walk between ticks:

```
a_i(t | t−dt) = a_i(t−dt | t−dt)
P_i(t | t−dt) = P_i(t−dt | t−dt) + σ²_ω · dt
```

`P_i` is the uncertainty about our estimate of `a_i`. It grows slightly every
tick (by σ²_ω · dt) because we become slightly less certain about team strength
as time passes without new information.

**Update equation — when a goal is scored:**

```
λ_i = exp(a_i + f(t) + γ_i[X] + δ_i(ΔS))    (current intensity)

K   = P_i(t|t−dt) / (P_i(t|t−dt) · λ_i + 1)  (Kalman gain)

a_i(t|t) = a_i(t|t−dt) + K · (1 − λ_i · dt)  (state update)
P_i(t|t) = (1 − K · λ_i) · P_i(t|t−dt)        (uncertainty update)
```

The Kalman gain K automatically determines how much to update: if we were
very uncertain about `a_i` (P_i large), the gain is large and the update is
strong. If we were already confident, the gain is small.

A goal scored by team i is positive evidence that their intensity is higher
than we thought — `a_i` moves up. The innovation term `(1 − λ_i · dt)` is
positive when a goal is scored (we observed 1 but expected λ_i · dt ≈ 0.02).

**Update equation — when no goal is scored (every tick):**

```
innovation = 0 − λ_i · dt                           (expected goals minus observed)
K_0 = P_i(t|t−dt) · λ_i / (P_i(t|t−dt) · λ_i + 1)
a_i(t|t) = a_i(t|t−dt) + K_0 · innovation
P_i(t|t) = P_i(t|t−dt)                              (no change when evidence is weak)
```

Every tick without a goal is weak negative evidence — `a_i` moves down very
slightly. Over 40 minutes without a goal, these tiny adjustments accumulate
into a meaningful downward revision. This is the key advantage over a model
that only updates on goal events.

#### Parameter: σ²_ω (process noise)

This single parameter controls how much we expect team strength to drift
within a match. A large σ²_ω means we allow the estimate to change rapidly;
a small value means we treat team strength as nearly fixed.

It is estimated by a walk-forward procedure on historical matches:

```
For each match in training data:
  1. Estimate a_H_first using only first-half goals (MLE)
  2. Estimate a_H_second using only second-half goals (MLE)
  3. Record the drift: Δa_H = a_H_second − a_H_first

σ²_ω = Var(Δa_H across all matches) / 45 minutes
```

#### Parameter: P_0 (initial uncertainty)

P_0 is how uncertain we are about `a_i` at kickoff. It is set from the
residual error of the Phase 2 backsolve — matches where pre-match odds
were hard to fit get a higher P_0.

---

### 4.4 Time Profile: f(t) — When Goals Are Scored

#### What it represents

Goals are not uniformly distributed across the match. They cluster near
halftime and at the end, with elevated rates in stoppage periods. The time
profile f(t) captures these patterns as a piecewise constant function over
8 time periods.

#### The 8 periods

```
Period  Minutes    Typical b value    Interpretation
──────  ───────    ───────────────    ──────────────────────────────
b[0]    0–15       −0.17              Early game suppression
b[1]    15–30      −0.13              Building phase
b[2]    30–45      −0.10              Late first half
b[3]    45–60      +0.06              Second half restart energy
b[4]    60–75      −0.00              Mid second half
b[5]    75–85      −0.08              Late game, teams tiring
b[6]    85–90      −0.05              Kalshi anchoring zone (see §1.3)
b[7]    90–T_exp   +0.20              Stoppage time spike ← KEY EDGE ZONE
```

The b[7] = +0.20 value is critical to the trading thesis. It means goals are
22% more likely per minute during stoppage time than at the match baseline.
Kalshi participants stop updating prices around minute 87, creating a
systematic underpricing of events in the final minutes.

#### How the parameters are estimated

All eight b values are estimated jointly in the NLL optimization (§4.8)
from 11,531 historical matches. The optimizer finds the values that make
the observed distribution of goal times most probable.

---

### 4.5 Red Card Effect: γ_i[X_t] — Player Count States

#### The Markov state

When a player is sent off, the number of players on each side changes
and the scoring rates shift. We model this using a **Markov chain** over
four states representing current player counts:

```
State 0: 11 vs 11  (normal play)
State 1: 10 vs 11  (home team reduced)
State 2: 11 vs 10  (away team reduced)
State 3: 10 vs 10  (both teams reduced)
```

The γ term adjusts log-intensity for each state. For example:

| State | Home λ multiplier | Away λ multiplier |
|-------|------------------|------------------|
| 11v11 (state 0) | ×1.00 | ×1.00 |
| 10v11 (state 1, home red) | ×0.67 | ×1.20 |
| 11v10 (state 2, away red) | ×1.20 | ×0.67 |
| 10v10 (state 3) | ×0.80 | ×0.80 |

These multipliers are approximate values from Vecer et al. (2009), who
reported ×0.67 and ×1.2 from World Cup/Euro betting data. Titman et al.
(2015) found broadly consistent effects in EPL data. These values serve
as initialization — the NLL optimizer on our 11,531-match dataset refines
them to league-specific estimates.

#### The Q matrix (transition rates)

The Q matrix is a 4×4 generator matrix that describes how quickly the
system transitions between red card states. It is estimated separately
from the NLL optimization, by directly counting red card events and
computing empirical transition rates per minute of play.

State transitions:
- Home red card: state 0→1, or state 2→3
- Away red card: state 0→2, or state 1→3

In state 3 (10v10), data is sparse since both teams being reduced is rare.
We use an **additivity assumption** to fill these gaps:
the rate of the second red card is assumed unchanged by the first.

---

### 4.6 Score State: δ_i(ΔS_t) — Who Is Winning

#### What it represents

The score differential ΔS = home goals − away goals affects how teams play.
A team leading by two goals defends more conservatively; a team trailing
chases the game more aggressively. The δ term captures this.

#### Asymmetric effects

A key insight is that these effects are **not symmetric**. When the home
team leads 1-0, the adjustment is different from when the away team leads
0-1, even though both have ΔS = ±1. The home team plays differently when
chasing than when protecting a lead — the home crowd, tactical setup, and
psychological factors all differ.

We therefore use four independent parameters:

```
δ_H⁺(ΔS):  Home scoring rate when ΔS > 0 (home team leading)
δ_H⁻(ΔS):  Home scoring rate when ΔS < 0 (home team trailing)
δ_A⁺(ΔS):  Away scoring rate when ΔS > 0 (away team trailing)
δ_A⁻(ΔS):  Away scoring rate when ΔS < 0 (away team leading)
```

Expected values from the literature (Heuer et al. 2012, Dixon & Robinson 1998):

| Situation | Home λ multiplier | Away λ multiplier |
|-----------|------------------|------------------|
| Home leads by 1 | ×0.85 (conservative) | ×1.15 (chasing) |
| Home leads by 2+ | ×0.70 | ×1.35 |
| Away leads by 1 | ×1.25 (home crowd + chasing) | ×0.90 (protecting) |
| Away leads by 2+ | ×1.40 | ×0.75 |

The asymmetry in the "home leading" case is smaller than "away leading" because
home advantage reinforces the chasing team's motivation when they are behind.

#### ΔS bins

Score differential is discretized into five bins for estimation:

```
Bin 0: ΔS ≤ −2  (home trailing by 2 or more)
Bin 1: ΔS = −1  (home trailing by 1)
Bin 2: ΔS =  0  (level)
Bin 3: ΔS = +1  (home leading by 1)
Bin 4: ΔS ≥ +2  (home leading by 2 or more)
```

---

### 4.7 Stoppage Time: η — Added Minutes

#### What it represents

Dixon and Robinson (1998) documented that goal-scoring rates spike during
stoppage time — the additional minutes played at the end of each half.
Titman et al. (2015) measured this spike at 1.4–1.8× the normal rate.

The time profile b[7] already captures part of this (b[7] = +0.20). The
additional η term provides precision by allowing the stoppage intensity
to differ between first and second half stoppage periods:

```
log λ_i(t) += η_i¹ · 1[t ∈ (45, 45 + α₁)]   (first half stoppage)
            + η_i² · 1[t ∈ (90, 90 + α₂)]   (second half stoppage)
```

where α₁ and α₂ are the announced stoppage durations, updated dynamically
when Goalserve reports them.

The η parameters are estimated by MLE on intervals classified as stoppage
time in the historical commentary data.

---

### 4.8 How All Parameters Are Learned: NLL Optimization

#### The likelihood function

All the parameters in the intensity function except σ²_ω are estimated
jointly by maximizing the **Poisson process log-likelihood** over all
historical matches.

The core idea: if the intensity model is correct, then goals should occur
at times and rates consistent with that model. We find the parameters that
make the observed history of goals most probable.

For a single match, the log-likelihood has two components:

**Point event term** — each observed goal increases the likelihood:
```
Σ_goals log λ_i(t_goal)
```
This says: goals should occur when intensity is high.

**Integration term** — the total expected goals must match reality:
```
− ∫₀ᵀ λ_i(t) dt
```
This says: if the model predicts too many goals overall, the likelihood drops.

Combined for all matches across all leagues, the loss function is:

```
NLL = −Σ_matches [Σ_home_goals log λ_H(t_g)
                 + Σ_away_goals log λ_A(t_g)
                 − ∫ (λ_H(t) + λ_A(t)) dt]

    + (1/2σ²_a) Σ_matches [(a_H − a_H_init)² + (a_A − a_A_init)²]

    + λ_reg (‖b‖² + ‖γ‖² + ‖δ‖²)
```

The second line is **regularization** — it keeps per-match team strengths
close to their Betfair-backsolve prior estimates. Without this, team strength
values for matches with few goals (or goals clustered in one half) would drift
unreliably away from the market's informed starting point.

The third line is **L2 regularization** — it prevents the structural
parameters from overfitting to noise in the training data.

#### Optimization algorithm

The NLL is minimized using **Adam** (gradient descent with adaptive learning
rate) implemented in PyTorch. Adam handles the non-convex loss landscape
better than second-order methods for the large number of parameters involved.

Parameters are initialized from the Betfair backsolve for a_H and a_A (or
league-average MLE if Betfair odds are unavailable), and from zero for the
structural parameters (b, γ, δ). The optimizer runs for 1,000 epochs.

---

### 4.9 Validation: Does It Actually Work?

After optimization, parameters are validated using **walk-forward
cross-validation**:

```
For k = 1, 2, ..., N_seasons:
  Train on seasons 1 through k
  Validate on season k+1
  Record Brier Score on validation season

Brier Score = (1/N) Σ [(p_home − y_home)² + (p_draw − y_draw)² + (p_away − y_away)²]
```

where y_home = 1 if home team won, else 0 (similarly for draw and away).

**GO/NO-GO decision:**
- GO if model Brier Score < uniform baseline (0.33, 0.33, 0.33) → model is useful
- NO-GO if not better than random → do not save parameters, investigate

The walk-forward design respects time order — the model is never validated
on data that precedes its training data. This is essential for honest
evaluation of a model that will be deployed in real time.

---

### 4.10 What Gets Saved to the Database

After a successful GO verdict, the following parameters are saved per league:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `b` | [8] | Time profile coefficients |
| `gamma_H`, `gamma_A` | [4] each | Red card intensity effects |
| `Q` | [4×4] | Red card transition rate matrix |
| `delta_H_pos`, `delta_H_neg` | [5] each | Asymmetric home score-state effect |
| `delta_A_pos`, `delta_A_neg` | [5] each | Asymmetric away score-state effect |
| `eta_H`, `eta_A`, `eta_H2`, `eta_A2` | scalar each | Stoppage time multipliers |
| `sigma_omega_sq` | scalar | EKF process noise |
| `sigma_a` | scalar | Regularization strength (also sets EKF P_0) |
| `brier_score` | scalar | Validation metric |
| `match_count` | integer | Number of training matches |

These parameters are version-stamped and pinned per match — a match that
starts using version 42 of EPL parameters will use version 42 for its
entire 90 minutes, even if version 43 is saved to the database mid-match.


---


---

## 5. Phase 2: Pre-Match Setup

### 5.1 Purpose and Timing

Phase 1 produces league-level parameters — values that apply to all matches
in a league. Phase 2 takes those parameters and applies them to a specific
match that is about to be played.

**When it runs:** 65 minutes before kickoff of each match. This timing is
deliberate: early enough to have settled pre-match odds, late enough to capture
any last-minute team news (injuries, lineup changes) that shift the odds.

**What it does in one sentence:** Given the current betting market's view of
the match, find the team strength values `a_H` and `a_A` that make the model's
pre-match probability agree with the market, then identify which Kalshi
contracts correspond to this match.

**Why 65 minutes, not earlier?** Pre-match odds are most informative close to
kickoff. Odds posted days before include more uncertainty and are less reliable
as team strength estimates. Lineups are often released 60–75 minutes before
kickoff, which shifts the odds significantly.

---

### 5.2 The Core Problem: Estimating Team Strength for This Match

Pre-match betting odds are the most information-rich signal available about
a specific match. Betfair's closing odds reflect millions of dollars of trading
by professional bettors, arbitrageurs, and market makers — far more signal than
any statistical model trained on historical data alone.

Converting odds to initial team strength values `a_H(0)` and `a_A(0)` requires
three steps: vig removal using the Shin method, a Dixon-Coles inverse to recover
goal intensities, and a time-profile correction to obtain the final initial values.

---

#### Step 1 — Vig removal: the Shin method

Raw bookmaker odds contain a margin (vig). The naive approach — dividing
each raw implied probability by the overround — is theoretically wrong because
it treats all outcomes symmetrically. Favorites are actually underpriced and
longshots overpriced relative to their true probabilities, a phenomenon known
as the **favourite-longshot bias**.

The **Shin method** (Shin, 1992; 1993) models this correctly. It assumes a
fraction `z` of bettors are informed insiders who always bet on the true
outcome. The bookmaker, knowing this, shades odds to protect themselves —
which is exactly what creates the favourite-longshot bias. The model is
internally consistent and theoretically grounded in information asymmetry.

Given raw implied probabilities `q_i = 1/odds_i` and overround `O = Σq_i`,
define the normalized implied probabilities `ρ_i = q_i / O` (these sum to 1).

**Estimate z** by solving numerically:

```
Σ_{i=1}^{3} √(z² + 4(1−z) · ρ_i²) = 2(1−z)/O + 3z
```

This has a unique solution in `z ∈ (0, 1)` and is found efficiently by
bisection. For typical soccer markets z ≈ 0.02–0.08.

**Recover true probabilities:**

```
p_i = O · [ √(z² + 4(1−z) · ρ_i²) − z ] / [ 2(1−z) ]
```

The result: favorites (high q_i) get slightly **higher** probability than
naive normalization; longshots (low q_i) get slightly **lower** probability.
The Shin-corrected probabilities sum exactly to 1.

---

#### Step 2 — Dixon-Coles inverse: probabilities to goal intensities

Given the corrected true probabilities `(p_home, p_draw, p_away)`, we
recover the Poisson goal intensity pair `(λ_H, λ_A)` that produces them.

Under the independent Poisson model:

```
p_home = Σ_{i > j} Poisson(i; λ_H) × Poisson(j; λ_A)
p_draw = Σ_{i}     Poisson(i; λ_H) × Poisson(i; λ_A)
p_away = Σ_{i < j} Poisson(i; λ_H) × Poisson(j; λ_A)
```

We solve for `(λ_H, λ_A)` numerically using `scipy.optimize.minimize`,
initialized from the implied goal counts:

```
λ_H_0 ≈ 1.5 × p_home + 1.0 × p_draw + 0.5 × p_away
λ_A_0 ≈ 0.5 × p_home + 1.0 × p_draw + 1.5 × p_away
```

The optimizer minimizes the squared residual between the model's output
probabilities and the Shin-corrected targets. Convergence is fast because
the starting point is already close to the solution.

---

#### Step 3 — Time-profile correction: intensities to initial strength

`λ_H` and `λ_A` represent the full-match expected goals (the Poisson
parameters from Step 2). To recover `a_H(0)`, we integrate the MMPP
intensity over the full match at kickoff conditions (X = 0, ΔS = 0):

```
E[goals_H] = exp(a_H) × Σ_{k=0}^{5} exp(b[k]) × Δτ_k = exp(a_H) × C_time
```

where `C_time = Σ exp(b[k]) × 15` is the integrated time profile
(6 bins × 15 minutes each). Therefore:

```
a_H(0) = log(λ_H) − log(C_time) = log(λ_H / C_time)
a_A(0) = log(λ_A) − log(C_time) = log(λ_A / C_time)
```

Dividing by `C_time` (equivalently, subtracting `log(C_time)`) is essential.
It correctly accounts for the entire time-varying intensity profile, not just
the first bin. The resulting `a_H(0)` and `a_A(0)` represent pure team
strength, independent of the time period.

---

#### Why Betfair closing odds, not a statistical model

The Betfair pre-game market is semi-strong efficient (Croxson & Reade, 2014).
Its closing price already incorporates team form, injuries, lineup news, home
advantage, and everything else a statistical model might use — plus information
from professional bettors that no historical model can access.

This three-step pipeline (Shin → Dixon-Coles inverse → b[0] correction)
produces initial `a_H(0)` and `a_A(0)` values that are theoretically grounded
and empirically more accurate than any regression model. The MMPP's job during
the match is to track how team strength *changes* relative to this prior —
not to compete with the market's pre-match assessment.

---

#### Where the odds come from: tiered fallback

Not all odds sources are available for every match. Phase 2 tries three
tiers in order, using the first one that succeeds:

```
Tier 1: Betfair Exchange closing odds (football-data.co.uk CSV)
  The primary and most accurate source.
  Fully-traded, efficient market closing price.
  Available for all 8 leagues via football-data.co.uk historical CSVs.
  For live matches: supplement with Odds-API real-time Betfair feed
  (captures late lineup news not yet in the CSV).

Tier 2: Bet365 / Odds-API live feed
  Used when Betfair data is unavailable or stale.
  Less sharp than Betfair Exchange but still a strong signal.
  Most accurate for leagues where Betfair liquidity is lower
  (MLS, Brasileirão, Argentina).

Tier 3: League average MLE
  Simple fallback: league-average home and away goal rates.
  Used only when no odds data is available at all.
  Triggers a wide sanity check threshold — low confidence.
```

The prediction method is recorded in Phase2Result so the live system knows
how much to trust the initial `a_H(0)` and `a_A(0)` values.

---

### 5.3 EKF Initialization: P_0

When the EKF starts at kickoff, it needs an initial uncertainty value `P_0`
— how confident are we in our estimate of `a_H` before any goals are scored?

This is derived from the backsolve quality. If the optimizer converged cleanly
from Tier 1 Betfair odds, `P_0` is small — we trust the starting estimate.
If we fell back to Tier 2 (Bet365) or Tier 3 (league average), `P_0` is
larger, reflecting our lower confidence in the initial team strength estimate.

A larger `P_0` means the EKF will update `a_H` and `a_A` more aggressively
in response to early goals, because we start with less certainty.

---

### 5.4 Kalshi Ticker Matching

Kalshi creates a separate contract for each outcome of each match. A typical
EPL match has five tradeable markets:

```
KXEPLGAME-26APR19ARSCHE-ARS   (Arsenal wins)
KXEPLGAME-26APR19ARSCHE-TIE   (draw)
KXEPLGAME-26APR19ARSCHE-CHE   (Chelsea wins)
KXEPLTOTAL-26APR19ARSCHE-OV   (over 2.5 goals)
KXEPLBTTS-26APR19ARSCHE-YES   (both teams score)
```

Matching the Goalserve fixture data to the correct Kalshi tickers is
non-trivial because team names differ across data sources. Phase 2 uses
a 251-team alias table to normalize names across sources, then applies
time-window matching to ensure we find the right contract for the right
match date.

---

### 5.5 Sanity Check: When to Skip a Match

Before committing to trade a match, Phase 2 runs a sanity check:

```
For each market (home win, draw, away win):
  |P_model − P_market| must be < 0.15

If any market exceeds 0.15 → verdict = SKIP
```

A large divergence means either the model parameters are stale, the odds
source is unusual, or there is a data problem. Trading into a match where
we do not understand the pre-match state is unwise — we skip it.

---

### 5.6 Output: Phase2Result

Phase 2 produces a `Phase2Result` record that is passed to the match container:

```
a_H, a_A          — initial team strength values (log scale)
mu_H, mu_A        — expected goals for each team (exp(a_H) × C_time)
ekf_P0            — initial EKF uncertainty
param_version     — which Phase 1 parameters to use (version pinned)
kalshi_tickers    — map of {market_type → ticker string}
market_implied    — the odds-implied probabilities used for backsolve
prediction_method — which tier produced a_H and a_A
verdict           — "GO" or "SKIP"
```

---

## 6. Phase 3: The Live Engine

### 6.1 Overview

Phase 3 is the heart of the system. It runs continuously from kickoff to
final whistle, updating the model state every second and producing a
complete probability estimate for every market every second.

Three concurrent processes run in parallel for every match:

```
┌─────────────────────────────────────────────────────┐
│  goalserve_poller    (every 3 seconds)              │
│  Polls the Goalserve API for match events and       │
│  live statistics. Updates model state when goals,  │
│  red cards, or period changes are detected.         │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  kalshi_ob_sync      (continuous, WebSocket)        │
│  Subscribes to Kalshi's orderbook and trade feed.  │
│  Maintains current P_kalshi for each market.        │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│  tick_loop           (every 1 second)               │
│  Runs the three mathematical layers. Produces       │
│  TickPayload. Sends to Phase 4 execution engine.   │
└─────────────────────────────────────────────────────┘
```

All three share a single `LiveMatchModel` object in memory. Because Python's
asyncio is single-threaded, no locking is needed — only one coroutine runs
at any moment.

---

### 6.2 What Happens Every Second: The Tick Loop

Each tick follows the same sequence:

```
1. Update effective match time (model.t)
   t = (wall_clock − kickoff_time − halftime_duration) / 60
   This gives effective play minutes, excluding halftime.

2. EKF prediction step
   For each team i:
     P_i += σ²_ω × dt    (uncertainty grows slightly)
   This reflects that team strength drifts slightly every second.

3. Layer 2 update (when new live_stats data arrived from Goalserve)
   Update HMM state probabilities from new shot/corner/possession data
   OR update DomIndex if HMM is not available

4. Compute adjusted intensities
   λ_H_adj = λ_H(t) × exp(φ_H × Z_t)
   λ_A_adj = λ_A(t) × exp(φ_A × Z_t)

5. Run Monte Carlo simulation (in parallel thread, non-blocking)
   50,000 simulated match completions using current λ_adj(t)
   → P_model (home_win, draw, away_win, over_25, btts)
   → σ_MC (standard error for each market)

6. Compute σ²_p (total probability uncertainty)
   σ²_p = σ²_MC + σ²_model
   This is used by Phase 4 to size positions.

7. Assemble TickPayload and send to Phase 4 queue
```

---

### 6.3 How Time Is Tracked

A subtle but important detail: `model.t` is **effective play time** in minutes,
not wall clock time. The difference matters because halftime can last 12–20
minutes, during which the match clock is paused.

```python
# At kickoff:
kickoff_wall_clock = time.monotonic()

# At halftime break start:
halftime_start = time.monotonic()

# When second half begins:
halftime_accumulated = time.monotonic() - halftime_start

# Every tick during the second half:
model.t = (time.monotonic() - kickoff_wall_clock - halftime_accumulated) / 60
```

Tick scheduling uses absolute times rather than `sleep(1)`, so if the Monte
Carlo simulation takes slightly more than 1 second, the next tick fires at
the correct absolute time anyway. No clock drift accumulates.

---

### 6.4 Event Handling

When Goalserve reports a match event, Phase 3 handles it with specific logic
before the next tick loop iteration:

#### Goal events

```
1. Save rollback snapshot: current (a_H, a_A, P_H, P_A, score, n_H, n_A)
2. Update score and ΔS
3. EKF goal update: a_H or a_A shifts in response to the observation
4. Layer 2: force HMM state toward the scoring team's dominant state
5. Compute SurpriseScore = 1 − P_model(scoring_team_wins | state before goal)
6. Activate cooldown: no new orders for cooldown_ticks seconds after goal
```

**Cooldown design — dynamic, not fixed:**

The original rationale for a 50-second cooldown was to avoid filling against
stale book orders immediately after a goal. But we use limit orders posted at
P_model fair value — stale fill risk is already controlled by the price itself,
not by waiting. The real question is whether meaningful edge remains after
the market partially reprices.

From Choi & Hui (2014), market mispricing decays approximately as exp(−ρ × t)
with ρ ≈ 0.5/min on Betfair. At t = 50 seconds (≈ 0.83 min), the fraction
remaining is exp(−0.5 × 0.83) ≈ 0.66 — meaning roughly two-thirds of the
initial overreaction is still present at the 50-second mark. This ρ value is
replaced by ρ_kalshi once estimated from the 307-match backtest.

However, the ρ ≈ 0.5/min value is from Betfair data. Kalshi has lower
liquidity, a smaller participant pool, and fewer market makers. The true
Kalshi ρ could be meaningfully different in either direction.

**Therefore: cooldown is set dynamically based on the ρ estimated from the
307-match Kalshi backtest:**

```
cooldown_seconds = max(15, min(60, (1/ρ_kalshi) × ln(2)))
```

This sets the cooldown to the half-life of the mispricing — the point at
which roughly 50% of the initial overreaction has corrected. If Kalshi
corrects faster than Betfair (larger ρ), the cooldown shortens. If it
corrects more slowly (smaller ρ), it lengthens.

Until ρ_kalshi is empirically estimated from the backtest, the default is
30 seconds (corresponding to ρ ≈ 0.5/min: half-life = ln(2)/0.5 ≈ 83s,
but clamped down to 30s for conservatism during early testing).

#### Red card events

```
1. Update Markov state X_t (11v11 → 10v11 or 11v10)
2. γ_i[X_t] in the intensity formula changes automatically
3. Activate shorter cooldown (~30 seconds)
```

#### Penalty events

```
1. Freeze all order activity (ob_freeze = True)
2. Wait for outcome from Goalserve (Package 1 score change, or
   Package 2 summary.goals.player[].penalty confirmation)
3. If scored: process as a normal goal via handle_goal()
4. If missed: unfreeze, resume normal tick loop
```

We do not attempt to adjust P_model during the 30-second penalty window.
Penalty conversion rates vary significantly by player (roughly 60–90%)
and Goalserve does not reliably provide the kicker's identity in real time.
The ob_freeze already prevents any orders during this window, so the
P_model imprecision carries no trading risk.

#### VAR cancellations

VAR cancellations are detected via two independent signals from Goalserve,
both checked on every poll:

```
Detection method 1 — Score decrease (Package 1, every 3-5s):
  if current_score < previous_score:
    → goal was cancelled, trigger VAR rollback immediately

Detection method 2 — Explicit flag (Package 2, every 30s):
  if summary.goals.player[].var_cancelled == "True":
    → confirms VAR cancellation for a specific goal
```

Method 1 catches the cancellation faster (Package 1 polls every 3–5s vs
Package 2 every 30s). Method 2 provides explicit confirmation. Both trigger
the same rollback procedure:

```
1. Pop the rollback snapshot from the goal event
2. Restore: a_H, a_A, P_H, P_A, score, n_H, n_A to pre-goal values
3. Restore HMM state to pre-goal position
4. Clear the SurpriseScore from the cancelled goal
5. Signal Phase 4 to exit any positions opened on that goal
```

The rollback stack stores the last 3 snapshots. A VAR review window
typically resolves within 90 seconds, well within the stack's coverage.

#### Period changes

```
FIRST_HALF → HALFTIME:   record halftime_start time, pause tick loop
HALFTIME → SECOND_HALF:  compute halftime_accumulated, resume tick loop
SECOND_HALF → FINISHED:  send final TickPayload, terminate all loops
```

---

### 6.5 Layer 2: The In-Play State Estimator

#### Why Layer 1 alone is not enough

Layer 1 answers "how strong is each team?" based on historical data and
match events so far. But it cannot observe the current tactical situation.

Consider: a team with high attacking strength (high `a_H`) might be playing
a low-block defensive formation today. Their historical scoring rate says
they should be creating chances, but the live match data — few shots on target,
low possession — tells a different story. Layer 2 captures this.

#### The HMM: inferring momentum from observable signals

We use a **Hidden Markov Model (HMM)** with three hidden states:

```
Z_t = −1:  Away team dominant
Z_t =  0:  Balanced
Z_t = +1:  Home team dominant
```

The state is "hidden" because we cannot directly observe which team is
dominating — we can only observe signals that correlate with dominance.
The signals come from Goalserve's live_stats feed, updated every 3–5 seconds:

```
Signal 1: Δ shots on target (home minus away since last poll)
Signal 2: Δ corners (home minus away)
Signal 3: Δ dangerous attacks (home minus away)
Signal 4: possession difference (home % minus 50, centered at zero)
```

For each hidden state, these signals have characteristic distributions. When
home is dominant (Z = +1), we expect more home shots, more home corners, more
home attacks, and higher home possession. The HMM infers the hidden state by
finding which state best explains the stream of observed signals.

#### How Layer 2 adjusts Layer 1

The HMM state modifies the Layer 1 intensity values:

```
λ_H_adj(t) = λ_H(t) × exp(φ_H × Z_t)
λ_A_adj(t) = λ_A(t) × exp(φ_A × Z_t)
```

When home team is dominant (Z_t = +1), φ_H > 0 increases λ_H and
φ_A < 0 decreases λ_A. This makes the model more responsive to the
current flow of the game, not just historical team quality.

#### Goal-forced transitions

When a goal is scored, the HMM state is pushed immediately toward the
scoring team's dominant state:

```
Home goal → P(Z = +1) increases strongly → [0.10, 0.25, 0.65]
Away goal → P(Z = −1) increases strongly → [0.65, 0.25, 0.10]
```

A goal is strong evidence about which team is currently dominant.

#### DomIndex: the simple fallback

When HMM is not available (league without live_stats coverage, or HMM not
yet trained on recordings), we use a simpler momentum signal:

```
DomIndex(t) = Σ_home_goals exp(−κ × (t − t_g))
            − Σ_away_goals exp(−κ × (t − t_g))
```

This exponentially-decays the influence of each goal over time. A goal scored
10 minutes ago contributes less than a goal scored 2 minutes ago. The decay
rate κ ≈ 0.1/min means a goal's influence halves every ~7 minutes.

The DomIndex is then squashed to (−1, +1) using tanh and used the same way
as the HMM state Z_t.

#### Graceful degradation

Layer 2 is an enhancement, not a requirement:

```
live_stats available AND HMM trained  →  use HMM state
live_stats available, HMM not trained →  use DomIndex
live_stats unavailable                →  Layer 1 only (φ = 0)
```

The system produces valid probabilities in all three cases.

---

### 6.6 Monte Carlo Simulation: From Intensities to Probabilities

Given the adjusted intensities λ_H_adj(t) and λ_A_adj(t) for the remaining
time `[t, T_exp]`, how do we compute probabilities for each market?

We use Monte Carlo simulation: simulate 50,000 independent completions of
the remaining match and count outcomes.

For each simulated path:

```
s = current time t
Current score: (S_H, S_A)
Current Markov state: X_t

While s < T_exp:
  Compute λ_H, λ_A, λ_redcard for current state
  λ_total = λ_H + λ_A + λ_redcard

  Sample time to next event: dt ~ Exponential(λ_total)
  s_next = s + dt

  If s_next crosses a basis boundary: advance to boundary and continue

  Sample event type:
    With prob λ_H/λ_total: home scores, S_H += 1
    With prob λ_A/λ_total: away scores, S_A += 1
    With prob λ_rc/λ_total: red card, update X_t

  s = s_next

Record final (S_H, S_A)
```

After 50,000 paths:

```
P(home_win)  = fraction of paths where final S_H > S_A
P(draw)      = fraction where final S_H = S_A
P(away_win)  = fraction where final S_H < S_A
P(over_25)   = fraction where final S_H + S_A ≥ 3
P(btts)      = fraction where both final S_H ≥ 1 and S_A ≥ 1
```

The Monte Carlo engine runs on the CPU using Numba JIT compilation, completing
50,000 paths in approximately 7.6 milliseconds. It runs in a separate thread
to avoid blocking the 1-second tick loop.

**Asymmetric δ and computational load:**

Each simulated path applies the asymmetric score-state effect δ_H⁺/δ_H⁻/δ_A⁺/δ_A⁻
whenever a goal is scored inside the simulation. This adds one array index lookup
and one exp() call per simulated goal event. Across 50,000 paths with an average
of 2–3 goals per path, this is roughly 100,000–150,000 additional exp() calls per
tick — negligible inside a Numba JIT-compiled loop where the call is a single
native instruction.

The realistic stress scenario is a high-scoring match in a late-game red-card
state (e.g. 10v10, score 4-3). Simulated paths contain more events and more
basis boundary crossings than average. Even in this case, MC time is expected
to reach at most 15–20ms — still less than 2% of the 1-second tick budget, and
non-blocking because MC runs in a separate executor thread.

**Critical: Numba JIT warmup.**

Numba compiles the simulation kernel on first call, which can take 200–800ms
depending on code complexity. If this happens on the first live tick, the tick
loop will stall visibly. The container startup sequence must include a warmup
call before kickoff:

```python
# Required in container initialization, before engine starts:
_ = run_mc_simulation(
    dummy_state,    # arbitrary valid state
    n_paths=1000,   # small N — just enough to trigger compilation
)
# After this call, all subsequent JIT calls are fast.
```

This warmup must use the same code path (same δ lookup tables, same Q matrix
structure) as the live simulation. Changing function signatures between warmup
and live calls will trigger recompilation.

---

### 6.7 The TickPayload: What Phase 3 Sends to Phase 4

Every second, Phase 3 assembles and sends a complete snapshot of the model
state to Phase 4. This includes:

```
Timing:
  t              — effective play time in minutes
  engine_phase   — FIRST_HALF | HALFTIME | SECOND_HALF | FINISHED

Model state:
  score          — (home_goals, away_goals)
  X              — Markov state (0=11v11, 1=10v11, 2=11v10, 3=10v10)
  delta_S        — score differential (home − away)
  mu_H, mu_A     — remaining expected goals for each team

EKF state:
  a_H_current    — current team strength estimate for home
  a_A_current    — current team strength estimate for away
  ekf_P_H        — current EKF uncertainty for home
  ekf_P_A        — current EKF uncertainty for away

Layer 2 state:
  hmm_state      — current HMM state (−1, 0, +1)
  dom_index      — DomIndex value (fallback momentum)

Probabilities:
  P_model        — (home_win, draw, away_win, over_25, btts)
  sigma_MC       — Monte Carlo standard error for each market
  surprise_score — SurpriseScore from last goal (0 if no goal yet)

Trading controls:
  order_allowed  — False during cooldown, ob_freeze, or VAR review
  cooldown       — True for dynamic cooldown window after any event
                   (default 30s; set to (1/ρ_kalshi)×ln(2) after backtest)
  ob_freeze      — True while penalty is in progress or VAR is ongoing
  event_state    — IDLE | CONFIRMED | PENALTY_PENDING | VAR_REVIEW
```

Phase 4 receives this payload and decides whether and how much to trade.
The full model state is transmitted every second so Phase 4 has everything
it needs without requiring any additional data sources.


---


---

## 7. Phase 4: The Execution Engine

### 7.1 Overview

Phase 4 sits downstream of Phase 3. Every second, it receives a TickPayload
containing the complete current model state and asks one question:

> **"Given the current P_model and the current Kalshi market price,
> is there a trade worth making?"**

If yes, it calculates how large that trade should be, places a limit order,
and monitors the position until it should be exited.

Phase 4 is entirely driven by P_model. There is no separate "consensus" or
"reference" price — the mathematical model is the sole authority.

---

### 7.2 The Full Decision Loop

Each tick follows this sequence:

```
Receive TickPayload from Phase 3 queue
         │
         ▼
Is order_allowed? ─── No ──→ Skip this tick (cooldown / penalty / VAR)
         │
        Yes
         ▼
For each active market (home_win, draw, away_win, over_25, btts):

  1. Get P_model and P_kalshi for this market
  2. Compute EV in both directions
  3. Compute dynamic entry threshold θ_entry(t)
  4. If max(EV) > θ_entry → signal detected, size and place order
  5. For all open positions: check 6 exit triggers
         │
         ▼
Publish positions and P&L to Redis → Dashboard
```

---

## 8. Layer 3: Probability → Edge → Sizing

### 8.1 Edge Detection

The basic edge calculation is straightforward. For each market, Kalshi offers
a YES contract (pays $1 if outcome occurs, costs P_kalshi) and a NO contract
(pays $1 if outcome does not occur, costs 1 − P_kalshi).

```
EV if we buy YES = P_model − P_kalshi
EV if we buy NO  = (1 − P_model) − (1 − P_kalshi) = P_kalshi − P_model

Best direction = whichever has higher EV
Trade if max(EV_YES, EV_NO) > θ_entry(t)
```

If P_model = 0.62 and P_kalshi = 0.55, then EV_YES = 0.07 (7¢). Buying YES
means we pay 55¢ for a contract worth 62¢ by our model's estimate — a 7¢
theoretical profit per contract.

---

### 8.2 The Dynamic Entry Threshold

A fixed threshold (e.g. "trade if EV > 3¢") ignores a crucial reality: the
model's probability estimate is more reliable at some moments than others.
In the 5th minute with 0 goals scored, the model has very little information
— it is essentially reciting its pre-match estimate. In the 82nd minute with
3 goals already scored and well-updated EKF values, it is far more confident.

The dynamic threshold adapts to this uncertainty:

```
θ_entry(t) = c_spread + c_slippage + z_α × σ_p(t)

c_spread   = 0.01   (Kalshi effective spread, measured from trade data)
c_slippage = 0.005  (cost of limit order execution delay)
z_α        = 1.645  (95% one-tailed confidence level)
σ_p(t)     = total probability uncertainty at this tick
```

The total uncertainty `σ_p(t)` has two components — one from the Monte Carlo
simulation and one from uncertainty in the team strength estimates:

```
σ²_p = σ²_MC + σ²_model

σ²_MC    = p̂ × (1 − p̂) / N_MC
         (standard error of proportion from 50,000 simulations)

σ²_model = P_H × (p̂ × (1 − p̂) × μ_H_remaining)²
         (EKF uncertainty P_H, propagated through the model via Delta method)
```

The second term is the key insight from Baker and McHale (2013): uncertainty
in team strength `a_H` translates into uncertainty in the final probability
estimate `p̂`. The larger μ_H_remaining (more goals still expected), the more
this uncertainty matters — a small error in `a_H` creates a larger error in
the win probability when there is more match left to play.

**In practice:** early in a match, σ_p is high and so is θ_entry — the
threshold demands a larger apparent edge before trading. Late in the match,
σ_p falls and θ_entry falls with it — smaller edges become tradeable.

---

### 8.3 SurpriseScore: Classifying What Just Happened

When a goal is scored, the immediate question is: how surprising was this?
The answer determines how aggressively we trade the Kalshi overreaction.

**Definition:**

```
SurpriseScore = 1 − P_model(scoring_team_wins | state just before goal)
```

This is the probability that the scoring team would **lose** just before
they scored. A team with a 10% chance of winning just scored — SurpriseScore
is 0.90 (very surprising). A team with an 80% chance of winning just scored
— SurpriseScore is 0.20 (not surprising).

The value ranges from 0 to 1 and is computed from the TickPayload at the
moment the goal is detected. It persists in the TickPayload for subsequent
ticks until the next goal.

| SurpriseScore | What it means | Expected market overreaction |
|--------------|--------------|------------------------------|
| > 0.65 | Strong underdog scored | Large (>8¢ possible) |
| 0.40 – 0.65 | Moderate surprise | Medium (3–8¢) |
| < 0.40 | Favourite scored | Small or none |

**Why this is better than a fixed pre-match label:**

Consider a team that started the match as a 70% favourite. By minute 75,
they trail 0-2. Their current model win probability is perhaps 12%. If they
now score, their SurpriseScore is 0.88 — genuinely surprising given the
current game state — even though they were the pre-match favourite.

A system that labels teams as "favourite" or "underdog" at kickoff and keeps
that label for 90 minutes would miss this. SurpriseScore reflects the current
in-game situation.

---

### 8.4 Kelly Criterion with Uncertainty Shrinkage

#### The standard Kelly fraction

The Kelly Criterion gives the mathematically optimal fraction of bankroll to
bet to maximize long-run growth:

```
f* = (b × p̂ − q) / b

where:
  p̂ = P_model (our probability estimate)
  q  = 1 − p̂
  b  = (1 / P_kalshi) − 1   (decimal odds minus 1)
```

If P_model = 0.62 and P_kalshi = 0.55, then b = (1/0.55) − 1 = 0.818, and:

```
f* = (0.818 × 0.62 − 0.38) / 0.818 = (0.507 − 0.38) / 0.818 = 0.155
```

Full Kelly says bet 15.5% of bankroll. This is theoretically optimal but
practically too aggressive — Kelly assumes perfect knowledge of p̂, which
we do not have.

#### Baker-McHale shrinkage for parameter uncertainty

Baker and McHale (2013) showed that when p̂ is estimated with uncertainty,
the optimal bet size is smaller than Kelly by a factor that depends on that
uncertainty:

```
f_optimal = f* × shrinkage

shrinkage = 1 − σ²_p / (p̂ − P_kalshi)²
shrinkage = max(0, shrinkage)   (clamp to non-negative)
```

The logic: if our model uncertainty σ²_p is large relative to the apparent
edge `(p̂ − P_kalshi)²`, the edge might not be real — it could just be
estimation noise. The shrinkage factor reflects this skepticism.

**Example:** P_model = 0.62, P_kalshi = 0.55, σ_p = 0.04

```
apparent edge = (0.62 − 0.55)² = 0.0049
σ²_p          = 0.04² = 0.0016

shrinkage = 1 − 0.0016 / 0.0049 = 1 − 0.33 = 0.67

f_optimal = 0.155 × 0.67 = 0.104   (10.4% of bankroll, not 15.5%)
```

When σ_p equals the apparent edge (σ_p = |p̂ − P_kalshi|), shrinkage drops
to zero and the optimal bet is nothing — the edge is entirely explained by
model noise. This is the right behavior: if we are not confident, we should
not trade.

#### SurpriseScore multiplier

On top of the Kelly shrinkage, we apply a multiplier based on SurpriseScore.
This scales the position size with our conviction about the market's
overreaction:

```
kelly_mult = α_base + α_surprise × SurpriseScore

f_final = f_optimal × kelly_mult
```

`α_base = 0.10` means we use at most one-tenth Kelly even in the baseline
case — extremely conservative. A typical surprise goal (SurpriseScore ≈ 0.70)
with calibrated `α_surprise ≈ 0.15` gives:

```
kelly_mult = 0.10 + 0.15 × 0.70 = 0.205
```

About one-fifth Kelly. This conservatism is intentional: the Kelly formula
assumes we will play many identical bets, but in-play soccer events are
rare and the true edge is uncertain.

`α_surprise` is calibrated from the 307-match Kalshi backtest, not set
arbitrarily.

#### Final position size

```
dollar_amount = f_final × bankroll
contracts     = floor(dollar_amount / P_kalshi)
```

All of the following hard caps must pass before any order is placed:

```
Per-order cap:     max $50
Per-match cap:     max 10% of bankroll
Total exposure:    max 20% of bankroll across all active positions
Liquidity gate:    contracts ≤ depth in Kalshi orderbook at our target price
```

If any cap is binding, the order size is reduced to the cap. If the liquidity
gate fails entirely (orderbook is too thin), the order is skipped.

---

### 8.5 Limit Orders: Becoming the Market Maker

A critical design choice is to use **limit orders** rather than market orders.

#### Why this matters

Kalshi's orderbook typically shows a wide bid-ask spread — often 20–60¢
between the best bid and best ask. A market order fills at the best ask
price, paying that full spread. In many cases, the spread would consume most
of the theoretical edge.

Instead, we post a limit order at our model's fair value and wait for
someone to trade against us:

```
P_model says home_win probability = 0.62
Kalshi best_ask = 0.73   (someone selling YES contracts at 73¢)

Market order: buy at 0.73, edge = 0.62 − 0.73 = −0.11   (losing trade!)
Limit order:  post bid at 0.62, wait for fill

If filled at 0.62: edge = 0.62 − 0.62 = 0¢ at fill, but position is
now at fair value. We exit when Kalshi converges toward 0.62 from below.
```

The distinction is subtle but important. We are not betting that Kalshi is
wrong right now — we are providing liquidity at fair value and profiting when
the market corrects toward fair value.

#### Order lifetime management

A limit order that is not filled within 30 seconds is cancelled. If the
model's probability has moved significantly since the order was placed, a
fresh order is posted at the new price. This prevents stale orders from
filling at prices that are no longer relevant.

```
On each tick for open limit orders:
  If |P_model_now − P_order| > 0.02: cancel and re-post at new P_model
  If order_age > 30 seconds: cancel
  If P_model_now < θ_exit: cancel (edge has evaporated)
```

---

### 8.6 When to Exit: The Bias Decay Model

Every open position is evaluated every tick against six exit triggers.
The first trigger that fires causes an immediate exit.

#### The theoretical basis for exit timing

The three behavioral biases described in §1.3 all share a common structure:
the mispricing is largest immediately after the event and decays over time
as market participants process the new information.

Choi and Hui (2014) quantified this for surprise goals on Betfair:

```
Mispricing at time Δt after goal = Initial mispricing × exp(−ρ × Δt)
```

With ρ ≈ 0.4/min, about 40% of the initial overreaction corrects within the
first minute. Given this decay, the optimal time to exit is:

```
τ* = (1/ρ) × ln(Initial EV / (ρ × exit_cost))
```

where exit_cost is the round-trip transaction cost (spread + slippage).

We estimate ρ from the 307-match Kalshi dataset by measuring how quickly
Kalshi prices converge after goal events. The optimal hold time τ* then
follows analytically.

**In practice:** τ* is used as a soft guide. The six triggers below
override it if the situation warrants an earlier or later exit.

#### The six exit triggers

Positions are monitored every tick. The first trigger that fires wins.

**Trigger 1 — EDGE_DECAY**
```
Current EV = P_model − P_kalshi < θ_exit

θ_exit is computed the same way as θ_entry (dynamic, uncertainty-adjusted).
Meaning: Kalshi has converged close to our model's probability.
The mispricing we entered to exploit has largely corrected.
Action: Take profit.
```

**Trigger 2 — EDGE_REVERSAL**
```
EV direction has flipped:
  We entered BUY_YES (P_model > P_kalshi)
  Now P_model < P_kalshi

Meaning: Either P_model fell (new information revised our estimate down)
or P_kalshi rose above P_model (market overshot past fair value).
Either way, we are now on the wrong side.
Action: Exit immediately, cut loss.
```

**Trigger 3 — POSITION_TRIM**
```
Current position size > 2× Kelly optimal at current P_model

Meaning: P_model has fallen since entry. The position we sized for
P_model = 0.62 is now oversized for P_model = 0.55.
Action: Partially reduce position to current Kelly optimal size.
```

**Trigger 4 — OPPORTUNITY_COST**
```
The opposite direction on the same market has EV > θ_entry

Meaning: It would be better to exit our current position and enter
the opposite side than to hold. The market has moved against us far
enough that the other direction now looks attractive.
Action: Exit current position, re-enter opposite direction.
```

**Trigger 5 — EXPIRY_EVAL**
```
Match time > 85 minutes

Meaning: The match is nearly over. The economics of holding to
settlement vs exiting now change near the end — the remaining
time value of the edge must be weighed against the cost of exit.

Compute: Expected P&L of holding to settlement = current edge × decay
If expected P&L < exit cost: sell now.
Otherwise: hold.
```

**Trigger 6 — EKF_DIVERGENCE**
```
EKF uncertainty P_H or P_A has grown above threshold (1.5)

Meaning: The model has lost confidence in its team strength estimate.
This happens when unexpected events (multiple quick goals, unusual
scoring patterns) push the EKF state far from its prior. When the
model is highly uncertain, P_model is unreliable.
Action: Exit defensively. Wait for model to restabilize.
```

**Minimum hold time:** No position is exited within 50 ticks (~150 seconds)
of entry, except for Trigger 2 (edge reversal) which is allowed to fire
immediately. This prevents the system from churning in and out of the same
position on small fluctuations.

**Cooldown after exit:** After any exit, the system waits 100 ticks (~5 minutes)
before entering the same market again. This prevents immediately re-entering
the same trade after a stop-out.

---

### 8.7 Execution Mechanics: Reserve-Confirm-Release

When multiple match containers run simultaneously, a race condition can arise:
two containers might both try to allocate capital to the same trade. The
reserve-confirm-release pattern prevents this.

```
1. reserve_exposure(amount)
   Write a RESERVED row to the exposure_reservations table.
   This is a fast DB write (<10ms) that atomically claims the capital.
   If the total exposure limit would be exceeded, this fails immediately.

2. execute_order()
   Submit the limit order to Kalshi.
   Wait up to 5 seconds for a fill (or partial fill).
   No database lock held during this wait.

3. confirm_exposure(fill_amount) or release_exposure()
   On fill: update reservation to CONFIRMED with actual fill amount.
   On no fill: delete the reservation, releasing the capital.
```

A CRON job runs every 5 minutes to release any RESERVED rows older than
60 seconds — these represent cases where the process died between reserve
and release.

---

### 8.8 Settlement and P&L

When a match ends, Phase 4 polls Kalshi for the settlement result of all
open markets. Kalshi resolves contracts 10–30 minutes after the final whistle.

For each settled market:

```
if we held YES contracts and outcome occurred:  profit = (1.00 − entry_price) × contracts
if we held YES contracts and outcome did not:   loss   = entry_price × contracts
if we exited before settlement:                  P&L computed at exit price
```

Realized P&L is recorded to the `positions` table and published to Redis for
the dashboard.

---

### 8.9 Multi-Match Portfolio Kelly

#### The problem with independent per-match Kelly

The Kelly sizing described in §8.4 treats each match independently. When
only one match is running, this is correct. But when several matches run
simultaneously — a common scenario across 8 leagues — independent Kelly
sizing has a structural flaw: it ignores correlations between positions.

Consider two EPL matches running at the same time. Both might involve
Premier League teams, both might be affected by the same weather event,
and their outcomes are mildly correlated. Independent Kelly applied to each
match allocates capital as if the positions are unrelated, which can lead to
overexposure when several positions move against us simultaneously.

More fundamentally, the Kelly Criterion's theoretical optimality applies
to a **single repeated bet**. When sizing multiple simultaneous positions,
the correct formulation is a portfolio optimization problem.

#### The portfolio Kelly formulation

Busseti, Ryu and Boyd (2016) showed that for simultaneous positions, the
growth-optimal allocation solves:

```
maximize    E[log(1 + r^T f)]
subject to  Σ_i f_i ≤ F_max
            f_i ≥ 0  for all i
```

where `f_i` is the fraction of bankroll allocated to position i, `r_i` is
the random return of position i, and `F_max` is the total exposure cap.

For binary Kalshi contracts (pay $1 or $0), the return of a YES position
in market i is:

```
r_i = (1 − P_kalshi_i) / P_kalshi_i    with probability  P_model_i
r_i = −1                                with probability  1 − P_model_i
```

The expected log-growth is then:

```
E[log(1 + r_i × f_i)] = P_model_i × log(1 + f_i × (1−P_kalshi_i)/P_kalshi_i)
                       + (1−P_model_i) × log(1 − f_i)
```

Summing across all active positions and maximizing over `(f_1, ..., f_N)`
subject to the constraints gives the portfolio-optimal allocation. This
is a convex optimization problem and solves efficiently with standard
solvers even for N = 20–30 simultaneous positions.

#### Key properties of portfolio Kelly

**Lower drawdown than independent Kelly:** When positions are positively
correlated (e.g. two home-win bets in the same league on the same day),
portfolio Kelly reduces both positions relative to the independent
allocation. The math recognizes that the combined downside is larger than
the sum of individual downsides.

**Automatic diversification:** If one position has a very strong apparent
edge, portfolio Kelly will still limit it if it consumes too much of the
exposure budget relative to the rest of the portfolio. This prevents a
single high-conviction trade from dominating the bankroll.

**Degenerates to single-match Kelly:** When N = 1, the portfolio Kelly
solution is identical to the standard Kelly formula from §8.4. The
generalization is backward compatible.

#### Practical implementation

In practice, portfolio Kelly is solved once per tick across all active
positions simultaneously:

```
Active positions: all open positions + candidate new positions this tick
Inputs per position:
  P_model_i    (from TickPayload)
  P_kalshi_i   (from Kalshi orderbook)
  sigma_p_i    (total uncertainty, same as §8.2)

Solver: CVXPY with ECOS (fast convex solver, <5ms for N ≤ 30)

Output: f_i for each position
  → If f_i increased vs current: add to position
  → If f_i decreased: trim position (Trigger 3 from §8.6)
  → If f_i = 0 for candidate: do not enter
```

The Baker-McHale shrinkage from §8.4 is applied to each `f_i` before
passing to the solver, so model uncertainty is already incorporated in
the inputs.

**Constraints enforced by the solver:**
```
Σ_i f_i  ≤  0.20   (total exposure cap: 20% of bankroll)
f_i      ≤  0.10   (per-match cap: 10% of bankroll)
f_i      ≥  0      (no short positions on Kalshi)
```

#### Current status and rollout plan

Portfolio Kelly requires multiple simultaneous matches, which only occurs
reliably once the orchestrator is running across all 8 leagues. The
implementation plan is:

```
Phase 1 (now):    Per-match independent Kelly (§8.4)
                  Correct for single-match testing and paper trading

Phase 2 (post-backtest): Portfolio Kelly activated
                  Once orchestrator is live and ≥2 matches run simultaneously
                  A/B test: compare Sharpe ratio vs independent Kelly
                  over 4-week paper trading window before applying to live capital
```

---

Phase 4 operates in one of two modes:

**Paper mode:** All trade logic runs exactly as in live mode, including
EV calculation, Kelly sizing, and order generation. Instead of submitting
orders to Kalshi's API, they are recorded to the database as simulated fills
at the current Kalshi market price. This produces a realistic P&L simulation
without risking capital.

**Live mode:** Orders are submitted to Kalshi's API using RSA-PSS
authentication. Fills from the real orderbook are recorded.

The mode is set per-system and is reflected in the `trading_mode` field
of every match and position record. **No live capital until the 307-match
backtest and 2-week paper trading period both pass their success criteria.**


---


---

## 9. Phase 5: The Orchestrator

### 9.1 Purpose

The orchestrator is the system's traffic controller. It runs continuously,
asking one question every few hours:

> **"Are there any upcoming soccer matches on Kalshi that we should be
> trading, and are they properly set up?"**

If the answer is yes, it ensures Phase 2 runs at the right time, launches
the match container at kickoff, monitors it throughout the match, and cleans
up afterward.

---

### 9.2 Match Lifecycle

Every match the system trades goes through a defined sequence of states:

```
SCHEDULED
  The match has been discovered on Kalshi and Goalserve.
  Kickoff is more than 65 minutes away.
  No computation has run yet.

PHASE2_RUNNING
  Phase 2 has been triggered (65 minutes before kickoff).
  Backsolve is running, Kalshi tickers being matched.

PHASE2_DONE
  Phase 2 completed successfully. a_H, a_A, tickers all set.
  Waiting for kickoff.

PHASE2_SKIPPED
  Phase 2 ran but returned SKIP verdict (sanity check failed,
  no odds data, or no Kalshi market found).
  No container will be launched for this match.

PHASE3_RUNNING
  Match container launched ~2 minutes before kickoff.
  Phase 3 and Phase 4 are active inside the container.
  All three concurrent coroutines are running.

FINISHED
  Final whistle detected. Tick loop terminated.
  Settlement polling underway.

ARCHIVED
  Settlement complete, P&L recorded.
  Container stopped and removed.
```

---

### 9.3 How Matches Are Discovered

The orchestrator queries two sources every 6 hours:

**Kalshi:** Lists all active markets for our 8 league series prefixes
(KXEPLGAME, KXLALIGAGAME, etc.). Each market has a close time, which
approximates the kickoff time.

**Goalserve:** Lists upcoming fixtures for each league. Provides the
canonical home and away team names and confirmed kickoff time.

The orchestrator matches these two lists using a 251-team alias table
that handles name variations across data sources (e.g. "Man City" vs
"Manchester City" vs "Man. City"). Only matches that appear in both
sources are scheduled for trading.

---

### 9.4 Container Architecture

Each match runs in its own isolated Docker container. This design means:

- Matches never interfere with each other (separate memory, separate state)
- A crash in one match container does not affect other running matches
- The orchestrator can stop, inspect, or restart individual match containers
- Resource usage scales linearly with the number of simultaneous matches

The orchestrator maintains the `mmpp-net` Docker network. All match
containers and infrastructure services (PostgreSQL, Redis) are attached
to this network so they can communicate.

**What is inside a match container:**
- Phase 3 live engine (tick loop, Goalserve poller, Kalshi WS listener)
- Phase 4 execution engine (signal generation, order management, settlement)
- All model parameters loaded from the database at startup
- A copy of `Phase2Result` (team strength values, Kalshi tickers)

**What the orchestrator does not do:**
The orchestrator does not run any live trading logic. If the orchestrator
process dies for 30 minutes, all running match containers continue
normally — they are fully self-contained. The orchestrator is only needed
for match discovery, Phase 2 triggering, and container lifecycle.

---

### 9.5 Recovery on Restart

If the orchestrator restarts, it scans the `match_schedule` database table
for matches in intermediate states:

- **PHASE2_RUNNING** past its trigger time → re-run Phase 2 immediately
- **PHASE2_DONE** past kickoff with no container running → launch container
- **PHASE3_RUNNING** with no heartbeat → container may have crashed, investigate

This ensures no match is silently dropped due to an orchestrator restart.

---

## 10. Phase 6: The Dashboard

### 10.1 Purpose

The dashboard provides real-time visibility into everything the system is doing.
It is built for a single operator who needs to understand, at a glance:

- Which matches are currently trading
- What the model currently thinks about each match
- What positions are open and how they are performing
- Whether any part of the system has a problem

---

### 10.2 Architecture

The dashboard has two components:

**Dashboard API (FastAPI, Python)**
- Reads historical data from PostgreSQL via REST endpoints
- Subscribes to Redis pub/sub channels for real-time updates
- Forwards real-time updates to connected browser clients via WebSocket

**Dashboard UI (React)**
- Connects to the Dashboard API on load (gets initial state via REST)
- Then maintains a WebSocket connection for live updates
- Reconnects automatically with exponential backoff if the connection drops

The UI never touches the database directly. All data flows through the API.

---

### 10.3 What the Dashboard Shows

**Command Center (main view)**

One card per active match, showing:
- Current score, match minute, engine phase
- P_model for each market (home win, draw, away win) as a live bar
- Current SurpriseScore (highlighted in orange when > 0.5)
- EKF uncertainty values (P_H, P_A)
- HMM state (home dominant / balanced / away dominant)
- Open positions with current unrealized P&L
- order_allowed status (green = can trade, red = cooldown/frozen)

**Match Deep Dive**

For a selected match:
- Full minute-by-minute P_model chart overlaid with Kalshi prices
- Team strength trajectory (a_H, a_A over time, showing EKF updates)
- All goal events annotated with SurpriseScore
- Position entry/exit times marked on the chart
- Monte Carlo uncertainty band around P_model

**P&L Analytics**

- Cumulative P&L (paper and live) over rolling windows
- P&L broken down by: market type, league, SurpriseScore bucket, match phase
- Edge realization rate (realized P&L / theoretical EV)
- Win rate by trade category

**System Operations**

- Status of all match containers (running, finished, crashed)
- Phase 1 last run time and Brier Score per league
- System alerts (any errors or warnings from any container)
- Bankroll balance (paper and live separate)

---

## 11. Before Going Live: Validation Gates

No real capital is deployed until two sequential validation stages pass.

### 11.1 Stage 1: The 307-Match Historical Backtest

We have historical Kalshi trade data for 307 EPL matches. This allows us
to replay what our system would have done on each match, using the actual
Kalshi prices that were available, and measure the result.

**How the backtest works:**

```
For each of the 307 EPL matches:

  1. Reconstruct minute-by-minute Kalshi prices
     (90-second VWAP window centered on each minute)

  2. Run the MMPP model forward through the match
     Apply EKF updates at each goal event
     Produce P_model at each minute

  3. Simulate trades
     When |P_model − P_kalshi| > θ_entry → enter
     Monitor exit triggers each minute
     Record P&L at exit or settlement

Aggregate across all 307 matches:
  → Total P&L, Sharpe ratio, win rates, edge realization
```

**All six criteria must pass before proceeding:**

| Criterion | Target | Meaning |
|-----------|--------|---------|
| Total P&L | > 0 | System is profitable over this dataset |
| Sharpe ratio | > 1.0 | Risk-adjusted return is meaningful |
| Win rate (high SurpriseScore trades) | ≥ 55% | The core edge hypothesis holds |
| Edge realization | ≥ 30% | We capture at least 30¢ of every $1 theoretical EV |
| Max drawdown | < 20% | The system doesn't blow up on a bad run |
| Stoppage edge | >3¢ gap in ≥30% of matches | Late-game bias is real and consistent |

If any criterion fails, we diagnose the problem before proceeding. If
Stage 1 fails entirely (negative P&L), the core hypothesis needs revision.

### 11.2 Stage 2: Two-Week Paper Trading

After the backtest passes, we run the complete live system for two weeks
in paper trading mode. All logic runs exactly as in live mode — the only
difference is that orders are recorded to the database rather than submitted
to Kalshi's API.

**What we measure during paper trading:**

| Metric | Target | Why it matters |
|--------|--------|---------------|
| Limit order fill rate | > 40% | If too few orders fill, the edge may be theoretical only |
| EV retention at fill | > 70% | How much of the signal survives to execution |
| Kalshi price convergence time | Matches ρ estimate | Validates our exit timing model |
| live_stats availability | All 8 leagues | Ensures Layer 2 can actually run |
| EKF stability | P_H, P_A in [0.01, 1.5] | No runaway estimates |

Paper trading also catches operational issues (container crashes, API rate
limits, network timeouts) that cannot be simulated in backtesting.

**Decision matrix after paper trading:**

| Backtest | Paper fill rate | Paper EV retention | Decision |
|----------|----------------|-------------------|---------|
| Pass | > 40% | > 70% | **GO LIVE** |
| Pass | > 40% | 40–70% | **Investigate execution, may proceed carefully** |
| Pass | < 40% | Any | **Revise limit order strategy before going live** |
| Fail | Any | Any | **Do not proceed. Revise model.** |

---

## 12. Risks and How We Manage Them

Every assumption in the system can be wrong. These are the most important
risks and what we do about each:

| Risk | How likely | How bad | What we do |
|------|-----------|---------|-----------|
| Kalshi bias is smaller than Betfair research suggests | Medium | Fatal | The 307-match backtest measures this directly. If it's too small, don't proceed. |
| EKF σ²_ω is poorly estimated, causing unstable updates | Low | High | Toggle to disable EKF and fall back to fixed a_H/a_A. A/B test both. |
| live_stats not available for Americas leagues | Medium | Medium | DomIndex fallback works without live_stats. Test each league during setup. |
| Limit orders don't fill often enough | Medium | Medium | Measure fill rate in paper trading. Adjust pricing if needed. |
| Goalserve live_stats feed drops during a match | Low | Low | Graceful fallback to Layer 1 only. System continues without Layer 2. |
| VAR cancel detected late (>90 seconds after goal) | Low | Low | Rollback stack covers 3 snapshots. Most VAR decisions come within 60s. |
| EKF diverges (P_H grows uncontrollably) | Low | Medium | Hard clamp P_H to [0.00001, 1.5] in code. Log and alert if triggered. |
| Kalshi market is illiquid at the moment we want to trade | Medium | Medium | Liquidity gate: always check orderbook depth before placing any order. |
| Multiple simultaneous matches cause capital overcommitment | Low | High | Total exposure cap of 20% of bankroll across all active positions. |

---

## 13. Complete Parameter Reference

This table lists every tunable parameter in the system, where it comes from,
and its current status.

### 13.1 Layer 1 Parameters (Estimated in Phase 1)

| Parameter | Symbol | Shape | Estimated From | Status |
|-----------|--------|-------|---------------|--------|
| Time profile | b | [8] | NLL optimization on 11,531 matches | ✅ Calibrated |
| Home red card effect | γ_H | [4] | NLL optimization | ✅ Calibrated |
| Away red card effect | γ_A | [4] | NLL optimization | ✅ Calibrated |
| Red card transition rates | Q | [4×4] | Red card frequency MLE | ✅ Calibrated |
| Home score-state (symmetric) | δ_H | [5] | NLL optimization | ✅ Calibrated |
| Away score-state (symmetric) | δ_A | [5] | NLL optimization | ✅ Calibrated |
| Home score-state when leading | δ_H⁺ | [5] | Asymmetric MLE | 🔲 TODO |
| Home score-state when trailing | δ_H⁻ | [5] | Asymmetric MLE | 🔲 TODO |
| Away score-state when trailing | δ_A⁺ | [5] | Asymmetric MLE | 🔲 TODO |
| Away score-state when leading | δ_A⁻ | [5] | Asymmetric MLE | 🔲 TODO |
| First-half stoppage multiplier | η_H, η_A | scalar | Stoppage segment MLE | 🔲 TODO |
| Second-half stoppage multiplier | η_H², η_A² | scalar | Stoppage segment MLE | 🔲 TODO |
| EKF process noise | σ²_ω | scalar | Walk-forward MLE on half-match splits | 🔲 TODO |
| Initial EKF uncertainty | P_0 | scalar | Phase 2 backsolve residual | ✅ Per-match |

### 13.2 Layer 2 Parameters (Estimated from Recordings)

| Parameter | Symbol | Estimated From | Status |
|-----------|--------|---------------|--------|
| HMM emission means | μ_k(z) for k=1..4, z=−1,0,+1 | Baum-Welch EM on live_stats recordings | 🔲 TODO (needs data) |
| HMM emission variances | σ_k² | Same | 🔲 TODO (needs data) |
| HMM transition matrix | Γ | Same | 🔲 TODO (needs data) |
| HMM intensity adjustment | φ_H, φ_A | Same | 🔲 TODO (needs data) |
| DomIndex decay rate | κ | MLE on historical goal sequences | 🔲 TODO |

### 13.3 Layer 3 Parameters (Estimated from Kalshi Data)

| Parameter | Symbol | Estimated From | Status |
|-----------|--------|---------------|--------|
| Bias decay rate | ρ | Exponential fit to 307-match Kalshi price convergence | 🔲 TODO |
| Optimal hold time | τ* | Analytic formula using ρ and c_exit | 🔲 TODO (depends on ρ) |
| Kelly base multiplier | α_base | Conservative fixed value | ✅ 0.10 |
| Kelly surprise scale | α_surprise | Sharpe optimization on 307-match backtest | 🔲 TODO |
| Kalshi effective spread | c_spread | Measured from 307-match trade data | ✅ ~0.010 |
| Slippage cost | c_slippage | Measured from paper trading | ✅ ~0.005 |

### 13.4 Fixed System Constants

These are not estimated from data but are design decisions:

| Constant | Value | Rationale |
|----------|-------|-----------|
| Monte Carlo paths | 50,000 | Validated: 7.6ms on CPU, sufficient precision |
| Confidence level for threshold | 95% (z = 1.645) | Conservative: large enough to filter noise |
| Minimum hold ticks | 50 (~150 seconds) | Prevents churn on small fluctuations |
| Cooldown after exit | 100 ticks (~5 minutes) | Prevents re-entry on same momentum |
| Max order lifetime | 30 seconds | Prevents stale fills at old prices |
| Post-goal cooldown | 50 ticks | Allows Kalshi to begin repricing before we enter |
| Post-red-card cooldown | 30 ticks | Shorter: red card effect is more predictable |
| Per-order cap | $50 | Hard limit regardless of Kelly calculation |
| Per-match cap | 10% of bankroll | Prevents overexposure to a single match |
| Total exposure cap | 20% of bankroll | Hard portfolio-level risk limit |

---

## 14. Academic References

Every modeling decision in the system is grounded in published research.
This table maps each academic paper to where it is applied:

| Paper | Key finding | Applied in |
|-------|-------------|-----------|
| Dixon & Coles (1997) | Bivariate Poisson with low-score correction; foundation of soccer modeling | Phase 1 calibration baseline |
| Dixon & Robinson (1998) | Non-homogeneous Poisson; score-state effects; stoppage goal spike | Layer 1: time profile, score state, η |
| Koopman & Lit (2015), JRSSA | Dynamic team strength via state-space model; Kalman filter motivation | Layer 1: EKF design |
| Vecer, Kopřiva & Ichiba (2009) | Red card effect empirically: ×0.67 / ×1.25 | Layer 1: γ initialization |
| Titman et al. (2015), JRSSA | Stoppage time intensity 1.4–1.8× baseline | Layer 1: η estimation |
| Heuer, Müller & Rubner (2010) | Goals are Poisson within a match; excess draws from team heterogeneity | Validates Poisson framework |
| Heuer & Rubner (2012) | Asymmetric score-state effects when trailing vs leading | Layer 1: asymmetric δ design |
| Choi & Hui (2014), JEBO | Surprise goal overreaction; bias decays ~40%/min | Layer 3: SurpriseScore, τ* |
| Ötting, Langrock & Maruotti (2023) | HMM 3-state momentum model for football | Layer 2: HMM design |
| Decroos et al. (2019), KDD | VAEP: action value for momentum estimation | Layer 2: DomIndex motivation |
| Baker & McHale (2013), Decision Analysis | Kelly shrinkage under parameter uncertainty | Layer 3: Baker-McHale formula |
| Busseti, Ryu & Boyd (2016) | Risk-constrained Kelly via convex optimization | Layer 3: future portfolio Kelly |
| Croxson & Reade (2014), Economic Journal | Betfair semi-strong efficiency; rapid price updates after goals | §1.1: why speed edge failed |
| Goddard & Asimakopoulos (2004) | Fixed-odds market efficiency; end-of-season effects | Background: market structure |
| Moskowitz (2021), Journal of Finance | Asset pricing and sports betting; behavioral biases in betting markets | §1.2: theoretical basis for biases |

---

*This concludes the five-part explanation of how the MMPP v5 system works.*
*Parts 1–5 together describe the complete system from first principles to full parameter reference.*
