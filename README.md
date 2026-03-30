# Real-Time MMPP(Markov-Modulated Poisson Process) Pricing Engine for Football(Soccer) Prediction Markets

A statistical model for real-time soccer match outcome pricing that exploits structural inefficiencies in prediction markets. The system estimates true outcome probabilities using a Markov-Modulated Poisson Process (MMPP) with regime-switching dynamics, validates edge against the sharpest available bookmaker (Pinnacle), and executes Kelly-optimal positions on Kalshi prediction markets at 1-second frequency.

---

## Thesis

Soccer match outcome markets reprice slowly after in-match events (goals, red cards) relative to the speed at which the true probability distribution shifts. A well-calibrated generative model of the goal-scoring process — one that explicitly accounts for time-varying intensity, regime changes, and score-differential feedback — can produce probability estimates that lead the market by seconds to minutes, creating systematically exploitable edge.

The key insight is that most market participants (including bookmaker algorithms) reprice reactively using heuristic adjustments, while a properly specified stochastic process model reprices generatively from first principles in under 1ms.

---

## Model Specification

### Scoring Intensity

Each team's instantaneous goal-scoring rate follows a log-linear intensity function modulated by three state variables:

$$\lambda_H(t \mid X, \Delta S) = \exp\!\left(a_H + b_{i(t)} + \gamma^H_{X(t)} + \delta_H(\Delta S(t))\right)$$

$$\lambda_A(t \mid X, \Delta S) = \exp\!\left(a_A + b_{i(t)} + \gamma^A_{X(t)} + \delta_A(\Delta S(t))\right)$$

| Parameter | Interpretation | Dimension |
|-----------|---------------|-----------|
| $a_H, a_A$ | Match-level baseline intensity (team strength prior) | 2 per match |
| $b_{i(t)}$ | Piecewise-constant time profile across six 15-minute intervals | 6 (shared) |
| $\gamma^H, \gamma^A$ | Red-card state adjustments: dismissal $\leftrightarrow$ scoring rate shift | 4 each (8 total) |
| $\delta_H, \delta_A$ | Score-difference tactical effect: trailing teams attack, leading teams defend | 4 each (8 total) |

The Markov chain $X(t) \in \{0, 1, 2, 3\}$ tracks the red-card state (11v11, 10v11, 11v10, 10v10) with transition rate matrix $Q$ estimated from empirical dismissal rates. The score difference $\Delta S(t) = S_H(t) - S_A(t)$ indexes the $\delta$ lookup.

### Why MMPP over alternatives

| Alternative | Limitation |
|------------|-----------|
| Static Poisson (Dixon-Coles) | No within-match dynamics; only prices pre-match |
| Time-varying Poisson (no regime) | Cannot capture discontinuous intensity jumps on red cards |
| Neural network / sequence model | Uninterpretable parameters; failure modes undiagnosable; requires orders of magnitude more data for comparable calibration on rare events (red cards occur in ~5% of matches) |
| Simple Bayesian update on market odds | No generative model of the scoring process; cannot price conditional outcomes or extrapolate under novel state configurations |

The MMPP provides an explicit generative model with interpretable parameters, closed-form likelihood, and the ability to price any derivative of the final score distribution — not just the outcomes that happen to have liquid markets.

---

## Estimation

### Data

~50,000 historical matches across 500+ leagues, 5+ seasons from a unified API source (Goalserve). Each match contributes a segmented timeline of constant-intensity intervals tagged with $(X(t),\; \Delta S(t))$, plus a four-tier feature vector for the XGBoost prior:

- **Tier 1:** Team rolling stats including xG (expected goals from the underlying shot model)
- **Tier 2:** Player-level per-90 metrics aggregated by position group (FW/MF/DF/GK)
- **Tier 3:** Implied probabilities from 20+ bookmakers (overround-removed), capturing the market's consensus strength estimate
- **Tier 4:** Contextual covariates (home/away, rest days, head-to-head)

### Two-stage estimation

**Stage 1 — XGBoost prior (cross-sectional):** A gradient-boosted tree with Poisson deviance loss maps the feature vector to match-level expected goals $\hat{\mu}_H, \hat{\mu}_A$. This serves as a warm start for the per-match baseline:

$$\hat{a}_H = \ln(\hat{\mu}_H) - \ln(C_{\text{time}}), \qquad \hat{a}_A = \ln(\hat{\mu}_A) - \ln(C_{\text{time}})$$

where $C_{\text{time}} = \sum_i \exp(b_i) \cdot \Delta\tau_i$ normalizes the time profile. XGBoost is preferred over neural alternatives because the feature space is tabular, the target has known distributional structure, and feature importances provide direct diagnostic value.

**Stage 2 — Joint NLL optimization (within-match, PyTorch):** All shared MMPP parameters are estimated by maximizing the likelihood of the observed goal sequences across the full historical dataset. The log-likelihood decomposes as:

$$\ell = \sum_{m} \left[ \sum_{\text{goals}} \ln \lambda_T(t_k) - \sum_{\text{intervals}} \int \lambda_H(t) + \lambda_A(t) \, dt \right]$$

The first sum captures **point-event terms** (log-intensity at each goal timestamp) and the second captures **survival terms** (integral of intensity over event-free intervals). Home and away goals contribute to separate NLL branches with team-specific $\gamma$ and $\delta$, enforcing the asymmetry between scoring and conceding under dismissals.

**Key estimation details:**

- Goals scored at time $t$ use **pre-goal** $\Delta S$ in the point-event term (causality: the goal cannot be caused by the state it creates)
- Own goals are excluded from the point-event term entirely (not driven by the attributed team's attacking intensity) but still trigger interval boundaries via $\Delta S$ change
- VAR-cancelled goals are excluded from both terms (the event did not occur)
- Multi-start optimization (5 random initializations) + $L_2$ regularization ($\lambda_{\text{reg}} = 0.01$) + parameter clamping ($|\gamma| \leq 1.0$, $|\delta| \leq 0.5$) to avoid pathological optima
- Halftime is excised via effective play-time transform; stoppage time per half is observed directly from data

### Transition matrix $Q$

The $4 \times 4$ continuous-time Markov chain rate matrix is estimated from empirical red-card transition counts, stratified by league tier. Off-diagonal rates are on the order of 0.02–0.05 per match-minute (red cards are rare), making the matrix exponential $e^{Q \cdot \Delta t}$ close to the identity for short intervals — which is why precomputation on a minute-by-minute grid is sufficient.

---

## Validation

### Walk-forward cross-validation

Three expanding-window folds with strict temporal ordering (no future data leakage). Within each fold, Stage 1 (XGBoost) and Stage 2 (NLL) are retrained from scratch on the training window.

### Benchmark: Pinnacle closing line

The primary benchmark is the Pinnacle close line — widely regarded as the sharpest odds in the market. Pinnacle implied probabilities (overround-removed) serve as the efficient-market baseline:

$$BS_{\text{model}} = \frac{1}{N}\sum_n (P_{\text{model},n} - O_n)^2, \qquad BS_{\text{Pinnacle}} = \frac{1}{N}\sum_n (P_{\text{Pinnacle},n} - O_n)^2$$

$$\Delta BS = BS_{\text{model}} - BS_{\text{Pinnacle}}$$

The system must demonstrate $\Delta BS < 0$ to justify any claim of informational advantage.

### Diagnostic metrics

| Metric | What it reveals | Pass criterion |
|--------|----------------|----------------|
| **Brier Score vs. Pinnacle** | Calibration + discrimination vs. efficient market | $\Delta BS < 0$ across all folds |
| **Calibration reliability diagram** | Predicted probabilities match empirical frequencies | $\pm 5\%$ of diagonal across all bins |
| **Multi-market consistency** | $\mu_H, \mu_A$ jointly predict 1X2, Over/Under, BTTS | Improve vs. market odds in all three |
| **$\gamma$ sign validation** | Red-card effects align with tactical priors | $\gamma^H_1 < 0$ (own dismissal hurts), $\gamma^H_2 > 0$ (opponent dismissal helps), symmetric for away |
| **$\delta$ sign validation** | Score-difference effects align with tactical priors | $\delta_H(-1) > 0$ (trailing $\to$ attack), $\delta_H(+1) < 0$ (leading $\to$ defend), symmetric for away |
| **$\delta$ likelihood ratio test** | Statistical significance of score-difference effect | $LR = -2(\mathcal{L}_{\delta=0} - \mathcal{L}_{\delta\neq 0}) \sim \chi^2(df)$, reject at $p < 0.05$ |
| **$b$ half-split validation** | Learned first/second-half intensity ratio matches empirical shot data | Within $\pm 10\%$ of observed shot split |
| **Simulated P&L + max drawdown** | Edge survives transaction costs and position sizing | Positive return in all folds; max drawdown $\leq 20\%$ |

The $\delta$ significance flag is a **model selection decision**: if the LRT fails to reject $\delta = 0$, the simpler model (without score-difference feedback) is used, and the in-play pricing engine can use exact analytic formulas rather than Monte Carlo. If $\delta$ is significant, the coupling between home and away intensities via $\Delta S$ breaks the independence assumption, requiring simulation.

### Symmetric vs. asymmetric parameter comparison

Both $\gamma$ and $\delta$ are tested in symmetric and asymmetric forms:

- **Symmetric $\gamma$ (2 params):** $\gamma^A_1 = -\gamma^H_2$, $\gamma^A_2 = -\gamma^H_1$ (home losing a player is the mirror of away gaining advantage)
- **Asymmetric $\gamma$ (4 params):** Independent estimation; justified if validation Log Loss improves

- **Symmetric $\delta$ (4 params):** $\delta_A(\Delta S) = \delta_H(-\Delta S)$ (trailing behavior is symmetric)
- **Asymmetric $\delta$ (8 params):** Independent; captures home-field asymmetry in tactical response

Model complexity is chosen by validation Log Loss, not by in-sample fit.

---

## In-Play Pricing

### Remaining expected goals

At match time $t$ with state $(X, \Delta S)$, the expected remaining goals for each team integrate the intensity over $[t, T]$, splitting at time-profile basis boundaries and marginalizing over uncertain future Markov states:

$$\mu_H(t, T) = \sum_{\ell=1}^{L} \sum_{j=0}^{3} \bar{P}_{X(t),j}^{(\ell)} \cdot \exp\!\left(a_H + b_{i_\ell} + \gamma^H_j + \delta_H(\Delta S)\right) \cdot \Delta\tau_\ell$$

where $\bar{P}_{X(t),j}^{(\ell)}$ denotes the time-averaged transition probability from current state $X(t)$ to state $j$ during subinterval $\ell$, obtained via matrix exponential lookup. $\Delta S$ is held constant at the current value — future score changes are handled by the pricing layer.

### Hybrid pricing

| State condition | Method | Rationale |
|----------------|--------|-----------|
| $X = 0,\; \Delta S = 0,\; \delta$ not significant | **Analytic:** Poisson CDF (Over/Under), Skellam distribution (match-winner) | Independence holds; exact closed-form solution |
| $X = 0,\; \Delta S = 0,\; \delta$ significant | Analytic (first-order approximation) | Ignores future $\delta$ feedback; acceptable when $\|\delta\| < 0.1$ |
| $X \neq 0$ or $\Delta S \neq 0$ | **Monte Carlo:** 50,000 forward simulations (Numba JIT, ~0.5ms) | Full path simulation with Markov transitions + score coupling; quantified standard errors ($< 0.3$pp) |

**Analytic pricing formulas.** Let $G = S_H + S_A$ denote current total goals:

$$P(\text{Over } N.5) = \begin{cases} 1 & \text{if } G > N \\\\ 1 - \sum_{k=0}^{N-G} \frac{\mu_{\text{total}}^{\,k} \cdot e^{-\mu_{\text{total}}}}{k!} & \text{if } G \leq N \end{cases}$$

$$P(\text{Home Win}) = \sum_{D=1}^{\infty} e^{-(\mu_H + \mu_A)} \left(\frac{\mu_H}{\mu_A}\right)^{D/2} I_{|D|}(2\sqrt{\mu_H \mu_A})$$

where $I_{|D|}$ is the modified Bessel function of the first kind (Skellam distribution).

The analytic/MC boundary is not a performance hack — it's a correctness requirement. Once a goal occurs ($\Delta S \neq 0$) and $\delta$ is significant, scoring a second goal changes both teams' intensities simultaneously, creating a feedback loop that analytic formulas cannot capture. The MC engine samples complete match continuations including future goals, red cards, and their cascading effects on intensities.

### In-play backtest (200+ matches)

Before live deployment, the full pricing engine is replayed through historical matches with reconstructed event timelines. Eight diagnostic criteria must pass:

| Criterion | Threshold | Failure implication |
|-----------|-----------|---------------------|
| In-play Brier Score (home_win) | $< 0.20$ | Poor in-play accuracy |
| BS by time bin (0–15, 15–30, ..., 75–90+) | Decreasing trend | More information should improve predictions |
| Calibration max deviation | $\leq 7\%$ | In-play probability estimates are biased |
| Monotonicity violations | $< 1\%$ of event-free ticks | Numerical instability or basis-boundary artifacts |
| MC vs. analytic divergence (at $X=0, \Delta S=0$) | $\leq 1$pp | Implementation bug in one pricing path |
| Directional correctness after goals | $100\%$ | Any failure is a critical bug (sign error in $\delta$ or event handler) |
| Directional correctness after red cards | $100\%$ | Sign error in $\gamma$ or Markov transition logic |
| Simulated P&L (if market prices available) | $> 0$ | Edge exists against the market |

---

## Edge Exploitation

### Conservative probability adjustment

MC estimates carry sampling uncertainty $\sigma_{\text{MC}}$. To avoid systematically overestimating edge, the system uses a directional conservative bound:

$$P_{\text{cons}}^{\text{Yes}} = P_{\text{true}} - z \cdot \sigma_{\text{MC}} \qquad \text{(lower bound; higher } P \text{ favors Yes)}$$

$$P_{\text{cons}}^{\text{No}} = P_{\text{true}} + z \cdot \sigma_{\text{MC}} \qquad \text{(upper bound; lower } P \text{ favors No)}$$

Using a single lower bound for both directions would artificially inflate the $(1 - P_{\text{cons}})$ term for No positions, creating a systematic bias toward overbetting No when MC uncertainty is high. The directional adjustment eliminates this.

### Expected value

Fee-adjusted EV is computed using VWAP effective prices (not best bid/ask) to account for the actual fill cost at the target quantity:

$$EV_{\text{yes}} = P_{\text{cons}} \cdot (1-c) \cdot (1 - P_{\text{eff}}) \;-\; (1 - P_{\text{cons}}) \cdot P_{\text{eff}}$$

$$EV_{\text{no}} = (1 - P_{\text{cons}}) \cdot (1-c) \cdot P_{\text{eff}} \;-\; P_{\text{cons}} \cdot (1 - P_{\text{eff}})$$

A 2-pass computation resolves the circularity between position size and VWAP: Pass 1 estimates rough quantity at best bid/ask, Pass 2 recomputes EV at the VWAP for that quantity.

### Position sizing (fractional Kelly)

$$f^* = \frac{EV}{W \times L}$$

With direction-specific payoffs:

$$\text{Buy Yes:} \quad W = (1-c)(1-P_{\text{kalshi}}), \quad L = P_{\text{kalshi}}$$

$$\text{Buy No:} \quad W = (1-c)\,P_{\text{kalshi}}, \quad L = (1-P_{\text{kalshi}})$$

Raw Kelly is scaled by $K_{\text{frac}} \in [0.25, 0.50]$ and a market alignment multiplier ($0.8$ when the model agrees directionally with bet365 in-play odds, $0.5$ when divergent). bet365 is treated as a market alignment check rather than independent validation — both the model and bet365 derive from the same underlying data feed, limiting true independence.

### Risk management

Three-layer exposure caps: $3\%$ per order, $5\%$ per match (correlated markets pro-rated), $20\%$ portfolio-wide. A 15-second post-event cooldown blocks new orders while markets stabilize. An order-book freeze system halts trading within 0.5 seconds of detecting any event, with 3-tick stabilization and 10-second timeout release conditions.

---

## Alpha Characterization

### Source of edge

The model's informational advantage comes from three sources:

1. **Structural:** The MMPP explicitly models regime changes (red cards, score-differential feedback) that most market participants price heuristically. The generative model reprices from first principles in $<1$ms; the market takes seconds to minutes.

2. **Temporal:** Sub-second event detection (via Goalserve Live Odds WebSocket) combined with precomputed post-event $\mu$ means the model has an updated probability estimate before most market participants have even registered the event.

3. **Calibration:** Walk-forward validation against Pinnacle ensures the model's probability estimates are not just directionally correct but accurately calibrated — a prerequisite for Kelly-optimal sizing to generate positive long-run geometric growth.

### Expected alpha decay

| Timeframe | Edge type | Persistence |
|-----------|-----------|-------------|
| 0–5s post-event | Structural + temporal | Strongest; market hasn't repriced yet |
| 5–30s post-event | Structural | Market reprices heuristically; model's generative estimate may still be more accurate |
| 30s–3min post-event | Residual structural | Diminishing; market converges to true probability |
| Steady-state (no events) | Calibration | Small but persistent; model's time-decay is more accurate than market's heuristic $\theta$ |

The 15-second post-event cooldown deliberately sacrifices the strongest temporal edge in exchange for protection against stale-price fills and VAR cancellations — a conservative choice that can be relaxed as empirical data accumulates.

### Limitations and known weaknesses

- **Shared data source:** Both the model and bet365 (used as alignment check) consume Goalserve data, limiting the independence of the cross-validation signal.
- **$\delta$ estimation on rare events:** Score differences of $\pm 3$ or beyond are sparse in the training data, making $\delta$ estimates at extreme $\Delta S$ less reliable.
- **Substitution effects not modeled:** The current MMPP does not account for tactical substitutions, which can materially change team intensity in the final 20 minutes.
- **Liquidity constraints:** Kalshi prediction markets have limited depth; VWAP degradation at larger sizes caps the practical capacity of the strategy.
- **Single-match correlation:** Markets within the same match (home win, over 2.5, etc.) are correlated through the shared $\mu_H, \mu_A$; the match-level exposure cap partially addresses this but does not fully decorrelate.

---

## System Evolution

| Stage | Sizing | Filters | What's being measured |
|-------|--------|---------|----------------------|
| Paper (2–4 wk) | $K_{\text{frac}} = 0.25$ | Block divergent entries; no rapid entry | Calibration accuracy, edge existence, slippage estimates |
| Conservative (1–2 mo) | $K_{\text{frac}} = 0.25$ | Same | Realized P&L, Brier Score drift, fill quality |
| Adaptive (2–4 mo) | $K_{\text{frac}} \to 0.25\text{–}0.50$ | Allow divergent at $0.5\times$; tune $z$ directionally | Divergent trade performance, optimal $z$ by direction |
| Mature (ongoing) | Data-driven | Conditional rapid entry with VAR safety; bet365 auto-exit | Full auto-tuning; seasonal retraining |

Each transition is gated by quantitative criteria from post-match analytics — 11 metrics feeding 7 adaptive parameters — not by elapsed time.
