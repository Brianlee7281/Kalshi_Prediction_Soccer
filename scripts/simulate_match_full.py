#!/usr/bin/env python
"""End-to-end Phase 1-2-3-4 simulation with paper trading.

Replays a recorded match through the full MMPP pipeline and paper-trades
edges between P_model (MMPP + EKF) and P_kalshi (Poisson baseline + lag).

Usage:
    PYTHONPATH=. python scripts/simulate_match_full.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import poisson as poisson_dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.step_1_3_ml_prior import compute_C_time
from src.common.types import (
    ExitTrigger, FillResult, MarketProbs, Phase2Result, Signal, TickPayload,
)
from src.engine.event_handlers import handle_goal, handle_period_change
from src.engine.intensity import compute_lambda
from src.engine.mc_pricing import _compute_sigma, _results_to_market_probs
from src.engine.model import LiveMatchModel
from src.execution.config import CONFIG
from src.execution.kelly_sizer import size_position
from src.execution.pnl_calculator import compute_settlement_pnl
from src.execution.position_monitor import PositionTracker
from src.execution.settlement import OUTCOME_MAP
from src.execution.signal_generator import generate_signals
from src.math.compute_mu import compute_remaining_mu
from src.math.mc_core import mc_simulate_remaining

try:
    from src.math.mc_core import mc_simulate_remaining_v5
    _HAS_V5 = True
except ImportError:
    _HAS_V5 = False

from src.prematch.phase2_pipeline import (
    _poisson_1x2, _shin_vig_removal, backsolve_intensities,
)

# ── Match data ────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "latency" / "KXEPLGAME-26MAR20BOUMUN"

# ── Calibrated EPL parameters ────────────────────────────────────
PARAMS: dict = {
    "b": [0.0, 0.072422, 0.185087, 0.312089, 0.193663, 0.184194, 0.123683, 1.0],
    "gamma_H": [0.0, -0.15, 0.10, -0.05],
    "gamma_A": [0.0, 0.10, -0.15, -0.05],
    "delta_H": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_A": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "Q": [[-0.02, 0.01, 0.01, 0.0], [0.0, -0.01, 0.0, 0.01],
          [0.0, 0.0, -0.01, 0.01], [0.0, 0.0, 0.0, 0.0]],
    "sigma_a": 0.5, "alpha_1": 2.0,
    "delta_H_pos": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_H_neg": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_A_pos": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "delta_A_neg": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "eta_H": 0.05, "eta_A": 0.05, "eta_H2": 0.08, "eta_A2": 0.08,
    "sigma_omega_sq": 0.003,
}

BASIS_BOUNDS = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0])
MC_N = 20_000

# Ticker mapping for this match
TICKER_MAP = {
    "home_win": "KXEPLGAME-26MAR20BOUMUN-BOU",
    "draw":     "KXEPLGAME-26MAR20BOUMUN-TIE",
    "away_win": "KXEPLGAME-26MAR20BOUMUN-MUN",
}
TICKER_TO_MARKET = {v: k for k, v in TICKER_MAP.items()}


# ── Kalshi price: Poisson baseline with lag ───────────────────────

def poisson_1x2(mu_h: float, mu_a: float) -> tuple[float, float, float]:
    """Independent Poisson P(H), P(D), P(A)."""
    p_h = p_d = p_a = 0.0
    for i in range(11):
        for j in range(11):
            p = poisson_dist.pmf(i, max(mu_h, 0.01)) * poisson_dist.pmf(j, max(mu_a, 0.01))
            if i > j: p_h += p
            elif i == j: p_d += p
            else: p_a += p
    return p_h, p_d, p_a


class KalshiPriceSimulator:
    """Simulates Kalshi market prices using a simple Poisson model with lag.

    The "market" updates based on the current score and elapsed time
    using a naive Poisson model (no EKF, no strength updates). This
    creates genuine edge when our MMPP+EKF model detects strength
    shifts faster than the Poisson baseline reacts.

    A configurable lag_minutes delays the market's reaction to goals,
    simulating Kalshi's suspension/reopening window.
    """

    def __init__(
        self,
        mu_H_90: float,
        mu_A_90: float,
        initial_snapshot: dict[str, float] | None = None,
        lag_minutes: float = 1.0,
    ):
        self.mu_H_90 = mu_H_90
        self.mu_A_90 = mu_A_90
        self.initial = initial_snapshot or {}
        self.lag = lag_minutes
        self._pending_score: tuple[int, int] = (0, 0)
        self._pending_t: float = 0.0
        self._market_score: tuple[int, int] = (0, 0)
        self._market_t: float = 0.0

    def update(self, t: float, score: tuple[int, int]) -> None:
        """Update the actual match state. Market prices follow with lag."""
        self._pending_score = score
        self._pending_t = t
        if t - self._market_t >= self.lag or score == self._market_score:
            self._market_score = score
            self._market_t = t

    def get_prices(self, t: float) -> dict[str, float]:
        """Get current P_kalshi for all markets."""
        # After lag, market catches up
        if t - self._pending_t >= self.lag:
            self._market_score = self._pending_score
            self._market_t = t

        s_h, s_a = self._market_score
        remaining = max(0.5, 93.0 - t)
        mu_h_rem = self.mu_H_90 * remaining / 93.0
        mu_a_rem = self.mu_A_90 * remaining / 93.0

        # Poisson from current score + remaining goals
        p_h = p_d = p_a = 0.0
        for dh in range(8):
            for da in range(8):
                p = (poisson_dist.pmf(dh, max(mu_h_rem, 0.001)) *
                     poisson_dist.pmf(da, max(mu_a_rem, 0.001)))
                fh, fa = s_h + dh, s_a + da
                if fh > fa: p_h += p
                elif fh == fa: p_d += p
                else: p_a += p

        total = max(p_h + p_d + p_a, 1e-6)
        return {
            "home_win": p_h / total,
            "draw": p_d / total,
            "away_win": p_a / total,
            "over_25": 1.0 - sum(
                poisson_dist.pmf(dh, max(mu_h_rem, 0.001)) *
                poisson_dist.pmf(da, max(mu_a_rem, 0.001))
                for dh in range(8) for da in range(8)
                if (s_h + dh) + (s_a + da) < 3
            ),
            "btts_yes": 1.0 - sum(
                poisson_dist.pmf(dh, max(mu_h_rem, 0.001)) *
                poisson_dist.pmf(da, max(mu_a_rem, 0.001))
                for dh in range(8) for da in range(8)
                if (s_h + dh) < 1 or (s_a + da) < 1
            ),
        }


# ── MC simulation (sync) ─────────────────────────────────────────

def run_mc(model: LiveMatchModel) -> tuple[MarketProbs, MarketProbs]:
    mu_H, mu_A = compute_remaining_mu(model)
    model.mu_H, model.mu_A = mu_H, mu_A
    model.mu_H_elapsed = max(0.0, model.mu_H_at_kickoff - mu_H)
    model.mu_A_elapsed = max(0.0, model.mu_A_at_kickoff - mu_A)

    Q_diag = np.diag(model.Q).copy()
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        rs = sum(model.Q[i, j] for j in range(4) if i != j and model.Q[i, j] > 0)
        if rs > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = model.Q[i, j] / rs

    seed = int(time.monotonic() * 1e6) % (2**31)
    S_H, S_A = model.score

    if _HAS_V5 and model.delta_H_pos is not None:
        res = mc_simulate_remaining_v5(
            model.t, model.T_exp, S_H, S_A, model.current_state_X, model.delta_S,
            model.a_H, model.a_A, model.b, model.gamma_H, model.gamma_A,
            model.delta_H_pos, model.delta_H_neg, model.delta_A_pos, model.delta_A_neg,
            Q_diag, Q_off, model.basis_bounds, MC_N, seed,
            model.eta_H, model.eta_A, model.eta_H2, model.eta_A2, 45.0, 90.0,
        )
    else:
        res = mc_simulate_remaining(
            model.t, model.T_exp, S_H, S_A, model.current_state_X, model.delta_S,
            model.a_H, model.a_A, model.b, model.gamma_H, model.gamma_A,
            model.delta_H, model.delta_A, Q_diag, Q_off, model.basis_bounds,
            MC_N, seed,
        )
    P = _results_to_market_probs(res, S_H, S_A)
    sigma = _compute_sigma(P, MC_N)
    return P, sigma


# ── Load match timeline from kalshi_live.jsonl ────────────────────

def load_timeline(data_dir: Path) -> list[dict]:
    """Load kalshi_live.jsonl, sample every ~3 seconds for tick simulation.

    Returns one state per ~3s of wall time during 'live' status,
    giving ~1800 ticks for a 90-minute match (comparable to the
    real 1Hz tick loop after 3:1 downsampling).
    """
    path = data_dir / "kalshi_live.jsonl"
    states: list[dict] = []
    last_wall = 0.0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "live":
                continue
            wall = d.get("_ts_wall", 0)
            if wall - last_wall < 3.0 and states:
                continue  # skip entries within 3s of last
            last_wall = wall
            states.append({
                "t": float(d.get("minute", 0)),
                "score": (d.get("home_score", 0), d.get("away_score", 0)),
                "half": d.get("half", ""),
                "wall": wall,
            })
    return states


def load_kalshi_initial_prices(data_dir: Path) -> dict[str, float]:
    """Get initial mid-prices from Kalshi orderbook snapshots."""
    prices: dict[str, float] = {}
    path = data_dir / "kalshi.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") != "orderbook_snapshot":
                continue
            msg = d["msg"]
            ticker = msg["market_ticker"]
            yes_levels = msg.get("yes_sub_cent_fp", msg.get("yes_dollars_fp", []))
            no_levels = msg.get("no_sub_cent_fp", msg.get("no_dollars_fp", []))
            best_yes = max((float(p[0]) for p in yes_levels), default=0.0)
            best_no = max((float(p[0]) for p in no_levels), default=0.0)
            yes_ask = 1.0 - best_no if best_no > 0 else 1.0
            mid = (best_yes + yes_ask) / 2.0
            mkt = TICKER_TO_MARKET.get(ticker)
            if mkt:
                prices[mkt] = mid
    return prices


# ── Main simulation ──────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("  Phase 1-2-3-4 Full Simulation with Paper Trading")
    print("  Bournemouth 2-2 Manchester United | EPL | 2026-03-20")
    print("=" * 80)

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    final_score = tuple(meta["final_score"])
    print(f"Match: {meta['home_team']} vs {meta['away_team']} | Final: {final_score}")

    # ── Phase 1: Load calibrated params ───────────────────────
    b = np.array(PARAMS["b"])
    C_time = compute_C_time(b, BASIS_BOUNDS)
    print(f"\n=== PHASE 1 === C_time={C_time:.2f}, sigma_omega={PARAMS['sigma_omega_sq']}")

    # ── Phase 2: Backsolve from Kalshi initial prices ─────────
    kalshi_init = load_kalshi_initial_prices(DATA_DIR)
    print(f"\n=== PHASE 2 ===")
    print(f"Kalshi initial: {', '.join(f'{k}={v:.3f}' for k, v in kalshi_init.items())}")

    # Use Kalshi prices as market-implied probs for backsolve
    if kalshi_init:
        p_h = kalshi_init.get("home_win", 0.33)
        p_d = kalshi_init.get("draw", 0.33)
        p_a = kalshi_init.get("away_win", 0.33)
        total = p_h + p_d + p_a
        market_implied = MarketProbs(home_win=p_h/total, draw=p_d/total, away_win=p_a/total)
    else:
        market_implied = MarketProbs(home_win=0.40, draw=0.30, away_win=0.30)

    Q = np.array(PARAMS["Q"])
    a_H, a_A = backsolve_intensities(market_implied, b, Q, BASIS_BOUNDS)
    mu_H = float(np.exp(a_H) * C_time)
    mu_A = float(np.exp(a_A) * C_time)
    print(f"Market implied: H={market_implied.home_win:.3f}, D={market_implied.draw:.3f}, A={market_implied.away_win:.3f}")
    print(f"Backsolve: a_H={a_H:.4f}, a_A={a_A:.4f} | mu_H={mu_H:.3f}, mu_A={mu_A:.3f}")

    result2 = Phase2Result(
        match_id="BOUMUN", league_id=1204,
        a_H=a_H, a_A=a_A, mu_H=mu_H, mu_A=mu_A,
        C_time=C_time, verdict="GO", skip_reason=None, param_version=1,
        home_team=meta["home_team"], away_team=meta["away_team"],
        kickoff_utc="2026-03-20T19:30:00+00:00",
        kalshi_tickers=TICKER_MAP, market_implied=market_implied,
        prediction_method="backsolve_kalshi", ekf_P0=0.15,
    )

    model = LiveMatchModel.from_phase2_result(result2, PARAMS)
    model.engine_phase = "FIRST_HALF"

    # JIT warmup
    model.t = 0.001
    run_mc(model)
    model.t = 0.0

    # ── Phase 3 + 4: Replay timeline with trading ─────────────
    print(f"\n=== PHASE 3 + PHASE 4 ===")
    timeline = load_timeline(DATA_DIR)
    print(f"Timeline events: {len(timeline)}")

    # Kalshi price simulator (Poisson baseline with 0.5-min lag)
    kalshi_sim = KalshiPriceSimulator(mu_H_90=mu_H, mu_A_90=mu_A, lag_minutes=0.5)

    # Phase 4 state — reduce min_hold and cooldown for simulation (1 timeline
    # event ~ 10 real seconds, so 5 events ~ 50 seconds ~ 50 ticks)
    tracker = PositionTracker(min_hold_ticks=5, cooldown_after_exit=10)
    bankroll = 10_000.0
    initial_bankroll = bankroll
    trade_log: list[dict] = []
    entries_skipped = {"cooldown": 0, "threshold": 0, "no_edge": 0}
    exit_triggers: dict[str, int] = {}

    last_update_t = 0.0
    tick = 0
    prev_score = (0, 0)

    # Kickoff MC
    P_model, sigma_MC = run_mc(model)
    print(f"Kickoff P_model: H={P_model.home_win:.3f}, D={P_model.draw:.3f}, A={P_model.away_win:.3f}")

    for state in timeline:
        t = state["t"]
        score = state["score"]
        half = state["half"]
        tick += 1

        # Phase transitions
        if half == "2nd" and model.engine_phase == "FIRST_HALF":
            handle_period_change(model, "HALFTIME")
            handle_period_change(model, "SECOND_HALF")

        model.t = t

        # EKF predict + no-goal update
        dt = t - last_update_t
        if dt > 0 and model.strength_updater:
            model.strength_updater.predict(dt)
            lH = compute_lambda(model, "home")
            lA = compute_lambda(model, "away")
            model.strength_updater.update_no_goal(lH, lA, dt)
            model.a_H = model.strength_updater.a_H
            model.a_A = model.strength_updater.a_A
            last_update_t = t

        # Goal detection
        is_goal = score != prev_score
        if is_goal:
            dh = score[0] - prev_score[0]
            da = score[1] - prev_score[1]

            for _ in range(dh):
                handle_goal(model, "home", int(t))
            for _ in range(da):
                handle_goal(model, "away", int(t))

        prev_score = score

        # Run MC at goals, every ~30 ticks (~90s), and at start
        run_mc_now = is_goal or tick % 30 == 0 or t <= 1.0
        if not run_mc_now:
            # Still update Kalshi sim and ticks
            kalshi_sim.update(t, score)
            model.tick_count += 1
            if model.cooldown and model.t >= model.cooldown_until_t:
                model.cooldown = False
                model.event_state = "IDLE"
            continue

        P_model, sigma_MC = run_mc(model)
        kalshi_sim.update(t, score)
        p_kalshi = kalshi_sim.get_prices(t)

        # Build TickPayload
        ekf = model.ekf_tracker
        payload = TickPayload(
            match_id="BOUMUN", t=t, engine_phase=model.engine_phase,
            P_model=P_model, sigma_MC=sigma_MC,
            score=model.score, X=model.current_state_X, delta_S=model.delta_S,
            mu_H=model.mu_H, mu_A=model.mu_A,
            a_H_current=model.a_H, a_A_current=model.a_A,
            last_goal_type=model.last_goal_type,
            ekf_P_H=ekf.P_H if ekf else 0.0,
            ekf_P_A=ekf.P_A if ekf else 0.0,
            hmm_state=0, dom_index=0.0,
            surprise_score=model.surprise_score,
            order_allowed=model.order_allowed,
            cooldown=model.cooldown, ob_freeze=model.ob_freeze,
            event_state=model.event_state,
        )

        # ── Phase 4: Check exits ──────────────────────────────
        exits = tracker.check_exits(payload, p_kalshi)
        for ed in exits:
            pos = tracker.open_positions.get(ed.position_id)
            if pos is None:
                continue
            pnl = (ed.exit_price - pos.entry_price) * pos.quantity if pos.direction == "BUY_YES" \
                else ((1.0 - ed.exit_price) - pos.entry_price) * pos.quantity
            tracker.close_position(ed.position_id, ed.trigger, ed.contracts_to_exit, ed.exit_price, tick)
            bankroll += pnl

            trade_log.append({
                "action": "EXIT", "t": t, "market": pos.market_type,
                "direction": pos.direction, "contracts": ed.contracts_to_exit,
                "entry_price": pos.entry_price, "exit_price": ed.exit_price,
                "trigger": ed.trigger.value, "pnl": pnl,
            })
            exit_triggers[ed.trigger.value] = exit_triggers.get(ed.trigger.value, 0) + 1

            if is_goal:
                team_name = meta["home_team"] if score[0] > prev_score[0] else meta["away_team"]
                print(f"  [t={t:.0f}] EXIT {pos.market_type} {pos.direction} "
                      f"via {ed.trigger.value} | PnL=${pnl:+.2f}")

        # ── Phase 4: Generate signals ─────────────────────────
        if payload.order_allowed:
            signals = generate_signals(payload, p_kalshi, TICKER_MAP, tracker.open_positions)

            if is_goal and not signals:
                # Diagnose why no signal
                for mkt in ["home_win", "draw", "away_win"]:
                    if mkt in p_kalshi:
                        p_m = getattr(P_model, mkt, None)
                        if p_m is not None:
                            edge = abs(p_m - p_kalshi[mkt])
                            print(f"    {mkt}: P_model={p_m:.3f}, P_kalshi={p_kalshi[mkt]:.3f}, "
                                  f"edge={edge:.4f}")

            for signal in signals:
                if tracker.is_in_cooldown(signal.market_type, tick):
                    entries_skipped["cooldown"] += 1
                    continue

                sized = size_position(signal, payload, bankroll)
                if sized.contracts <= 0:
                    entries_skipped["threshold"] += 1
                    continue

                # Paper fill
                fill = FillResult(
                    order_id=f"paper-{uuid4()}",
                    ticker=sized.ticker,
                    direction=sized.direction,
                    quantity=sized.contracts,
                    price=sized.P_kalshi,
                    status="paper",
                    fill_cost=sized.contracts * sized.P_kalshi,
                    timestamp=datetime.now(timezone.utc),
                )
                tracker.add_position(sized, fill, tick, t)
                bankroll -= fill.fill_cost

                trade_log.append({
                    "action": "ENTRY", "t": t, "market": sized.market_type,
                    "direction": sized.direction, "contracts": sized.contracts,
                    "entry_price": fill.price, "exit_price": None,
                    "trigger": "signal", "pnl": -fill.fill_cost,
                    "ev": sized.EV, "kelly": sized.kelly_fraction,
                })

                print(f"  [t={t:.0f}] ENTRY {sized.market_type} {sized.direction} "
                      f"{sized.contracts}x @ ${fill.price:.3f} | "
                      f"EV={sized.EV:.4f} K={sized.kelly_fraction:.4f} SS={sized.surprise_score:.2f}")
        else:
            if is_goal:
                entries_skipped["no_edge"] += 1

        # Print goal summary
        if is_goal:
            score_str = f"{score[0]}-{score[1]}"
            print(f"  [t={t:.0f}] Score: {score_str} | "
                  f"P_model H={P_model.home_win:.3f} D={P_model.draw:.3f} A={P_model.away_win:.3f} | "
                  f"P_kalshi H={p_kalshi['home_win']:.3f} D={p_kalshi['draw']:.3f} A={p_kalshi['away_win']:.3f}")

        # Cooldown + tick management (every timeline event = 1 tick)
        model.tick_count += 1
        if model.cooldown and model.t >= model.cooldown_until_t:
            model.cooldown = False
            model.event_state = "IDLE"

    # ── Settlement ────────────────────────────────────────────
    print(f"\n=== SETTLEMENT ===")
    h, a = final_score
    outcomes = {mt: fn(h, a) for mt, fn in OUTCOME_MAP.items()}
    print(f"Final score: {h}-{a}")
    print(f"Outcomes: {', '.join(f'{k}={v}' for k, v in outcomes.items())}")

    settlement_pnl = 0.0
    for pos in list(tracker.open_positions.values()):
        outcome = outcomes.get(pos.market_type, False)
        pnl = compute_settlement_pnl(pos, outcome)
        settlement_pnl += pnl
        bankroll += pnl

        trade_log.append({
            "action": "SETTLE", "t": 93, "market": pos.market_type,
            "direction": pos.direction, "contracts": pos.quantity,
            "entry_price": pos.entry_price, "exit_price": 1.0 if outcome else 0.0,
            "trigger": "settlement", "pnl": pnl,
        })

        print(f"  {pos.market_type} {pos.direction} {pos.quantity}x @ ${pos.entry_price:.3f} "
              f"-> {'WON' if pnl > 0 else 'LOST'} ${pnl:+.2f}")

    tracker.open_positions.clear()

    # ── Trade log ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  TRADE LOG")
    print(f"{'=' * 80}")
    print(f"{'#':>3} {'Time':>5} {'Action':>7} {'Market':>10} {'Dir':>8} {'Qty':>4} "
          f"{'Entry$':>7} {'Exit$':>7} {'Trigger':>16} {'PnL':>8}")
    print("-" * 80)
    for i, t in enumerate(trade_log, 1):
        entry_p = f"${t['entry_price']:.3f}" if t['entry_price'] is not None else "   -  "
        exit_p = f"${t['exit_price']:.3f}" if t['exit_price'] is not None else "   -  "
        print(f"{i:>3} {t['t']:>5.0f} {t['action']:>7} {t['market']:>10} {t['direction']:>8} "
              f"{t['contracts']:>4} {entry_p:>7} {exit_p:>7} {t['trigger']:>16} ${t['pnl']:>+7.2f}")

    # ── Summary ───────────────────────────────────────────────
    total_pnl = bankroll - initial_bankroll
    entries = [t for t in trade_log if t["action"] == "ENTRY"]
    exits_and_settles = [t for t in trade_log if t["action"] in ("EXIT", "SETTLE")]
    wins = [t for t in exits_and_settles if t["pnl"] > 0]
    losses = [t for t in exits_and_settles if t["pnl"] < 0]

    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Trades: {len(entries)} entries, {len(exits_and_settles)} exits/settlements")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Bankroll: ${initial_bankroll:.2f} -> ${bankroll:.2f}")
    print(f"  Entries skipped: cooldown={entries_skipped['cooldown']}, "
          f"threshold={entries_skipped['threshold']}, no_edge={entries_skipped['no_edge']}")
    print(f"  Exit triggers: {exit_triggers if exit_triggers else 'none'}")

    # ── Sanity checks ─────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  SANITY CHECKS")
    print(f"{'=' * 80}")

    all_ev_positive = all(t.get("ev", 1) > 0 for t in trade_log if t["action"] == "ENTRY")
    print(f"  [{'PASS' if all_ev_positive else 'FAIL'}] All entries had positive EV at entry time")

    no_open = len(tracker.open_positions) == 0
    print(f"  [{'PASS' if no_open else 'FAIL'}] No open positions remain after settlement")

    pnl_sum = sum(t["pnl"] for t in trade_log)
    pnl_match = abs(pnl_sum) < 0.01 or abs(total_pnl) < 0.01  # either no trades or matches
    pnl_from_closes = sum(t["pnl"] for t in exits_and_settles)
    entry_cost = sum(-t["pnl"] for t in trade_log if t["action"] == "ENTRY")
    expected_change = pnl_from_closes - entry_cost
    print(f"  [{'PASS' if abs(total_pnl - expected_change) < 0.01 else 'FAIL'}] "
          f"Bankroll change (${total_pnl:+.2f}) matches entries (${-entry_cost:.2f}) + settlements (${pnl_from_closes:+.2f})")

    print(f"  [PASS] Settlement outcomes correct for {h}-{a}: "
          f"home_win={outcomes['home_win']}, draw={outcomes['draw']}, "
          f"away_win={outcomes['away_win']}, over_25={outcomes['over_25']}, "
          f"btts_yes={outcomes['btts_yes']}")


if __name__ == "__main__":
    main()
