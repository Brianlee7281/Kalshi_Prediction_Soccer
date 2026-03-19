#!/usr/bin/env python
"""Full pipeline backtest: Phase 1→2→3→4 on recorded matches.

Runs end-to-end MMPP v5 pipeline in PAPER mode. Uses REAL recorded Kalshi
orderbook data from data/latency/{match_id}/kalshi.jsonl by default.
Falls back to KalshiPriceSimulator with --use-simulator.

Usage:
    PYTHONPATH=. python scripts/backtest_full_pipeline.py
    PYTHONPATH=. python scripts/backtest_full_pipeline.py --lag-analysis
    PYTHONPATH=. python scripts/backtest_full_pipeline.py --use-simulator --sensitivity

Output:
    Stdout: per-match trading summary + aggregate PnL + edge analysis
    Disk:   data/backtest_results/{timestamp}_{config}/  (JSON + CSV)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.step_1_3_ml_prior import compute_C_time
from src.common.types import (
    ExitTrigger,
    FillResult,
    MarketProbs,
    MatchPnL,
    Phase2Result,
    Signal,
    TickPayload,
    TradingMode,
)
from src.engine.event_handlers import (
    detect_events_from_poll,
    handle_goal,
    handle_period_change,
)
from src.engine.intensity import compute_lambda
from src.engine.mc_pricing import _compute_sigma, _results_to_market_probs
from src.engine.model import LiveMatchModel
from src.execution.config import CONFIG
from src.execution.kalshi_replay import KalshiOrderbookReplay
from src.execution.kalshi_sim import KalshiPriceSimulator
from src.execution.kelly_sizer import size_position
from src.execution.mock_db import MockDBPool
from src.execution.pnl_calculator import compute_settlement_pnl
from src.execution.position_monitor import PositionTracker
from src.execution.settlement import OUTCOME_MAP
from src.execution.signal_generator import (
    _get_market_ekf_P,
    _get_market_mu,
    compute_dynamic_threshold,
    compute_edge,
    generate_signals,
)
from src.math.compute_mu import compute_remaining_mu
from src.math.mc_core import mc_simulate_remaining

try:
    from src.math.mc_core import mc_simulate_remaining_v5

    _HAS_V5_MC = True
except ImportError:
    _HAS_V5_MC = False

from src.prematch.phase2_pipeline import (
    _shin_vig_removal,
    backsolve_intensities,
)

# ── Constants ─────────────────────────────────────────────────────
MC_N = 20_000
MC_SKIP_BETWEEN_EVENTS = 5  # run MC every Nth tick between events

CALIBRATED_PARAMS: dict = {
    "b": [0.0, 0.072422, 0.185087, 0.312089, 0.193663, 0.184194, 0.123683, 1.0],
    "gamma_H": [0.0, -0.15, 0.10, -0.05],
    "gamma_A": [0.0, 0.10, -0.15, -0.05],
    "delta_H": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_A": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "Q": [
        [-0.02, 0.01, 0.01, 0.00],
        [0.00, -0.01, 0.00, 0.01],
        [0.00, 0.00, -0.01, 0.01],
        [0.00, 0.00, 0.00, 0.00],
    ],
    "sigma_a": 0.5,
    "alpha_1": 2.0,
    "delta_H_pos": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_H_neg": [-0.283441, -0.017133, 0.0, 0.000177, 0.233356],
    "delta_A_pos": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "delta_A_neg": [-0.129124, -0.148479, 0.0, 0.06128, 0.273054],
    "eta_H": 0.05,
    "eta_A": 0.05,
    "eta_H2": 0.08,
    "eta_A2": 0.08,
    "sigma_omega_sq": 0.003,
}

ALPHA_1 = 2.0
BASIS_BOUNDS = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0])

MARKET_TYPES = ["home_win", "draw", "away_win", "over_25", "btts_yes"]

# Kalshi ticker templates (backtesting only — not real tickers)
BACKTEST_TICKERS = {
    "home_win": "BT-HOME",
    "draw": "BT-DRAW",
    "away_win": "BT-AWAY",
    "over_25": "BT-O25",
    "btts_yes": "BT-BTTS",
}


# ── Data structures ───────────────────────────────────────────────


@dataclass
class TradeRecord:
    """Single trade (entry + exit or settlement)."""

    trade_num: int
    entry_tick: int
    entry_t: float
    market_type: str
    direction: str
    entry_price: float
    entry_p_model: float
    entry_p_kalshi: float
    entry_ev: float
    entry_kelly_fraction: float
    contracts: int
    surprise_score_at_entry: float
    edge_source: str  # "event_lag" | "noise"
    exit_tick: int | None = None
    exit_t: float | None = None
    exit_price: float | None = None
    exit_trigger: str = ""
    realized_pnl: float = 0.0
    hold_ticks: int = 0


@dataclass
class SignalRecord:
    """Every signal candidate (including filtered ones)."""

    tick: int
    t: float
    market_type: str
    direction: str
    p_model: float
    p_kalshi: float
    ev: float
    dynamic_threshold: float
    passed_threshold: bool
    filtered_reason: str | None  # None = traded, else reason
    result: str  # "traded" | "skipped"
    edge_source: str  # "event_lag" | "noise"


@dataclass
class EdgeWindow:
    """Period where edge existed between P_model and P_kalshi."""

    start_tick: int
    end_tick: int
    market_type: str
    avg_ev: float
    converted_to_trade: bool


@dataclass
class MatchBacktestResult:
    """Full result for one match backtest."""

    match_id: str
    home_team: str
    away_team: str
    final_score: tuple[int, int]
    outcome_str: str

    # Trading results
    trades: list[TradeRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_count: int = 0
    loss_count: int = 0

    # Signals (all candidates, including filtered)
    signals: list[SignalRecord] = field(default_factory=list)

    # Edge analysis
    edge_windows: list[EdgeWindow] = field(default_factory=list)
    signals_generated: int = 0
    signals_filtered_threshold: int = 0
    signals_filtered_cooldown: int = 0
    signals_filtered_positioned: int = 0
    signals_filtered_sizing: int = 0
    signals_filtered_exposure: int = 0

    # Per-market breakdown
    market_stats: dict[str, dict] = field(default_factory=dict)

    # Exit trigger distribution
    exit_triggers: dict[str, int] = field(default_factory=dict)

    # Timing
    elapsed_s: float = 0.0
    total_ticks: int = 0
    total_mc_calls: int = 0

    red_flags: list[str] = field(default_factory=list)


# ── MC helper (sync, from existing scripts) ───────────────────────


def run_mc_sync(
    model: LiveMatchModel,
    N: int = MC_N,
    seed: int | None = None,
) -> tuple[MarketProbs, MarketProbs]:
    """Synchronous MC pricing — returns (P_model, sigma_MC)."""
    mu_H, mu_A = compute_remaining_mu(model)
    model.mu_H = mu_H
    model.mu_A = mu_A
    model.mu_H_elapsed = max(0.0, model.mu_H_at_kickoff - model.mu_H)
    model.mu_A_elapsed = max(0.0, model.mu_A_at_kickoff - model.mu_A)

    Q_diag = np.diag(model.Q).copy()
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        rs = sum(model.Q[i, j] for j in range(4) if i != j and model.Q[i, j] > 0)
        if rs > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = model.Q[i, j] / rs

    if seed is None:
        seed = int(time.monotonic() * 1_000_000) % (2**31)
    S_H, S_A = model.score

    if _HAS_V5_MC and model.delta_H_pos is not None:
        results = mc_simulate_remaining_v5(
            t_now=model.t,
            T_end=model.T_exp,
            S_H=S_H,
            S_A=S_A,
            state=model.current_state_X,
            score_diff=model.delta_S,
            a_H=model.a_H,
            a_A=model.a_A,
            b=model.b,
            gamma_H=model.gamma_H,
            gamma_A=model.gamma_A,
            delta_H_pos=model.delta_H_pos,
            delta_H_neg=model.delta_H_neg,
            delta_A_pos=model.delta_A_pos,
            delta_A_neg=model.delta_A_neg,
            Q_diag=Q_diag,
            Q_off=Q_off,
            basis_bounds=model.basis_bounds,
            N=N,
            seed=seed,
            eta_H=model.eta_H,
            eta_A=model.eta_A,
            eta_H2=model.eta_H2,
            eta_A2=model.eta_A2,
            stoppage_1_start=45.0,
            stoppage_2_start=90.0,
        )
    else:
        results = mc_simulate_remaining(
            t_now=model.t,
            T_end=model.T_exp,
            S_H=S_H,
            S_A=S_A,
            state=model.current_state_X,
            score_diff=model.delta_S,
            a_H=model.a_H,
            a_A=model.a_A,
            b=model.b,
            gamma_H=model.gamma_H,
            gamma_A=model.gamma_A,
            delta_H=model.delta_H,
            delta_A=model.delta_A,
            Q_diag=Q_diag,
            Q_off=Q_off,
            basis_bounds=model.basis_bounds,
            N=N,
            seed=seed,
        )

    P_model = _results_to_market_probs(results, S_H, S_A)
    sigma_MC = _compute_sigma(P_model, N)
    return P_model, sigma_MC


# ── Phase 2 helpers ───────────────────────────────────────────────


def extract_bet365_odds(data_dir: Path) -> tuple[float, float, float] | None:
    """Extract first Bet365 (or Sbobet fallback) ML odds from odds_api.jsonl."""
    odds_path = data_dir / "odds_api.jsonl"
    if not odds_path.exists():
        return None
    for bookie in ("Bet365", "Sbobet"):
        with open(odds_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("bookie") != bookie:
                    continue
                for mkt in entry.get("markets", []):
                    if mkt.get("name") == "ML" and mkt.get("odds"):
                        o = mkt["odds"][0]
                        try:
                            return float(o["home"]), float(o["draw"]), float(o["away"])
                        except (KeyError, ValueError):
                            continue
    return None


def build_model_from_data(
    data_dir: Path,
    params: dict,
) -> tuple[LiveMatchModel, str, str, tuple[int, int], str] | None:
    """Run Phase 1+2 and build LiveMatchModel."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    match_id = str(meta.get("match_id", data_dir.name))
    home = meta.get("home_team", "Home")
    away = meta.get("away_team", "Away")
    final = tuple(meta.get("final_score", [0, 0]))

    odds = extract_bet365_odds(data_dir)
    b = np.array(params["b"])
    Q = np.array(params["Q"])
    C_time = compute_C_time(b, BASIS_BOUNDS)

    if odds:
        p_h, p_d, p_a = _shin_vig_removal(*odds)
        market_implied = MarketProbs(home_win=p_h, draw=p_d, away_win=p_a)
        a_H, a_A = backsolve_intensities(market_implied, b, Q, BASIS_BOUNDS)
    else:
        a_H = float(np.log(1.4 / C_time))
        a_A = float(np.log(1.1 / C_time))
        market_implied = None

    mu_H = float(np.exp(a_H) * C_time)
    mu_A = float(np.exp(a_A) * C_time)

    result2 = Phase2Result(
        match_id=match_id,
        league_id=1204,
        a_H=a_H,
        a_A=a_A,
        mu_H=mu_H,
        mu_A=mu_A,
        C_time=C_time,
        verdict="GO",
        skip_reason=None,
        param_version=1,
        home_team=home,
        away_team=away,
        kickoff_utc="2026-03-16T20:00:00+00:00",
        kalshi_tickers=BACKTEST_TICKERS,
        market_implied=market_implied,
        prediction_method="backsolve_odds_api" if odds else "league_mle",
        ekf_P0=0.15 if odds else 0.50,
    )

    model = LiveMatchModel.from_phase2_result(result2, params)
    return model, home, away, final, match_id


# ── Helpers ───────────────────────────────────────────────────────


def _sync_paper_fill(signal: Signal) -> FillResult:
    """Create a paper fill synchronously (avoids asyncio)."""
    return FillResult(
        order_id=f"paper-{uuid4()}",
        ticker=signal.ticker,
        direction=signal.direction,
        quantity=signal.contracts,
        price=signal.P_kalshi,
        status="paper",
        fill_cost=signal.contracts * signal.P_kalshi,
        timestamp=datetime.now(timezone.utc),
    )


def _classify_edge_source(
    ticks_since_last_goal: int, event_window: float = 30.0
) -> str:
    """Classify whether an edge comes from event lag or between-event noise.

    event_window: ticks after a goal within which edge is event-driven.
    Default 30 ticks (~30 seconds at 1Hz) matches typical market reaction.
    """
    if ticks_since_last_goal <= event_window:
        return "event_lag"
    return "noise"


def _compute_exit_pnl(pos_direction: str, entry_price: float, exit_price: float, qty: int) -> float:
    """Compute realized PnL for an exit."""
    if pos_direction == "BUY_YES":
        return (exit_price - entry_price) * qty
    else:
        return ((1.0 - exit_price) - entry_price) * qty


# ── Core: single-match backtest ───────────────────────────────────


def backtest_single_match(
    data_dir: Path,
    params: dict,
    kalshi_sim_config: dict | None = None,
    initial_bankroll: float = 10_000.0,
    cooldown_after_exit: int = 300,
    min_hold_ticks: int = 150,
    use_simulator: bool = False,
) -> MatchBacktestResult | None:
    """Run full Phase 1→2→3→4 pipeline on one recorded match.

    Uses real Kalshi orderbook data from kalshi.jsonl when available.
    Falls back to KalshiPriceSimulator when use_simulator=True or no
    kalshi.jsonl exists.
    """
    built = build_model_from_data(data_dir, params)
    if built is None:
        return None
    model, home, away, final_score, match_id = built

    outcome_str = (
        "Home Win"
        if final_score[0] > final_score[1]
        else "Draw"
        if final_score[0] == final_score[1]
        else "Away Win"
    )

    result = MatchBacktestResult(
        match_id=match_id,
        home_team=home,
        away_team=away,
        final_score=final_score,
        outcome_str=outcome_str,
    )
    t0 = time.monotonic()

    # ── Initialize Phase 4 components ─────────────────────────────
    db_pool = MockDBPool(initial_bankroll)
    tracker = PositionTracker(
        min_hold_ticks=min_hold_ticks,
        cooldown_after_exit=cooldown_after_exit,
    )
    bankroll = initial_bankroll

    # ── Kalshi price source: real data or simulator ───────────────
    kalshi_jsonl = data_dir / "kalshi.jsonl"
    kalshi_replay: KalshiOrderbookReplay | None = None
    kalshi_sim: KalshiPriceSimulator | None = None

    if not use_simulator and kalshi_jsonl.exists():
        # Build ticker→market_type map from metadata
        meta_path = data_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        ticker_list = meta.get("kalshi_tickers", [])
        # Convention: last 3 chars of ticker = team abbrev or "TIE"
        ticker_to_market: dict[str, str] = {}
        for tk in ticker_list:
            suffix = tk.rsplit("-", 1)[-1].upper()
            if suffix == "TIE":
                ticker_to_market[tk] = "draw"
            elif tk == model.kalshi_tickers.get("home_win", ""):
                ticker_to_market[tk] = "home_win"
            elif tk == model.kalshi_tickers.get("away_win", ""):
                ticker_to_market[tk] = "away_win"
            elif tk == model.kalshi_tickers.get("draw", ""):
                ticker_to_market[tk] = "draw"
        # Fallback: match by position if model tickers are empty
        if not ticker_to_market:
            # Heuristic from metadata: [away, tie, home] ordering
            if len(ticker_list) >= 3:
                ticker_to_market[ticker_list[0]] = "away_win"
                ticker_to_market[ticker_list[1]] = "draw"
                ticker_to_market[ticker_list[2]] = "home_win"

        kalshi_replay = KalshiOrderbookReplay(kalshi_jsonl, ticker_to_market)
        using_real_data = True
    else:
        sim_cfg = kalshi_sim_config or {"seed": 42}
        kalshi_sim = KalshiPriceSimulator(**sim_cfg)
        using_real_data = False

    # ── JIT warmup ────────────────────────────────────────────────
    model.engine_phase = "FIRST_HALF"
    model.t = 0.001
    run_mc_sync(model, N=100)

    # ── Kickoff ───────────────────────────────────────────────────
    model.t = 0.0
    model.engine_phase = "FIRST_HALF"
    P_model, sigma_MC = run_mc_sync(model)
    result.total_mc_calls = 1

    if kalshi_sim is not None:
        kalshi_sim.initialize(P_model)
        model.p_kalshi = kalshi_sim.get_prices()
    else:
        model.p_kalshi = {}  # will be populated on first poll with _ts_wall

    # ── Load goalserve polls ──────────────────────────────────────
    polls: list[dict] = []
    gs_path = data_dir / "goalserve.jsonl"
    if gs_path.exists():
        with open(gs_path) as f:
            for line in f:
                try:
                    polls.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # ── Tracking state ────────────────────────────────────────────
    tick_counter = 0
    last_update_t = 0.0
    last_mc_t = 0.0
    in_play = False
    saw_halftime = False
    running_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    trade_records: list[TradeRecord] = []
    signal_records: list[SignalRecord] = []
    trajectory_rows: list[dict] = []
    active_edges: dict[str, dict] = {}
    edge_windows: list[EdgeWindow] = []
    ticks_since_last_goal = 1000  # start high so no lag at kickoff

    # ── Process each poll (simulates 1Hz ticks) ───────────────────
    for poll in polls:
        data = poll.get("data")
        if data is None:
            continue
        status = str(data.get("@status", "")).strip()

        if status.isdigit():
            minute = int(status)
            new_t = float(minute)
            if not in_play:
                in_play = True
            if saw_halftime and minute <= 45:
                continue
            model.t = new_t
        elif status == "HT":
            if not saw_halftime:
                saw_halftime = True
                handle_period_change(model, "HALFTIME")
            in_play = False
            continue
        elif status == "FT":
            if model.engine_phase != "FINISHED":
                handle_period_change(model, "FINISHED")
            in_play = False
            continue
        else:
            continue

        if not in_play:
            continue

        if not saw_halftime and model.engine_phase != "FIRST_HALF":
            handle_period_change(model, "FIRST_HALF")
        elif saw_halftime and model.t > 45 and model.engine_phase != "SECOND_HALF":
            handle_period_change(model, "SECOND_HALF")

        tick_counter += 1
        ticks_since_last_goal += 1
        model.tick_count = tick_counter

        # ── EKF predict + no-goal update ──────────────────────────
        dt_min = model.t - last_update_t
        if dt_min > 0 and model.strength_updater is not None:
            model.strength_updater.predict(dt_min)
            lH = compute_lambda(model, "home")
            lA = compute_lambda(model, "away")
            model.strength_updater.update_no_goal(lH, lA, dt_min)
            model.a_H = model.strength_updater.a_H
            model.a_A = model.strength_updater.a_A
            last_update_t = model.t

        # ── Event detection ───────────────────────────────────────
        detected = detect_events_from_poll(model, data)
        is_event_tick = False

        for evt in detected:
            if evt["type"] == "goal":
                team = evt["team"]
                minute = evt.get("minute", int(model.t))
                pre_P, _ = run_mc_sync(model)
                result.total_mc_calls += 1

                handle_goal(model, team, minute)

                if model.strength_updater:
                    surprise = model.strength_updater.compute_surprise_score(
                        team, pre_P.home_win, pre_P.away_win,
                    )
                    model.surprise_score = surprise

                is_event_tick = True
                ticks_since_last_goal = 0

            elif evt["type"] == "period_change":
                new_phase = evt.get("new_phase", "")
                if new_phase and new_phase != model.engine_phase:
                    handle_period_change(model, new_phase)

        # ── MC pricing (adaptive frequency) ───────────────────────
        run_mc = False
        if is_event_tick:
            run_mc = True
        elif ticks_since_last_goal < 25:  # ~20 ticks near event is enough for MC
            run_mc = True
        elif tick_counter % MC_SKIP_BETWEEN_EVENTS == 0:
            run_mc = True
        elif model.t - last_mc_t >= 5.0:
            run_mc = True

        if run_mc:
            P_model, sigma_MC = run_mc_sync(model, N=MC_N)
            result.total_mc_calls += 1
            last_mc_t = model.t

        # ── Update Kalshi prices ──────────────────────────────────
        poll_wall_time = poll.get("_ts_wall", 0.0)
        if kalshi_replay is not None:
            real_prices = kalshi_replay.get_prices_at(poll_wall_time)
            if real_prices:
                model.p_kalshi = real_prices
        elif kalshi_sim is not None:
            model.p_kalshi = kalshi_sim.update(tick_counter, P_model, is_event_tick)

        # ── Build TickPayload for Phase 4 logic ───────────────────
        # KEY FIX: order_allowed is based on engine phase only, NOT
        # Phase 3's 50-tick goal cooldown. The cooldown is a Phase 3
        # concept for MC re-pricing throttle. Phase 4's signal
        # generator + dynamic threshold + Phase 4 cooldown handle
        # trading decisions independently.
        ekf = model.ekf_tracker
        phase4_order_allowed = (
            model.engine_phase in ("FIRST_HALF", "SECOND_HALF")
            and not getattr(model, "ob_freeze", False)
        )

        payload = TickPayload(
            match_id=match_id,
            t=model.t,
            engine_phase=model.engine_phase,
            P_model=P_model,
            sigma_MC=sigma_MC,
            score=model.score,
            X=model.current_state_X,
            delta_S=model.delta_S,
            mu_H=model.mu_H,
            mu_A=model.mu_A,
            a_H_current=model.a_H,
            a_A_current=model.a_A,
            last_goal_type=getattr(model, "last_goal_type", "NEUTRAL"),
            ekf_P_H=ekf.P_H if ekf else 0.0,
            ekf_P_A=ekf.P_A if ekf else 0.0,
            hmm_state=getattr(model, "hmm_state", 0),
            dom_index=getattr(model, "dom_index", 0.0),
            surprise_score=getattr(model, "surprise_score", 0.0),
            order_allowed=phase4_order_allowed,
            cooldown=model.cooldown,
            ob_freeze=getattr(model, "ob_freeze", False),
            event_state=getattr(model, "event_state", "IDLE"),
        )

        # ── Phase 4: Check exits ──────────────────────────────────
        tick_signal_generated = ""
        tick_trade_executed = ""

        exits = tracker.check_exits(payload, model.p_kalshi)
        for exit_decision in exits:
            if exit_decision.position_id not in tracker.open_positions:
                continue
            pos = tracker.open_positions[exit_decision.position_id]

            fill = FillResult(
                order_id=f"paper-exit-{uuid4()}",
                ticker=pos.ticker,
                direction="BUY_NO" if pos.direction == "BUY_YES" else "BUY_YES",
                quantity=exit_decision.contracts_to_exit,
                price=exit_decision.exit_price,
                status="paper",
                fill_cost=exit_decision.contracts_to_exit * exit_decision.exit_price,
                timestamp=datetime.now(timezone.utc),
            )

            realized_pnl = _compute_exit_pnl(
                pos.direction, pos.entry_price, fill.price, fill.quantity
            )
            hold_ticks = pos.ticks_held

            tracker.close_position(
                pos.id,
                exit_decision.trigger,
                exit_decision.contracts_to_exit,
                fill.price,
                tick_counter,
            )

            bankroll += realized_pnl
            running_pnl += realized_pnl
            peak_pnl = max(peak_pnl, running_pnl)
            max_drawdown = min(max_drawdown, running_pnl - peak_pnl)

            trigger_name = exit_decision.trigger.value
            result.exit_triggers[trigger_name] = (
                result.exit_triggers.get(trigger_name, 0) + 1
            )

            # Find matching trade record and update exit fields
            for tr in trade_records:
                if tr.exit_tick is None and tr.market_type == pos.market_type:
                    tr.exit_tick = tick_counter
                    tr.exit_t = model.t
                    tr.exit_price = fill.price
                    tr.exit_trigger = trigger_name
                    tr.realized_pnl = realized_pnl
                    tr.hold_ticks = hold_ticks
                    break

        # ── Phase 4: Signal evaluation + execution ────────────────
        # Evaluate ALL markets for signals, logging each candidate
        if phase4_order_allowed:
            for mt in MARKET_TYPES:
                if mt not in model.p_kalshi:
                    continue
                p_m = getattr(P_model, mt)
                if p_m is None:
                    continue
                p_k = model.p_kalshi[mt]

                direction, ev = compute_edge(p_m, p_k)
                if direction == "HOLD":
                    continue

                # Compute dynamic threshold
                mu_market = _get_market_mu(mt, model.mu_H, model.mu_A)
                ekf_P = _get_market_ekf_P(
                    mt, ekf.P_H if ekf else 0.0, ekf.P_A if ekf else 0.0
                )
                sigma_mc_val = getattr(sigma_MC, mt) or 0.0
                theta = compute_dynamic_threshold(p_m, sigma_mc_val, ekf_P, mu_market)

                edge_source = _classify_edge_source(ticks_since_last_goal)

                passed = ev >= theta
                if not passed:
                    result.signals_filtered_threshold += 1
                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=False,
                            filtered_reason="below_threshold",
                            result="skipped",
                            edge_source=edge_source,
                        )
                    )
                    continue

                result.signals_generated += 1

                # Check if already positioned in this market
                has_position = any(
                    getattr(pos, "market_type", None) == mt
                    for pos in tracker.open_positions.values()
                )
                if has_position:
                    result.signals_filtered_positioned += 1
                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=True,
                            filtered_reason="already_positioned",
                            result="skipped",
                            edge_source=edge_source,
                        )
                    )
                    continue

                # Check Phase 4 cooldown
                if tracker.is_in_cooldown(mt, tick_counter):
                    result.signals_filtered_cooldown += 1
                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=True,
                            filtered_reason="cooldown_after_exit",
                            result="skipped",
                            edge_source=edge_source,
                        )
                    )
                    continue

                # Build signal and size
                signal = Signal(
                    match_id=match_id,
                    ticker=BACKTEST_TICKERS[mt],
                    market_type=mt,
                    direction=direction,
                    P_kalshi=p_k,
                    P_model=p_m,
                    EV=ev,
                    kelly_fraction=0.0,
                    kelly_amount=0.0,
                    contracts=0,
                    surprise_score=getattr(model, "surprise_score", 0.0),
                )
                signal = size_position(signal, payload, bankroll)
                if signal.contracts <= 0:
                    result.signals_filtered_sizing += 1
                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=True,
                            filtered_reason="zero_contracts",
                            result="skipped",
                            edge_source=edge_source,
                        )
                    )
                    continue

                # Exposure check
                amount = signal.contracts * signal.P_kalshi
                current_exposure = sum(
                    r["reserved_amount"]
                    for r in db_pool.reservations
                    if r["status"] in ("RESERVED", "CONFIRMED")
                )
                if current_exposure + amount > bankroll * CONFIG.TOTAL_EXPOSURE_CAP_FRAC:
                    result.signals_filtered_exposure += 1
                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=True,
                            filtered_reason="exposure_cap",
                            result="skipped",
                            edge_source=edge_source,
                        )
                    )
                    continue

                # Execute paper fill
                fill = _sync_paper_fill(signal)

                if fill is not None and fill.quantity > 0:
                    pos = tracker.add_position(
                        signal, fill, tick_counter, payload.t
                    )
                    bankroll -= fill.fill_cost
                    tick_signal_generated = mt
                    tick_trade_executed = mt

                    trade_records.append(
                        TradeRecord(
                            trade_num=len(trade_records) + 1,
                            entry_tick=tick_counter,
                            entry_t=model.t,
                            market_type=mt,
                            direction=direction,
                            entry_price=pos.entry_price,
                            entry_p_model=p_m,
                            entry_p_kalshi=p_k,
                            entry_ev=ev,
                            entry_kelly_fraction=signal.kelly_fraction,
                            contracts=signal.contracts,
                            surprise_score_at_entry=getattr(
                                model, "surprise_score", 0.0
                            ),
                            edge_source=edge_source,
                        )
                    )

                    signal_records.append(
                        SignalRecord(
                            tick=tick_counter,
                            t=model.t,
                            market_type=mt,
                            direction=direction,
                            p_model=p_m,
                            p_kalshi=p_k,
                            ev=ev,
                            dynamic_threshold=theta,
                            passed_threshold=True,
                            filtered_reason=None,
                            result="traded",
                            edge_source=edge_source,
                        )
                    )

        # ── Trajectory row ────────────────────────────────────────
        # Record per-tick state for CSV output
        positions_by_market: dict[str, int] = {}
        for pos in tracker.open_positions.values():
            positions_by_market[pos.market_type] = pos.quantity

        row = {
            "tick": tick_counter,
            "t": model.t,
            "score_H": model.score[0],
            "score_A": model.score[1],
            "P_model_H": P_model.home_win,
            "P_model_D": P_model.draw,
            "P_model_A": P_model.away_win,
            "P_kalshi_H": model.p_kalshi.get("home_win", 0),
            "P_kalshi_D": model.p_kalshi.get("draw", 0),
            "P_kalshi_A": model.p_kalshi.get("away_win", 0),
            "edge_H": P_model.home_win - model.p_kalshi.get("home_win", P_model.home_win),
            "edge_D": P_model.draw - model.p_kalshi.get("draw", P_model.draw),
            "edge_A": P_model.away_win - model.p_kalshi.get("away_win", P_model.away_win),
            "a_H": model.a_H,
            "a_A": model.a_A,
            "ekf_P_H": ekf.P_H if ekf else 0.0,
            "ekf_P_A": ekf.P_A if ekf else 0.0,
            "position_H": positions_by_market.get("home_win", 0),
            "position_D": positions_by_market.get("draw", 0),
            "position_A": positions_by_market.get("away_win", 0),
            "signal_generated": tick_signal_generated,
            "trade_executed": tick_trade_executed,
        }
        trajectory_rows.append(row)

        # ── Edge window tracking ──────────────────────────────────
        for mt in MARKET_TYPES:
            p_m = getattr(P_model, mt)
            p_k = model.p_kalshi.get(mt)
            if p_m is None or p_k is None:
                continue
            ev = abs(p_m - p_k)
            if ev > 0.02:
                if mt not in active_edges:
                    active_edges[mt] = {
                        "start_tick": tick_counter,
                        "ev_sum": 0.0,
                        "count": 0,
                        "traded": False,
                    }
                active_edges[mt]["ev_sum"] += ev
                active_edges[mt]["count"] += 1
                for tr in trade_records:
                    if (
                        tr.market_type == mt
                        and tr.entry_tick >= active_edges[mt]["start_tick"]
                    ):
                        active_edges[mt]["traded"] = True
            else:
                if mt in active_edges:
                    ew = active_edges.pop(mt)
                    if ew["count"] > 0:
                        edge_windows.append(
                            EdgeWindow(
                                start_tick=ew["start_tick"],
                                end_tick=tick_counter,
                                market_type=mt,
                                avg_ev=ew["ev_sum"] / ew["count"],
                                converted_to_trade=ew["traded"],
                            )
                        )

        # ── Phase 3 cooldown management ───────────────────────────
        if model.cooldown and model.tick_count >= model.cooldown_until_tick:
            model.cooldown = False
            model.event_state = "IDLE"

    # ── Close remaining edge windows ──────────────────────────────
    for mt, ew in active_edges.items():
        if ew["count"] > 0:
            edge_windows.append(
                EdgeWindow(
                    start_tick=ew["start_tick"],
                    end_tick=tick_counter,
                    market_type=mt,
                    avg_ev=ew["ev_sum"] / ew["count"],
                    converted_to_trade=ew["traded"],
                )
            )

    # ── Settlement ────────────────────────────────────────────────
    h, a = final_score
    score_outcomes = {mt: fn(h, a) for mt, fn in OUTCOME_MAP.items()}

    for pos in list(tracker.open_positions.values()):
        outcome = score_outcomes.get(pos.market_type, False)
        realized_pnl = compute_settlement_pnl(pos, outcome)
        running_pnl += realized_pnl
        peak_pnl = max(peak_pnl, running_pnl)
        max_drawdown = min(max_drawdown, running_pnl - peak_pnl)

        result.exit_triggers["settlement"] = (
            result.exit_triggers.get("settlement", 0) + 1
        )

        # Update matching trade record
        for tr in trade_records:
            if tr.exit_tick is None and tr.market_type == pos.market_type:
                tr.exit_tick = tick_counter
                tr.exit_t = model.t
                tr.exit_price = 1.0 if outcome else 0.0
                tr.exit_trigger = "settlement"
                tr.realized_pnl = realized_pnl
                tr.hold_ticks = pos.ticks_held
                break

    tracker.open_positions.clear()

    # ── Compile results ───────────────────────────────────────────
    result.trades = trade_records
    result.signals = signal_records
    result.total_pnl = running_pnl
    result.max_drawdown = max_drawdown
    result.edge_windows = edge_windows
    result.total_ticks = tick_counter
    result.elapsed_s = time.monotonic() - t0

    for tr in trade_records:
        if tr.realized_pnl > 0:
            result.win_count += 1
        elif tr.realized_pnl < 0:
            result.loss_count += 1

    for mt in MARKET_TYPES:
        mt_trades = [tr for tr in trade_records if tr.market_type == mt]
        if mt_trades:
            result.market_stats[mt] = {
                "trades": len(mt_trades),
                "avg_ev": sum(tr.entry_ev for tr in mt_trades) / len(mt_trades),
                "avg_hold": sum(tr.hold_ticks for tr in mt_trades) / len(mt_trades),
                "pnl": sum(tr.realized_pnl for tr in mt_trades),
                "win_pct": sum(1 for tr in mt_trades if tr.realized_pnl > 0)
                / len(mt_trades)
                * 100,
            }

    if result.total_pnl > 100 * max(len(trade_records), 1):
        result.red_flags.append("extremely_high_pnl")
    if len(trade_records) == 0:
        result.red_flags.append("zero_trades")
    if ekf and (ekf.P_H > 1.5 or ekf.P_A > 1.5):
        result.red_flags.append(
            f"ekf_divergence: P_H={ekf.P_H:.2f} P_A={ekf.P_A:.2f}"
        )

    # Stash trajectory for saving later
    result._trajectory_rows = trajectory_rows  # type: ignore[attr-defined]

    return result


# ── Result saving ─────────────────────────────────────────────────


def save_results(
    results: list[MatchBacktestResult],
    config: dict,
    config_name: str,
    output_base: Path,
) -> Path:
    """Save all results to disk. Returns output directory path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / f"{ts}_{config_name}"
    matches_dir = out_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    all_pnl = [r.total_pnl for r in results]
    total_pnl = sum(all_pnl)
    n = len(results)
    avg_pnl = total_pnl / n if n else 0
    std_pnl = float(np.std(all_pnl)) if n > 1 else 0.0
    sharpe = (avg_pnl / std_pnl * math.sqrt(380)) if std_pnl > 0 else 0.0
    all_trades = [tr for r in results for tr in r.trades]
    total_trades = len(all_trades)
    win_rate = sum(1 for t in all_trades if t.realized_pnl > 0) / total_trades if total_trades else 0
    max_dd = min((r.max_drawdown for r in results), default=0)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "calibrated_params": {
            "b": CALIBRATED_PARAMS["b"],
            "delta_H": CALIBRATED_PARAMS["delta_H"],
            "delta_A": CALIBRATED_PARAMS["delta_A"],
            "gamma_H": CALIBRATED_PARAMS["gamma_H"],
            "gamma_A": CALIBRATED_PARAMS["gamma_A"],
            "sigma_omega_sq": CALIBRATED_PARAMS["sigma_omega_sq"],
        },
        "matches": [
            {
                "match_id": r.match_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "final_score": list(r.final_score),
                "total_pnl": round(r.total_pnl, 4),
                "trade_count": len(r.trades),
                "win_count": r.win_count,
                "loss_count": r.loss_count,
            }
            for r in results
        ],
        "aggregate": {
            "total_pnl": round(total_pnl, 4),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-match files
    for r in results:
        # trades.json
        trades_data = []
        for tr in r.trades:
            trades_data.append({
                "trade_id": tr.trade_num,
                "market_type": tr.market_type,
                "direction": tr.direction,
                "entry_tick": tr.entry_tick,
                "entry_t": round(tr.entry_t, 2),
                "entry_price": round(tr.entry_price, 4),
                "entry_p_model": round(tr.entry_p_model, 4),
                "entry_p_kalshi": round(tr.entry_p_kalshi, 4),
                "entry_ev": round(tr.entry_ev, 4),
                "entry_kelly_fraction": round(tr.entry_kelly_fraction, 6),
                "contracts": tr.contracts,
                "exit_tick": tr.exit_tick,
                "exit_t": round(tr.exit_t, 2) if tr.exit_t else None,
                "exit_price": round(tr.exit_price, 4) if tr.exit_price is not None else None,
                "exit_trigger": tr.exit_trigger,
                "realized_pnl": round(tr.realized_pnl, 4),
                "hold_time_ticks": tr.hold_ticks,
                "surprise_score_at_entry": round(tr.surprise_score_at_entry, 4),
                "edge_source": tr.edge_source,
            })
        with open(matches_dir / f"{r.match_id}_trades.json", "w") as f:
            json.dump(trades_data, f, indent=2)

        # signals.json
        signals_data = []
        for sr in r.signals:
            signals_data.append({
                "tick": sr.tick,
                "t": round(sr.t, 2),
                "market_type": sr.market_type,
                "direction": sr.direction,
                "p_model": round(sr.p_model, 4),
                "p_kalshi": round(sr.p_kalshi, 4),
                "ev": round(sr.ev, 4),
                "dynamic_threshold": round(sr.dynamic_threshold, 4),
                "passed_threshold": sr.passed_threshold,
                "filtered_reason": sr.filtered_reason,
                "result": sr.result,
                "edge_source": sr.edge_source,
            })
        with open(matches_dir / f"{r.match_id}_signals.json", "w") as f:
            json.dump(signals_data, f, indent=2)

        # trajectory.csv
        traj_rows = getattr(r, "_trajectory_rows", [])
        if traj_rows:
            csv_path = matches_dir / f"{r.match_id}_trajectory.csv"
            fieldnames = [
                "tick", "t", "score_H", "score_A",
                "P_model_H", "P_model_D", "P_model_A",
                "P_kalshi_H", "P_kalshi_D", "P_kalshi_A",
                "edge_H", "edge_D", "edge_A",
                "a_H", "a_A", "ekf_P_H", "ekf_P_A",
                "position_H", "position_D", "position_A",
                "signal_generated", "trade_executed",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in traj_rows:
                    # Round floats for readability
                    out = {}
                    for k in fieldnames:
                        v = row.get(k, "")
                        if isinstance(v, float):
                            out[k] = round(v, 6)
                        else:
                            out[k] = v
                    writer.writerow(out)

        # summary.json per match
        match_summary = {
            "match_id": r.match_id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "final_score": list(r.final_score),
            "outcome": r.outcome_str,
            "total_pnl": round(r.total_pnl, 4),
            "max_drawdown": round(r.max_drawdown, 4),
            "trade_count": len(r.trades),
            "win_count": r.win_count,
            "loss_count": r.loss_count,
            "signals_generated": r.signals_generated,
            "signals_filtered_threshold": r.signals_filtered_threshold,
            "signals_filtered_cooldown": r.signals_filtered_cooldown,
            "signals_filtered_positioned": r.signals_filtered_positioned,
            "signals_filtered_sizing": r.signals_filtered_sizing,
            "signals_filtered_exposure": r.signals_filtered_exposure,
            "edge_windows": len(r.edge_windows),
            "exit_triggers": r.exit_triggers,
            "total_ticks": r.total_ticks,
            "total_mc_calls": r.total_mc_calls,
            "elapsed_s": round(r.elapsed_s, 1),
            "red_flags": r.red_flags,
        }
        with open(matches_dir / f"{r.match_id}_summary.json", "w") as f:
            json.dump(match_summary, f, indent=2)

    return out_dir


# ── Output formatting ─────────────────────────────────────────────


def print_match_result(r: MatchBacktestResult) -> None:
    """Print per-match trading summary."""
    print()
    print("=" * 65)
    print(f"  Match: {r.home_team} {r.final_score[0]}-{r.final_score[1]} {r.away_team}")
    print(f"  Final Score: {r.final_score[0]}-{r.final_score[1]} ({r.outcome_str})")
    print("=" * 65)

    if r.trades:
        print()
        print("  Trades:")
        print("  " + "-" * 78)
        print(
            f"  {'#':>2}  {'Tick':>5}  {'Market':<10} {'Dir':<8} {'Entry':>6} "
            f"{'Exit':>6} {'Trigger':<18} {'PnL':>8} {'Src':<9}"
        )
        print("  " + "-" * 78)
        for tr in r.trades:
            exit_p = f"${tr.exit_price:.2f}" if tr.exit_price is not None else "  N/A"
            pnl_str = f"{'+'if tr.realized_pnl >= 0 else ''}${tr.realized_pnl:.2f}"
            print(
                f"  {tr.trade_num:>2}  t={tr.entry_tick:<4} {tr.market_type:<10} "
                f"{tr.direction:<8} ${tr.entry_price:.2f} {exit_p:>6} "
                f"{tr.exit_trigger:<18} {pnl_str:>8} {tr.edge_source:<9}"
            )
        print("  " + "-" * 78)

    total_trades = len(r.trades)
    win_pct = (r.win_count / total_trades * 100) if total_trades > 0 else 0

    print()
    print("  Summary:")
    print(f"    Total trades: {total_trades}")
    print(f"    Win/Loss: {r.win_count}/{r.loss_count} ({win_pct:.0f}% win rate)")
    print(f"    Total PnL: {'+'if r.total_pnl >= 0 else ''}${r.total_pnl:.2f}")
    print(f"    Max drawdown: -${abs(r.max_drawdown):.2f}")

    # Signal funnel
    total_candidates = (
        r.signals_generated + r.signals_filtered_threshold
    )
    print(f"    Signal funnel: {total_candidates} candidates -> "
          f"{r.signals_generated} passed threshold -> {total_trades} traded")
    print(f"      Filtered by threshold:  {r.signals_filtered_threshold}")
    print(f"      Filtered by cooldown:   {r.signals_filtered_cooldown}")
    print(f"      Filtered by position:   {r.signals_filtered_positioned}")
    print(f"      Filtered by sizing:     {r.signals_filtered_sizing}")
    print(f"      Filtered by exposure:   {r.signals_filtered_exposure}")

    if r.edge_windows:
        avg_dur = sum(ew.end_tick - ew.start_tick for ew in r.edge_windows) / len(
            r.edge_windows
        )
        converted = sum(1 for ew in r.edge_windows if ew.converted_to_trade)
        print(
            f"    Edge windows: {len(r.edge_windows)} "
            f"(avg {avg_dur:.0f} ticks, {converted} converted)"
        )

    print(f"    MC calls: {r.total_mc_calls} | Ticks: {r.total_ticks} | Time: {r.elapsed_s:.1f}s")

    if r.market_stats:
        print()
        print("  Edge Analysis:")
        print(f"    {'Market':<10} {'Trades':>6} {'Win%':>5} {'Avg EV':>8} {'Avg Hold':>9} {'PnL':>10}")
        for mt, st in r.market_stats.items():
            print(
                f"    {mt:<10} {st['trades']:>6} {st['win_pct']:>4.0f}% "
                f"{st['avg_ev']:>8.4f} {st['avg_hold']:>7.0f} tk "
                f"{'+'if st['pnl'] >= 0 else ''}${st['pnl']:>8.2f}"
            )

    if r.exit_triggers:
        print()
        print("  Exit Triggers:")
        for trigger, count in sorted(r.exit_triggers.items(), key=lambda x: -x[1]):
            pct = count / total_trades * 100 if total_trades > 0 else 0
            print(f"    {trigger:<20} {count:>3} ({pct:.0f}%)")

    # Edge source breakdown with PnL
    event_trades = [t for t in r.trades if t.edge_source == "event_lag"]
    noise_trades = [t for t in r.trades if t.edge_source == "noise"]
    if r.trades:
        el_pnl = sum(t.realized_pnl for t in event_trades)
        n_pnl = sum(t.realized_pnl for t in noise_trades)
        print()
        print("  Edge Source Breakdown:")
        print(f"    Event-lag trades:     {len(event_trades)} trades, "
              f"{'+'if el_pnl >= 0 else ''}${el_pnl:.2f} PnL (this is real alpha)")
        print(f"    Between-event trades: {len(noise_trades)} trades, "
              f"{'+'if n_pnl >= 0 else ''}${n_pnl:.2f} PnL (should be ~$0)")

    if r.red_flags:
        print()
        print(f"  RED FLAGS: {', '.join(r.red_flags)}")


def print_aggregate_report(
    results: list[MatchBacktestResult],
    initial_bankroll: float = 10_000.0,
) -> None:
    """Print aggregate backtest results across all matches."""
    n = len(results)
    if n == 0:
        print("No matches to report.")
        return

    print()
    print("=" * 65)
    print(f"  AGGREGATE BACKTEST RESULTS ({n} matches)")
    print("=" * 65)

    all_pnl = [r.total_pnl for r in results]
    total_pnl = sum(all_pnl)
    avg_pnl = total_pnl / n
    std_pnl = float(np.std(all_pnl)) if n > 1 else 0.0
    sharpe = (avg_pnl / std_pnl * math.sqrt(380)) if std_pnl > 0 else 0.0
    max_loss = min(all_pnl)
    match_wins = sum(1 for p in all_pnl if p > 0)

    print()
    print("  PnL Summary:")
    print(f"    Total PnL:            {'+'if total_pnl >= 0 else ''}${total_pnl:.2f}")
    print(f"    Per-match avg:        {'+'if avg_pnl >= 0 else ''}${avg_pnl:.2f}")
    print(f"    Std dev:              ${std_pnl:.2f}")
    print(f"    Sharpe (approx):      {sharpe:.2f}  (annualized to ~380 EPL matches/season)")
    print(f"    Max single-match loss: -${abs(max_loss):.2f}")
    print(f"    Win rate (matches):   {match_wins}/{n}")

    roc = total_pnl / initial_bankroll * 100
    print()
    print("  Capital Efficiency:")
    print(f"    Starting bankroll:      ${initial_bankroll:,.0f}")
    print(f"    Return on capital:      {roc:.2f}%")

    all_trades = [tr for r in results for tr in r.trades]
    total_trades = len(all_trades)
    avg_trades = total_trades / n if n > 0 else 0
    trade_wins = sum(1 for tr in all_trades if tr.realized_pnl > 0)
    trade_win_pct = trade_wins / total_trades * 100 if total_trades > 0 else 0
    avg_hold = (
        sum(tr.hold_ticks for tr in all_trades) / total_trades
        if total_trades > 0
        else 0
    )

    print()
    print("  Trading Activity:")
    print(f"    Total trades:     {total_trades}")
    print(f"    Avg trades/match: {avg_trades:.1f}")
    print(f"    Win rate (trades): {trade_win_pct:.0f}%")
    print(f"    Avg hold time:    {avg_hold:.0f} ticks ({avg_hold / 60:.1f} minutes)")

    # Edge source
    event_lag = [t for t in all_trades if t.edge_source == "event_lag"]
    noise = [t for t in all_trades if t.edge_source == "noise"]
    print(f"    Edge source:      {len(event_lag)} event_lag, {len(noise)} noise")

    all_windows = [ew for r in results for ew in r.edge_windows]
    if all_windows:
        avg_duration = (
            sum(ew.end_tick - ew.start_tick for ew in all_windows) / len(all_windows)
        )
        avg_edge_ev = sum(ew.avg_ev for ew in all_windows) / len(all_windows)
        converted = sum(1 for ew in all_windows if ew.converted_to_trade)
        conv_rate = converted / len(all_windows) * 100

        print()
        print("  Edge Windows:")
        print(f"    Total detected:   {len(all_windows)}")
        print(f"    Avg duration:     {avg_duration:.0f} ticks")
        print(f"    Avg edge (EV):    {avg_edge_ev:.3f} ({avg_edge_ev * 100:.1f} cents)")
        print(f"    Conversion rate:  {conv_rate:.0f}% (windows -> trades)")

    print()
    print("  By Market:")
    print(f"    {'Market':<10} {'Trades':>6} {'Win%':>5} {'Total PnL':>12}")
    for mt in MARKET_TYPES:
        mt_trades = [tr for tr in all_trades if tr.market_type == mt]
        if mt_trades:
            mt_wins = sum(1 for tr in mt_trades if tr.realized_pnl > 0)
            mt_pnl = sum(tr.realized_pnl for tr in mt_trades)
            mt_win_pct = mt_wins / len(mt_trades) * 100
            print(
                f"    {mt:<10} {len(mt_trades):>6} {mt_win_pct:>4.0f}% "
                f"{'+'if mt_pnl >= 0 else ''}${mt_pnl:>10.2f}"
            )
        else:
            print(f"    {mt:<10} {'--':>6} {'--':>5} {'--':>12}")

    all_triggers: dict[str, int] = {}
    for r in results:
        for trigger, count in r.exit_triggers.items():
            all_triggers[trigger] = all_triggers.get(trigger, 0) + count

    if all_triggers:
        print()
        print("  Exit Trigger Distribution:")
        for trigger, count in sorted(all_triggers.items(), key=lambda x: -x[1]):
            pct = count / total_trades * 100 if total_trades > 0 else 0
            print(f"    {trigger:<20} {count:>3} ({pct:.0f}%)")

    # Signal funnel aggregate
    total_thresh = sum(r.signals_filtered_threshold for r in results)
    total_cooldown = sum(r.signals_filtered_cooldown for r in results)
    total_positioned = sum(r.signals_filtered_positioned for r in results)
    total_sizing = sum(r.signals_filtered_sizing for r in results)
    total_exposure = sum(r.signals_filtered_exposure for r in results)
    total_generated = sum(r.signals_generated for r in results)

    print()
    print("  Signal Funnel:")
    print(f"    Below threshold:   {total_thresh}")
    print(f"    Passed threshold:  {total_generated}")
    print(f"      -> cooldown:     {total_cooldown}")
    print(f"      -> positioned:   {total_positioned}")
    print(f"      -> sizing:       {total_sizing}")
    print(f"      -> exposure:     {total_exposure}")
    print(f"      -> TRADED:       {total_trades}")

    flagged = [r for r in results if r.red_flags]
    if flagged:
        print()
        print(f"  Red Flags ({len(flagged)}/{n} matches):")
        for r in flagged:
            print(
                f"    {r.match_id} ({r.home_team} v {r.away_team}): "
                f"{', '.join(r.red_flags)}"
            )


def print_sensitivity_report(
    configs: list[tuple[str, dict]],
    all_results: list[list[MatchBacktestResult]],
) -> None:
    """Print sensitivity analysis comparing different Kalshi sim configs."""
    print()
    print("=" * 65)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 65)
    print()
    print(
        f"  {'Config':<12} {'t½':>4} {'react':>6} {'spread':>7} "
        f"{'PnL':>10} {'Trades':>7} {'EL/N':>5}"
    )
    print("  " + "-" * 58)

    pnl_values = []
    for (name, config), results in zip(configs, all_results):
        total_pnl = sum(r.total_pnl for r in results)
        total_trades = sum(len(r.trades) for r in results)
        event_lag_trades = sum(
            1 for r in results for t in r.trades if t.edge_source == "event_lag"
        )
        noise_trades = total_trades - event_lag_trades
        pnl_str = f"{'+'if total_pnl >= 0 else ''}${total_pnl:.2f}"
        hl = config.get("market_reaction_half_life", "?")
        ir = config.get("market_initial_reaction", "?")
        sp = config.get("base_spread", config.get("spread", "?"))
        print(
            f"  {name:<12} {hl:>4} {ir:>6} "
            f"{sp:>7} {pnl_str:>10} {total_trades:>7} "
            f"{event_lag_trades}/{noise_trades}"
        )
        pnl_values.append(total_pnl)

    print()
    # Check ordering: optimistic >= baseline >= pessimistic
    if len(pnl_values) == 3:
        correct_order = pnl_values[0] >= pnl_values[1] >= pnl_values[2]
        all_positive = all(p > 0 for p in pnl_values)
        if correct_order and all_positive:
            print("  VERDICT: Correct ordering (opt > base > pess) and all positive")
            print("           -> edge appears robust and sensitivity-consistent")
        elif all_positive:
            print("  VERDICT: All positive but ordering unexpected")
            print("           -> check edge_source breakdown for noise-trading")
        else:
            print("  VERDICT: Not all positive -> edge may be fragile")
    print("           (reminder: simulated Kalshi prices, not real orderbook data)")


# ── Lag analysis ──────────────────────────────────────────────────


def _run_lag_analysis(match_dirs: list[Path]) -> None:
    """Measure real market reaction speed after goals using recorded data.

    For each goal, compares P_model (from MC) vs P_kalshi (from real book)
    at offsets 0, +1, +2, +5, +10, +15, +30, +60 seconds after the event.
    """
    offsets = [0, 1, 2, 5, 10, 15, 30, 60]
    all_rows: list[dict] = []

    for match_dir in match_dirs:
        meta_path = match_dir / "metadata.json"
        events_path = match_dir / "events.jsonl"
        kalshi_path = match_dir / "kalshi.jsonl"
        if not all(p.exists() for p in [meta_path, events_path, kalshi_path]):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        ticker_list = meta.get("kalshi_tickers", [])
        # Build ticker map: same logic as backtest
        ticker_to_market: dict[str, str] = {}
        if len(ticker_list) >= 3:
            ticker_to_market[ticker_list[0]] = "away_win"
            ticker_to_market[ticker_list[1]] = "draw"
            ticker_to_market[ticker_list[2]] = "home_win"

        # Load goal events with wall-clock timestamps
        goals: list[dict] = []
        with open(events_path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("type") == "goal":
                    goals.append(e)

        if not goals:
            continue

        match_label = f"{meta.get('home_team', '?')} v {meta.get('away_team', '?')}"

        # For each goal, create a fresh replay and measure prices at offsets
        for gi, goal in enumerate(goals):
            goal_ts = goal.get("ts_wall", 0)
            team = goal.get("team", "?")
            new_score = goal.get("new_score", [0, 0])
            if goal_ts == 0:
                continue

            row = {
                "match": match_label,
                "goal_num": gi + 1,
                "team": team,
                "score": f"{new_score[0]}-{new_score[1]}",
                "goal_ts": goal_ts,
            }

            replay = KalshiOrderbookReplay(kalshi_path, ticker_to_market)

            for dt in offsets:
                prices = replay.get_prices_at(goal_ts + dt)
                spread_h = replay.get_spread("home_win")
                row[f"H_{dt}s"] = prices.get("home_win", 0)
                row[f"D_{dt}s"] = prices.get("draw", 0)
                row[f"A_{dt}s"] = prices.get("away_win", 0)
                row[f"spread_{dt}s"] = spread_h

            all_rows.append(row)

    if not all_rows:
        print("\n  No goals found for lag analysis.")
        return

    # Print analysis
    print()
    print("=" * 75)
    print("  AGGREGATE LAG ANALYSIS")
    print("=" * 75)

    # Per-goal detail
    print()
    print("  Goal Reaction Detail:")
    for row in all_rows:
        print(f"    {row['match']} — Goal {row['goal_num']} ({row['team']}) → {row['score']}")
        parts = []
        for dt in offsets:
            h = row.get(f"H_{dt}s", 0)
            sp = row.get(f"spread_{dt}s", 0)
            parts.append(f"H={h:.3f}(sp={sp:.3f})")
        print(f"      t=0s: {parts[0]}")
        for i, dt in enumerate(offsets[1:], 1):
            print(f"      +{dt:>2}s: {parts[i]}")

    # Aggregate: average absolute edge change from pre-goal
    print()
    print("  Seconds after goal    Avg |delta_H|   Avg spread    N")
    print("  " + "-" * 55)
    for dt in offsets:
        deltas = []
        spreads = []
        for row in all_rows:
            h_at_dt = row.get(f"H_{dt}s", 0)
            h_pre = row.get("H_0s", 0)
            if h_at_dt > 0 and h_pre > 0:
                deltas.append(abs(h_at_dt - h_pre))
                spreads.append(row.get(f"spread_{dt}s", 0))
        if deltas:
            avg_d = sum(deltas) / len(deltas)
            avg_s = sum(spreads) / len(spreads)
            label = "pre-goal" if dt == 0 else f"+{dt}s"
            print(f"  {label:>20}    {avg_d:.4f}        {avg_s:.4f}        {len(deltas)}")

    print()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline backtest: Phase 1->2->3->4"
    )
    parser.add_argument("--matches", nargs="*", help="Paths to match data directories")
    parser.add_argument("--use-simulator", action="store_true",
                        help="Use KalshiPriceSimulator instead of real orderbook data")
    parser.add_argument("--lag-analysis", action="store_true",
                        help="Run lag analysis: measure real market reaction speed after goals")
    # Simulator-only params (ignored when using real data)
    parser.add_argument("--half-life", type=float, default=5.0,
                        help="[simulator] Market reaction half-life in ticks")
    parser.add_argument("--initial-reaction", type=float, default=0.30,
                        help="[simulator] Fraction of move market captures instantly")
    parser.add_argument("--tick-vol", type=float, default=0.002,
                        help="[simulator] Per-tick random walk std")
    parser.add_argument("--spread", type=float, default=0.02,
                        help="[simulator] Base bid-ask spread")
    parser.add_argument("--event-spread-mult", type=float, default=3.0,
                        help="[simulator] Spread multiplier during events")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--bankroll", type=float, default=10_000.0, help="Initial bankroll"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=300,
        help="Phase 4 cooldown_after_exit in ticks (default 300)",
    )
    parser.add_argument(
        "--min-hold",
        type=int,
        default=150,
        help="MIN_HOLD_TICKS for exit triggers (default 150)",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis (simulator mode only)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results to disk"
    )
    args = parser.parse_args()

    # Discover match directories
    if args.matches:
        match_dirs = [Path(m) for m in args.matches]
    else:
        data_dir = PROJECT_ROOT / "data" / "latency"
        if not data_dir.exists():
            print("ERROR: data/latency/ not found")
            sys.exit(1)
        match_dirs = sorted(
            [
                d
                for d in data_dir.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()
            ]
        )

    price_source = "simulator" if args.use_simulator else "real orderbook"
    print("=" * 65)
    print("  Full Pipeline Backtest: Phase 1->2->3->4 (Paper Trading)")
    print(f"  Matches: {len(match_dirs)}")
    print(f"  Kalshi prices: {price_source}")
    if args.use_simulator:
        print(f"  Sim params: half_life={args.half_life} react={args.initial_reaction} "
              f"spread={args.spread}")
    print(f"  Bankroll: ${args.bankroll:,.0f} | MC: N={MC_N}")
    print(f"  Phase 4: cooldown={args.cooldown} min_hold={args.min_hold}")
    print("=" * 65)

    baseline_config: dict | None = None
    if args.use_simulator:
        baseline_config = {
            "market_reaction_half_life": args.half_life,
            "market_initial_reaction": args.initial_reaction,
            "tick_volatility": args.tick_vol,
            "base_spread": args.spread,
            "event_spread_mult": args.event_spread_mult,
            "seed": args.seed,
        }

    # Run baseline backtest
    results: list[MatchBacktestResult] = []
    for match_dir in match_dirs:
        print(f"\n  Processing {match_dir.name}...")
        r = backtest_single_match(
            match_dir,
            CALIBRATED_PARAMS,
            baseline_config,
            args.bankroll,
            cooldown_after_exit=args.cooldown,
            min_hold_ticks=args.min_hold,
            use_simulator=args.use_simulator,
        )
        if r is None:
            print(f"    SKIPPED (no metadata)")
            continue
        results.append(r)
        print_match_result(r)

    print_aggregate_report(results, args.bankroll)

    # Lag analysis (real data only)
    if args.lag_analysis and not args.use_simulator and results:
        _run_lag_analysis(match_dirs)

    # Save results
    output_dir = None
    if not args.no_save and results:
        output_base = PROJECT_ROOT / "data" / "backtest_results"
        save_config = baseline_config or {}
        full_config = {**save_config, "price_source": price_source,
                       "bankroll": args.bankroll, "mc_n": MC_N,
                       "cooldown_after_exit": args.cooldown, "min_hold_ticks": args.min_hold}
        config_label = "real_data" if not args.use_simulator else "simulator"
        output_dir = save_results(results, full_config, config_label, output_base)
        print(f"\n  Results saved to: {output_dir}")

    # Sensitivity analysis (simulator mode only)
    if args.sensitivity and args.use_simulator and len(results) > 0:
        sensitivity_configs: list[tuple[str, dict]] = [
            (
                "Optimistic",
                {
                    "market_reaction_half_life": 8.0,
                    "market_initial_reaction": 0.20,
                    "tick_volatility": 0.002,
                    "base_spread": 0.015,
                    "event_spread_mult": 2.0,
                    "seed": 42,
                },
            ),
            ("Baseline", baseline_config),
            (
                "Pessimistic",
                {
                    "market_reaction_half_life": 3.0,
                    "market_initial_reaction": 0.50,
                    "tick_volatility": 0.002,
                    "base_spread": 0.030,
                    "event_spread_mult": 4.0,
                    "seed": 42,
                },
            ),
        ]
        all_sensitivity_results: list[list[MatchBacktestResult]] = []

        for name, config in sensitivity_configs:
            if config == baseline_config:
                all_sensitivity_results.append(results)
                continue
            print(f"\n  Running sensitivity: {name}...")
            sens_results = []
            for match_dir in match_dirs:
                r = backtest_single_match(
                    match_dir,
                    CALIBRATED_PARAMS,
                    config,
                    args.bankroll,
                    cooldown_after_exit=args.cooldown,
                    min_hold_ticks=args.min_hold,
                    use_simulator=True,
                )
                if r is not None:
                    sens_results.append(r)
            all_sensitivity_results.append(sens_results)

            # Save each sensitivity config
            if not args.no_save and sens_results:
                output_base = PROJECT_ROOT / "data" / "backtest_results"
                full_config = {**config, "bankroll": args.bankroll, "mc_n": MC_N,
                               "cooldown_after_exit": args.cooldown, "min_hold_ticks": args.min_hold}
                sens_dir = save_results(
                    sens_results, full_config, name.lower(), output_base
                )
                print(f"  Results saved to: {sens_dir}")

        print_sensitivity_report(sensitivity_configs, all_sensitivity_results)

    print()


if __name__ == "__main__":
    main()
