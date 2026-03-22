#!/usr/bin/env python
"""Multi-match backtest: validate Phase 1->2->3 pipeline across recorded matches.

Discovers all recorded matches in data/recordings/, runs the full pipeline with
calibrated parameters, and prints per-match and aggregate accuracy metrics.

Usage:
    PYTHONPATH=. python scripts/backtest_multi_match.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.step_1_3_ml_prior import compute_C_time
from src.common.types import MarketProbs, Phase2Result
from src.engine.event_handlers import (
    detect_events_from_poll,
    handle_goal,
    handle_period_change,
)
from src.engine.intensity import compute_lambda
from src.engine.mc_pricing import _compute_sigma, _results_to_market_probs
from src.engine.model import LiveMatchModel
from src.math.compute_mu import compute_remaining_mu
from src.math.mc_core import mc_simulate_remaining

try:
    from src.math.mc_core import mc_simulate_remaining_v5
    _HAS_V5_MC = True
except ImportError:
    _HAS_V5_MC = False

from src.prematch.phase2_pipeline import (
    _poisson_1x2, _shin_vig_removal, backsolve_intensities, sanity_check,
)

# -- Calibrated EPL parameters ------------------------------------
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
    "eta_H": 0.05, "eta_A": 0.05, "eta_H2": 0.08, "eta_A2": 0.08,
    "sigma_omega_sq": 0.003,
}

ALPHA_1 = 2.0
BASIS_BOUNDS = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 87.0, 92.0, 93.0])
MC_N = 20_000


# -- Data structures ----------------------------------------------

@dataclass
class GoalSnapshot:
    minute: float
    team: str
    score_after: tuple[int, int]
    pre_goal_probs: tuple[float, float, float]
    post_goal_probs: tuple[float, float, float]
    surprise_score: float
    ekf_K: float
    a_shift: float


@dataclass
class MatchBacktestResult:
    match_id: str
    home_team: str
    away_team: str
    league: str
    final_score: tuple[int, int]
    pre_match_odds: tuple[float, float, float] | None
    a_H: float
    a_A: float
    phase2_status: str
    odds_source: str
    kickoff_probs: tuple[float, float, float]
    final_probs: tuple[float, float, float]
    goal_events: list[GoalSnapshot] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    kickoff_market_error: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    red_flags: list[str] = field(default_factory=list)


# -- MC simulation (sync) -----------------------------------------

def run_mc_sync(model: LiveMatchModel, N: int = MC_N) -> MarketProbs:
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

    seed = int(time.monotonic() * 1_000_000) % (2**31)
    S_H, S_A = model.score

    if _HAS_V5_MC and model.delta_H_pos is not None:
        results = mc_simulate_remaining_v5(
            t_now=model.t, T_end=model.T_exp, S_H=S_H, S_A=S_A,
            state=model.current_state_X, score_diff=model.delta_S,
            a_H=model.a_H, a_A=model.a_A, b=model.b,
            gamma_H=model.gamma_H, gamma_A=model.gamma_A,
            delta_H_pos=model.delta_H_pos, delta_H_neg=model.delta_H_neg,
            delta_A_pos=model.delta_A_pos, delta_A_neg=model.delta_A_neg,
            Q_diag=Q_diag, Q_off=Q_off, basis_bounds=model.basis_bounds,
            N=N, seed=seed,
            eta_H=model.eta_H, eta_A=model.eta_A,
            eta_H2=model.eta_H2, eta_A2=model.eta_A2,
            stoppage_1_start=45.0, stoppage_2_start=90.0,
        )
    else:
        results = mc_simulate_remaining(
            t_now=model.t, T_end=model.T_exp, S_H=S_H, S_A=S_A,
            state=model.current_state_X, score_diff=model.delta_S,
            a_H=model.a_H, a_A=model.a_A, b=model.b,
            gamma_H=model.gamma_H, gamma_A=model.gamma_A,
            delta_H=model.delta_H, delta_A=model.delta_A,
            Q_diag=Q_diag, Q_off=Q_off, basis_bounds=model.basis_bounds,
            N=N, seed=seed,
        )
    return _results_to_market_probs(results, S_H, S_A)


# -- Per-match simulation -----------------------------------------

def extract_bet365_odds(data_dir: Path) -> tuple[float, float, float] | None:
    """Extract first Bet365 ML odds from odds_api.jsonl."""
    odds_path = data_dir / "odds_api.jsonl"
    if not odds_path.exists():
        return None
    with open(odds_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("bookie") != "Bet365":
                continue
            for mkt in entry.get("markets", []):
                if mkt.get("name") == "ML" and mkt.get("odds"):
                    o = mkt["odds"][0]
                    try:
                        return float(o["home"]), float(o["draw"]), float(o["away"])
                    except (KeyError, ValueError):
                        continue
    # Try Sbobet as fallback
    with open(odds_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("bookie") != "Sbobet":
                continue
            for mkt in entry.get("markets", []):
                if mkt.get("name") == "ML" and mkt.get("odds"):
                    o = mkt["odds"][0]
                    try:
                        return float(o["home"]), float(o["draw"]), float(o["away"])
                    except (KeyError, ValueError):
                        continue
    return None


def derive_goal_minutes(data_dir: Path) -> list[dict]:
    """Derive goal minutes from events.jsonl using wall-clock timestamps.

    Returns list of {minute: float, team: str, score: [h, a]}.
    """
    events_path = data_dir / "events.jsonl"
    if not events_path.exists():
        return []

    events = []
    with open(events_path) as f:
        for line in f:
            events.append(json.loads(line))

    # Find kickoff: first status_change with numeric new_status, or first goal
    kickoff_wall = None
    for e in events:
        if e.get("type") == "status_change":
            ns = str(e.get("new_status", ""))
            if ns.isdigit():
                kickoff_wall = e.get("ts_wall")
                break

    # If no numeric status, estimate kickoff from first event wall time
    if kickoff_wall is None:
        for e in events:
            if e.get("ts_wall"):
                kickoff_wall = e["ts_wall"]
                break
    if kickoff_wall is None:
        return []

    # Find halftime wall time to account for break
    ht_start_wall = None
    ht_end_wall = None
    for e in events:
        if e.get("type") == "status_change" and e.get("new_status") == "HT":
            if ht_start_wall is None:
                ht_start_wall = e.get("ts_wall")
        if e.get("type") == "status_change":
            ns = str(e.get("new_status", ""))
            if ns.isdigit() and int(ns) > 45:
                ht_end_wall = e.get("ts_wall")
                break

    ht_duration = 0.0
    if ht_start_wall and ht_end_wall:
        ht_duration = ht_end_wall - ht_start_wall

    goals = []
    seen_scores = set()
    for e in events:
        if e.get("type") != "goal":
            continue
        score = tuple(e.get("new_score", [0, 0]))
        # Deduplicate: same score appearing multiple times = VAR reversal
        if score in seen_scores:
            continue
        seen_scores.add(score)

        wall = e.get("ts_wall", kickoff_wall)
        elapsed_s = wall - kickoff_wall
        # Subtract halftime if goal is after HT
        if ht_start_wall and wall > ht_start_wall:
            elapsed_s -= ht_duration
        minute = max(1.0, elapsed_s / 60.0)
        minute = min(minute, 93.0)  # cap at regulation

        goals.append({
            "minute": round(minute, 1),
            "team": e.get("team", "home"),
            "score": list(score),
        })

    return goals


def simulate_match(data_dir: Path, params: dict) -> MatchBacktestResult | None:
    """Run full Phase 1->2->3 pipeline on a single recorded match."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    match_id = meta.get("match_id", data_dir.name)
    home = meta.get("home_team", "Home")
    away = meta.get("away_team", "Away")
    league = meta.get("league", "?")
    final = tuple(meta.get("final_score", [0, 0]))

    # -- Phase 2: Get odds and backsolve -----------------------
    odds = extract_bet365_odds(data_dir)
    b = np.array(params["b"])
    Q = np.array(params["Q"])
    basis_bounds = BASIS_BOUNDS
    C_time = compute_C_time(b, basis_bounds)

    if odds:
        p_h, p_d, p_a = _shin_vig_removal(*odds)
        market_implied = MarketProbs(home_win=p_h, draw=p_d, away_win=p_a)
        a_H, a_A = backsolve_intensities(market_implied, b, Q, basis_bounds)
        odds_source = "Bet365"
        phase2_status = "GO"
    else:
        a_H = float(np.log(1.4 / C_time))
        a_A = float(np.log(1.1 / C_time))
        market_implied = None
        p_h, p_d, p_a = 0.40, 0.30, 0.30
        odds_source = "league_average"
        phase2_status = "GO (no-odds)"

    mu_H = float(np.exp(a_H) * C_time)
    mu_A = float(np.exp(a_A) * C_time)

    result2 = Phase2Result(
        match_id=match_id, league_id=1204,
        a_H=a_H, a_A=a_A, mu_H=mu_H, mu_A=mu_A,
        C_time=C_time, verdict=phase2_status, skip_reason=None,
        param_version=1, home_team=home, away_team=away,
        kickoff_utc="2026-03-16T20:00:00+00:00",
        kalshi_tickers={}, market_implied=market_implied,
        prediction_method="backsolve_odds_api" if odds else "league_mle",
        ekf_P0=0.15 if odds else 0.50,
    )

    # -- Build model -------------------------------------------
    model = LiveMatchModel.from_phase2_result(result2, params)
    model.engine_phase = "FIRST_HALF"
    model.t = 0.001

    # JIT warmup (first call only)
    run_mc_sync(model, N=100)

    # -- Phase 3: Simulate using derived goal minutes ----------
    model.t = 0.0
    P0 = run_mc_sync(model)
    kickoff_probs = (P0.home_win, P0.draw, P0.away_win)

    # Compute kickoff market error
    kickoff_error = abs(P0.home_win - p_h) if odds else 0.0

    goals = derive_goal_minutes(data_dir)
    goal_snapshots: list[GoalSnapshot] = []
    trajectory: list[dict] = [_snap(model, P0, 0.0)]
    last_update_t = 0.0

    # Checkpoints
    checkpoints = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90}
    recorded_checkpoints: set[int] = set()

    for goal in goals:
        t_goal = goal["minute"]
        team = goal["team"]

        # Advance time with EKF updates
        model.t = max(model.t, t_goal - 0.1)
        if model.t > 45 and model.engine_phase == "FIRST_HALF":
            handle_period_change(model, "HALFTIME")
            handle_period_change(model, "SECOND_HALF")

        dt = model.t - last_update_t
        if dt > 0 and model.strength_updater:
            model.strength_updater.predict(dt)
            lH = compute_lambda(model, "home")
            lA = compute_lambda(model, "away")
            model.strength_updater.update_no_goal(lH, lA, dt)
            model.a_H = model.strength_updater.a_H
            model.a_A = model.strength_updater.a_A
            last_update_t = model.t

        # Record checkpoints we passed
        for cp in sorted(checkpoints - recorded_checkpoints):
            if cp <= model.t:
                model_cp = model.t
                model.t = float(cp)
                P_cp = run_mc_sync(model)
                trajectory.append(_snap(model, P_cp, float(cp)))
                model.t = model_cp
                recorded_checkpoints.add(cp)

        # Pre-goal MC
        model.t = t_goal
        pre_P = run_mc_sync(model)

        # EKF state before goal
        ekf = model.ekf_tracker
        pre_P_team = ekf.P_H if team == "home" else ekf.P_A
        pre_lam = compute_lambda(model, team)
        K = pre_P_team / (pre_P_team * pre_lam + 1.0) if pre_lam > 0 else 0.0

        old_a = model.a_H if team == "home" else model.a_A

        # Handle goal
        handle_goal(model, team, int(t_goal))

        new_a = model.a_H if team == "home" else model.a_A
        a_shift = new_a - old_a

        # Post-goal MC
        post_P = run_mc_sync(model)

        # Surprise score
        surprise = 0.0
        if model.strength_updater:
            surprise = model.strength_updater.compute_surprise_score(
                team, pre_P.home_win, pre_P.away_win,
            )
        model.surprise_score = surprise

        goal_snapshots.append(GoalSnapshot(
            minute=t_goal, team=team,
            score_after=model.score,
            pre_goal_probs=(pre_P.home_win, pre_P.draw, pre_P.away_win),
            post_goal_probs=(post_P.home_win, post_P.draw, post_P.away_win),
            surprise_score=surprise, ekf_K=K, a_shift=a_shift,
        ))
        trajectory.append(_snap(model, post_P, t_goal, f"GOAL {team}"))
        last_update_t = t_goal

        # Cooldown management
        model.tick_count += 50
        if model.cooldown and model.t >= model.cooldown_until_t:
            model.cooldown = False
            model.event_state = "IDLE"

    # Final pricing
    model.t = min(92.9, model.T_exp - 0.1)
    if model.t > 45 and model.engine_phase == "FIRST_HALF":
        handle_period_change(model, "HALFTIME")
        handle_period_change(model, "SECOND_HALF")
    P_final = run_mc_sync(model)
    final_probs = (P_final.home_win, P_final.draw, P_final.away_win)
    trajectory.append(_snap(model, P_final, model.t))

    # -- Compute accuracy metrics ------------------------------
    # Brier score: (p_outcome - 1)^2 + sum(p_other^2)
    if final[0] > final[1]:
        outcome_idx = 0  # home win
    elif final[0] == final[1]:
        outcome_idx = 1  # draw
    else:
        outcome_idx = 2  # away win

    brier = sum(
        (final_probs[i] - (1.0 if i == outcome_idx else 0.0)) ** 2
        for i in range(3)
    )

    p_outcome = max(1e-10, final_probs[outcome_idx])
    log_loss_val = -math.log(p_outcome)

    # Red flags
    flags: list[str] = []
    if kickoff_error > 0.02:
        flags.append(f"kickoff_mismatch={kickoff_error:.3f}")
    if p_outcome < 0.10:
        flags.append(f"low_outcome_prob={p_outcome:.3f}")
    for gs in goal_snapshots:
        for p in gs.pre_goal_probs + gs.post_goal_probs:
            if math.isnan(p) or math.isinf(p):
                flags.append("nan_or_inf_in_probs")
                break
    if not (-6.0 <= model.a_H <= 0.0) or not (-6.0 <= model.a_A <= 0.0):
        flags.append(f"a_out_of_range: a_H={model.a_H:.2f}, a_A={model.a_A:.2f}")
    if ekf and (ekf.P_H > 2.0 or ekf.P_A > 2.0):
        flags.append(f"ekf_P_large: P_H={ekf.P_H:.2f}, P_A={ekf.P_A:.2f}")

    return MatchBacktestResult(
        match_id=match_id, home_team=home, away_team=away,
        league=league, final_score=final,
        pre_match_odds=odds, a_H=a_H, a_A=a_A,
        phase2_status=phase2_status, odds_source=odds_source,
        kickoff_probs=kickoff_probs, final_probs=final_probs,
        goal_events=goal_snapshots, trajectory=trajectory,
        kickoff_market_error=kickoff_error,
        brier_score=brier, log_loss=log_loss_val,
        red_flags=flags,
    )


def _snap(model: LiveMatchModel, P: MarketProbs, t: float, event: str = "") -> dict:
    return {
        "t": t, "P_H": P.home_win, "P_D": P.draw, "P_A": P.away_win,
        "score": model.score, "a_H": model.a_H, "a_A": model.a_A,
        "event": event,
    }


# -- Main ---------------------------------------------------------

def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "recordings"
    if not data_dir.exists():
        print("ERROR: data/recordings/ not found")
        sys.exit(1)

    match_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    print("=" * 80)
    print("  Multi-Match Backtest: Phase 1->2->3 Pipeline Validation")
    print(f"  Matches found: {len(match_dirs)}")
    print(f"  Parameters: calibrated b/delta/gamma + sigma_omega=0.003")
    print(f"  MC simulations: N={MC_N}")
    print("=" * 80)

    results: list[MatchBacktestResult] = []

    for match_dir in match_dirs:
        match_id = match_dir.name
        print(f"\n{'-' * 70}")
        print(f"Processing {match_id}...")

        t0 = time.monotonic()
        r = simulate_match(match_dir, CALIBRATED_PARAMS)
        elapsed = time.monotonic() - t0

        if r is None:
            print(f"  SKIPPED (no metadata)")
            continue

        results.append(r)

        # Per-match summary
        outcome = "H" if r.final_score[0] > r.final_score[1] else "D" if r.final_score[0] == r.final_score[1] else "A"
        print(f"  {r.home_team} {r.final_score[0]}-{r.final_score[1]} {r.away_team} ({r.league})")
        print(f"  Odds: {r.odds_source}" + (f" ({r.pre_match_odds[0]:.2f}/{r.pre_match_odds[1]:.2f}/{r.pre_match_odds[2]:.2f})" if r.pre_match_odds else ""))
        print(f"  Kickoff P(H/D/A): {r.kickoff_probs[0]:.3f} / {r.kickoff_probs[1]:.3f} / {r.kickoff_probs[2]:.3f}")
        print(f"  Final   P(H/D/A): {r.final_probs[0]:.3f} / {r.final_probs[1]:.3f} / {r.final_probs[2]:.3f}")
        print(f"  Outcome: {outcome} | P(outcome)={r.final_probs[{'H':0,'D':1,'A':2}[outcome]]:.3f} | Brier={r.brier_score:.4f} | LogLoss={r.log_loss:.4f}")

        for gs in r.goal_events:
            team_name = r.home_team if gs.team == "home" else r.away_team
            print(f"    t={gs.minute:5.1f} GOAL {team_name:>20s} {gs.score_after[0]}-{gs.score_after[1]}  "
                  f"P(H):{gs.pre_goal_probs[0]:.3f}->{gs.post_goal_probs[0]:.3f}  "
                  f"K={gs.ekf_K:.3f}  da={gs.a_shift:+.3f}  SS={gs.surprise_score:.3f}")

        if r.red_flags:
            print(f"  RED FLAGS: {', '.join(r.red_flags)}")
        print(f"  [{elapsed:.1f}s]")

    # -- Aggregate report --------------------------------------
    print(f"\n{'=' * 80}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 80}")

    n = len(results)
    if n == 0:
        print("No matches processed")
        return

    # 2a. Kickoff calibration
    odds_results = [r for r in results if r.pre_match_odds]
    print(f"\n  2a. Kickoff calibration ({len(odds_results)} matches with odds):")
    print(f"  {'Match':<35} {'Odds P(H)':>9} {'Model P(H)':>10} {'Error':>7}")
    print(f"  {'-'*35} {'-'*9} {'-'*10} {'-'*7}")
    for r in odds_results:
        p_h_mkt = r.pre_match_odds[0]
        p_h_shin, _, _ = _shin_vig_removal(*r.pre_match_odds)
        print(f"  {r.home_team+' v '+r.away_team:<35} {p_h_shin:9.4f} {r.kickoff_probs[0]:10.4f} {r.kickoff_market_error:+7.4f}")
    if odds_results:
        mae = np.mean([r.kickoff_market_error for r in odds_results])
        print(f"  MAE: {mae:.4f} (target < 0.010)")

    # 2b. Brier scores
    print(f"\n  2b. Brier score per match:")
    print(f"  {'Match':<35} {'Score':>6} {'Out':>3} {'P(out)':>6} {'Brier':>7} {'LogLoss':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*3} {'-'*6} {'-'*7} {'-'*8}")
    for r in results:
        outcome = "H" if r.final_score[0] > r.final_score[1] else "D" if r.final_score[0] == r.final_score[1] else "A"
        p_out = r.final_probs[{"H":0,"D":1,"A":2}[outcome]]
        sc = f"{r.final_score[0]}-{r.final_score[1]}"
        print(f"  {r.home_team+' v '+r.away_team:<35} {sc:>6} {outcome:>3} {p_out:6.3f} {r.brier_score:7.4f} {r.log_loss:8.4f}")
    avg_brier = np.mean([r.brier_score for r in results])
    avg_logloss = np.mean([r.log_loss for r in results])
    print(f"  Average Brier: {avg_brier:.4f} (bookmaker benchmark: ~0.19-0.21)")
    print(f"  Average LogLoss: {avg_logloss:.4f}")

    # 2c. Goal-event analysis
    all_goals = [gs for r in results for gs in r.goal_events]
    print(f"\n  2c. Goal-event analysis ({len(all_goals)} goals):")
    if all_goals:
        print(f"  Mean |a_shift|: {np.mean([abs(gs.a_shift) for gs in all_goals]):.4f}")
        buckets = {"0-30": [], "30-60": [], "60-90+": []}
        for gs in all_goals:
            if gs.minute < 30:
                buckets["0-30"].append(gs.ekf_K)
            elif gs.minute < 60:
                buckets["30-60"].append(gs.ekf_K)
            else:
                buckets["60-90+"].append(gs.ekf_K)
        for label, ks in buckets.items():
            if ks:
                print(f"    {label}: mean K = {np.mean(ks):.3f} (N={len(ks)})")
        # Check K increases with time
        means = [np.mean(buckets[b]) if buckets[b] else 0 for b in ["0-30", "30-60", "60-90+"]]
        if means[0] > 0 and means[2] > means[0]:
            print(f"    K increases with time: YES ({means[0]:.3f} -> {means[2]:.3f})")
        else:
            print(f"    K increases with time: CHECK ({means})")

    # 2d. Late-game P(D) check
    late_draws = [(r, gs) for r in results for gs in r.goal_events
                  if gs.minute >= 75 and gs.score_after[0] == gs.score_after[1] and gs.score_after[0] > 0]
    if late_draws:
        print(f"\n  2d. Late-equaliser P(D) ({len(late_draws)} events):")
        for r, gs in late_draws:
            print(f"    {r.home_team} v {r.away_team}: {gs.score_after[0]}-{gs.score_after[1]} at t={gs.minute:.0f}"
                  f" -> P(D)={gs.post_goal_probs[1]:.3f}")

    # Red flags summary
    flagged = [r for r in results if r.red_flags]
    print(f"\n  Red flags: {len(flagged)}/{n} matches")
    for r in flagged:
        print(f"    {r.match_id} ({r.home_team} v {r.away_team}): {', '.join(r.red_flags)}")

    # Verdict
    print(f"\n{'=' * 80}")
    if not flagged:
        print("  VERDICT: ALL MATCHES PASS -- model generalizes well, ready for Phase 4")
    else:
        print(f"  VERDICT: {len(flagged)}/{n} MATCHES HAVE ISSUES -- investigate before Phase 4")
        for r in flagged:
            print(f"    - {r.match_id}: {', '.join(r.red_flags)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
