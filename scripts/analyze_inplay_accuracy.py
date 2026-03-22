#!/usr/bin/env python3
"""
In-Play MMPP Accuracy vs Kalshi Prices
========================================
Uses the Brentford 2-2 Wolves recording (match 4190023) to measure how
accurately the MMPP model prices P(home_win) during live play.

Reconstructs:
  1. MMPP P_model timeline: minute-by-minute MC simulation using actual score
  2. Kalshi price timeline: orderbook mid-price from recorded WS data
  3. Gap analysis: where model diverges most from market

Data sources:
  - data/recordings/4190023/kalshi_ob.jsonl (orderbook snapshots + deltas)
  - data/recordings/4190023/events.jsonl (goals with exact timestamps)
  - data/recordings/4190023/goalserve.jsonl (minute-level status + score)
  - MMPP model with EPL-average default params (Phase 1 not yet trained)

Match: Brentford 2-2 Wolves, 2026-03-16
Goals: min 22 (1-0 H), min 37 (2-0 H), min 44 (2-1 A), min 77 (2-2 A)

Usage:
  PYTHONPATH=. python scripts/analyze_inplay_accuracy.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.engine.strength_updater import InPlayStrengthUpdater

# --- Config -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MATCH_DIR = PROJECT_ROOT / "data" / "recordings" / "4190023"
BRE_TICKER_SUFFIX = "-BRE"  # Brentford home win market (suffix to avoid matching BREWOL event)

# MMPP default params -- EPL-calibrated (8-period basis)
# b vector from Phase 1 calibration on 2,278 EPL matches
DEFAULT_PARAMS = {
    "b": [-0.1688, -0.1282, -0.1008, 0.0581, -0.0025, -0.0797, -0.0499, 0.1982],
    "gamma_H": [0.0, -0.30, 0.10, -0.20],
    "gamma_A": [0.0, 0.10, -0.30, -0.20],
    "delta_H": [0.15, 0.08, 0.0, -0.05, -0.12],
    "delta_A": [-0.12, -0.05, 0.0, 0.08, 0.15],
    "Q": [
        [-0.0009, 0.00045, 0.00045, 0.0],
        [0.0, -0.00045, 0.0, 0.00045],
        [0.0, 0.0, -0.00045, 0.00045],
        [0.0, 0.0, 0.0, 0.0],
    ],
    # Per-minute log-intensities (EPL average)
    "a_H_avg": -4.09,  # exp(-4.09)*93 ~ 1.55 goals/match
    "a_A_avg": -4.33,  # exp(-4.33)*93 ~ 1.22 goals/match
}

# Match-specific overrides: Brentford home at Gtech vs Wolves
# Pre-match Kalshi: BRE ~0.62, WOL ~0.15, TIE ~0.21
# This implies Brentford is a moderate home favorite.
# Backsolve: P(home_win)=0.62 ->scale a_H up, a_A down
# Using the logit-based scaling from analyze_structural_edge.py:
_LOGIT_DEFAULT_HOME = math.log(0.456 / 0.544)
_LOGIT_BRE = math.log(0.62 / 0.38)
_LOGIT_DEFAULT_AWAY = math.log(0.302 / 0.698)
_LOGIT_WOL = math.log(0.15 / 0.85)
_SCALE = 0.25

A_H_BRE = DEFAULT_PARAMS["a_H_avg"] + _SCALE * (_LOGIT_BRE - _LOGIT_DEFAULT_HOME)
A_A_WOL = DEFAULT_PARAMS["a_A_avg"] + _SCALE * (_LOGIT_WOL - _LOGIT_DEFAULT_AWAY)


# --- MMPP Simplified Pricing -------------------------------------------------

def mmpp_mc_prices(
    t_min: float,
    score_h: int,
    score_a: int,
    state_X: int,
    a_H: float,
    a_A: float,
    N: int = 10_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Simplified MMPP MC for P(home_win), P(draw), P(away_win).

    Uses Poisson approximation for speed (no red card transitions).
    Returns dict with keys: home_win, draw, away_win, mu_H, mu_A.
    """
    p = DEFAULT_PARAMS
    b = p["b"]
    gamma_H = p["gamma_H"]
    gamma_A = p["gamma_A"]
    delta_H = p["delta_H"]
    delta_A = p["delta_A"]

    T_exp = 93.0
    if t_min >= T_exp:
        if score_h > score_a:
            return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0, "mu_H": 0.0, "mu_A": 0.0}
        elif score_h < score_a:
            return {"home_win": 0.0, "draw": 0.0, "away_win": 1.0, "mu_H": 0.0, "mu_A": 0.0}
        return {"home_win": 0.0, "draw": 1.0, "away_win": 0.0, "mu_H": 0.0, "mu_A": 0.0}

    ds = max(-2, min(2, score_h - score_a))
    di_H = ds + 2
    di_A = -ds + 2  # away perspective to match MLE calibration
    basis_bounds = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, T_exp]
    mu_H = 0.0
    mu_A = 0.0
    for k in range(len(basis_bounds) - 1):
        seg_start = max(t_min, basis_bounds[k])
        seg_end = min(T_exp, basis_bounds[k + 1])
        if seg_start >= seg_end:
            continue
        dt = seg_end - seg_start
        mu_H += dt * math.exp(a_H + b[k] + gamma_H[state_X] + delta_H[di_H])
        mu_A += dt * math.exp(a_A + b[k] + gamma_A[state_X] + delta_A[di_A])

    mu_H = max(0.001, mu_H)
    mu_A = max(0.001, mu_A)

    rng = np.random.default_rng(seed if seed is not None else int(t_min * 1000) % (2**31))
    rem_H = rng.poisson(mu_H, size=N)
    rem_A = rng.poisson(mu_A, size=N)
    final_H = score_h + rem_H
    final_A = score_a + rem_A

    p_home = float(np.sum(final_H > final_A)) / N
    p_draw = float(np.sum(final_H == final_A)) / N
    p_away = float(np.sum(final_A > final_H)) / N

    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "mu_H": mu_H,
        "mu_A": mu_A,
    }


# --- Kalshi Orderbook Mid-Price ----------------------------------------------

def build_kalshi_mid_timeline(
    match_dir: Path, ticker_suffix: str = "-BRE"
) -> list[tuple[float, float, float, float]]:
    """Returns [(ts_wall, mid, best_bid, best_ask), ...].

    Reuses the approach from analyze_drift.py:
    best_bid = max(yes_dollars_fp prices with qty > 0)
    best_ask = 1 - max(no_dollars_fp prices with qty > 0)
    mid = (best_bid + best_ask) / 2

    IMPORTANT: Uses ticker suffix matching (e.g., "-BRE") to avoid
    matching other markets that contain "BRE" in the event name.
    """
    yes_book: dict[str, float] = {}
    no_book: dict[str, float] = {}
    tl: list[tuple[float, float, float, float]] = []

    with open(match_dir / "kalshi_ob.jsonl") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("msg", data)
            ticker = msg.get("market_ticker", "")
            if not ticker.endswith(ticker_suffix):
                continue
            ts = data.get("_ts_wall", 0)
            msg_type = data.get("type", "")

            if msg_type == "orderbook_snapshot":
                yes_book.clear()
                no_book.clear()
                for p, q in msg.get("yes_dollars_fp", []):
                    if float(q) > 0:
                        yes_book[p] = float(q)
                for p, q in msg.get("no_dollars_fp", []):
                    if float(q) > 0:
                        no_book[p] = float(q)

            elif msg_type == "orderbook_delta":
                side = msg.get("side", "")
                p = msg.get("price_dollars", "")
                delta = float(msg.get("delta_fp", "0"))
                if not p:
                    continue
                book = yes_book if side == "yes" else no_book
                cur = book.get(p, 0)
                new = cur + delta
                if new > 0:
                    book[p] = new
                else:
                    book.pop(p, None)
            else:
                continue

            if yes_book and no_book:
                best_bid = max(float(p) for p in yes_book)
                best_ask = 1.0 - max(float(p) for p in no_book)
                mid = (best_bid + best_ask) / 2.0
                tl.append((ts, mid, best_bid, best_ask))

    return tl


# --- Event Timeline ----------------------------------------------------------

def load_events(match_dir: Path) -> tuple[list[dict], dict[int, float], float, float]:
    """Load events, build minute->ts_wall map, find kickoff and halftime.

    Returns:
        (goals, minute_to_ts, kickoff_ts, halftime_start_ts)
    """
    goals: list[dict] = []
    minute_to_ts: dict[int, float] = {}
    halftime_start_ts = 0.0

    with open(match_dir / "events.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["type"] == "goal":
                goals.append(d)
            elif d["type"] == "status_change":
                new = d.get("new_status", "")
                if new.isdigit():
                    minute_to_ts[int(new)] = d["ts_wall"]
                elif new == "HT" and halftime_start_ts == 0.0:
                    halftime_start_ts = d["ts_wall"]
                elif new == "FT":
                    minute_to_ts[999] = d["ts_wall"]  # sentinel

    # Infer kickoff from first minute event
    # Status goes 4->5->6... so minute N starts at the event ts
    # Kickoff (minute 0) ~ ts(minute_5) - 5*60
    first_min = min(minute_to_ts.keys())
    kickoff_ts = minute_to_ts[first_min] - first_min * 60.0

    return goals, minute_to_ts, kickoff_ts, halftime_start_ts


def load_goalserve_score_timeline(match_dir: Path) -> list[dict]:
    """Load minute-level score from goalserve.jsonl.

    Returns [{ts_wall, minute, score_h, score_a}, ...] for in-play entries.
    """
    entries = []
    with open(match_dir / "goalserve.jsonl") as f:
        for line in f:
            d = json.loads(line)
            data = d.get("data") or {}
            status = data.get("@status", "")
            if not status or not status.isdigit():
                continue
            minute = int(status)
            lt = data.get("localteam") or {}
            vt = data.get("visitorteam") or {}
            try:
                sh = int(lt.get("@goals", "0"))
                sa = int(vt.get("@goals", "0"))
            except ValueError:
                continue
            entries.append({
                "ts_wall": d["_ts_wall"],
                "minute": minute,
                "score_h": sh,
                "score_a": sa,
            })
    return entries


# --- Main Analysis ------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("IN-PLAY MMPP ACCURACY: Brentford 2-2 Wolves (2026-03-16)")
    print("=" * 72)

    # -- Step 0: Load data -----------------------------------------------------
    print("\n[Loading data...]")

    goals, minute_to_ts, kickoff_ts, ht_start_ts = load_events(MATCH_DIR)
    gs_timeline = load_goalserve_score_timeline(MATCH_DIR)

    print(f"  Goals: {len(goals)}")
    for g in goals:
        utc = g["utc"][:19]
        print(f"    {utc} | {g['prev_score']} ->{g['new_score']} ({g['team']})")

    print(f"  Goalserve in-play entries: {len(gs_timeline)}")
    print(f"  Kickoff (inferred): {datetime.fromtimestamp(kickoff_ts, tz=timezone.utc).strftime('%H:%M:%S')} UTC")

    # Build minute->score mapping from goalserve data
    minute_score: dict[int, tuple[int, int]] = {}
    for entry in gs_timeline:
        m = entry["minute"]
        minute_score[m] = (entry["score_h"], entry["score_a"])

    # Fill gaps by propagating last known score
    last_score = (0, 0)
    for m in range(1, 100):
        if m in minute_score:
            last_score = minute_score[m]
        else:
            minute_score[m] = last_score

    print("\n  Score progression:")
    prev = None
    for m in range(1, 100):
        s = minute_score.get(m, (0, 0))
        if s != prev:
            print(f"    Min {m}: {s[0]}-{s[1]}")
            prev = s

    # -- Step 1: Build Kalshi mid-price timeline -------------------------------
    print("\n[Building Kalshi BRE mid-price timeline...]")
    kalshi_tl = build_kalshi_mid_timeline(MATCH_DIR, BRE_TICKER_SUFFIX)
    print(f"  Total orderbook updates: {len(kalshi_tl):,}")

    # Build minute->median_mid mapping
    # Map ts_wall to match minute using minute_to_ts
    kalshi_by_minute: dict[int, list[float]] = defaultdict(list)

    # Use halftime timing to properly map second half
    # First half: kickoff_ts to ht_start_ts
    # Second half: ht_start_ts + ~16min break to FT
    # We'll map each kalshi timestamp to the nearest Goalserve minute

    # Build sorted list of (minute, ts_wall) boundaries
    sorted_minutes = sorted(minute_to_ts.items())

    for ts, mid, _, _ in kalshi_tl:
        if ts < kickoff_ts:
            continue
        # Find which minute this timestamp falls in
        assigned_min = None
        for i in range(len(sorted_minutes) - 1):
            m1, t1 = sorted_minutes[i]
            m2, t2 = sorted_minutes[i + 1]
            if t1 <= ts < t2:
                assigned_min = m1
                break
        if assigned_min is not None and 1 <= assigned_min <= 99:
            kalshi_by_minute[assigned_min].append(mid)

    kalshi_minute_mid: dict[int, float] = {}
    for m, mids in kalshi_by_minute.items():
        kalshi_minute_mid[m] = statistics.median(mids)

    print(f"  Minutes with Kalshi data: {len(kalshi_minute_mid)}")
    if kalshi_minute_mid:
        print(f"  Range: min {min(kalshi_minute_mid)} to {max(kalshi_minute_mid)}")

    # -- Step 2: Compute MMPP P_model for each minute -------------------------
    print("\n[Computing MMPP P_model for each minute...]")
    print(f"  Using match-specific params: a_H={A_H_BRE:.3f}, a_A={A_A_WOL:.3f}")
    print(f"  (Backsolved from pre-match Kalshi: BRE=0.62, WOL=0.15)")

    model_timeline: dict[int, dict] = {}
    for m in range(1, 100):
        score = minute_score.get(m, (0, 0))
        result = mmpp_mc_prices(
            t_min=float(m),
            score_h=score[0],
            score_a=score[1],
            state_X=0,  # no red cards this match
            a_H=A_H_BRE,
            a_A=A_A_WOL,
            N=10_000,
            seed=m * 31 + 7,  # deterministic per-minute seed
        )
        model_timeline[m] = {
            "minute": m,
            "score": score,
            "p_home": result["home_win"],
            "p_draw": result["draw"],
            "p_away": result["away_win"],
            "mu_H": result["mu_H"],
            "mu_A": result["mu_A"],
        }

    print(f"  Computed {len(model_timeline)} minutes")

    # -- Step 2b: v3 -- 8-period + InPlayStrengthUpdater with correction cap --
    print("\n[Computing v3: 8-period + strength updater...]")

    # Compute mu_at_kickoff (t=0, score 0-0, initial a_H/a_A)
    kickoff_result = mmpp_mc_prices(
        t_min=0.0, score_h=0, score_a=0, state_X=0,
        a_H=A_H_BRE, a_A=A_A_WOL, N=10_000, seed=9999,
    )
    mu_H_at_kickoff = kickoff_result["mu_H"]
    mu_A_at_kickoff = kickoff_result["mu_A"]
    print(f"  mu_at_kickoff: mu_H={mu_H_at_kickoff:.3f}, mu_A={mu_A_at_kickoff:.3f}")

    # Initialize updater
    SIGMA_A = 0.5
    updater = InPlayStrengthUpdater(
        a_H_init=A_H_BRE,
        a_A_init=A_A_WOL,
        sigma_a_sq=SIGMA_A ** 2,
        pre_match_home_prob=0.62,
    )

    # Goals with their minutes and teams
    match_goals = [
        (22, "home"),   # 1-0 Brentford
        (37, "home"),   # 2-0 Brentford
        (44, "away"),   # 2-1 Wolves
        (77, "away"),   # 2-2 Wolves
    ]
    goal_minutes_sorted = [g[0] for g in match_goals]

    # Track a_H/a_A trajectory and goal update details
    a_H_current = A_H_BRE
    a_A_current = A_A_WOL
    goal_update_log: list[dict] = []
    a_trajectory: dict[int, tuple[float, float]] = {}  # minute ->(a_H, a_A)

    # Build v3 timeline: recompute minute by minute with dynamic a_H/a_A
    v3_timeline: dict[int, dict] = {}
    goal_idx = 0

    for m in range(1, 100):
        # Check if a goal happened at this minute -- update BEFORE computing this minute
        while goal_idx < len(match_goals) and match_goals[goal_idx][0] == m:
            goal_min, goal_team = match_goals[goal_idx]

            # Compute mu_elapsed at the moment of the goal using CURRENT a_H/a_A
            pre_goal_result = mmpp_mc_prices(
                t_min=float(goal_min),
                score_h=minute_score.get(goal_min - 1, (0, 0))[0],
                score_a=minute_score.get(goal_min - 1, (0, 0))[1],
                state_X=0,
                a_H=a_H_current, a_A=a_A_current,
                N=10_000, seed=goal_min * 31 + 3,
            )
            mu_H_remaining = pre_goal_result["mu_H"]
            mu_A_remaining = pre_goal_result["mu_A"]
            mu_H_elapsed = max(0.0, mu_H_at_kickoff - mu_H_remaining)
            mu_A_elapsed = max(0.0, mu_A_at_kickoff - mu_A_remaining)

            a_H_before = a_H_current
            a_A_before = a_A_current
            new_a_H, new_a_A = updater.update_on_goal(goal_team, mu_H_elapsed, mu_A_elapsed)
            a_H_current = new_a_H
            a_A_current = new_a_A

            classification = updater.classify_goal(goal_team)
            shrink_H = mu_H_elapsed / (mu_H_elapsed + SIGMA_A ** 2)
            shrink_A = mu_A_elapsed / (mu_A_elapsed + SIGMA_A ** 2)

            goal_update_log.append({
                "minute": goal_min,
                "team": goal_team,
                "a_H_before": a_H_before,
                "a_A_before": a_A_before,
                "a_H_after": new_a_H,
                "a_A_after": new_a_A,
                "mu_H_elapsed": mu_H_elapsed,
                "mu_A_elapsed": mu_A_elapsed,
                "shrink_H": shrink_H,
                "shrink_A": shrink_A,
                "classification": classification.label,
                "n_H": updater.n_H,
                "n_A": updater.n_A,
            })

            goal_idx += 1

        # Record trajectory at key minutes
        a_trajectory[m] = (a_H_current, a_A_current)

        # Compute v3 model prices with current (possibly updated) a_H/a_A
        score = minute_score.get(m, (0, 0))
        result = mmpp_mc_prices(
            t_min=float(m),
            score_h=score[0], score_a=score[1],
            state_X=0,
            a_H=a_H_current, a_A=a_A_current,
            N=10_000,
            seed=m * 31 + 7,
        )
        v3_timeline[m] = {
            "minute": m,
            "score": score,
            "p_home": result["home_win"],
            "p_draw": result["draw"],
            "p_away": result["away_win"],
            "mu_H": result["mu_H"],
            "mu_A": result["mu_A"],
            "a_H": a_H_current,
            "a_A": a_A_current,
        }

    print(f"  Computed {len(v3_timeline)} minutes with dynamic a_H/a_A")

    # -- Step 3: Full comparison timeline --------------------------------------
    print("\n" + "=" * 72)
    print("MINUTE-BY-MINUTE COMPARISON: P(Brentford win)")
    print("=" * 72)
    print(f"\n  {'Min':>3} {'Score':>5} {'P_model':>8} {'P_kalshi':>9} {'Gap':>7} {'Notes'}")
    print("  " + "-" * 65)

    gaps: list[dict] = []
    goal_minutes = {22, 37, 44, 77}

    for m in range(4, 96):  # minutes with reliable data (incl. stoppage)
        model = model_timeline.get(m)
        kalshi_mid = kalshi_minute_mid.get(m)

        if model is None or kalshi_mid is None:
            continue

        score = model["score"]
        p_model = model["p_home"]
        gap = p_model - kalshi_mid

        note = ""
        if m in goal_minutes:
            note = " <-- GOAL"
        elif m in {m + 1 for m in goal_minutes}:
            note = " (post-goal)"
        elif m > 90:
            note = " (STOPPAGE)"
        elif 85 <= m <= 90:
            note = " (pre-stoppage)"

        gaps.append({
            "minute": m,
            "score": score,
            "p_model": p_model,
            "p_kalshi": kalshi_mid,
            "gap": gap,
            "abs_gap": abs(gap),
        })

        # Print every minute
        score_str = f"{score[0]}-{score[1]}"
        print(f"  {m:>3} {score_str:>5} {p_model:>8.3f} {kalshi_mid:>9.3f} {gap:>+7.3f}{note}")

    # -- Step 4: Summary statistics --------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY STATISTICS")
    print("=" * 72)

    if not gaps:
        print("  No overlapping data to compare!")
        return

    abs_gaps = [g["abs_gap"] for g in gaps]
    signed_gaps = [g["gap"] for g in gaps]

    print(f"\n  Minutes compared: {len(gaps)}")
    print(f"  Mean absolute gap: {statistics.mean(abs_gaps)*100:.1f}c")
    print(f"  Median absolute gap: {statistics.median(abs_gaps)*100:.1f}c")
    print(f"  Max absolute gap: {max(abs_gaps)*100:.1f}c")
    print(f"  Mean signed gap (model - Kalshi): {statistics.mean(signed_gaps)*100:+.1f}c")
    print(f"  Std dev of gap: {statistics.stdev(signed_gaps)*100:.1f}c")

    # Bias analysis
    model_higher = sum(1 for g in signed_gaps if g > 0)
    model_lower = sum(1 for g in signed_gaps if g < 0)
    print(f"\n  Model higher than Kalshi: {model_higher}/{len(gaps)} minutes ({100*model_higher/len(gaps):.0f}%)")
    print(f"  Model lower than Kalshi: {model_lower}/{len(gaps)} minutes ({100*model_lower/len(gaps):.0f}%)")

    if abs(statistics.mean(signed_gaps)) > 0.03:
        direction = "OVERESTIMATES" if statistics.mean(signed_gaps) > 0 else "UNDERESTIMATES"
        print(f"\n  ->SYSTEMATIC BIAS: Model {direction} Brentford win probability")
        print(f"    by {abs(statistics.mean(signed_gaps))*100:.1f}c on average")
    else:
        print(f"\n  ->No strong systematic bias (mean gap within +/-3c)")

    # -- Step 5: Top 10 biggest discrepancies ----------------------------------
    print("\n" + "=" * 72)
    print("TOP 10 BIGGEST DISCREPANCIES")
    print("=" * 72)

    sorted_gaps = sorted(gaps, key=lambda g: g["abs_gap"], reverse=True)
    print(f"\n  {'Min':>3} {'Score':>5} {'P_model':>8} {'P_kalshi':>9} {'Gap':>7} Context")
    print("  " + "-" * 65)

    for g in sorted_gaps[:10]:
        m = g["minute"]
        s = g["score"]
        context = ""
        if m in goal_minutes:
            context = "GOAL MINUTE"
        elif m - 1 in goal_minutes:
            context = "just after goal"
        elif m - 2 in goal_minutes or m - 3 in goal_minutes:
            context = f"~{m - max(gm for gm in goal_minutes if gm < m)}min after goal"
        elif 85 <= m:
            context = "stoppage zone"
        elif 45 <= m <= 47:
            context = "around halftime"
        else:
            context = "normal play"

        print(f"  {m:>3} {s[0]}-{s[1]:>3} {g['p_model']:>8.3f} {g['p_kalshi']:>9.3f} {g['gap']:>+7.3f} {context}")

    # -- Step 6: Goal reaction analysis ----------------------------------------
    print("\n" + "=" * 72)
    print("GOAL REACTION ANALYSIS")
    print("=" * 72)

    goal_info = [
        (22, "1-0 (Brentford)", "home"),
        (37, "2-0 (Brentford)", "home"),
        (44, "2-1 (Wolves)", "away"),
        (77, "2-2 (Wolves)", "away"),
    ]

    for goal_min, description, team in goal_info:
        pre_min = goal_min - 1
        post_min = goal_min + 1

        pre_model = model_timeline.get(pre_min, {}).get("p_home")
        post_model = model_timeline.get(post_min, {}).get("p_home")
        pre_kalshi = kalshi_minute_mid.get(pre_min)
        post_kalshi = kalshi_minute_mid.get(post_min)

        print(f"\n  Goal at min {goal_min}: {description}")

        if pre_model is not None and post_model is not None:
            model_adj = post_model - pre_model
            print(f"    Model:  {pre_model:.3f} ->{post_model:.3f} (Delta = {model_adj:+.3f} = {model_adj*100:+.1f}c)")
        else:
            model_adj = None
            print(f"    Model:  data missing for pre/post")

        if pre_kalshi is not None and post_kalshi is not None:
            kalshi_adj = post_kalshi - pre_kalshi
            print(f"    Kalshi: {pre_kalshi:.3f} ->{post_kalshi:.3f} (Delta = {kalshi_adj:+.3f} = {kalshi_adj*100:+.1f}c)")
        else:
            kalshi_adj = None
            print(f"    Kalshi: data missing for pre/post")

        if model_adj is not None and kalshi_adj is not None:
            diff = model_adj - kalshi_adj
            print(f"    Adjustment gap: {diff:+.3f} ({diff*100:+.1f}c)")
            expected_dir = "+" if team == "home" else "-"
            model_correct = (model_adj > 0 and team == "home") or (model_adj < 0 and team == "away")
            kalshi_correct = (kalshi_adj > 0 and team == "home") or (kalshi_adj < 0 and team == "away")
            print(f"    Direction correct? Model={model_correct}, Kalshi={kalshi_correct}")

    # -- Step 7: Late-game + stoppage time analysis -----------------------------
    print("\n" + "=" * 72)
    print("LATE-GAME + STOPPAGE TIME ANALYSIS (min 78-95)")
    print("=" * 72)

    print("\n  Score at min 78+: 2-2 (draw)")
    print("  MMPP 8-period basis: b[5]=-0.08 [75-85], b[6]=-0.05 [85-90], b[7]=+0.20 [90+]")
    print("  Kalshi: how does market price the draw/trailing team in late minutes?")

    print(f"\n  {'Min':>3} {'P_model':>8} {'P_kalshi':>9} {'Gap':>7} {'mu_H':>6} {'mu_A':>6}")
    print("  " + "-" * 55)

    late_gaps = []  # min 78-90
    stoppage_gaps = []  # min 90+
    for m in range(78, 96):
        model = model_timeline.get(m)
        kalshi = kalshi_minute_mid.get(m)
        if model and kalshi:
            gap = model["p_home"] - kalshi
            if m <= 90:
                late_gaps.append({"minute": m, "gap": gap, "abs_gap": abs(gap)})
            else:
                stoppage_gaps.append({"minute": m, "gap": gap, "abs_gap": abs(gap)})
            zone = "STOP" if m > 90 else ""
            print(f"  {m:>3} {model['p_home']:>8.3f} {kalshi:>9.3f} {gap:>+7.3f} {model['mu_H']:>6.3f} {model['mu_A']:>6.3f}  {zone}")
        elif model:
            print(f"  {m:>3} {model['p_home']:>8.3f} {'---':>9} {'---':>7} {model['mu_H']:>6.3f} {model['mu_A']:>6.3f}  (no Kalshi)")

    if late_gaps:
        print(f"\n  Mean |gap| min 78-90: {statistics.mean([g['abs_gap'] for g in late_gaps])*100:.1f}c")
    if stoppage_gaps:
        print(f"  Mean |gap| min 90+:   {statistics.mean([g['abs_gap'] for g in stoppage_gaps])*100:.1f}c")

    # -- Step 8: Sanity check -- specific minutes with raw data ------------------
    print("\n" + "=" * 72)
    print("SANITY CHECK: KEY MINUTES WITH RAW DATA")
    print("=" * 72)

    check_minutes = [10, 40, 70, 85, 88, 90, 92]
    for m in check_minutes:
        model = model_timeline.get(m)
        kalshi = kalshi_minute_mid.get(m)
        n_kalshi = len(kalshi_by_minute.get(m, []))

        print(f"\n  --- Minute {m} ---")
        if model:
            print(f"  Score: {model['score'][0]}-{model['score'][1]}")
            print(f"  Model inputs: t={m}, state_X=0, "
                  f"a_H={A_H_BRE:.3f}, a_A={A_A_WOL:.3f}")
            print(f"  Model output: P(home)={model['p_home']:.4f}, "
                  f"P(draw)={model['p_draw']:.4f}, P(away)={model['p_away']:.4f}")
            print(f"  Remaining mu: mu_H={model['mu_H']:.3f}, mu_A={model['mu_A']:.3f}")
            print(f"  Sanity: mu_H+mu_A = {model['mu_H']+model['mu_A']:.3f} "
                  f"(expected ~{(93-m)/93*2.77:.2f} remaining goals)")
        if kalshi is not None:
            print(f"  Kalshi mid: {kalshi:.4f} (median of {n_kalshi} OB updates in this minute)")
        if model and kalshi is not None:
            print(f"  Gap: {model['p_home'] - kalshi:+.4f}")

        # Does model goal reaction make directional sense?
        if m == 40:
            print(f"\n  Directional check: at 2-0, min 40, model says P(home)={model['p_home']:.3f}")
            print(f"  This should be very high (~0.90+) since Brentford leads by 2 with 53min left")
            if model["p_home"] > 0.80:
                print(f"  OK: Looks reasonable")
            else:
                print(f"  WARN: Seems too low -- model may be underreacting to 2-goal lead")

    # -- Step 9: Phase analysis -- where are gaps biggest? ----------------------
    print("\n" + "=" * 72)
    print("PHASE ANALYSIS: WHERE DO GAPS CLUSTER?")
    print("=" * 72)

    phases = {
        "Pre-goal 1 (min 4-21)": [g for g in gaps if 4 <= g["minute"] <= 21],
        "After goal 1 (min 23-36)": [g for g in gaps if 23 <= g["minute"] <= 36],
        "After goal 2 (min 38-43)": [g for g in gaps if 38 <= g["minute"] <= 43],
        "After goal 3 (min 45-76)": [g for g in gaps if 45 <= g["minute"] <= 76],
        "After goal 4 (min 78-90)": [g for g in gaps if 78 <= g["minute"] <= 90],
        "Stoppage time (min 90+)": [g for g in gaps if g["minute"] > 90],
    }

    print(f"\n  {'Phase':<30} {'N':>3} {'Mean|gap|':>9} {'Mean gap':>9} {'Max|gap|':>9}")
    print("  " + "-" * 65)

    for phase_name, phase_gaps in phases.items():
        if not phase_gaps:
            print(f"  {phase_name:<30} {'0':>3}")
            continue
        ag = [g["abs_gap"] for g in phase_gaps]
        sg = [g["gap"] for g in phase_gaps]
        print(f"  {phase_name:<30} {len(phase_gaps):>3} "
              f"{statistics.mean(ag)*100:>8.1f}c {statistics.mean(sg)*100:>+8.1f}c "
              f"{max(ag)*100:>8.1f}c")

    # -- Step 10: Three-version comparison -------------------------------------
    print("\n" + "=" * 72)
    print("THREE-VERSION COMPARISON: v1 (6-period) vs v2 (8-period) vs v3 (8-period + strength updater)")
    print("=" * 72)

    # Previous results from 6-period model (hardcoded from prior run)
    v1_mean_gap = 3.4
    v1_late_gap = 6.9

    # v2 stats (already computed above as 'gaps')
    v2_mean_gap = statistics.mean(abs_gaps) * 100
    v2_late_entries = [g for g in gaps if 78 <= g["minute"] <= 90]
    v2_late_gap = statistics.mean([g["abs_gap"] for g in v2_late_entries]) * 100 if v2_late_entries else 0.0
    v2_stoppage_entries = [g for g in gaps if g["minute"] > 90]
    v2_stoppage_gap = statistics.mean([g["abs_gap"] for g in v2_stoppage_entries]) * 100 if v2_stoppage_entries else 0.0

    # v3 stats
    v3_gaps: list[dict] = []
    for m in range(4, 96):
        v3m = v3_timeline.get(m)
        kalshi_mid = kalshi_minute_mid.get(m)
        if v3m is None or kalshi_mid is None:
            continue
        gap3 = v3m["p_home"] - kalshi_mid
        v3_gaps.append({
            "minute": m,
            "score": v3m["score"],
            "p_model": v3m["p_home"],
            "p_kalshi": kalshi_mid,
            "gap": gap3,
            "abs_gap": abs(gap3),
        })

    v3_mean_gap = statistics.mean([g["abs_gap"] for g in v3_gaps]) * 100 if v3_gaps else 0.0
    v3_late_entries = [g for g in v3_gaps if 78 <= g["minute"] <= 90]
    v3_late_gap = statistics.mean([g["abs_gap"] for g in v3_late_entries]) * 100 if v3_late_entries else 0.0
    v3_stoppage_entries = [g for g in v3_gaps if g["minute"] > 90]
    v3_stoppage_gap = statistics.mean([g["abs_gap"] for g in v3_stoppage_entries]) * 100 if v3_stoppage_entries else 0.0

    # Per-goal adjustment comparison (v2 vs v3)
    v2_goal_gaps: list[float] = []
    v3_goal_gaps: list[float] = []
    for gm in [22, 37, 44, 77]:
        for g in gaps:
            if g["minute"] == gm + 1:
                v2_goal_gaps.append(g["gap"] * 100)
        for g in v3_gaps:
            if g["minute"] == gm + 1:
                v3_goal_gaps.append(g["gap"] * 100)

    # Goal 4 specifically (min 78, post 2-2 equalizer)
    v2_goal4_gap = next((g["gap"] * 100 for g in gaps if g["minute"] == 78), None)
    v3_goal4_gap = next((g["gap"] * 100 for g in v3_gaps if g["minute"] == 78), None)

    print(f"""
  {'Metric':<35} {'v1 (6-period)':>14} {'v2 (8-period)':>14} {'v3 (8p+updater)':>16}
  {'-'*83}
  {'Mean |gap| (all minutes)':<35} {v1_mean_gap:>13.1f}c {v2_mean_gap:>13.1f}c {v3_mean_gap:>15.1f}c
  {'Late-game |gap| (min 78-90)':<35} {v1_late_gap:>13.1f}c {v2_late_gap:>13.1f}c {v3_late_gap:>15.1f}c
  {'Stoppage |gap| (min 90+)':<35} {'N/A':>14} {v2_stoppage_gap:>13.1f}c {v3_stoppage_gap:>15.1f}c""")

    if v2_goal_gaps and v3_goal_gaps:
        print(f"  {'Mean post-goal |gap| (4 goals)':<35} {'N/A':>14} {statistics.mean([abs(g) for g in v2_goal_gaps]):>13.1f}c {statistics.mean([abs(g) for g in v3_goal_gaps]):>15.1f}c")
    if v2_goal4_gap is not None and v3_goal4_gap is not None:
        print(f"  {'Goal 4 gap (min 78, 2-2 eq.)':<35} {'N/A':>14} {v2_goal4_gap:>+13.1f}c {v3_goal4_gap:>+15.1f}c")

    # -- Step 10b: a_H trajectory ----------------------------------------------
    print("\n" + "=" * 72)
    print("a_H / a_A TRAJECTORY THROUGH MATCH")
    print("=" * 72)

    traj_minutes = [1, 25, 40, 47, 80]
    print(f"\n  {'Min':>3} {'Score':>5} {'a_H':>8} {'a_A':>8} {'Deltaa_H':>8} {'Deltaa_A':>8} {'Notes'}")
    print("  " + "-" * 55)
    for m in traj_minutes:
        a_h, a_a = a_trajectory.get(m, (A_H_BRE, A_A_WOL))
        score = minute_score.get(m, (0, 0))
        da_h = a_h - A_H_BRE
        da_a = a_a - A_A_WOL
        note = ""
        if m < 22:
            note = "pre-goal"
        elif m < 37:
            note = "after 1-0"
        elif m < 44:
            note = "after 2-0"
        elif m < 77:
            note = "after 2-1"
        else:
            note = "after 2-2"
        print(f"  {m:>3} {score[0]}-{score[1]:>3} {a_h:>8.4f} {a_a:>8.4f} {da_h:>+8.4f} {da_a:>+8.4f}  {note}")

    # -- Step 10c: Raw a_H/a_A before/after each goal (SANITY CHECK) ----------
    print("\n" + "=" * 72)
    print("SANITY CHECK: RAW a_H, a_A BEFORE AND AFTER EACH GOAL")
    print("=" * 72)

    for entry in goal_update_log:
        m = entry["minute"]
        team = entry["team"]
        cls = entry["classification"]
        print(f"\n  Goal at min {m}: {team} scores ({cls})")
        print(f"    n_H={entry['n_H']}, n_A={entry['n_A']}")
        print(f"    mu_H_elapsed={entry['mu_H_elapsed']:.4f}, mu_A_elapsed={entry['mu_A_elapsed']:.4f}")
        print(f"    shrink_H={entry['shrink_H']:.4f}, shrink_A={entry['shrink_A']:.4f}")
        print(f"    a_H: {entry['a_H_before']:.4f} ->{entry['a_H_after']:.4f} (Delta={entry['a_H_after'] - entry['a_H_before']:+.4f})")
        print(f"    a_A: {entry['a_A_before']:.4f} ->{entry['a_A_after']:.4f} (Delta={entry['a_A_after'] - entry['a_A_before']:+.4f})")

    # -- Step 10d: v3 minute-by-minute for goal windows ------------------------
    print("\n" + "=" * 72)
    print("v2 vs v3 AROUND EACH GOAL (+/-2 minutes)")
    print("=" * 72)
    print(f"\n  {'Min':>3} {'Score':>5} {'v2_P':>7} {'v3_P':>7} {'Kalshi':>7} {'v2_gap':>7} {'v3_gap':>7} {'Improv':>7}")
    print("  " + "-" * 60)

    for goal_min, _ in match_goals:
        for m in range(goal_min - 1, goal_min + 3):
            v2m = model_timeline.get(m)
            v3m = v3_timeline.get(m)
            km = kalshi_minute_mid.get(m)
            if v2m is None or v3m is None or km is None:
                continue
            v2g = v2m["p_home"] - km
            v3g = v3m["p_home"] - km
            improv = abs(v2g) - abs(v3g)
            score = v2m["score"]
            marker = " <-- GOAL" if m == goal_min else ""
            print(f"  {m:>3} {score[0]}-{score[1]:>3} {v2m['p_home']:>7.3f} {v3m['p_home']:>7.3f} {km:>7.3f} {v2g:>+7.3f} {v3g:>+7.3f} {improv:>+7.3f}{marker}")
        print()

    # -- Step 11: Verdict -----------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    mean_abs = statistics.mean(abs_gaps) * 100
    mean_signed = statistics.mean(signed_gaps) * 100
    max_gap_entry = max(gaps, key=lambda g: g["abs_gap"])

    print(f"""
  OVERALL ACCURACY:
    Mean |gap|: {mean_abs:.1f}c (model vs Kalshi mid-price)
    Mean bias: {mean_signed:+.1f}c ({'model too high' if mean_signed > 0 else 'model too low'})
    Max gap: {max_gap_entry['abs_gap']*100:.1f}c at min {max_gap_entry['minute']} ({max_gap_entry['score'][0]}-{max_gap_entry['score'][1]})

  INTERPRETATION:
    - Mean |gap| < 5c: Model is competitive with market
    - Mean |gap| 5-10c: Model has significant calibration work needed
    - Mean |gap| > 10c: Model is fundamentally mispriced

  BIGGEST GAP SITUATIONS:""")

    # Categorize gaps
    post_goal_gaps = [g for g in gaps if g["minute"] in {m + 1 for m in goal_minutes} or g["minute"] in goal_minutes]
    normal_gaps = [g for g in gaps if g["minute"] not in goal_minutes and g["minute"] not in {m + 1 for m in goal_minutes}]

    if post_goal_gaps:
        print(f"    Post-goal minutes: mean |gap| = {statistics.mean([g['abs_gap'] for g in post_goal_gaps])*100:.1f}c")
    if normal_gaps:
        print(f"    Normal play minutes: mean |gap| = {statistics.mean([g['abs_gap'] for g in normal_gaps])*100:.1f}c")

    print(f"""
  WHERE MODEL NEEDS IMPROVEMENT:
    1. Match-specific a_H/a_A: Using approximate backsolve from pre-match price.
       Phase 2 full optimization (XGBoost + Betfair odds) would improve baseline.
    2. Score-differential effects: delta_H/delta_A are EPL averages. Training on
       actual data could sharpen the model's response to score changes.
    3. Late-game dynamics: T_exp=93 is an average; this match had {(minute_to_ts.get(999,0) - minute_to_ts.get(90,0))/60:.1f}min
       of actual stoppage time.

  WHAT COULD MAKE THESE RESULTS MISLEADING:
    - MMPP uses Poisson approximation (no within-match red card transitions)
    - Kalshi mid-price != true fair value (spread + thin liquidity)
    - Minute-level bucketing hides sub-minute dynamics (goal reactions)
    - Single match may not be representative of typical model performance
    - Default EPL params (not trained) add systematic uncertainty
""")


if __name__ == "__main__":
    main()
