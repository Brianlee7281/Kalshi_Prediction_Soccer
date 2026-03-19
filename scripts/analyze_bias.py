#!/usr/bin/env python3
"""
Behavioral Bias Test: Surprise Goal Overreaction + Stoppage Time Anchoring
==========================================================================
Uses the Brentford 2-2 Wolves recording (match 4190023) to test two
specific behavioral biases in Kalshi pricing.

BIAS 1: Surprise Goal Overreaction
  - Wolves (underdog) scored at min 44 and min 77
  - Does Kalshi overshoot P(Wolves win) after these goals?
  - Compare with Brentford (favorite) goals at min 22 and min 37

BIAS 2: Stoppage Time Anchoring
  - At 2-2 from min 78+, does Kalshi price P(Draw) as if "90 = game over"?
  - Compare Kalshi's P(Draw) trajectory vs MMPP model (which knows about stoppage)

Match: Brentford 2-2 Wolves, 2026-03-16
Goals: min 22 (1-0 H), min 37 (2-0 H), min 44 (2-1 A), min 77 (2-2 A)

Usage:
  PYTHONPATH=. python scripts/analyze_bias.py data/latency/4190023
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ─── MMPP Params (reused from analyze_inplay_accuracy.py) ─────────────────────

DEFAULT_PARAMS = {
    "b": [-0.1688, -0.1282, -0.1008, 0.0581, -0.0025, -0.0797, -0.0499, 0.1982],
    "gamma_H": [0.0, -0.30, 0.10, -0.20],
    "gamma_A": [0.0, 0.10, -0.30, -0.20],
    "delta_H": [0.15, 0.08, 0.0, -0.05, -0.12],
    "delta_A": [-0.12, -0.05, 0.0, 0.08, 0.15],
    "a_H_avg": -4.09,
    "a_A_avg": -4.33,
}

_LOGIT_DEFAULT_HOME = math.log(0.456 / 0.544)
_LOGIT_BRE = math.log(0.62 / 0.38)
_LOGIT_DEFAULT_AWAY = math.log(0.302 / 0.698)
_LOGIT_WOL = math.log(0.15 / 0.85)
_SCALE = 0.25

A_H_BRE = DEFAULT_PARAMS["a_H_avg"] + _SCALE * (_LOGIT_BRE - _LOGIT_DEFAULT_HOME)
A_A_WOL = DEFAULT_PARAMS["a_A_avg"] + _SCALE * (_LOGIT_WOL - _LOGIT_DEFAULT_AWAY)


def mmpp_mc_prices(
    t_min: float,
    score_h: int,
    score_a: int,
    T_exp: float = 93.0,
    N: int = 50_000,
    seed: int | None = None,
) -> dict[str, float]:
    """MMPP Monte Carlo: P(home_win), P(draw), P(away_win)."""
    p = DEFAULT_PARAMS
    b = p["b"]
    gamma_H = p["gamma_H"]
    gamma_A = p["gamma_A"]
    delta_H = p["delta_H"]
    delta_A = p["delta_A"]

    if t_min >= T_exp:
        if score_h > score_a:
            return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}
        elif score_h < score_a:
            return {"home_win": 0.0, "draw": 0.0, "away_win": 1.0}
        return {"home_win": 0.0, "draw": 1.0, "away_win": 0.0}

    ds = max(-2, min(2, score_h - score_a))
    di_H = ds + 2
    di_A = -ds + 2  # away perspective to match MLE calibration
    state_X = 0  # no red cards
    basis_bounds = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0, T_exp]
    mu_H = 0.0
    mu_A = 0.0
    for k in range(len(basis_bounds) - 1):
        seg_start = max(t_min, basis_bounds[k])
        seg_end = min(T_exp, basis_bounds[k + 1])
        if seg_start >= seg_end:
            continue
        dt = seg_end - seg_start
        mu_H += dt * math.exp(A_H_BRE + b[k] + gamma_H[state_X] + delta_H[di_H])
        mu_A += dt * math.exp(A_A_WOL + b[k] + gamma_A[state_X] + delta_A[di_A])

    mu_H = max(0.001, mu_H)
    mu_A = max(0.001, mu_A)

    rng = np.random.default_rng(seed if seed is not None else int(t_min * 1000) % (2**31))
    rem_H = rng.poisson(mu_H, size=N)
    rem_A = rng.poisson(mu_A, size=N)
    final_H = score_h + rem_H
    final_A = score_a + rem_A

    return {
        "home_win": float(np.mean(final_H > final_A)),
        "draw": float(np.mean(final_H == final_A)),
        "away_win": float(np.mean(final_A > final_H)),
    }


# ─── Orderbook Mid-Price Builder ─────────────────────────────────────────────

def build_mid_timeline(
    match_dir: Path, ticker_suffix: str
) -> list[tuple[float, float]]:
    """Returns [(ts_wall, mid), ...] for a given ticker suffix."""
    yes_book: dict[str, float] = {}
    no_book: dict[str, float] = {}
    tl: list[tuple[float, float]] = []

    with open(match_dir / "kalshi.jsonl") as f:
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
                if best_ask > best_bid:
                    mid = (best_bid + best_ask) / 2.0
                    tl.append((ts, mid))

    return tl


# ─── Event Timeline ──────────────────────────────────────────────────────────

def load_events(match_dir: Path) -> tuple[list[dict], dict[int, float], float]:
    """Load events. Returns (goals, minute_to_ts, kickoff_ts)."""
    goals: list[dict] = []
    minute_to_ts: dict[int, float] = {}

    with open(match_dir / "events.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["type"] == "goal":
                goals.append(d)
            elif d["type"] == "status_change":
                new = d.get("new_status", "")
                if new.isdigit():
                    minute_to_ts[int(new)] = d["ts_wall"]

    first_min = min(minute_to_ts.keys())
    kickoff_ts = minute_to_ts[first_min] - first_min * 60.0
    return goals, minute_to_ts, kickoff_ts


def ts_to_match_minute(ts: float, kickoff_ts: float, minute_to_ts: dict[int, float]) -> float:
    """Convert wall timestamp to fractional match minute using status events."""
    sorted_mins = sorted(minute_to_ts.items())
    for i in range(len(sorted_mins) - 1):
        m1, t1 = sorted_mins[i]
        m2, t2 = sorted_mins[i + 1]
        if t1 <= ts < t2:
            frac = (ts - t1) / (t2 - t1)
            return m1 + frac * (m2 - m1)
    if ts < sorted_mins[0][1]:
        return sorted_mins[0][0] - (sorted_mins[0][1] - ts) / 60.0
    return sorted_mins[-1][0] + (ts - sorted_mins[-1][1]) / 60.0


def get_prices_in_window(
    timeline: list[tuple[float, float]], t_start: float, t_end: float
) -> list[tuple[float, float]]:
    """Get all (ts, price) in [t_start, t_end]."""
    return [(ts, p) for ts, p in timeline if t_start <= ts <= t_end]


def window_median(
    timeline: list[tuple[float, float]], t_start: float, t_end: float
) -> float | None:
    """Median price in time window, or None if no data."""
    prices = [p for ts, p in timeline if t_start <= ts <= t_end]
    if not prices:
        return None
    return float(np.median(prices))


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        match_dir = Path(sys.argv[1])
    else:
        match_dir = Path(__file__).parent.parent / "data" / "latency" / "4190023"

    print("=" * 78)
    print("BEHAVIORAL BIAS TEST: Brentford 2-2 Wolves (2026-03-16)")
    print("=" * 78)

    # ── Load data ─────────────────────────────────────────────────────────────
    goals, minute_to_ts, kickoff_ts = load_events(match_dir)

    # Build timelines for all 3 markets
    tl_bre = build_mid_timeline(match_dir, "-BRE")
    tl_wol = build_mid_timeline(match_dir, "-WOL")
    tl_tie = build_mid_timeline(match_dir, "-TIE")

    print(f"\nData loaded:")
    print(f"  Kalshi OB updates: BRE={len(tl_bre):,}, WOL={len(tl_wol):,}, TIE={len(tl_tie):,}")
    print(f"  Goals: {len(goals)}")
    for g in goals:
        print(f"    {g['utc'][:19]} | {g['prev_score']} -> {g['new_score']} ({g['team']})")
    print(f"  Kickoff (inferred): {datetime.fromtimestamp(kickoff_ts, tz=timezone.utc).strftime('%H:%M:%S')} UTC")

    # Goal timestamps
    goal_ts = {
        22: next(g["ts_wall"] for g in goals if g["new_score"] == [1, 0]),
        37: next(g["ts_wall"] for g in goals if g["new_score"] == [2, 0]),
        44: next(g["ts_wall"] for g in goals if g["new_score"] == [2, 1]),
        77: next(g["ts_wall"] for g in goals if g["new_score"] == [2, 2]),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BIAS 1: SURPRISE GOAL OVERREACTION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("BIAS 1: SURPRISE GOAL OVERREACTION")
    print("=" * 78)

    # Step 1: Determine favorite from early Kalshi prices
    early_end = kickoff_ts + 5 * 60
    early_bre = window_median(tl_bre, kickoff_ts, early_end)
    early_wol = window_median(tl_wol, kickoff_ts, early_end)
    early_tie = window_median(tl_tie, kickoff_ts, early_end)

    print(f"\n  STEP 1: Who is the favorite?")
    print(f"  Kalshi mid-prices in first 5 minutes of play:")
    print(f"    P(Brentford win) = {early_bre:.3f}" if early_bre else "    P(Brentford win) = N/A")
    print(f"    P(Wolves win)    = {early_wol:.3f}" if early_wol else "    P(Wolves win)    = N/A")
    print(f"    P(Draw)          = {early_tie:.3f}" if early_tie else "    P(Draw)          = N/A")

    if early_bre and early_bre > 0.50:
        print(f"  -> Brentford is FAVORITE (P={early_bre:.3f} > 0.50)")
        print(f"  -> Wolves is UNDERDOG (P={early_wol:.3f})")
        print(f"  -> Wolves goals at min 44 and 77 are SURPRISE GOALS")
    elif early_wol and early_wol > 0.50:
        print(f"  -> Wolves is FAVORITE — surprise goal hypothesis reversed")
    else:
        print(f"  -> No clear favorite")

    # Step 2-3: Analyze each goal's Kalshi reaction
    goal_analyses = [
        {"minute": 22, "desc": "1-0 Brentford (favorite scores)", "team": "home",
         "score_before": (0, 0), "score_after": (1, 0)},
        {"minute": 37, "desc": "2-0 Brentford (favorite scores)", "team": "home",
         "score_before": (1, 0), "score_after": (2, 0)},
        {"minute": 44, "desc": "2-1 Wolves (UNDERDOG surprise goal)", "team": "away",
         "score_before": (2, 0), "score_after": (2, 1)},
        {"minute": 77, "desc": "2-2 Wolves (UNDERDOG equalizer)", "team": "away",
         "score_before": (2, 1), "score_after": (2, 2)},
    ]

    overshoot_results: list[dict] = []

    for ga in goal_analyses:
        gmin = ga["minute"]
        gts = goal_ts[gmin]

        print(f"\n  {'─' * 72}")
        print(f"  GOAL at minute {gmin}: {ga['desc']}")
        print(f"  {'─' * 72}")

        # Track the scoring team's win probability
        if ga["team"] == "home":
            tl = tl_bre
            label = "P(Brentford win)"
        else:
            tl = tl_wol
            label = "P(Wolves win)"

        # Windows: 2min before, then 0-1, 1-2, 2-3, 3-5, 5-10 after
        windows = [
            ("BEFORE (2min)", gts - 120, gts),
            ("AFTER 0-1min", gts, gts + 60),
            ("AFTER 1-2min", gts + 60, gts + 120),
            ("AFTER 2-3min", gts + 120, gts + 180),
            ("AFTER 3-5min", gts + 180, gts + 300),
            ("AFTER 5-10min", gts + 300, gts + 600),
        ]

        print(f"\n    {label} by time window:")
        print(f"    {'Window':<18} {'Median':>8} {'N_obs':>6}")
        print(f"    {'─' * 34}")

        window_prices: dict[str, float | None] = {}
        for wname, t0, t1 in windows:
            pts = get_prices_in_window(tl, t0, t1)
            med = float(np.median([p for _, p in pts])) if pts else None
            window_prices[wname] = med
            n = len(pts)
            if med is not None:
                print(f"    {wname:<18} {med:>8.4f} {n:>6}")
            else:
                print(f"    {wname:<18} {'---':>8} {n:>6}")

        # Compute overshoot
        before_p = window_prices.get("BEFORE (2min)")
        peak_windows = ["AFTER 0-1min", "AFTER 1-2min", "AFTER 2-3min", "AFTER 3-5min"]
        peak_prices = [window_prices[w] for w in peak_windows if window_prices.get(w) is not None]
        settled_p = window_prices.get("AFTER 5-10min")

        if peak_prices and settled_p is not None and before_p is not None:
            peak_p = max(peak_prices)
            overshoot = peak_p - settled_p
            jump = peak_p - before_p

            # Check if another goal contaminates the settled window
            settled_start = gts + 300
            settled_end = gts + 600
            contaminating_goals = [
                m for m, t in goal_ts.items()
                if m != gmin and settled_start <= t <= settled_end
            ]

            print(f"\n    Analysis:")
            print(f"      Pre-goal:          {before_p:.4f}")
            print(f"      Peak (0-5min):     {peak_p:.4f}  (jump = {jump:+.4f} = {jump*100:+.1f}¢)")
            print(f"      Settled (5-10min): {settled_p:.4f}")
            print(f"      Overshoot = peak - settled = {overshoot:+.4f} = {overshoot*100:+.1f}¢")

            clean_overshoot = overshoot
            if contaminating_goals:
                clean_settled = window_prices.get("AFTER 3-5min")
                print(f"\n      WARNING: Goal at min {contaminating_goals[0]} falls inside the")
                print(f"        5-10min settled window — overshoot may be CONTAMINATED.")
                if clean_settled is not None:
                    clean_overshoot = peak_p - clean_settled
                    print(f"        Clean settled (3-5min, before next goal): {clean_settled:.4f}")
                    print(f"        Clean overshoot = {clean_overshoot:+.4f} = {clean_overshoot*100:+.1f}¢")

            if clean_overshoot > 0.005:
                print(f"      -> OVERREACTION detected: Kalshi overshot by {clean_overshoot*100:.1f}¢")
                profitable = True
            elif clean_overshoot < -0.005:
                print(f"      -> UNDERREACTION: Price continued rising after initial move")
                profitable = False
            else:
                print(f"      -> Efficient pricing: no significant overshoot")
                profitable = False

            role = "UNDERDOG" if ga["team"] == "away" else "FAVORITE"
            overshoot_results.append({
                "minute": gmin,
                "role": role,
                "jump_cents": jump * 100,
                "overshoot_cents": clean_overshoot * 100,
                "profitable": profitable,
            })

            if profitable:
                print(f"      Profitable trade: SELL {label} at peak ({peak_p:.3f}),")
                print(f"        it reverts to {settled_p:.3f} -> profit {clean_overshoot*100:.1f}¢/contract")
        else:
            print(f"\n    Analysis: Insufficient data for overshoot calculation")

        # Sanity check: 3 raw OB updates around the event
        print(f"\n    SANITY CHECK — 3 raw OB updates around goal (ts_wall={gts:.1f}):")
        nearby = sorted(
            [(abs(ts - gts), ts, p) for ts, p in tl if abs(ts - gts) < 30],
            key=lambda x: x[0],
        )
        if not nearby:
            nearby = sorted(
                [(abs(ts - gts), ts, p) for ts, p in tl if abs(ts - gts) < 120],
                key=lambda x: x[0],
            )
        for i, (_, ts, p) in enumerate(nearby[:3]):
            rel = ts - gts
            match_min = ts_to_match_minute(ts, kickoff_ts, minute_to_ts)
            utc = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
            print(f"      [{i+1}] {utc} (t={rel:+.1f}s from goal, min={match_min:.1f}): {label}={p:.4f}")

    # Actual outcome comparison
    print(f"\n  {'─' * 72}")
    print(f"  ACTUAL OUTCOME vs POST-GOAL PRICING")
    print(f"  {'─' * 72}")
    print(f"""
    Goal at min 44 (2-1 Wolves ahead):
      Wolves eventually DREW 2-2, did NOT win.
      -> Anyone buying Wolves WIN at the inflated post-goal price LOST money.

    Goal at min 77 (2-2 equalizer):
      Match ended 2-2. Draw was the correct outcome.
      -> P(Wolves win) should have moved DOWN (equalizer, not go-ahead goal)
      -> But initial surge in P(WOL) is natural before market digests the score

    Brentford goals at min 22 and 37:
      Brentford eventually drew 2-2 (did NOT win).
      -> Anyone buying BRE WIN at post-goal peaks also LOST money.""")

    # Favorite vs underdog comparison
    print(f"\n  {'─' * 72}")
    print(f"  FAVORITE vs UNDERDOG OVERREACTION SUMMARY")
    print(f"  {'─' * 72}")

    if overshoot_results:
        print(f"\n    {'Min':>4} {'Role':>9} {'Jump':>8} {'Overshoot':>10} {'Profitable?':>12}")
        print(f"    {'─' * 48}")
        for r in overshoot_results:
            prof = "YES" if r["profitable"] else "no"
            print(f"    {r['minute']:>4} {r['role']:>9} {r['jump_cents']:>+7.1f}¢ "
                  f"{r['overshoot_cents']:>+9.1f}¢ {prof:>12}")

        fav_overshoots = [r["overshoot_cents"] for r in overshoot_results if r["role"] == "FAVORITE"]
        und_overshoots = [r["overshoot_cents"] for r in overshoot_results if r["role"] == "UNDERDOG"]

        if fav_overshoots and und_overshoots:
            avg_fav = sum(fav_overshoots) / len(fav_overshoots)
            avg_und = sum(und_overshoots) / len(und_overshoots)
            print(f"\n    Avg overshoot — Favorite goals: {avg_fav:+.1f}¢")
            print(f"    Avg overshoot — Underdog goals: {avg_und:+.1f}¢")
            if avg_und > avg_fav + 1.0:
                print(f"    -> BIAS CONFIRMED: Underdog goals cause {avg_und - avg_fav:.1f}¢ MORE overshoot")
                print(f"      Direction: Kalshi OVERprices underdog win probability after surprise goals")
            elif avg_fav > avg_und + 1.0:
                print(f"    -> REVERSE BIAS: Favorite goals cause MORE overshoot ({avg_fav - avg_und:.1f}¢)")
            else:
                print(f"    -> NO CLEAR BIAS: Overshoot similar for both ({abs(avg_und - avg_fav):.1f}¢ difference)")

    # ══════════════════════════════════════════════════════════════════════════
    # BIAS 2: STOPPAGE TIME ANCHORING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 78)
    print("BIAS 2: STOPPAGE TIME ANCHORING")
    print("=" * 78)

    print(f"\n  Context: Score is 2-2 from min 78. Does Kalshi price P(Draw)")
    print(f"  as if '90 = game over' (anchoring bias) or account for stoppage?")

    # Step 1: Extract Kalshi probabilities at each minute 85-93
    target_minutes = [85, 87, 89, 90, 91, 92, 93]

    print(f"\n  STEP 1: Kalshi prices at key minutes (score: 2-2)")
    print(f"  {'Min':>4} {'P(Draw)':>9} {'P(Home)':>9} {'P(Away)':>9} {'N_obs':>6}")
    print(f"  {'─' * 45}")

    kalshi_draw_by_min: dict[int, float] = {}
    kalshi_home_by_min: dict[int, float] = {}
    kalshi_away_by_min: dict[int, float] = {}

    for m in target_minutes:
        if m in minute_to_ts:
            t_start = minute_to_ts[m]
            next_mins = [mm for mm in sorted(minute_to_ts.keys()) if mm > m]
            t_end = minute_to_ts[next_mins[0]] if next_mins else t_start + 60
        else:
            # For minutes after 90 (no status events), estimate from minute 90
            m90_ts = minute_to_ts.get(90)
            if m90_ts is None:
                continue
            t_start = m90_ts + (m - 90) * 60
            t_end = t_start + 60

        p_draw = window_median(tl_tie, t_start, t_end)
        p_home = window_median(tl_bre, t_start, t_end)
        p_away = window_median(tl_wol, t_start, t_end)

        n_obs = len(get_prices_in_window(tl_tie, t_start, t_end))

        if p_draw is not None:
            kalshi_draw_by_min[m] = p_draw
        if p_home is not None:
            kalshi_home_by_min[m] = p_home
        if p_away is not None:
            kalshi_away_by_min[m] = p_away

        d_str = f"{p_draw:.4f}" if p_draw is not None else "---"
        h_str = f"{p_home:.4f}" if p_home is not None else "---"
        a_str = f"{p_away:.4f}" if p_away is not None else "---"
        zone = "  STOPPAGE" if m > 90 else ""
        print(f"  {m:>4} {d_str:>9} {h_str:>9} {a_str:>9} {n_obs:>6}{zone}")

    # Step 2: MMPP model probabilities
    print(f"\n  STEP 2: MMPP model prices at same minutes (score: 2-2, T_exp=93)")
    print(f"  {'Min':>4} {'P(Draw)':>9} {'P(Home)':>9} {'P(Away)':>9}")
    print(f"  {'─' * 38}")

    mmpp_draw_by_min: dict[int, float] = {}
    mmpp_home_by_min: dict[int, float] = {}
    mmpp_away_by_min: dict[int, float] = {}

    for m in target_minutes:
        result = mmpp_mc_prices(float(m), 2, 2, T_exp=93.0, N=50_000, seed=m * 37 + 11)
        mmpp_draw_by_min[m] = result["draw"]
        mmpp_home_by_min[m] = result["home_win"]
        mmpp_away_by_min[m] = result["away_win"]
        zone = "  STOPPAGE" if m > 90 else ""
        print(f"  {m:>4} {result['draw']:>9.4f} {result['home_win']:>9.4f} {result['away_win']:>9.4f}{zone}")

    # Step 3: Kalshi vs MMPP comparison
    print(f"\n  STEP 3: Kalshi vs MMPP P(Draw) gap")
    print(f"  {'Min':>4} {'Kalshi':>8} {'MMPP':>8} {'Gap':>8} {'Assessment':>16}")
    print(f"  {'─' * 50}")

    for m in target_minutes:
        kd = kalshi_draw_by_min.get(m)
        md = mmpp_draw_by_min.get(m)
        if kd is not None and md is not None:
            gap = kd - md
            if gap > 0.02:
                interp = "Kalshi too high"
            elif gap < -0.02:
                interp = "Kalshi too low"
            else:
                interp = "~aligned"
            print(f"  {m:>4} {kd:>8.4f} {md:>8.4f} {gap*100:>+7.1f}¢ {interp:>16}")

    # Step 4: Slope analysis
    print(f"\n  STEP 4: Slope analysis — dP(Draw)/dmin")

    def compute_slope(prices_by_min: dict[int, float], m_start: int, m_end: int) -> float | None:
        vals = [(m, p) for m, p in prices_by_min.items() if m_start <= m <= m_end]
        if len(vals) < 2:
            return None
        vals.sort()
        # Use endpoints for simple slope
        return (vals[-1][1] - vals[0][1]) / (vals[-1][0] - vals[0][0])

    slopes = {}
    for label, data in [("Kalshi", kalshi_draw_by_min), ("MMPP", mmpp_draw_by_min)]:
        s1 = compute_slope(data, 85, 90)
        s2 = compute_slope(data, 90, 93)
        slopes[label] = (s1, s2)

    print(f"\n  {'Source':<8} {'dP/dmin (85->90)':>16} {'dP/dmin (90->93)':>16} {'Ratio 90+/pre':>14}")
    print(f"  {'─' * 58}")

    for label in ["Kalshi", "MMPP"]:
        s1, s2 = slopes[label]
        s1_str = f"{s1*100:+.2f}¢/min" if s1 is not None else "---"
        s2_str = f"{s2*100:+.2f}¢/min" if s2 is not None else "---"
        if s1 is not None and s2 is not None and abs(s1) > 0.0001:
            ratio = s2 / s1
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "---"
        print(f"  {label:<8} {s1_str:>16} {s2_str:>16} {ratio_str:>14}")

    # Interpretation
    ks1, ks2 = slopes["Kalshi"]
    ms1, ms2 = slopes["MMPP"]

    print(f"\n  INTERPRETATION:")
    if ks1 is not None and ms1 is not None:
        if ks1 > ms1 + 0.002:
            print(f"    -> Kalshi P(Draw) rises FASTER than MMPP before min 90")
            print(f"      ({ks1*100:+.2f} vs {ms1*100:+.2f}¢/min)")
            print(f"      This suggests anchoring: Kalshi traders think '90 = game over'")
        elif ks1 < ms1 - 0.002:
            print(f"    -> Kalshi P(Draw) rises SLOWER than MMPP before min 90")
            print(f"      ({ks1*100:+.2f} vs {ms1*100:+.2f}¢/min)")
            print(f"      No evidence of 90-minute anchoring in pre-90 window")
        else:
            print(f"    -> Similar pre-90 slopes: Kalshi={ks1*100:+.2f}, MMPP={ms1*100:+.2f}¢/min")

    if ks2 is not None and ms2 is not None:
        if abs(ks2) < abs(ms2) * 0.5 and abs(ms2) > 0.001:
            print(f"    -> In stoppage (90->93): Kalshi barely moves ({ks2*100:+.2f}¢/min)")
            print(f"      while MMPP still adjusts ({ms2*100:+.2f}¢/min)")
            print(f"      Kalshi may think game is 'over' at 90 -> ANCHORING BIAS")
        else:
            print(f"    -> In stoppage (90->93): Kalshi={ks2*100:+.2f}, MMPP={ms2*100:+.2f}¢/min")

    # Magnitude
    print(f"\n  MAGNITUDE OF BIAS:")
    gap_vals = []
    for m in target_minutes:
        kd = kalshi_draw_by_min.get(m)
        md = mmpp_draw_by_min.get(m)
        if kd is not None and md is not None:
            gap_vals.append((m, (kd - md) * 100))

    if gap_vals:
        max_gap = max(gap_vals, key=lambda x: abs(x[1]))
        print(f"    Max |gap|: {abs(max_gap[1]):.1f}¢ at min {max_gap[0]}")
        avg_gap = sum(abs(g) for _, g in gap_vals) / len(gap_vals)
        print(f"    Mean |gap|: {avg_gap:.1f}¢ across {len(gap_vals)} minutes")

    # Step 5: Profitability
    print(f"\n  PROFITABILITY (actual result: Draw, so P(Draw)->1.00 at FT):")

    if 85 in kalshi_draw_by_min and 90 in kalshi_draw_by_min:
        buy_85 = kalshi_draw_by_min[85]
        buy_90 = kalshi_draw_by_min[90]
        print(f"    Buy DRAW at min 85 (Kalshi={buy_85:.3f}): profit = {(1.0 - buy_85)*100:.1f}¢/contract")
        print(f"    Buy DRAW at min 90 (Kalshi={buy_90:.3f}): profit = {(1.0 - buy_90)*100:.1f}¢/contract")

    if 85 in mmpp_draw_by_min and 85 in kalshi_draw_by_min:
        mmpp_85 = mmpp_draw_by_min[85]
        kalshi_85 = kalshi_draw_by_min[85]
        if kalshi_85 > mmpp_85:
            print(f"\n    At min 85: Kalshi={kalshi_85:.3f} > MMPP={mmpp_85:.3f}")
            print(f"    -> Kalshi OVERprices draw -> SELL draw for {(kalshi_85-mmpp_85)*100:.1f}¢ edge")
            print(f"    -> But draw WON, so this trade LOST money this time")
        else:
            print(f"\n    At min 85: Kalshi={kalshi_85:.3f} <= MMPP={mmpp_85:.3f}")
            print(f"    -> Kalshi UNDERprices draw -> BUY draw for {(mmpp_85-kalshi_85)*100:.1f}¢ edge")
            print(f"    -> Draw WON, so this trade was PROFITABLE")

    # Sanity check
    print(f"\n  SANITY CHECK — 3 raw OB updates around min 90:")
    ts_90 = minute_to_ts.get(90)
    if ts_90:
        nearby = sorted(
            [(abs(ts - ts_90), ts, p) for ts, p in tl_tie if abs(ts - ts_90) < 30],
            key=lambda x: x[0],
        )
        if not nearby:
            nearby = sorted(
                [(abs(ts - ts_90), ts, p) for ts, p in tl_tie if abs(ts - ts_90) < 120],
                key=lambda x: x[0],
            )
        for i, (_, ts, p) in enumerate(nearby[:3]):
            rel = ts - ts_90
            utc = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
            print(f"    [{i+1}] {utc} (t={rel:+.1f}s from min90): P(Draw)={p:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # NULL HYPOTHESES AND CAVEATS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 78)
    print("NULL HYPOTHESES AND CAVEATS")
    print("=" * 78)
    print(f"""
  BIAS 1 — Surprise Goal Overreaction:
    Null hypothesis: Kalshi adjusts instantly to the new correct probability
    after any goal, with no overshoot. Overshoot = 0 for all goals.
    What would no-bias look like: The price 0-5min after the goal should equal
    the price 5-10min after (i.e., no mean-reversion from the initial spike).

  BIAS 2 — Stoppage Time Anchoring:
    Null hypothesis: Kalshi's dP(Draw)/dmin matches the true mathematical
    trajectory, which accounts for ~5min of stoppage time. The slope from
    85->90 should be similar to the slope from 90->93 (per minute), since
    goals can still happen in stoppage.
    What would no-bias look like: Kalshi's P(Draw) slope in stoppage time
    should match MMPP's slope. If Kalshi slope drops to ~0 after min 90
    while MMPP still shows positive slope, that's anchoring.

  CAVEATS:
    1. SINGLE MATCH — This is n=1. Any finding could be noise.
    2. Kalshi mid-price != fair value (bid-ask spread + thin liquidity)
    3. MMPP model uses default EPL params, not match-specific trained params
    4. Orderbook mid-price can be manipulated by placing/canceling limit orders
    5. The 5-10min "settled" window may still contain post-goal drift
    6. Goal at min 44 is very close to halftime (min 45) — HT break may
       affect the settled price window for that goal
    7. Goalserve minute resolution (~60s polling) adds timing uncertainty
    8. No trade data available — only orderbook snapshots/deltas, so mid-price
       is inferred from best bid/ask, not actual transaction prices""")


if __name__ == "__main__":
    main()
