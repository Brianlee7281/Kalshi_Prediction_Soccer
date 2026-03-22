#!/usr/bin/env python3
"""Kalshi price adjustment curve analysis around goals.

Measures HOW Kalshi prices adjust after a goal — instant or gradual? —
to determine if early goal detection via orderbook moves provides a
tradeable edge window vs waiting for Goalserve confirmation.

Usage:
  PYTHONPATH=. python scripts/analyze_goal_adjustment_curve.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────

RECORDINGS_DIR = Path("data/recordings")
OUTPUT_DIR = Path("data/analysis/goal_adjustment_curve")

# Timestamps relative to t=0 (first Kalshi move) at which to sample prices
SAMPLE_OFFSETS = [
    -10, -5, -2, -1, 0,
    1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120,
]

# For detecting first significant Kalshi move (reuse analyze_latency logic)
KALSHI_MOVE_THRESHOLD = 0.03  # 3 cents
BASELINE_WINDOW = (180, 60)  # seconds before Goalserve detection
DETECTION_WINDOW = 60  # seconds before Goalserve to start scanning

# For false signal analysis
FALSE_SIGNAL_THRESHOLDS = [0.05, 0.08, 0.10]  # 5c, 8c, 10c


def _utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


# ─── Data loading ─────────────────────────────────────────────────────


def load_metadata(match_dir: Path) -> dict:
    with open(match_dir / "metadata.json") as f:
        return json.load(f)


def load_goals(match_dir: Path) -> list[dict]:
    """Load goal events from events.jsonl."""
    goals = []
    with open(match_dir / "events.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("type") == "goal":
                goals.append(evt)
    return goals


def load_latency_report(match_dir: Path) -> dict | None:
    path = match_dir / "latency_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def identify_ticker_roles(meta: dict) -> dict[str, str]:
    """Map market role to ticker substring.

    Returns: {"home": "BRE", "away": "WOL", "draw": "TIE"}

    Ticker format: KXEPLGAME-26MAR16BREWOL-BRE
    The middle segment has date + two 3-letter team codes (home first).
    """
    tickers = meta.get("kalshi_tickers", [])
    roles = {}
    candidates = []
    for t in tickers:
        suffix = t.rsplit("-", 1)[-1]
        if suffix == "TIE":
            roles["draw"] = suffix
        elif suffix not in candidates:
            candidates.append(suffix)

    if not candidates or not tickers:
        return roles

    # Extract the 6-char team code pair from the middle segment
    # e.g. KXEPLGAME-26MAR16BREWOL-BRE → middle = "26MAR16BREWOL" → last 6 = "BREWOL"
    parts = tickers[0].split("-")
    if len(parts) >= 3:
        middle = parts[-2]  # e.g. "26MAR16BREWOL"
        team_codes = middle[-6:]  # e.g. "BREWOL" (home 3 + away 3)
    else:
        team_codes = ""

    for c in candidates:
        if team_codes:
            idx = team_codes.find(c)
            if idx == 0:
                roles["home"] = c
            elif idx > 0:
                roles["away"] = c
            else:
                # Fallback
                if "home" not in roles:
                    roles["home"] = c
                else:
                    roles["away"] = c
        else:
            if "home" not in roles:
                roles["home"] = c
            else:
                roles["away"] = c

    return roles


# ─── Orderbook reconstruction ────────────────────────────────────────


def build_all_orderbook_timelines(
    match_dir: Path, ticker_substrs: dict[str, str]
) -> dict[str, list[dict]]:
    """Rebuild best_yes price + spread for ALL tickers in a single file pass.

    Args:
        match_dir: Path to match recording directory.
        ticker_substrs: {role: substr} e.g. {"home": "BRE", "away": "WOL", "draw": "TIE"}

    Returns: {role: [{ts_wall, mid, spread, best_yes, best_no}, ...]}
    """
    # Per-ticker orderbook state
    yes_books: dict[str, dict[str, float]] = {r: {} for r in ticker_substrs}
    no_books: dict[str, dict[str, float]] = {r: {} for r in ticker_substrs}
    timelines: dict[str, list[dict]] = {r: [] for r in ticker_substrs}

    # Map ticker substring → role for fast lookup
    substr_to_role: dict[str, str] = {v: k for k, v in ticker_substrs.items()}

    with open(match_dir / "kalshi_ob.jsonl") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("msg", data)
            ticker = msg.get("market_ticker", "")
            if not ticker:
                continue

            # Find which role this ticker belongs to (match suffix only)
            role = None
            for substr, r in substr_to_role.items():
                if ticker.endswith(f"-{substr}"):
                    role = r
                    break
            if role is None:
                continue

            ts = data.get("_ts_wall", 0)
            msg_type = data.get("type", "")
            yes_book = yes_books[role]
            no_book = no_books[role]

            if msg_type == "orderbook_snapshot":
                yes_book.clear()
                no_book.clear()
                for entry in msg.get("yes_dollars_fp", []):
                    if len(entry) >= 2:
                        qty = float(entry[1])
                        if qty > 0:
                            yes_book[entry[0]] = qty
                for entry in msg.get("no_dollars_fp", []):
                    if len(entry) >= 2:
                        qty = float(entry[1])
                        if qty > 0:
                            no_book[entry[0]] = qty

            elif msg_type == "orderbook_delta":
                side = msg.get("side", "")
                price_str = msg.get("price_dollars", "")
                delta = float(msg.get("delta_fp", "0"))
                if not price_str:
                    continue
                book = yes_book if side == "yes" else no_book
                current = book.get(price_str, 0)
                new_qty = current + delta
                if new_qty > 0:
                    book[price_str] = new_qty
                else:
                    book.pop(price_str, None)
            else:
                continue

            # Compute price from best_yes (primary, reliable metric)
            best_yes = max((float(p) for p in yes_book), default=None) if yes_book else None
            best_no = max((float(p) for p in no_book), default=None) if no_book else None

            if best_yes is None:
                continue

            mid = best_yes

            # Spread: (1 - best_no) - best_yes, clamp >= 0
            if best_no is not None:
                raw_spread = (1.0 - best_no) - best_yes
                spread = max(0.0, raw_spread)
            else:
                spread = None

            timelines[role].append({
                "ts_wall": ts,
                "mid": mid,
                "spread": spread,
                "best_yes": best_yes,
                "best_no": best_no,
            })

    return timelines


def sample_at_offset(
    timeline: list[dict], t0: float, offset_sec: float
) -> dict | None:
    """Find the most recent orderbook state at t0 + offset_sec.

    Returns the timeline entry with the largest ts_wall <= target,
    or None if no data exists before the target.
    """
    target = t0 + offset_sec
    best = None
    for entry in timeline:
        if entry["ts_wall"] <= target:
            best = entry
        elif entry["ts_wall"] > target:
            break  # timeline is sorted by ts_wall
    return best


# ─── Goal detection (reuses analyze_latency logic) ───────────────────


def detect_kalshi_first_move(
    timeline: list[dict], goal_ts: float
) -> float | None:
    """Detect first significant Kalshi mid-price move around a goal.

    Returns the wall-clock timestamp of the first move, or None.
    Uses the same baseline/detection window as analyze_latency.py.
    """
    # Baseline: average mid in [goal_ts - 180, goal_ts - 60]
    baseline_entries = [
        e for e in timeline
        if (goal_ts - BASELINE_WINDOW[0]) < e["ts_wall"] < (goal_ts - BASELINE_WINDOW[1])
    ]
    if not baseline_entries:
        return None
    baseline_mid = statistics.mean(e["mid"] for e in baseline_entries)

    # Scan from 60s before Goalserve to 300s after
    scan_entries = [
        e for e in timeline
        if (goal_ts - DETECTION_WINDOW) < e["ts_wall"] < (goal_ts + 300)
    ]
    for e in scan_entries:
        if abs(e["mid"] - baseline_mid) > KALSHI_MOVE_THRESHOLD:
            return e["ts_wall"]
    return None


# ─── Per-goal analysis ────────────────────────────────────────────────


def analyze_goal_curve(
    goal: dict,
    scoring_team_tl: list[dict],
    all_market_tls: dict[str, list[dict]],
    t_kalshi_first: float | None,
    goalserve_lag: float | None,
) -> dict | None:
    """Build the adjustment curve for a single goal.

    Returns dict with sampled prices, adjustment fractions, spreads,
    or None if insufficient data.
    """
    goal_ts = goal["ts_wall"]
    team = goal.get("team", "?")
    new_score = goal.get("new_score", [0, 0])

    # Use pre-computed t_kalshi_first as t=0
    if t_kalshi_first is None:
        # Try to detect from the scoring team's timeline
        t_kalshi_first = detect_kalshi_first_move(scoring_team_tl, goal_ts)
    if t_kalshi_first is None:
        return None

    t0 = t_kalshi_first

    # Sample the scoring team's market at each offset
    samples = {}
    for offset in SAMPLE_OFFSETS:
        entry = sample_at_offset(scoring_team_tl, t0, offset)
        if entry is not None:
            samples[offset] = {
                "mid": entry["mid"],
                "spread": entry["spread"],
                "ts_wall": entry["ts_wall"],
            }

    # Need at least baseline (-5s) and final (+120s) for normalization
    if -5 not in samples or 120 not in samples:
        return None

    baseline_mid = samples[-5]["mid"]
    final_mid = samples[120]["mid"]
    total_move = final_mid - baseline_mid

    if abs(total_move) < 0.005:
        # Move too small to analyze
        return None

    # Compute adjustment fraction at each offset
    adj_fractions = {}
    for offset, s in samples.items():
        adj_fractions[offset] = (s["mid"] - baseline_mid) / total_move

    # Also sample all 3 markets
    all_market_samples = {}
    for market_role, tl in all_market_tls.items():
        market_samples = {}
        for offset in SAMPLE_OFFSETS:
            entry = sample_at_offset(tl, t0, offset)
            if entry is not None:
                market_samples[offset] = {
                    "mid": entry["mid"],
                    "spread": entry["spread"],
                }
        all_market_samples[market_role] = market_samples

    gs_lag = goalserve_lag if goalserve_lag is not None else (goal_ts - t0)

    return {
        "goal_ts": goal_ts,
        "goal_utc": _utc_str(goal_ts),
        "team": team,
        "new_score": new_score,
        "t0": t0,
        "t0_utc": _utc_str(t0),
        "goalserve_lag": gs_lag,
        "baseline_mid": baseline_mid,
        "final_mid": final_mid,
        "total_move": total_move,
        "total_move_cents": total_move * 100,
        "samples": samples,
        "adj_fractions": adj_fractions,
        "all_market_samples": all_market_samples,
    }


# ─── False signal detection ──────────────────────────────────────────


def find_false_signals(
    timeline: list[dict],
    goal_timestamps: list[float],
    match_id: str,
    market_role: str,
) -> list[dict]:
    """Find non-goal price spikes > 5 cents in the mid-price timeline.

    Excludes windows within 120s of any goal's t_kalshi_first.
    Uses a trailing-pointer approach instead of list slicing for performance.
    """
    if len(timeline) < 2:
        return []

    spikes = []
    # trailing pointer for the ~5s-ago baseline
    trail = 0
    i = 0
    while i < len(timeline):
        entry = timeline[i]
        ts = entry["ts_wall"]

        # Skip if within 120s of any goal
        near_goal = False
        for gts in goal_timestamps:
            if abs(ts - gts) < 120:
                near_goal = True
                break
        if near_goal:
            i += 1
            continue

        # Advance trail to the latest entry that's >= 5s before current
        target_baseline = ts - 5.0
        while trail < i and timeline[trail + 1]["ts_wall"] <= target_baseline:
            trail += 1

        if timeline[trail]["ts_wall"] > target_baseline or trail == i:
            i += 1
            continue

        baseline_mid = timeline[trail]["mid"]
        move = abs(entry["mid"] - baseline_mid)

        if move > 0.05:
            # Check if it reversed within 10s (simple forward scan)
            future_ts = ts + 10.0
            future_entry = None
            for j in range(i, min(i + 200, len(timeline))):
                if timeline[j]["ts_wall"] >= future_ts:
                    future_entry = timeline[j]
                    break

            reversed_flag = False
            if future_entry is not None:
                revert = abs(future_entry["mid"] - baseline_mid)
                if revert < move * 0.5:
                    reversed_flag = True

            spikes.append({
                "match_id": match_id,
                "ts_wall": ts,
                "utc": _utc_str(ts),
                "market": market_role,
                "move_cents": round(move * 100, 1),
                "reversed": reversed_flag,
            })
            # Skip forward 10s to avoid double-counting
            while i < len(timeline) and timeline[i]["ts_wall"] < ts + 10:
                i += 1
            continue

        i += 1

    return spikes


# ─── Aggregation and reporting ────────────────────────────────────────


def aggregate_curves(goal_results: list[dict]) -> dict:
    """Compute mean/median adjustment fractions and spreads at each offset."""
    agg = {}
    for offset in SAMPLE_OFFSETS:
        fractions = []
        spreads = []
        for gr in goal_results:
            if offset in gr["adj_fractions"]:
                fractions.append(gr["adj_fractions"][offset])
            if offset in gr["samples"] and gr["samples"][offset]["spread"] is not None:
                spreads.append(gr["samples"][offset]["spread"] * 100)  # to cents

        n = len(fractions)
        agg[offset] = {
            "n": n,
            "mean_adj": statistics.mean(fractions) if fractions else None,
            "median_adj": statistics.median(fractions) if fractions else None,
            "mean_spread": statistics.mean(spreads) if spreads else None,
            "max_spread": max(spreads) if spreads else None,
            "pct_wide": (sum(1 for s in spreads if s > 5) / len(spreads) * 100) if spreads else None,
        }
    return agg


def format_offset(offset: int) -> str:
    if offset == 0:
        return "0s (detect)"
    sign = "+" if offset > 0 else ""
    return f"{sign}{offset}s"


def print_adjustment_curve_table(agg: dict, n_goals: int, avg_gs_lag: float) -> str:
    lines = []
    lines.append("")
    lines.append(f"KALSHI PRICE ADJUSTMENT CURVE ({n_goals} goals)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Time after':<15} {'Mean adj%':>10} {'Median adj%':>12} {'Mean spread':>12} {'N goals':>8}")
    lines.append(f"{'first move':<15} {'':>10} {'':>12} {'(cents)':>12} {'with data':>8}")
    lines.append("-" * 70)

    for offset in SAMPLE_OFFSETS:
        a = agg.get(offset, {})
        label = format_offset(offset)
        mean_a = f"{a['mean_adj']*100:.0f}%" if a.get("mean_adj") is not None else "N/A"
        med_a = f"{a['median_adj']*100:.0f}%" if a.get("median_adj") is not None else "N/A"
        spread = f"{a['mean_spread']:.1f}" if a.get("mean_spread") is not None else "N/A"
        n = a.get("n", 0)

        # Mark where Goalserve typically arrives
        gs_marker = ""
        if offset > 0 and abs(offset - avg_gs_lag) < 10:
            gs_marker = "  <- Goalserve"

        lines.append(f"{label:<15} {mean_a:>10} {med_a:>12} {spread:>12} {n:>8}{gs_marker}")

    text = "\n".join(lines)
    print(text)
    return text


def print_per_goal_table(goal_results: list[dict], match_metas: dict) -> str:
    lines = []
    lines.append("")
    lines.append("INDIVIDUAL GOAL ADJUSTMENT PROFILES")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"{'#':<4} {'Match':<25} {'Score':>6} {'Team':>5} "
        f"{'Total Move':>11} {'Adj@+5s':>8} {'Adj@+10s':>9} {'Adj@+30s':>9} {'GS lag':>7}"
    )
    lines.append("-" * 100)

    for i, gr in enumerate(goal_results, 1):
        match_id = gr.get("match_id", "?")
        meta = match_metas.get(match_id, {})
        match_name = f"{meta.get('home_team', '?')}-{meta.get('away_team', '?')}"
        score = f"{gr['new_score'][0]}-{gr['new_score'][1]}"
        team = gr["team"][0].upper()
        move = f"{gr['total_move_cents']:+.1f}c"

        adj5 = gr["adj_fractions"].get(5)
        adj10 = gr["adj_fractions"].get(10)
        adj30 = gr["adj_fractions"].get(30)
        a5 = f"{adj5*100:.0f}%" if adj5 is not None else "N/A"
        a10 = f"{adj10*100:.0f}%" if adj10 is not None else "N/A"
        a30 = f"{adj30*100:.0f}%" if adj30 is not None else "N/A"
        gs = f"{gr['goalserve_lag']:.0f}s"

        lines.append(
            f"{i:<4} {match_name:<25} {score:>6} {team:>5} "
            f"{move:>11} {a5:>8} {a10:>9} {a30:>9} {gs:>7}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_spread_table(agg: dict) -> str:
    offsets_for_spread = [-10, -5, 0, 1, 2, 3, 5, 10, 15, 30, 60]
    lines = []
    lines.append("")
    lines.append("SPREAD BEHAVIOR AROUND GOALS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"{'Time':<15} {'Mean spread':>12} {'Max spread':>11} {'% spread':>10}"
    )
    lines.append(
        f"{'':15} {'(cents)':>12} {'(cents)':>11} {'> 5 cents':>10}"
    )
    lines.append("-" * 70)

    for offset in offsets_for_spread:
        a = agg.get(offset, {})
        label = format_offset(offset)
        mean_s = f"{a['mean_spread']:.1f}" if a.get("mean_spread") is not None else "N/A"
        max_s = f"{a['max_spread']:.1f}" if a.get("max_spread") is not None else "N/A"
        pct = f"{a['pct_wide']:.0f}%" if a.get("pct_wide") is not None else "N/A"

        marker = ""
        if offset == 0:
            marker = "  <- goal detected"

        lines.append(f"{label:<15} {mean_s:>12} {max_s:>11} {pct:>10}{marker}")

    text = "\n".join(lines)
    print(text)
    return text


def print_edge_window(agg: dict, goal_results: list[dict]) -> str:
    lines = []
    lines.append("")
    lines.append("ESTIMATED TRADEABLE EDGE WINDOW")
    lines.append("=" * 70)
    lines.append("")

    # Average total move across goals (in cents)
    avg_total_move = statistics.mean(abs(gr["total_move_cents"]) for gr in goal_results)

    lines.append(f"Average total move: {avg_total_move:.1f} cents")
    lines.append("")
    lines.append(
        f"{'Time':>6} {'Remaining':>12} {'Spread cost':>12} {'Net edge':>10} {'Tradeable?':>11}"
    )
    lines.append(
        f"{'':>6} {'move (c)':>12} {'(c)':>12} {'(c)':>10} {'':>11}"
    )
    lines.append("-" * 70)

    edge_offsets = [1, 2, 3, 5, 10, 15, 30, 60]
    for offset in edge_offsets:
        a = agg.get(offset, {})
        mean_adj = a.get("mean_adj")
        mean_spread = a.get("mean_spread")

        if mean_adj is not None:
            remaining = (1.0 - mean_adj) * avg_total_move
        else:
            remaining = None

        if remaining is not None and mean_spread is not None:
            spread_cost = mean_spread / 2.0
            net = remaining - spread_cost
            tradeable = "YES" if net > 0 else "NO"
        else:
            spread_cost = None
            net = None
            tradeable = "N/A"

        rem_s = f"{remaining:.1f}" if remaining is not None else "N/A"
        spr_s = f"{spread_cost:.1f}" if spread_cost is not None else "N/A"
        net_s = f"{net:.1f}" if net is not None else "N/A"
        label = f"+{offset}s"

        lines.append(f"{label:>6} {rem_s:>12} {spr_s:>12} {net_s:>10} {tradeable:>11}")

    text = "\n".join(lines)
    print(text)
    return text


def print_false_signal_table(
    all_spikes: list[dict], n_goals: int
) -> str:
    lines = []
    lines.append("")
    lines.append("FALSE SIGNAL ANALYSIS (non-goal price spikes > 5 cents)")
    lines.append("=" * 90)
    lines.append("")

    if all_spikes:
        lines.append(
            f"{'Match':<14} {'Spike time':<12} {'Market':<12} "
            f"{'Move':>6} {'Reversed?':>10}"
        )
        lines.append("-" * 90)
        for spike in sorted(all_spikes, key=lambda s: s["ts_wall"]):
            lines.append(
                f"{spike['match_id']:<14} {spike['utc']:<12} {spike['market']:<12} "
                f"{spike['move_cents']:>+5.1f}c {('Yes' if spike['reversed'] else 'No'):>10}"
            )
    else:
        lines.append("No non-goal spikes > 5 cents detected.")

    lines.append("")
    for threshold in FALSE_SIGNAL_THRESHOLDS:
        tc = threshold * 100
        n_false = sum(1 for s in all_spikes if s["move_cents"] >= tc)
        total = n_false + n_goals
        rate = n_false / total * 100 if total > 0 else 0
        lines.append(
            f"Total non-goal spikes > {tc:.0f}c:  {n_false}"
        )
    lines.append(f"Total actual goals:           {n_goals}")
    lines.append("")
    for threshold in FALSE_SIGNAL_THRESHOLDS:
        tc = threshold * 100
        n_false = sum(1 for s in all_spikes if s["move_cents"] >= tc)
        total = n_false + n_goals
        rate = n_false / total * 100 if total > 0 else 0
        lines.append(
            f"False positive rate at {tc:.0f}c threshold:  "
            f"{n_false} / ({n_false} + {n_goals}) = {rate:.0f}%"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_verdict(
    agg: dict, goal_results: list[dict], all_spikes: list[dict]
) -> str:
    lines = []
    lines.append("")
    lines.append("VERDICT")
    lines.append("=" * 70)
    lines.append("")

    n_goals = len(goal_results)
    avg_total_move = statistics.mean(abs(gr["total_move_cents"]) for gr in goal_results)
    avg_gs_lag = statistics.mean(gr["goalserve_lag"] for gr in goal_results)

    # Adjustment speed
    adj5 = agg.get(5, {}).get("mean_adj")
    adj10 = agg.get(10, {}).get("mean_adj")

    if adj5 is not None and adj5 > 0.80:
        speed = "FAST (early detection barely helps)"
    elif adj10 is not None and adj10 < 0.50:
        speed = "GRADUAL (large tradeable window)"
    else:
        speed = "MIXED"
    lines.append(f"Adjustment speed:     {speed}")

    if adj5 is not None:
        lines.append(f"  At +5s:  {adj5*100:.0f}% adjusted")
    if adj10 is not None:
        lines.append(f"  At +10s: {adj10*100:.0f}% adjusted")

    # Tradeable window
    lines.append("")
    edge_offsets = [1, 2, 3, 5, 10, 15, 30, 60, 90, 120]
    tradeable_start = None
    tradeable_end = None
    for offset in edge_offsets:
        a = agg.get(offset, {})
        mean_adj = a.get("mean_adj")
        mean_spread = a.get("mean_spread")
        if mean_adj is not None and mean_spread is not None:
            remaining = (1.0 - mean_adj) * avg_total_move
            spread_cost = mean_spread / 2.0
            net = remaining - spread_cost
            if net > 0:
                if tradeable_start is None:
                    tradeable_start = offset
                tradeable_end = offset

    if tradeable_start is not None and tradeable_end is not None:
        lines.append(f"Tradeable window:     +{tradeable_start}s to +{tradeable_end}s (where net_edge > 0)")
    else:
        lines.append("Tradeable window:     NONE (no positive net edge found)")

    # GS comparison
    lines.append("")
    gs_adj = agg.get(30, {}).get("mean_adj")  # typical GS lag ~30s
    if gs_adj is not None:
        gs_remaining_pct = (1.0 - gs_adj) * 100
        lines.append(f"Current system (Goalserve, avg lag {avg_gs_lag:.0f}s): ~{gs_remaining_pct:.0f}% of move remaining")
    early_adj = agg.get(1, {}).get("mean_adj")
    if early_adj is not None:
        early_remaining_pct = (1.0 - early_adj) * 100
        lines.append(f"With Kalshi detection (+1s):  ~{early_remaining_pct:.0f}% of move remaining")
    if gs_adj is not None and early_adj is not None:
        improvement = (gs_adj - early_adj) * 100
        lines.append(f"Improvement: +{improvement:.0f} percentage points of capturable move")

    # Spread impact
    lines.append("")
    spread_5 = agg.get(5, {}).get("mean_spread")
    spread_15 = agg.get(15, {}).get("mean_spread")
    if spread_5 is not None and spread_5 < 3:
        spread_verdict = "LOW (can trade early, spread < 3c at +5s)"
    elif spread_15 is not None and spread_15 > 5:
        spread_verdict = "HIGH (must wait, spread > 5c at +15s)"
    else:
        spread_verdict = "MODERATE"
    lines.append(f"Spread impact:        {spread_verdict}")

    # False signal risk
    lines.append("")
    n_false_8 = sum(1 for s in all_spikes if s["move_cents"] >= 8)
    total_8 = n_false_8 + n_goals
    rate_8 = n_false_8 / total_8 * 100 if total_8 > 0 else 0
    if rate_8 < 10:
        fs_verdict = f"LOW (at 8c threshold: {rate_8:.0f}% false positive rate)"
    elif rate_8 < 30:
        fs_verdict = f"MODERATE (at 8c threshold: {rate_8:.0f}% false positive rate)"
    else:
        fs_verdict = f"HIGH (at 8c threshold: {rate_8:.0f}% false positive rate)"
    lines.append(f"False signal risk:    {fs_verdict}")

    # Recommendation
    lines.append("")
    has_window = tradeable_start is not None
    low_spread = spread_5 is not None and spread_5 < 5
    low_false = rate_8 < 30
    if has_window and low_spread and low_false:
        rec = "IMPLEMENT — meaningful edge window with manageable spread and false signal risk"
    elif has_window:
        rec = "NEEDS MORE DATA — edge window exists but spread/false signal concerns"
    else:
        rec = "SKIP — no tradeable edge window found"
    lines.append(f"Recommendation:       {rec}")

    text = "\n".join(lines)
    print(text)
    return text


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    if not RECORDINGS_DIR.exists():
        print(f"ERROR: {RECORDINGS_DIR} does not exist")
        sys.exit(1)

    match_dirs = sorted(
        p for p in RECORDINGS_DIR.iterdir()
        if p.is_dir() and (p / "kalshi_ob.jsonl").exists()
    )

    print("=" * 70)
    print("KALSHI GOAL ADJUSTMENT CURVE ANALYSIS")
    print(f"Matches: {len(match_dirs)}")
    print("=" * 70)

    all_goal_results: list[dict] = []
    all_false_spikes: list[dict] = []
    match_metas: dict[str, dict] = {}
    skipped_goals = 0

    for match_dir in match_dirs:
        meta = load_metadata(match_dir)
        match_id = meta.get("match_id", match_dir.name)
        match_metas[match_id] = meta
        home = meta.get("home_team", "?")
        away = meta.get("away_team", "?")

        print(f"\n--- {match_id}: {home} vs {away} ---")

        goals = load_goals(match_dir)
        if not goals:
            print("  No goals, skipping.")
            continue

        latency = load_latency_report(match_dir)
        ticker_roles = identify_ticker_roles(meta)

        print(f"  Tickers: home={ticker_roles.get('home')}, "
              f"away={ticker_roles.get('away')}, draw={ticker_roles.get('draw')}")
        print(f"  Goals: {len(goals)}, loading orderbook timelines...")

        # Build timelines for all 3 markets in a single file pass
        market_timelines = build_all_orderbook_timelines(match_dir, ticker_roles)
        for role in ticker_roles:
            print(f"    {role} ({ticker_roles[role]}): {len(market_timelines.get(role, []))} entries")

        # Process each goal
        for gi, goal in enumerate(goals):
            team = goal.get("team", "?")
            new_score = goal.get("new_score", [0, 0])

            # Determine scoring team's market
            scoring_market = team if team in ("home", "away") else "home"
            scoring_tl = market_timelines.get(scoring_market, [])

            if not scoring_tl:
                print(f"  Goal {gi+1} ({team} {new_score}): no orderbook data for {scoring_market}, skipping")
                skipped_goals += 1
                continue

            # Get t_kalshi_first from latency report if available
            t_kalshi_first = None
            if latency and gi < len(latency.get("results", [])):
                t_kalshi_first = latency["results"][gi].get("t_kalshi_first")

            gs_lag = None
            if t_kalshi_first is not None:
                gs_lag = goal["ts_wall"] - t_kalshi_first

            result = analyze_goal_curve(
                goal, scoring_tl, market_timelines, t_kalshi_first, gs_lag,
            )

            if result is None:
                print(f"  Goal {gi+1} ({team} {new_score}): insufficient data, skipping")
                skipped_goals += 1
                continue

            result["match_id"] = match_id
            all_goal_results.append(result)
            adj5 = result["adj_fractions"].get(5)
            adj5_s = f"{adj5*100:.0f}%" if adj5 is not None else "N/A"
            print(f"  Goal {gi+1} ({team} {new_score}): "
                  f"total_move={result['total_move_cents']:+.1f}c, "
                  f"adj@+5s={adj5_s}, gs_lag={result['goalserve_lag']:.0f}s")

        # False signal analysis: use home_win market (largest/most liquid)
        home_tl = market_timelines.get("home", [])
        goal_t0s = []
        for gr in all_goal_results:
            if gr["match_id"] == match_id:
                goal_t0s.append(gr["t0"])
        if home_tl:
            spikes = find_false_signals(home_tl, goal_t0s, match_id, "home_win")
            all_false_spikes.extend(spikes)
            if spikes:
                print(f"  False signals (>5c, home_win): {len(spikes)}")

    # ─── Reporting ────────────────────────────────────────────────
    n_goals = len(all_goal_results)
    if n_goals == 0:
        print("\nNo goals with sufficient data to analyze.")
        return

    print(f"\n\n{'='*70}")
    print(f"AGGREGATE RESULTS ({n_goals} goals analyzed, {skipped_goals} skipped)")
    print("=" * 70)

    avg_gs_lag = statistics.mean(gr["goalserve_lag"] for gr in all_goal_results)
    agg = aggregate_curves(all_goal_results)

    report_parts = []
    report_parts.append(print_adjustment_curve_table(agg, n_goals, avg_gs_lag))
    report_parts.append(print_per_goal_table(all_goal_results, match_metas))
    report_parts.append(print_spread_table(agg))
    report_parts.append(print_edge_window(agg, all_goal_results))
    report_parts.append(print_false_signal_table(all_false_spikes, n_goals))
    report_parts.append(print_verdict(agg, all_goal_results, all_false_spikes))

    # ─── Save outputs ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw JSON data
    json_data = {
        "n_goals": n_goals,
        "n_skipped": skipped_goals,
        "avg_goalserve_lag": avg_gs_lag,
        "aggregate": {
            str(k): v for k, v in agg.items()
        },
        "goals": [
            {
                "match_id": gr["match_id"],
                "goal_ts": gr["goal_ts"],
                "goal_utc": gr["goal_utc"],
                "team": gr["team"],
                "new_score": gr["new_score"],
                "t0": gr["t0"],
                "goalserve_lag": gr["goalserve_lag"],
                "total_move_cents": gr["total_move_cents"],
                "adj_fractions": {str(k): v for k, v in gr["adj_fractions"].items()},
                "samples": {
                    str(k): {
                        "mid": v["mid"],
                        "spread": v["spread"],
                    } for k, v in gr["samples"].items()
                },
            }
            for gr in all_goal_results
        ],
        "false_spikes": all_false_spikes,
    }
    json_path = OUTPUT_DIR / "adjustment_curve.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # 2. Aggregate report text
    report_path = OUTPUT_DIR / "aggregate_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_parts))
    print(f"Saved: {report_path}")

    # 3. Per-goal CSV
    csv_path = OUTPUT_DIR / "per_goal_detail.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "goal_num", "match_id", "team", "score", "total_move_cents",
            "adj_at_1s", "adj_at_2s", "adj_at_3s", "adj_at_5s",
            "adj_at_10s", "adj_at_15s", "adj_at_30s", "adj_at_60s",
            "goalserve_lag_s", "spread_at_0s_cents", "spread_at_5s_cents",
        ])
        for i, gr in enumerate(all_goal_results, 1):
            score_str = f"{gr['new_score'][0]}-{gr['new_score'][1]}"
            af = gr["adj_fractions"]
            s = gr["samples"]
            writer.writerow([
                i,
                gr["match_id"],
                gr["team"],
                score_str,
                round(gr["total_move_cents"], 1),
                round(af.get(1, 0), 3),
                round(af.get(2, 0), 3),
                round(af.get(3, 0), 3),
                round(af.get(5, 0), 3),
                round(af.get(10, 0), 3),
                round(af.get(15, 0), 3),
                round(af.get(30, 0), 3),
                round(af.get(60, 0), 3),
                round(gr["goalserve_lag"], 1),
                round(s.get(0, {}).get("spread", 0) * 100, 1) if s.get(0, {}).get("spread") is not None else "",
                round(s.get(5, {}).get("spread", 0) * 100, 1) if s.get(5, {}).get("spread") is not None else "",
            ])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
