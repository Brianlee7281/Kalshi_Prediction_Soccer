#!/usr/bin/env python3
"""Post-hoc latency analyzer for recorded match data.

Reads raw JSONL from data/recordings/{match_id}/ and computes cross-market
lag by retroactively detecting price movements from each source around
Goalserve-detected goals.

Usage:
  PYTHONPATH=. python scripts/analyze_latency.py data/recordings/4190023
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def load_goals(match_dir: Path) -> list[dict]:
    """Load goal events from events.jsonl."""
    goals = []
    events_path = match_dir / "events.jsonl"
    if not events_path.exists():
        return goals
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("type") == "goal":
                goals.append(evt)
    return goals


def find_match_event_ids(
    match_dir: Path, kickoff_hour: str = "T20:00"
) -> set[str]:
    """Find the single Odds-API event ID for the target match.

    When multiple matches share the same kickoff time, we disambiguate by
    finding the ID whose pre-match home odds most closely match the Kalshi
    pre-match best_yes price (both reflect the same implied probability).
    If no Kalshi data, returns all IDs for the kickoff hour.
    """
    # Collect all IDs with matching kickoff
    candidate_ids: set[str] = set()
    with open(match_dir / "odds_api.jsonl") as f:
        for line in f:
            data = json.loads(line)
            date = data.get("date", "")
            mid = data.get("id", "")
            if kickoff_hour in date and mid:
                candidate_ids.add(mid)

    if len(candidate_ids) <= 1:
        return candidate_ids

    # Get Kalshi pre-match best_yes as reference
    kalshi_file = match_dir / "kalshi_ob.jsonl"
    kalshi_ref = None
    if kalshi_file.exists():
        with open(kalshi_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") != "orderbook_snapshot":
                    continue
                msg = data.get("msg", {})
                ticker = msg.get("market_ticker", "")
                # Look for the home-win ticker (ends with team abbreviation)
                if not ticker.endswith("-TIE") and not ticker.endswith("-WOL"):
                    yes_book = msg.get("yes_dollars_fp", [])
                    if yes_book:
                        prices = [float(p) for p, q in yes_book if float(q) > 0]
                        if prices:
                            kalshi_ref = max(prices)
                            break

    if kalshi_ref is None:
        return candidate_ids

    # For each candidate ID, compute average pre-match home implied prob
    best_id = None
    best_diff = float("inf")
    for eid in candidate_ids:
        implieds: list[float] = []
        count = 0
        with open(match_dir / "odds_api.jsonl") as f:
            for line in f:
                data = json.loads(line)
                if data.get("id") != eid:
                    continue
                count += 1
                if count > 50:
                    break
                for mkt in data.get("markets", []):
                    if mkt.get("name") != "ML":
                        continue
                    odds = mkt.get("odds", [{}])[0]
                    h = odds.get("home")
                    if h:
                        try:
                            implieds.append(1.0 / float(h))
                        except (ValueError, ZeroDivisionError):
                            pass
        if implieds:
            avg_implied = sum(implieds) / len(implieds)
            diff = abs(avg_implied - kalshi_ref)
            if diff < best_diff:
                best_diff = diff
                best_id = eid

    if best_id:
        return {best_id}
    return candidate_ids


def build_odds_timeline(
    match_dir: Path, event_ids: set[str]
) -> list[dict]:
    """Build per-bookie implied probability timeline from Odds-API data.

    Returns list of {ts_wall, bookie, home_implied} sorted by ts_wall.
    """
    timeline = []
    with open(match_dir / "odds_api.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if data.get("id") not in event_ids:
                continue
            bookie = data.get("bookie", "")
            ts = data.get("_ts_wall", 0)
            for mkt in data.get("markets", []):
                if mkt.get("name") != "ML":
                    continue
                odds_list = mkt.get("odds", [])
                if not odds_list:
                    continue
                o = odds_list[0]
                home_str = o.get("home")
                if not home_str:
                    continue
                try:
                    home_odds = float(home_str)
                    if home_odds <= 0:
                        continue
                    implied = 1.0 / home_odds
                except (ValueError, ZeroDivisionError):
                    continue
                timeline.append({
                    "ts_wall": ts,
                    "bookie": bookie,
                    "home_implied": implied,
                    "home_odds": home_odds,
                })
    timeline.sort(key=lambda x: x["ts_wall"])
    return timeline


def build_kalshi_price_timeline(
    match_dir: Path, ticker_substr: str = "BRE"
) -> list[dict]:
    """Rebuild best yes price from Kalshi snapshots + deltas.

    Maintains an orderbook state and emits the best (highest) yes price
    with quantity > 0 after each update.
    """
    # Orderbook state: {price_str: qty} for yes side
    yes_book: dict[str, float] = {}
    timeline = []

    with open(match_dir / "kalshi_ob.jsonl") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("msg", data)
            ticker = msg.get("market_ticker", "")
            if ticker_substr not in ticker:
                continue

            ts = data.get("_ts_wall", 0)
            msg_type = data.get("type", "")

            if msg_type == "orderbook_snapshot":
                # Reset book from snapshot
                yes_book.clear()
                for entry in msg.get("yes_dollars_fp", []):
                    if len(entry) >= 2:
                        price_str = entry[0]
                        qty = float(entry[1])
                        if qty > 0:
                            yes_book[price_str] = qty

            elif msg_type == "orderbook_delta":
                side = msg.get("side", "")
                if side != "yes":
                    continue
                price_str = msg.get("price_dollars", "")
                delta = float(msg.get("delta_fp", "0"))
                if not price_str:
                    continue
                current = yes_book.get(price_str, 0)
                new_qty = current + delta
                if new_qty > 0:
                    yes_book[price_str] = new_qty
                else:
                    yes_book.pop(price_str, None)
            else:
                continue

            # Compute best (highest) yes price with quantity
            if yes_book:
                best_price = max(float(p) for p in yes_book.keys())
                timeline.append({
                    "ts_wall": ts,
                    "best_yes": best_price,
                    "ticker": ticker,
                })

    return timeline


def analyze_goal(
    goal: dict,
    odds_timeline: list[dict],
    kalshi_timeline: list[dict],
    window_after: float = 300.0,
) -> dict:
    """Analyze price reaction around a single goal event.

    Baseline is computed from 180s to 60s BEFORE the Goalserve detection
    (far enough back to avoid contamination from the actual event).
    Then we scan forward from 60s before Goalserve to find the first
    significant move (>3% / >3¢).
    """
    goal_ts = goal["ts_wall"]
    goal_utc = _utc_str(goal_ts)

    # --- Odds-API: find first significant move ---
    # Baseline: 180s to 60s before Goalserve detection
    pre_odds = [
        o for o in odds_timeline
        if (goal_ts - 180) < o["ts_wall"] < (goal_ts - 60)
    ]

    bookie_baselines: dict[str, list[float]] = defaultdict(list)
    for o in pre_odds:
        bookie_baselines[o["bookie"]].append(o["home_implied"])

    baselines: dict[str, float] = {}
    for bookie, values in bookie_baselines.items():
        if values:
            baselines[bookie] = sum(values) / len(values)

    # Scan from 60s before Goalserve through window_after
    t_odds_first = None
    odds_first_bookie = None
    odds_first_move = 0.0
    post_odds = [
        o for o in odds_timeline
        if (goal_ts - 60) < o["ts_wall"] < (goal_ts + window_after)
    ]
    for o in post_odds:
        baseline = baselines.get(o["bookie"])
        if baseline is None:
            continue
        move = abs(o["home_implied"] - baseline)
        if move > 0.03 and t_odds_first is None:
            t_odds_first = o["ts_wall"]
            odds_first_bookie = o["bookie"]
            odds_first_move = o["home_implied"] - baseline

    # --- Kalshi: find first significant move ---
    # Baseline: 180s to 60s before Goalserve detection
    pre_kalshi = [
        k for k in kalshi_timeline
        if (goal_ts - 180) < k["ts_wall"] < (goal_ts - 60)
    ]
    kalshi_baseline = None
    if pre_kalshi:
        kalshi_baseline = sum(k["best_yes"] for k in pre_kalshi) / len(pre_kalshi)

    t_kalshi_first = None
    kalshi_first_move = 0.0
    if kalshi_baseline is not None:
        post_kalshi = [
            k for k in kalshi_timeline
            if (goal_ts - 60) < k["ts_wall"] < (goal_ts + window_after)
        ]
        for k in post_kalshi:
            move = abs(k["best_yes"] - kalshi_baseline)
            if move > 0.03 and t_kalshi_first is None:
                t_kalshi_first = k["ts_wall"]
                kalshi_first_move = k["best_yes"] - kalshi_baseline

    return {
        "goal_ts": goal_ts,
        "goal_utc": goal_utc,
        "score": goal.get("new_score"),
        "team": goal.get("team"),
        "t_goalserve": goal_ts,
        "t_odds_first": t_odds_first,
        "odds_first_bookie": odds_first_bookie,
        "odds_first_move": odds_first_move,
        "t_kalshi_first": t_kalshi_first,
        "kalshi_first_move": kalshi_first_move,
        "kalshi_baseline": kalshi_baseline,
        "odds_baselines": baselines,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Post-hoc latency analysis")
    parser.add_argument("match_dir", type=str, help="Path to recording directory")
    parser.add_argument("--event-id", type=str, help="Force Odds-API event ID (skip auto-detection)")
    parser.add_argument("--home-ticker", type=str, default="BRE", help="Kalshi ticker substring for home team")
    args = parser.parse_args()

    match_dir = Path(args.match_dir)
    if not match_dir.exists():
        print(f"ERROR: {match_dir} does not exist")
        sys.exit(1)

    # Load metadata
    meta_path = match_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print("=" * 70)
    print(f"POST-HOC LATENCY ANALYSIS — {match_dir.name}")
    print(f"  {meta.get('home_team', '?')} vs {meta.get('away_team', '?')}")
    print("=" * 70)

    # 1. Load goals from Goalserve events
    goals = load_goals(match_dir)
    print(f"\nGoals detected by Goalserve: {len(goals)}")
    for g in goals:
        print(f"  {_utc_str(g['ts_wall'])} — {g.get('team')} "
              f"{g.get('prev_score')}→{g.get('new_score')}")

    if not goals:
        print("\nNo goals to analyze.")
        return

    # 2. Identify Odds-API event IDs for this match
    if args.event_id:
        event_ids = {args.event_id}
        print(f"\nOdds-API event ID (forced): {event_ids}")
    else:
        event_ids = set()
        for hour in ["T20:00", "T19:00", "T18:00", "T17:00", "T15:00"]:
            event_ids = find_match_event_ids(match_dir, hour)
            if event_ids:
                print(f"\nOdds-API event IDs (kickoff {hour}): {event_ids}")
                break
        if not event_ids:
            print("\nWARNING: Could not identify Odds-API event IDs. Using all data.")

    # 3. Build timelines
    print("\nBuilding Odds-API timeline...")
    odds_tl = build_odds_timeline(match_dir, event_ids) if event_ids else []
    print(f"  {len(odds_tl)} ML price points")

    home_ticker_substr = args.home_ticker
    print(f"Building Kalshi orderbook timeline for *{home_ticker_substr}*...")
    kalshi_tl = build_kalshi_price_timeline(match_dir, home_ticker_substr)
    print(f"  {len(kalshi_tl)} price snapshots")

    # 4. Analyze each goal
    print(f"\n{'='*70}")
    print("GOAL-BY-GOAL ANALYSIS")
    print("=" * 70)

    cross_lags = []
    results = []

    for i, goal in enumerate(goals):
        result = analyze_goal(goal, odds_tl, kalshi_tl)
        results.append(result)

        print(f"\n--- Goal #{i+1}: {result['team']} {result['score']} at {result['goal_utc']} ---")

        # Goalserve
        print(f"  Goalserve detected:  {_utc_str(result['t_goalserve'])}")

        # Odds-API
        if result["t_odds_first"]:
            delay_from_gs = result["t_odds_first"] - result["t_goalserve"]
            print(f"  Odds-API first move: {_utc_str(result['t_odds_first'])} "
                  f"({result['odds_first_bookie']}, {result['odds_first_move']:+.3f}) "
                  f"[{delay_from_gs:+.1f}s from Goalserve]")
        else:
            print(f"  Odds-API first move: NOT DETECTED")

        # Kalshi
        if result["t_kalshi_first"]:
            delay_from_gs = result["t_kalshi_first"] - result["t_goalserve"]
            print(f"  Kalshi first move:   {_utc_str(result['t_kalshi_first'])} "
                  f"({result['kalshi_first_move']:+.3f}) "
                  f"[{delay_from_gs:+.1f}s from Goalserve]")
        else:
            print(f"  Kalshi first move:   NOT DETECTED (baseline={result.get('kalshi_baseline')})")

        # Cross-market lag
        if result["t_odds_first"] and result["t_kalshi_first"]:
            lag = result["t_kalshi_first"] - result["t_odds_first"]
            cross_lags.append(lag)
            print(f"\n  >>> CROSS-MARKET LAG (Odds-API → Kalshi): {lag:.1f}s <<<")
        elif result["t_odds_first"] and not result["t_kalshi_first"]:
            print(f"\n  >>> Odds-API moved but Kalshi didn't (Kalshi may have been faster or no data)")
        elif not result["t_odds_first"] and result["t_kalshi_first"]:
            print(f"\n  >>> Kalshi moved but Odds-API didn't detect for this match ID")

    # 5. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if cross_lags:
        cross_lags_sorted = sorted(cross_lags)
        median = cross_lags_sorted[len(cross_lags_sorted) // 2]
        mean = sum(cross_lags) / len(cross_lags)
        print(f"\n  Cross-market lag (Odds-API → Kalshi):")
        print(f"    Measurements: {len(cross_lags)}")
        print(f"    Median:       {median:.1f}s")
        print(f"    Mean:         {mean:.1f}s")
        print(f"    Min:          {min(cross_lags):.1f}s")
        print(f"    Max:          {max(cross_lags):.1f}s")
        print(f"    Values:       {[round(l, 1) for l in cross_lags]}")

        print(f"\n  VERDICT (architecture.md §3.7.5 Metric 1):")
        if median > 10:
            print(f"    STRONG EDGE — median lag {median:.1f}s > 10s")
        elif median > 3:
            print(f"    CAUTIOUS — median lag {median:.1f}s (3-10s range)")
        else:
            print(f"    NO EDGE — median lag {median:.1f}s < 3s")
    else:
        print("\n  No cross-market lag measurements available.")
        print("  Check if Odds-API event IDs were correctly identified.")

    # Save report
    report = {
        "match_id": meta.get("match_id", match_dir.name),
        "goals": len(goals),
        "cross_market_lags": cross_lags,
        "results": [
            {k: v for k, v in r.items() if k != "odds_baselines"}
            for r in results
        ],
    }
    report_path = match_dir / "latency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
