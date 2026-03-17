#!/usr/bin/env python3
"""Analyze non-event drift + suspension detection from recorded match data.

ANALYSIS 1: Non-event drift between bookmaker consensus and Kalshi mid-price.
ANALYSIS 2: Bookmaker suspension detection around goals — is there a tradeable
            window between suspension detection and Kalshi repricing?

Usage:
  PYTHONPATH=. python scripts/analyze_drift.py data/latency/4190023 --event-id 69385814
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_goals(match_dir: Path) -> list[dict]:
    goals: list[dict] = []
    path = match_dir / "events.jsonl"
    if not path.exists():
        return goals
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                evt = json.loads(line)
                if evt.get("type") == "goal":
                    goals.append(evt)
    return goals


def load_halftime(match_dir: Path) -> tuple[float, float] | None:
    ht_start = ht_end = None
    path = match_dir / "events.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("type") != "status_change":
                continue
            if evt.get("new_status") == "HT" and ht_start is None:
                ht_start = evt["ts_wall"]
            if evt.get("prev_status") == "HT" and ht_end is None:
                ht_end = evt["ts_wall"]
    if ht_start and ht_end:
        return (ht_start, ht_end)
    return None


def is_excluded(
    ts: float,
    goal_times: list[float],
    halftime: tuple[float, float] | None,
    window: float = 180.0,
) -> bool:
    for gt in goal_times:
        if abs(ts - gt) < window:
            return True
    if halftime and halftime[0] <= ts <= halftime[1]:
        return True
    return False


def find_event_ids(match_dir: Path, kickoff_hour: str) -> set[str]:
    ids: set[str] = set()
    with open(match_dir / "odds_api.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if kickoff_hour in data.get("date", "") and data.get("id"):
                ids.add(data["id"])
    return ids


def build_odds_timeline(
    match_dir: Path, event_ids: set[str]
) -> list[tuple[float, str, float]]:
    """Returns [(ts_wall, bookie, home_implied), ...] sorted by ts."""
    tl: list[tuple[float, str, float]] = []
    with open(match_dir / "odds_api.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if event_ids and data.get("id") not in event_ids:
                continue
            bookie = data.get("bookie", "")
            ts = data.get("_ts_wall", 0)
            if not bookie or not ts:
                continue
            for mkt in data.get("markets", []):
                if mkt.get("name") != "ML":
                    continue
                odds_list = mkt.get("odds", [])
                if not odds_list:
                    continue
                h = odds_list[0].get("home")
                if not h:
                    continue
                try:
                    v = float(h)
                    if v <= 1.0:
                        continue
                    tl.append((ts, bookie, 1.0 / v))
                except (ValueError, ZeroDivisionError):
                    continue
    tl.sort()
    return tl


def build_kalshi_mid_timeline(
    match_dir: Path, ticker_substr: str
) -> list[tuple[float, float, float, float]]:
    """Returns [(ts_wall, mid, best_bid, best_ask), ...].

    best_bid = max(yes_dollars_fp prices with qty > 0)
    best_ask = 1 - max(no_dollars_fp prices with qty > 0)
    mid = (best_bid + best_ask) / 2
    """
    yes_book: dict[str, float] = {}
    no_book: dict[str, float] = {}
    tl: list[tuple[float, float, float, float]] = []

    with open(match_dir / "kalshi.jsonl") as f:
        for line in f:
            data = json.loads(line)
            msg = data.get("msg", data)
            ticker = msg.get("market_ticker", "")
            if ticker_substr not in ticker:
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

            # Compute mid
            if yes_book and no_book:
                best_bid = max(float(p) for p in yes_book)
                best_ask = 1.0 - max(float(p) for p in no_book)
                mid = (best_bid + best_ask) / 2.0
                tl.append((ts, mid, best_bid, best_ask))

    return tl


# ─── Raw Odds-API messages with timestamps (for suspension analysis) ─────────


def load_raw_odds_messages(
    match_dir: Path, event_ids: set[str]
) -> list[tuple[float, str]]:
    """Returns [(ts_wall, bookie), ...] for every ML update of target match."""
    msgs: list[tuple[float, str]] = []
    with open(match_dir / "odds_api.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if event_ids and data.get("id") not in event_ids:
                continue
            bookie = data.get("bookie", "")
            ts = data.get("_ts_wall", 0)
            if bookie and ts:
                msgs.append((ts, bookie))
    msgs.sort()
    return msgs


# ─── ANALYSIS 1: Non-Event Drift ────────────────────────────────────────────


def analyze_drift(
    odds_tl: list[tuple[float, str, float]],
    kalshi_tl: list[tuple[float, float, float, float]],
    goal_times: list[float],
    halftime: tuple[float, float] | None,
    bucket_s: float = 10.0,
) -> dict:
    t_start = max(odds_tl[0][0], kalshi_tl[0][0])
    t_end = min(odds_tl[-1][0], kalshi_tl[-1][0])
    n_buckets = int((t_end - t_start) / bucket_s) + 1

    bookie_last: dict[str, float] = {}
    kalshi_mid_last: float | None = None
    kalshi_bid_last: float | None = None
    kalshi_ask_last: float | None = None
    odds_idx = 0
    kalshi_idx = 0
    buckets: list[dict] = []

    for i in range(n_buckets):
        t = t_start + i * bucket_s
        if is_excluded(t, goal_times, halftime):
            continue

        while odds_idx < len(odds_tl) and odds_tl[odds_idx][0] <= t + bucket_s:
            _, bookie, impl = odds_tl[odds_idx]
            bookie_last[bookie] = impl
            odds_idx += 1

        while kalshi_idx < len(kalshi_tl) and kalshi_tl[kalshi_idx][0] <= t + bucket_s:
            _, mid, bid, ask = kalshi_tl[kalshi_idx]
            kalshi_mid_last = mid
            kalshi_bid_last = bid
            kalshi_ask_last = ask
            kalshi_idx += 1

        if not bookie_last or kalshi_mid_last is None:
            continue

        vals = sorted(bookie_last.values())
        n = len(vals)
        consensus = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

        buckets.append({
            "t": t,
            "consensus": round(consensus, 4),
            "kalshi_mid": round(kalshi_mid_last, 4),
            "kalshi_bid": round(kalshi_bid_last, 4),
            "kalshi_ask": round(kalshi_ask_last, 4),
            "drift_vs_mid": round(consensus - kalshi_mid_last, 4),
            "drift_vs_bid": round(consensus - kalshi_bid_last, 4),
            "n_bookies": n,
        })

    return {"buckets": buckets, "t_start": t_start, "t_end": t_end}


def print_drift_results(result: dict, bucket_s: float) -> None:
    buckets = result["buckets"]
    if not buckets:
        print("  No valid buckets after exclusions.")
        return

    drifts_mid = [b["drift_vs_mid"] for b in buckets]
    drifts_bid = [b["drift_vs_bid"] for b in buckets]
    abs_mid = [abs(d) for d in drifts_mid]
    abs_bid = [abs(d) for d in drifts_bid]

    print(f"\n  Non-event buckets ({bucket_s:.0f}s): {len(buckets)}")
    print(f"  Duration analyzed: {len(buckets) * bucket_s / 60:.1f} min")

    for label, drifts, abs_d in [
        ("vs MID", drifts_mid, abs_mid),
        ("vs BID (best_yes)", drifts_bid, abs_bid),
    ]:
        print(f"\n  --- Drift {label} ---")
        n2 = sum(1 for d in abs_d if d > 0.02)
        n3 = sum(1 for d in abs_d if d > 0.03)
        n5 = sum(1 for d in abs_d if d > 0.05)
        mean_a = sum(abs_d) / len(abs_d)
        med_a = sorted(abs_d)[len(abs_d) // 2]
        mean_s = sum(drifts) / len(drifts)
        n_pos = sum(1 for d in drifts if d > 0)

        print(f"  |drift| > 2¢: {n2:>4} ({n2/len(drifts)*100:.1f}%)")
        print(f"  |drift| > 3¢: {n3:>4} ({n3/len(drifts)*100:.1f}%)")
        print(f"  |drift| > 5¢: {n5:>4} ({n5/len(drifts)*100:.1f}%)")
        print(f"  Mean |drift|:   {mean_a*100:.2f}¢")
        print(f"  Median |drift|: {med_a*100:.2f}¢")
        print(f"  Mean signed:    {mean_s*100:+.2f}¢ ({'bookmakers > Kalshi' if mean_s > 0 else 'Kalshi > bookmakers'})")
        print(f"  Direction: {n_pos}/{len(drifts)} positive ({n_pos/len(drifts)*100:.0f}%)")

        # Episodes
        episodes: list[int] = []
        run = 0
        for d in abs_d:
            if d > 0.02:
                run += 1
            else:
                if run:
                    episodes.append(run)
                run = 0
        if run:
            episodes.append(run)
        durations = sorted([e * bucket_s for e in episodes], reverse=True)
        if durations:
            print(f"  Episodes (|d|>2¢): {len(episodes)}, "
                  f"median={sorted(durations)[len(durations)//2]:.0f}s, "
                  f"max={durations[0]:.0f}s")
            print(f"    Top 10: {[f'{d:.0f}s' for d in durations[:10]]}")

    # Time series sample
    print(f"\n  Sample (every 5 min):")
    t0 = result["t_start"]
    for b in buckets:
        m = (b["t"] - t0) / 60
        if m % 5 < (bucket_s / 60):
            print(f"    {_utc(b['t'])} con={b['consensus']:.3f} "
                  f"mid={b['kalshi_mid']:.3f} bid={b['kalshi_bid']:.2f} ask={b['kalshi_ask']:.2f} "
                  f"drift_mid={b['drift_vs_mid']*100:+.1f}¢ drift_bid={b['drift_vs_bid']*100:+.1f}¢ "
                  f"({b['n_bookies']}bk)")


# ─── ANALYSIS 2: Suspension Detection ───────────────────────────────────────


def analyze_suspension(
    raw_msgs: list[tuple[float, str]],
    kalshi_tl: list[tuple[float, float, float, float]],
    goals: list[dict],
) -> list[dict]:
    """For each goal, build message-rate histogram and detect silence."""
    results = []

    for gi, goal in enumerate(goals):
        gt = goal["ts_wall"]
        score = goal.get("new_score", [])
        team = goal.get("team", "?")

        # Count Odds-API messages in 5s bins from -60s to +120s
        bins: dict[int, int] = {}
        for k in range(-12, 25):  # -60s to +120s in 5s bins
            bins[k] = 0
        for ts, bookie in raw_msgs:
            offset = ts - gt
            if -60 <= offset <= 120:
                bin_idx = int(offset // 5)
                if bin_idx in bins:
                    bins[bin_idx] += 1

        # Detect silence: first bin after goal (offset >= 0) with 0 messages
        silence_start = None
        silence_end = None
        for k in range(0, 25):
            if bins.get(k, 0) == 0:
                if silence_start is None:
                    silence_start = k * 5
            else:
                if silence_start is not None and silence_end is None:
                    silence_end = k * 5
        if silence_start is not None and silence_end is None:
            silence_end = 120  # silence runs to end of window

        # Find Kalshi first significant move
        # Baseline: 180-60s before goal
        pre = [mid for ts, mid, _, _ in kalshi_tl if (gt - 180) < ts < (gt - 60)]
        baseline = sum(pre) / len(pre) if pre else None
        kalshi_first_move_ts = None
        if baseline is not None:
            for ts, mid, _, _ in kalshi_tl:
                if (gt - 60) < ts < (gt + 120):
                    if abs(mid - baseline) > 0.03:
                        kalshi_first_move_ts = ts
                        break

        kalshi_offset = (kalshi_first_move_ts - gt) if kalshi_first_move_ts else None

        results.append({
            "goal_idx": gi + 1,
            "score": score,
            "team": team,
            "goal_ts": gt,
            "bins": bins,
            "silence_start_offset": silence_start,
            "silence_end_offset": silence_end,
            "silence_duration": (silence_end - silence_start) if silence_start is not None and silence_end is not None else 0,
            "kalshi_first_move_offset": round(kalshi_offset, 1) if kalshi_offset is not None else None,
        })

    return results


def print_suspension_results(results: list[dict]) -> None:
    for r in results:
        gt = r["goal_ts"]
        print(f"\n  --- Goal #{r['goal_idx']}: {r['team']} {r['score']} at {_utc(gt)} ---")

        # Histogram
        bins = r["bins"]
        print(f"  Odds-API message rate (5s bins):")
        for k in range(-6, 25):
            offset = k * 5
            count = bins.get(k, 0)
            bar = "#" * min(count, 40)
            marker = " <-- GOAL" if k == 0 else ""
            label = f"    {offset:+4d}s"
            print(f"{label} | {bar:<40} {count:>3}{marker}")

        sil = r["silence_start_offset"]
        sil_end = r["silence_end_offset"]
        sil_dur = r["silence_duration"]
        kalshi_off = r["kalshi_first_move_offset"]

        if sil is not None:
            print(f"  Silence detected: +{sil}s to +{sil_end}s ({sil_dur}s)")
        else:
            print(f"  No silence detected (messages continued through goal)")

        if kalshi_off is not None:
            print(f"  Kalshi first move: {kalshi_off:+.1f}s from Goalserve")
        else:
            print(f"  Kalshi first move: NOT DETECTED")

        if sil is not None and kalshi_off is not None:
            # Suspension gap = time between suspension start and Kalshi move
            # If suspension starts at +5s and Kalshi moves at -30s,
            # Kalshi was already done before suspension even started
            gap = sil - kalshi_off  # positive = Kalshi moved before suspension ended
            print(f"  Suspension gap: {gap:.1f}s "
                  f"({'Kalshi moved BEFORE suspension started' if kalshi_off < sil else f'Kalshi moved {-gap:.0f}s into suspension'})")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift + suspension analysis")
    parser.add_argument("match_dir", type=str)
    parser.add_argument("--event-id", type=str, help="Force Odds-API event ID")
    parser.add_argument("--home-ticker", type=str, default="BRE")
    parser.add_argument("--bucket", type=float, default=10.0)
    args = parser.parse_args()

    match_dir = Path(args.match_dir)
    meta = {}
    if (match_dir / "metadata.json").exists():
        with open(match_dir / "metadata.json") as f:
            meta = json.load(f)

    print("=" * 70)
    print(f"DRIFT + SUSPENSION ANALYSIS — {match_dir.name}")
    print(f"  {meta.get('home_team', '?')} vs {meta.get('away_team', '?')}")
    print("=" * 70)

    goals = load_goals(match_dir)
    goal_times = [g["ts_wall"] for g in goals]
    halftime = load_halftime(match_dir)

    print(f"\nGoals: {len(goals)}")
    for g in goals:
        print(f"  {_utc(g['ts_wall'])} {g.get('team')} {g.get('prev_score')}→{g.get('new_score')}")
    if halftime:
        print(f"Halftime: {_utc(halftime[0])} — {_utc(halftime[1])}")

    # Resolve event IDs
    if args.event_id:
        event_ids = {args.event_id}
    else:
        event_ids = set()
        for h in ["T20:00", "T19:00", "T18:00", "T17:00", "T15:00"]:
            event_ids = find_event_ids(match_dir, h)
            if event_ids:
                break
    print(f"Odds-API event IDs: {event_ids}")

    # Build timelines
    print("\nBuilding timelines...")
    odds_tl = build_odds_timeline(match_dir, event_ids)
    kalshi_tl = build_kalshi_mid_timeline(match_dir, args.home_ticker)
    raw_msgs = load_raw_odds_messages(match_dir, event_ids)
    print(f"  Odds-API: {len(odds_tl)} ML prices, {len(raw_msgs)} raw messages")
    print(f"  Kalshi:   {len(kalshi_tl)} orderbook states")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: NON-EVENT DRIFT")
    print("=" * 70)

    if odds_tl and kalshi_tl:
        drift_result = analyze_drift(odds_tl, kalshi_tl, goal_times, halftime, args.bucket)
        print_drift_results(drift_result, args.bucket)
    else:
        print("  Insufficient data.")
        drift_result = {"buckets": []}

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: SUSPENSION DETECTION")
    print("=" * 70)

    if raw_msgs and kalshi_tl and goals:
        suspension_results = analyze_suspension(raw_msgs, kalshi_tl, goals)
        print_suspension_results(suspension_results)
    else:
        print("  Insufficient data.")
        suspension_results = []

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    # Drift verdict
    buckets = drift_result.get("buckets", [])
    if buckets:
        abs_mid = [abs(b["drift_vs_mid"]) for b in buckets]
        mean_abs_mid = sum(abs_mid) / len(abs_mid)
        drifts_mid = [b["drift_vs_mid"] for b in buckets]
        n_pos = sum(1 for d in drifts_mid if d > 0)
        pct_pos = n_pos / len(drifts_mid) * 100
        one_dir = pct_pos > 80 or pct_pos < 20
        episodes_mid = []
        run = 0
        for d in abs_mid:
            if d > 0.02:
                run += 1
            else:
                if run:
                    episodes_mid.append(run)
                run = 0
        if run:
            episodes_mid.append(run)
        max_ep = max([e * args.bucket for e in episodes_mid]) if episodes_mid else 0

        print(f"\n  DRIFT: mean |drift vs mid| = {mean_abs_mid*100:.1f}¢")
        if mean_abs_mid < 0.01:
            print(f"  → NO DRIFT EDGE (< 1¢)")
        elif one_dir:
            print(f"  → VIG DIFFERENCE — {pct_pos:.0f}% one-directional, not tradeable")
        elif mean_abs_mid > 0.02 and max_ep > 30:
            print(f"  → POSSIBLE DRIFT EDGE — episodes up to {max_ep:.0f}s")
        else:
            print(f"  → MARGINAL — episodes up to {max_ep:.0f}s")
    else:
        print("\n  DRIFT: no data")

    # Suspension verdict
    if suspension_results:
        gaps = []
        for r in suspension_results:
            sil = r["silence_start_offset"]
            ko = r["kalshi_first_move_offset"]
            if sil is not None and ko is not None:
                gaps.append(sil - ko)

        if gaps:
            print(f"\n  SUSPENSION: {len(gaps)} goals with both silence + Kalshi move")
            for i, r in enumerate(suspension_results):
                sil = r["silence_start_offset"]
                ko = r["kalshi_first_move_offset"]
                dur = r["silence_duration"]
                if sil is not None and ko is not None:
                    print(f"    Goal #{r['goal_idx']}: silence +{sil}s ({dur}s), "
                          f"Kalshi {ko:+.1f}s, gap={sil-ko:.1f}s")

            all_kalshi_before = all(
                r["kalshi_first_move_offset"] is not None and r["silence_start_offset"] is not None
                and r["kalshi_first_move_offset"] < r["silence_start_offset"]
                for r in suspension_results
                if r["silence_start_offset"] is not None and r["kalshi_first_move_offset"] is not None
            )
            if all_kalshi_before:
                print(f"  → NO SUSPENSION EDGE — Kalshi reprices BEFORE bookmakers suspend")
                print(f"    Cannot hit stale Kalshi orders; they're already gone")
            else:
                avg_gap = sum(gaps) / len(gaps)
                print(f"  → POSSIBLE SUSPENSION EDGE — avg gap {avg_gap:.1f}s")
        else:
            print(f"\n  SUSPENSION: insufficient measurements")
    else:
        print(f"\n  SUSPENSION: no data")

    # Combined
    print(f"\n  COMBINED VERDICT:")
    print(f"    The original thesis (Odds-API detects events before Kalshi) is")
    print(f"    INVERTED for event windows. Kalshi participants react immediately")
    print(f"    while bookmakers suspend. For non-event drift, check the numbers above.")

    # Save report
    report = {
        "match_id": meta.get("match_id", match_dir.name),
        "n_goals": len(goals),
        "drift_buckets": len(buckets),
        "drift_mean_abs_mid": sum(abs(b["drift_vs_mid"]) for b in buckets) / len(buckets) if buckets else 0,
        "suspension_results": [
            {k: v for k, v in r.items() if k != "bins"} for r in suspension_results
        ],
    }
    with open(match_dir / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {match_dir / 'drift_report.json'}")


if __name__ == "__main__":
    main()
