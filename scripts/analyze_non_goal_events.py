#!/usr/bin/env python3
"""Analyze non-goal event reactions and microstructure from recorded match data.

Five analyses:
1. Non-goal bookmaker moves → Kalshi reaction lag
2. Halftime transition repricing
3. Late-game drift (80+ minutes)
4. Cross-correlation (does Odds-API systematically lead Kalshi?)
5. Kalshi spread analysis (is the market tradeable at all?)

Usage:
  PYTHONPATH=. python scripts/analyze_non_goal_events.py data/latency/4190023 --event-id 69385814
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


# ─── Data Loading (reused helpers) ───────────────────────────────────────────


def load_goals(md: Path) -> list[float]:
    ts = []
    p = md / "events.jsonl"
    if not p.exists():
        return ts
    with open(p) as f:
        for line in f:
            e = json.loads(line.strip())
            if e.get("type") == "goal":
                ts.append(e["ts_wall"])
    return ts


def load_halftime(md: Path) -> tuple[float, float] | None:
    ht_s = ht_e = None
    p = md / "events.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        for line in f:
            e = json.loads(line.strip())
            if e.get("type") != "status_change":
                continue
            if e.get("new_status") == "HT" and ht_s is None:
                ht_s = e["ts_wall"]
            if e.get("prev_status") == "HT" and ht_e is None:
                ht_e = e["ts_wall"]
    return (ht_s, ht_e) if ht_s and ht_e else None


def in_goal_window(ts: float, goals: list[float], window: float = 300.0) -> bool:
    return any(abs(ts - g) < window for g in goals)


def build_odds_by_bookie(
    md: Path, event_ids: set[str]
) -> dict[str, list[tuple[float, float]]]:
    """Returns {bookie: [(ts, implied), ...]} sorted by ts."""
    data: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(md / "odds_api.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if event_ids and d.get("id") not in event_ids:
                continue
            b = d.get("bookie", "")
            ts = d.get("_ts_wall", 0)
            if not b or not ts:
                continue
            for m in d.get("markets", []):
                if m.get("name") != "ML":
                    continue
                h = m.get("odds", [{}])[0].get("home")
                if not h:
                    continue
                try:
                    v = float(h)
                    if v > 1.0:
                        data[b].append((ts, 1.0 / v))
                except (ValueError, ZeroDivisionError):
                    pass
    for b in data:
        data[b].sort()
    return dict(data)


def build_kalshi_book_timeline(
    md: Path, ticker_sub: str
) -> list[tuple[float, float, float]]:
    """Returns [(ts, best_bid_yes, best_ask_yes), ...].

    best_bid = max(yes prices with qty > 0)
    best_ask = 1 - max(no prices with qty > 0)
    """
    yes_bk: dict[str, float] = {}
    no_bk: dict[str, float] = {}
    tl: list[tuple[float, float, float]] = []

    with open(md / "kalshi.jsonl") as f:
        for line in f:
            d = json.loads(line)
            msg = d.get("msg", d)
            if ticker_sub not in msg.get("market_ticker", ""):
                continue
            ts = d.get("_ts_wall", 0)
            mt = d.get("type", "")

            if mt == "orderbook_snapshot":
                yes_bk.clear()
                no_bk.clear()
                for p, q in msg.get("yes_dollars_fp", []):
                    if float(q) > 0:
                        yes_bk[p] = float(q)
                for p, q in msg.get("no_dollars_fp", []):
                    if float(q) > 0:
                        no_bk[p] = float(q)
            elif mt == "orderbook_delta":
                side = msg.get("side", "")
                p = msg.get("price_dollars", "")
                delta = float(msg.get("delta_fp", "0"))
                if not p:
                    continue
                bk = yes_bk if side == "yes" else no_bk
                cur = bk.get(p, 0)
                new = cur + delta
                if new > 0:
                    bk[p] = new
                else:
                    bk.pop(p, None)
            else:
                continue

            if yes_bk and no_bk:
                bid = max(float(p) for p in yes_bk)
                ask = 1.0 - max(float(p) for p in no_bk)
                tl.append((ts, bid, ask))

    return tl


# ─── Analysis 1: Non-Goal Odds Moves ────────────────────────────────────────


def analyze_non_goal_moves(
    odds_by_bookie: dict[str, list[tuple[float, float]]],
    kalshi_tl: list[tuple[float, float, float]],
    goals: list[float],
    halftime: tuple[float, float] | None,
) -> list[dict]:
    """Find >2% bookmaker moves outside goal windows, measure Kalshi lag."""

    # Detect moves per bookie
    moves: list[dict] = []
    for bookie, tl in odds_by_bookie.items():
        for i in range(1, len(tl)):
            ts, impl = tl[i]
            prev_ts, prev_impl = tl[i - 1]
            diff = impl - prev_impl

            # Skip if within 5min of goal or halftime
            if in_goal_window(ts, goals, 300):
                continue
            if halftime and halftime[0] - 60 <= ts <= halftime[1] + 60:
                continue

            # Skip tiny moves and Betfair bid/ask oscillation
            if abs(diff) < 0.02:
                continue
            # Skip if the move reversed within 30s (oscillation)
            if i + 1 < len(tl) and abs(tl[i + 1][1] - prev_impl) < 0.01:
                continue

            moves.append({
                "ts": ts,
                "bookie": bookie,
                "move": diff,
                "impl": impl,
            })

    moves.sort(key=lambda x: x["ts"])

    # Cluster moves within 30s into single events
    events: list[dict] = []
    for m in moves:
        if events and m["ts"] - events[-1]["ts"] < 30:
            events[-1]["bookies"].append(m["bookie"])
            events[-1]["moves"].append(m["move"])
        else:
            events.append({
                "ts": m["ts"],
                "bookies": [m["bookie"]],
                "moves": [m["move"]],
                "direction": "up" if m["move"] > 0 else "down",
            })

    # For each event, find Kalshi reaction
    results = []
    for evt in events:
        t_odds = evt["ts"]
        direction = evt["direction"]

        # Find Kalshi bid change >1.5¢ in same direction within 60s
        t_kalshi = None
        # Get baseline bid
        baseline_bid = None
        for ts, bid, ask in kalshi_tl:
            if ts > t_odds - 30:
                baseline_bid = bid
                break

        if baseline_bid is not None:
            threshold = 0.015
            for ts, bid, ask in kalshi_tl:
                if ts < t_odds - 10:
                    continue
                if ts > t_odds + 120:
                    break
                move = bid - baseline_bid
                if direction == "up" and move > threshold:
                    t_kalshi = ts
                    break
                elif direction == "down" and move < -threshold:
                    t_kalshi = ts
                    break

        lag = (t_kalshi - t_odds) if t_kalshi else None
        results.append({
            "ts": t_odds,
            "utc": _utc(t_odds),
            "direction": direction,
            "n_bookies": len(set(evt["bookies"])),
            "max_move": max(abs(m) for m in evt["moves"]),
            "lag": round(lag, 1) if lag is not None else None,
        })

    return results


# ─── Analysis 2: Halftime Transition ────────────────────────────────────────


def analyze_halftime(
    odds_by_bookie: dict[str, list[tuple[float, float]]],
    kalshi_tl: list[tuple[float, float, float]],
    halftime: tuple[float, float],
) -> dict:
    """Compare prices just before HT and just after 2H start."""
    ht_start, ht_end = halftime

    # Bookmaker implied probs: last value before HT, first after
    pre_ht: dict[str, float] = {}
    post_ht: dict[str, float] = {}
    for b, tl in odds_by_bookie.items():
        for ts, impl in tl:
            if ts < ht_start:
                pre_ht[b] = impl
            elif ts > ht_end and b not in post_ht:
                post_ht[b] = impl

    # Kalshi: last bid/ask before HT, first after
    k_pre_bid = k_pre_ask = k_post_bid = k_post_ask = None
    for ts, bid, ask in kalshi_tl:
        if ts < ht_start:
            k_pre_bid, k_pre_ask = bid, ask
        elif ts > ht_end and k_post_bid is None:
            k_post_bid, k_post_ask = bid, ask

    return {
        "pre_ht_bookies": pre_ht,
        "post_ht_bookies": post_ht,
        "k_pre_bid": k_pre_bid,
        "k_pre_ask": k_pre_ask,
        "k_post_bid": k_post_bid,
        "k_post_ask": k_post_ask,
    }


# ─── Analysis 3: Late-Game Drift ────────────────────────────────────────────


def analyze_late_game(
    odds_by_bookie: dict[str, list[tuple[float, float]]],
    kalshi_tl: list[tuple[float, float, float]],
    goals: list[float],
    ft_time: float | None,
) -> list[dict]:
    """Rolling 30s drift for the last 15 minutes of play."""
    if ft_time is None:
        return []

    # Use minute 75 to FT (roughly last 15 min)
    t_start = ft_time - 15 * 60
    t_end = ft_time

    buckets: list[dict] = []
    bookie_last: dict[str, float] = {}
    kalshi_bid_last: float | None = None
    o_idx = k_idx = 0

    # Flatten odds timeline
    all_odds = []
    for b, tl in odds_by_bookie.items():
        for ts, impl in tl:
            all_odds.append((ts, b, impl))
    all_odds.sort()

    for t in range(int(t_start), int(t_end), 10):
        tf = float(t)
        if in_goal_window(tf, goals, 180):
            continue

        while o_idx < len(all_odds) and all_odds[o_idx][0] <= tf:
            _, b, impl = all_odds[o_idx]
            bookie_last[b] = impl
            o_idx += 1
        while k_idx < len(kalshi_tl) and kalshi_tl[k_idx][0] <= tf:
            kalshi_bid_last = kalshi_tl[k_idx][1]
            k_idx += 1

        if bookie_last and kalshi_bid_last is not None:
            vals = sorted(bookie_last.values())
            consensus = vals[len(vals) // 2]
            drift = consensus - kalshi_bid_last
            buckets.append({"t": tf, "consensus": consensus, "kalshi_bid": kalshi_bid_last, "drift": drift})

    return buckets


# ─── Analysis 4: Cross-Correlation ──────────────────────────────────────────


def analyze_cross_correlation(
    odds_by_bookie: dict[str, list[tuple[float, float]]],
    kalshi_tl: list[tuple[float, float, float]],
    goals: list[float],
    halftime: tuple[float, float] | None,
) -> dict[int, float]:
    """Compute correlation between consensus changes and Kalshi bid changes at various lags."""

    # Build 10s-sampled series for consensus and Kalshi bid
    all_odds = []
    for b, tl in odds_by_bookie.items():
        for ts, impl in tl:
            all_odds.append((ts, b, impl))
    all_odds.sort()

    if not all_odds or not kalshi_tl:
        return {}

    t0 = max(all_odds[0][0], kalshi_tl[0][0])
    t1 = min(all_odds[-1][0], kalshi_tl[-1][0])

    bookie_last: dict[str, float] = {}
    k_bid_last: float | None = None
    o_idx = k_idx = 0

    consensus_series: list[tuple[float, float]] = []
    kalshi_series: list[tuple[float, float]] = []

    for t in range(int(t0), int(t1), 10):
        tf = float(t)
        if in_goal_window(tf, goals, 300):
            continue
        if halftime and halftime[0] <= tf <= halftime[1]:
            continue

        while o_idx < len(all_odds) and all_odds[o_idx][0] <= tf:
            _, b, impl = all_odds[o_idx]
            bookie_last[b] = impl
            o_idx += 1
        while k_idx < len(kalshi_tl) and kalshi_tl[k_idx][0] <= tf:
            k_bid_last = kalshi_tl[k_idx][1]
            k_idx += 1

        if bookie_last and k_bid_last is not None:
            vals = sorted(bookie_last.values())
            consensus = vals[len(vals) // 2]
            consensus_series.append((tf, consensus))
            kalshi_series.append((tf, k_bid_last))

    if len(consensus_series) < 20:
        return {}

    # Compute changes (first differences)
    cons_changes = [consensus_series[i + 1][1] - consensus_series[i][1] for i in range(len(consensus_series) - 1)]
    kal_changes = [kalshi_series[i + 1][1] - kalshi_series[i][1] for i in range(len(kalshi_series) - 1)]

    # Cross-correlate at various offsets
    results: dict[int, float] = {}
    for offset in range(-6, 7):  # -60s to +60s in 10s steps
        if abs(offset) >= len(cons_changes):
            continue
        if offset >= 0:
            c = cons_changes[:len(cons_changes) - offset]
            k = kal_changes[offset:]
        else:
            c = cons_changes[-offset:]
            k = kal_changes[:len(kal_changes) + offset]
        n = min(len(c), len(k))
        if n < 10:
            continue
        c, k = c[:n], k[:n]
        # Pearson correlation
        mc = sum(c) / n
        mk = sum(k) / n
        num = sum((c[i] - mc) * (k[i] - mk) for i in range(n))
        den_c = math.sqrt(sum((c[i] - mc) ** 2 for i in range(n)))
        den_k = math.sqrt(sum((k[i] - mk) ** 2 for i in range(n)))
        if den_c > 0 and den_k > 0:
            results[offset * 10] = round(num / (den_c * den_k), 4)

    return results


# ─── Analysis 5: Spread Analysis ────────────────────────────────────────────


def analyze_spread(
    kalshi_tl: list[tuple[float, float, float]],
    goals: list[float],
    halftime: tuple[float, float] | None,
) -> dict:
    """Compute Kalshi bid-ask spread statistics during normal play."""
    spreads: list[float] = []
    for ts, bid, ask in kalshi_tl:
        if in_goal_window(ts, goals, 180):
            continue
        if halftime and halftime[0] <= ts <= halftime[1]:
            continue
        spread = bid - ask  # should be positive (bid > ask for crossed/normal book)
        spreads.append(spread)

    if not spreads:
        return {}

    # Sample every 100th to avoid overweighting (255K entries)
    sampled = spreads[::100]
    sampled.sort()
    return {
        "n_samples": len(sampled),
        "mean_spread": sum(sampled) / len(sampled),
        "median_spread": sampled[len(sampled) // 2],
        "min_spread": min(sampled),
        "max_spread": max(sampled),
        "p25_spread": sampled[len(sampled) // 4],
        "p75_spread": sampled[3 * len(sampled) // 4],
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("match_dir", type=str)
    parser.add_argument("--event-id", type=str)
    parser.add_argument("--home-ticker", type=str, default="BRE")
    args = parser.parse_args()

    md = Path(args.match_dir)
    meta = {}
    if (md / "metadata.json").exists():
        with open(md / "metadata.json") as f:
            meta = json.load(f)

    print("=" * 70)
    print(f"NON-GOAL EVENT ANALYSIS — {md.name}")
    print(f"  {meta.get('home_team', '?')} vs {meta.get('away_team', '?')}")
    print("=" * 70)

    goals = load_goals(md)
    halftime = load_halftime(md)
    event_ids = {args.event_id} if args.event_id else set()

    print(f"\nGoals: {[_utc(g) for g in goals]}")
    if halftime:
        print(f"Halftime: {_utc(halftime[0])} — {_utc(halftime[1])}")

    print("\nBuilding timelines...")
    odds_bb = build_odds_by_bookie(md, event_ids)
    kalshi_tl = build_kalshi_book_timeline(md, args.home_ticker)
    total_odds = sum(len(v) for v in odds_bb.values())
    print(f"  Odds-API: {total_odds} prices across {len(odds_bb)} bookmakers")
    print(f"  Kalshi:   {len(kalshi_tl)} orderbook states")
    for b, tl in odds_bb.items():
        print(f"    {b}: {len(tl)} updates")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: NON-GOAL BOOKMAKER MOVES → KALSHI REACTION")
    print("=" * 70)

    nge = analyze_non_goal_moves(odds_bb, kalshi_tl, goals, halftime)
    if nge:
        print(f"\n  Non-goal events detected: {len(nge)}")
        for e in nge:
            lag_str = f"{e['lag']:+.1f}s" if e['lag'] is not None else "no reaction"
            print(f"    {e['utc']} {e['direction']} {e['max_move']*100:.1f}% "
                  f"({e['n_bookies']} bookie{'s' if e['n_bookies']>1 else ''}) → Kalshi lag: {lag_str}")

        lags = [e["lag"] for e in nge if e["lag"] is not None]
        if lags:
            lags_s = sorted(lags)
            print(f"\n  Measured lags: {len(lags)}")
            print(f"    Median: {lags_s[len(lags_s)//2]:.1f}s")
            print(f"    Mean:   {sum(lags)/len(lags):.1f}s")
            print(f"    Range:  [{min(lags):.1f}s, {max(lags):.1f}s]")
            n_pos = sum(1 for l in lags if l > 0)
            print(f"    Odds-API first: {n_pos}/{len(lags)} ({n_pos/len(lags)*100:.0f}%)")
        else:
            print("\n  No measurable Kalshi reactions to non-goal events.")
    else:
        print("\n  No significant non-goal bookmaker moves detected.")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: HALFTIME TRANSITION")
    print("=" * 70)

    if halftime:
        ht = analyze_halftime(odds_bb, kalshi_tl, halftime)
        print(f"\n  Pre-HT bookmaker implied (last values):")
        for b, v in ht["pre_ht_bookies"].items():
            print(f"    {b}: {v:.3f}")
        print(f"  Post-HT bookmaker implied (first values):")
        for b, v in ht["post_ht_bookies"].items():
            print(f"    {b}: {v:.3f}")
        if ht["k_pre_bid"] is not None:
            print(f"  Kalshi pre-HT:  bid={ht['k_pre_bid']:.2f} ask={ht['k_pre_ask']:.2f}")
            print(f"  Kalshi post-HT: bid={ht['k_post_bid']:.2f} ask={ht['k_post_ask']:.2f}")
            bid_change = (ht["k_post_bid"] or 0) - (ht["k_pre_bid"] or 0)
            print(f"  Kalshi bid change: {bid_change*100:+.1f}¢")
            # Compare consensus change vs Kalshi change
            pre_vals = list(ht["pre_ht_bookies"].values())
            post_vals = list(ht["post_ht_bookies"].values())
            if pre_vals and post_vals:
                pre_con = sorted(pre_vals)[len(pre_vals) // 2]
                post_con = sorted(post_vals)[len(post_vals) // 2]
                con_change = post_con - pre_con
                print(f"  Consensus change: {con_change*100:+.1f}¢")
                gap = con_change - bid_change
                print(f"  Gap (consensus moved more than Kalshi): {gap*100:+.1f}¢")
    else:
        print("\n  No halftime data.")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 3: LATE-GAME DRIFT (last 15 min)")
    print("=" * 70)

    # Get FT time
    ft_ts = None
    p = md / "events.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                e = json.loads(line.strip())
                if e.get("type") == "status_change" and e.get("new_status") == "FT":
                    ft_ts = e["ts_wall"]

    if ft_ts:
        late = analyze_late_game(odds_bb, kalshi_tl, goals, ft_ts)
        if late:
            drifts = [b["drift"] for b in late]
            abs_d = [abs(d) for d in drifts]
            print(f"\n  Buckets (10s): {len(late)} (excluding goal windows)")
            print(f"  Mean drift: {sum(drifts)/len(drifts)*100:+.1f}¢ "
                  f"({'bookmakers > Kalshi' if sum(drifts) > 0 else 'Kalshi > bookmakers'})")
            print(f"  Mean |drift|: {sum(abs_d)/len(abs_d)*100:.1f}¢")
            n_gt_3 = sum(1 for d in abs_d if d > 0.03)
            print(f"  |drift| > 3¢: {n_gt_3}/{len(late)} ({n_gt_3/len(late)*100:.0f}%)")
            print(f"\n  Time series:")
            for b in late[::3]:  # every 30s
                print(f"    {_utc(b['t'])} con={b['consensus']:.3f} "
                      f"bid={b['kalshi_bid']:.2f} drift={b['drift']*100:+.1f}¢")
        else:
            print("\n  No late-game data (goal windows may overlap).")
    else:
        print("\n  No FT timestamp found.")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 4: CROSS-CORRELATION (who leads?)")
    print("=" * 70)

    xcorr = analyze_cross_correlation(odds_bb, kalshi_tl, goals, halftime)
    if xcorr:
        print(f"\n  Correlation of 10s price changes at various lags:")
        print(f"  (positive offset = Odds-API leads by N seconds)")
        best_offset = max(xcorr, key=lambda k: xcorr[k])
        for offset in sorted(xcorr):
            marker = " <<<" if offset == best_offset else ""
            print(f"    {offset:+4d}s: r = {xcorr[offset]:+.4f}{marker}")
        print(f"\n  Peak correlation at offset {best_offset:+d}s (r={xcorr[best_offset]:.4f})")
        if best_offset > 0:
            print(f"  → Odds-API leads Kalshi by ~{best_offset}s")
        elif best_offset < 0:
            print(f"  → Kalshi leads Odds-API by ~{-best_offset}s")
        else:
            print(f"  → Simultaneous (no lead)")
    else:
        print("\n  Insufficient data for cross-correlation.")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 5: KALSHI SPREAD (is the market tradeable?)")
    print("=" * 70)

    sp = analyze_spread(kalshi_tl, goals, halftime)
    if sp:
        print(f"\n  Samples: {sp['n_samples']}")
        print(f"  Spread (bid - ask) statistics:")
        print(f"    Mean:   {sp['mean_spread']*100:.1f}¢")
        print(f"    Median: {sp['median_spread']*100:.1f}¢")
        print(f"    p25:    {sp['p25_spread']*100:.1f}¢")
        print(f"    p75:    {sp['p75_spread']*100:.1f}¢")
        print(f"    Range:  [{sp['min_spread']*100:.1f}¢, {sp['max_spread']*100:.1f}¢]")

        if sp["median_spread"] > 0.10:
            print(f"\n  WARNING: Median spread {sp['median_spread']*100:.0f}¢ is VERY WIDE.")
            print(f"  Any edge < {sp['median_spread']*100/2:.0f}¢ is consumed by the spread.")
        elif sp["median_spread"] > 0.03:
            print(f"\n  Spread is moderate ({sp['median_spread']*100:.0f}¢). Tradeable for edges > {sp['median_spread']*100:.0f}¢.")
        else:
            print(f"\n  Tight spread ({sp['median_spread']*100:.1f}¢). Market is liquid.")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"""
  | Analysis              | Finding                                    |
  |-----------------------|--------------------------------------------|""")

    # Non-goal events
    if nge:
        lags = [e["lag"] for e in nge if e["lag"] is not None]
        if lags:
            med = sorted(lags)[len(lags) // 2]
            n_pos = sum(1 for l in lags if l > 0)
            print(f"  | Non-goal moves        | {len(lags)} events, median lag {med:+.1f}s, {n_pos}/{len(lags)} Odds-API first |")
        else:
            print(f"  | Non-goal moves        | {len(nge)} events, no Kalshi reaction   |")
    else:
        print(f"  | Non-goal moves        | None detected                            |")

    # Halftime
    if halftime:
        print(f"  | Halftime repricing    | See above                                |")

    # Late game
    if ft_ts:
        late = analyze_late_game(odds_bb, kalshi_tl, goals, ft_ts)
        if late:
            d = [b["drift"] for b in late]
            print(f"  | Late-game drift       | mean {sum(d)/len(d)*100:+.1f}¢                               |")

    # Cross-correlation
    if xcorr:
        best = max(xcorr, key=lambda k: xcorr[k])
        print(f"  | Cross-correlation peak | {best:+d}s (r={xcorr[best]:.3f})                          |")

    # Spread
    if sp:
        print(f"  | Kalshi spread          | median {sp['median_spread']*100:.0f}¢                                  |")

    # Verdict
    print(f"\n  VERDICT:")
    tradeable = False
    if sp and sp["median_spread"] > 0.10:
        print(f"  The Kalshi spread ({sp['median_spread']*100:.0f}¢) is the dominant constraint.")
        print(f"  No edge strategy can overcome a {sp['median_spread']*100:.0f}¢ round-trip cost.")
    elif xcorr and max(xcorr, key=lambda k: xcorr[k]) > 0:
        best = max(xcorr, key=lambda k: xcorr[k])
        print(f"  Odds-API leads by {best}s — potential micro-edge if spread is tight.")
        tradeable = True
    else:
        print(f"  No systematic lead detected. Need more data points across matches.")

    # Save report
    report = {
        "match_id": meta.get("match_id", md.name),
        "non_goal_events": len(nge) if nge else 0,
        "cross_correlation": xcorr,
        "spread": sp,
    }
    with open(md / "non_goal_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {md / 'non_goal_report.json'}")


if __name__ == "__main__":
    main()
