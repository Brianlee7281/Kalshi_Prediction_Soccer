#!/usr/bin/env python3
"""Analyze Kalshi soccer market spreads across historical events.

Pulls trade data from Kalshi API for settled EPL markets and computes
effective spreads from consecutive trade prices. Determines whether
the spread is tight enough to trade.

Usage:
  set -a && source .env && set +a
  PYTHONPATH=. python scripts/analyze_spreads.py
  PYTHONPATH=. python scripts/analyze_spreads.py --use-cache  # skip API, use saved trades

Output: data/feasibility/spread_report.json + terminal summary
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ─── Config ──────────────────────────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com"
OUTPUT_DIR = Path("data/feasibility")
CACHE_DIR = OUTPUT_DIR / "trade_cache"

_PK = None


def _load_pk():
    global _PK
    if _PK is not None:
        return _PK
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
    if not Path(key_path).exists():
        key_path = "keys/kalshi_private.pem"
    with open(key_path, "rb") as f:
        _PK = serialization.load_pem_private_key(f.read(), password=None)
    return _PK


def kalshi_sign(method: str, path: str) -> dict[str, str]:
    pk = _load_pk()
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(pk.sign(
        (ts + method + path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )).decode()
    return {
        "KALSHI-ACCESS-KEY": os.environ.get("KALSHI_API_KEY", ""),
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def kalshi_get(path: str, client: httpx.Client) -> dict:
    headers = kalshi_sign("GET", path)
    r = client.get(KALSHI_BASE + path, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


# ─── Data Collection ─────────────────────────────────────────────────────────


def get_settled_events(client: httpx.Client, prefix: str, limit: int = 200) -> list[dict]:
    """Get settled markets for a series prefix."""
    all_markets: list[dict] = []
    cursor = None
    for _ in range(20):
        path = f"/trade-api/v2/markets?series_ticker={prefix}&status=settled&limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"
        data = kalshi_get(path, client)
        batch = data.get("markets", [])
        if not batch:
            break
        all_markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.15)
    return all_markets


def pull_trades(ticker: str, client: httpx.Client) -> list[dict]:
    """Pull all trades for a ticker. Returns sorted by time."""
    all_trades: list[dict] = []
    cursor = None
    for _ in range(50):
        path = f"/trade-api/v2/markets/trades?ticker={ticker}&limit=100"
        if cursor:
            path += f"&cursor={cursor}"
        data = kalshi_get(path, client)
        trades = data.get("trades", [])
        if not trades:
            break
        all_trades.extend(trades)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.1)
    all_trades.sort(key=lambda t: t.get("created_time", ""))
    return all_trades


def load_or_pull_trades(
    ticker: str, client: httpx.Client | None, use_cache: bool
) -> list[dict]:
    """Load from cache or pull from API."""
    cache_file = CACHE_DIR / f"{ticker}.json"
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    if client is None:
        return []
    trades = pull_trades(ticker, client)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(trades, f)
    return trades


# ─── Spread Computation ─────────────────────────────────────────────────────


def compute_spreads(trades: list[dict]) -> dict:
    """Compute spread metrics from trade history."""
    if len(trades) < 5:
        return {"n_trades": len(trades), "valid": False}

    prices = []
    timestamps = []
    for t in trades:
        p = t.get("yes_price_dollars") or t.get("yes_price")
        ts = t.get("created_time", "")
        if p is not None:
            prices.append(float(p))
            timestamps.append(ts)

    if len(prices) < 5:
        return {"n_trades": len(prices), "valid": False}

    # 1. Tick-to-tick spread (consecutive trade price diff)
    tick_spreads = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]

    # 2. Rolling window bid-ask proxy (60s windows)
    window_spreads: list[float] = []
    # Parse timestamps
    parsed_ts: list[float] = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            parsed_ts.append(dt.timestamp())
        except (ValueError, AttributeError):
            parsed_ts.append(0)

    if parsed_ts and parsed_ts[0] > 0:
        t_start = parsed_ts[0]
        t_end = parsed_ts[-1]
        window = 60.0
        t = t_start
        while t < t_end:
            window_prices = [
                prices[i] for i in range(len(prices))
                if parsed_ts[i] > 0 and t <= parsed_ts[i] < t + window
            ]
            if len(window_prices) >= 2:
                window_spreads.append(max(window_prices) - min(window_prices))
            t += window

    # 3. Duration
    duration_s = 0
    if parsed_ts and parsed_ts[0] > 0 and parsed_ts[-1] > 0:
        duration_s = parsed_ts[-1] - parsed_ts[0]

    return {
        "n_trades": len(prices),
        "valid": True,
        "duration_s": duration_s,
        "duration_h": round(duration_s / 3600, 1),
        "tick_spread_median": round(statistics.median(tick_spreads), 4) if tick_spreads else 0,
        "tick_spread_mean": round(statistics.mean(tick_spreads), 4) if tick_spreads else 0,
        "tick_spread_p25": round(sorted(tick_spreads)[len(tick_spreads) // 4], 4) if tick_spreads else 0,
        "tick_spread_p75": round(sorted(tick_spreads)[3 * len(tick_spreads) // 4], 4) if tick_spreads else 0,
        "window_spread_median": round(statistics.median(window_spreads), 4) if window_spreads else 0,
        "window_spread_mean": round(statistics.mean(window_spreads), 4) if window_spreads else 0,
        "n_windows": len(window_spreads),
    }


def extract_match_info(ticker: str) -> dict:
    """Extract date, day of week from ticker pattern: KX...-26MAR16BREWOL-BRE."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return {}
    date_part = parts[1]  # e.g., "26MAR16BREWOL"
    # Extract date: first 7 chars = season(2) + month(3) + day(2)
    if len(date_part) >= 7:
        try:
            month_str = date_part[2:5]
            day_str = date_part[5:7]
            months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                       "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
            month = months.get(month_str.upper(), 0)
            day = int(day_str)
            if month and day:
                year = 2025 if month >= 8 else 2026  # season 25/26
                dt = datetime(year, month, day)
                return {
                    "date": dt.strftime("%Y-%m-%d"),
                    "day_of_week": dt.strftime("%A"),
                    "is_weekend": dt.weekday() >= 5,
                }
        except (ValueError, KeyError):
            pass
    return {}


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi soccer spread analysis")
    parser.add_argument("--use-cache", action="store_true", help="Use cached trade data")
    parser.add_argument("--max-events", type=int, default=100, help="Max events to analyze")
    parser.add_argument("--league", type=str, default="KXEPLGAME", help="Series prefix")
    args = parser.parse_args()

    print("=" * 70)
    print(f"KALSHI SPREAD ANALYSIS — {args.league}")
    print("=" * 70)

    client = None
    if not args.use_cache:
        api_key = os.environ.get("KALSHI_API_KEY", "")
        if not api_key:
            print("ERROR: KALSHI_API_KEY not set. Use --use-cache or set env vars.")
            sys.exit(1)
        client = httpx.Client(timeout=15)

    # Step 1: Get settled markets
    print(f"\n[1] Fetching settled {args.league} markets...")
    if args.use_cache and (CACHE_DIR / "_markets.json").exists():
        with open(CACHE_DIR / "_markets.json") as f:
            markets = json.load(f)
        print(f"  Loaded {len(markets)} markets from cache")
    else:
        markets = get_settled_events(client, args.league)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "_markets.json", "w") as f:
            json.dump(markets, f)
        print(f"  Found {len(markets)} settled markets")

    # Group by event (3 outcomes per match)
    events: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        event_key = m.get("event_ticker", m["ticker"].rsplit("-", 1)[0])
        events[event_key].append(m)

    print(f"  {len(events)} unique events (matches)")

    # Step 2: Pull trades and compute spreads
    print(f"\n[2] Pulling trades (max {args.max_events} events)...")
    results: list[dict] = []
    event_keys = sorted(events.keys())[:args.max_events]

    for i, ek in enumerate(event_keys):
        mkt_list = events[ek]
        # Use first market (typically home-win) for spread analysis
        # But also check all 3 for the tightest
        best_spread = None
        best_ticker = None
        best_metrics = None

        for mkt in mkt_list:
            ticker = mkt["ticker"]
            trades = load_or_pull_trades(ticker, client, args.use_cache)
            metrics = compute_spreads(trades)

            if metrics["valid"]:
                ts = metrics["tick_spread_median"]
                if best_spread is None or ts < best_spread:
                    best_spread = ts
                    best_ticker = ticker
                    best_metrics = metrics

            if (i * len(mkt_list)) % 50 == 0 and not args.use_cache:
                time.sleep(0.2)  # rate limit

        if best_metrics:
            info = extract_match_info(best_ticker)
            results.append({
                "event": ek,
                "ticker": best_ticker,
                **best_metrics,
                **info,
            })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(event_keys)} events processed...")

    print(f"  {len(results)} events with valid spread data")

    if not results:
        print("\nNo data. Run without --use-cache to pull from API.")
        return

    # Step 3: Statistics
    print(f"\n{'='*70}")
    print("SPREAD DISTRIBUTION")
    print("=" * 70)

    tick_spreads = [r["tick_spread_median"] for r in results]
    window_spreads = [r["window_spread_median"] for r in results if r["window_spread_median"] > 0]

    tick_sorted = sorted(tick_spreads)
    n = len(tick_sorted)

    print(f"\n  Tick-to-tick spread (median per event):")
    print(f"    Mean:   {statistics.mean(tick_spreads)*100:.1f}¢")
    print(f"    Median: {statistics.median(tick_spreads)*100:.1f}¢")
    print(f"    P25:    {tick_sorted[n//4]*100:.1f}¢")
    print(f"    P75:    {tick_sorted[3*n//4]*100:.1f}¢")
    print(f"    Min:    {min(tick_spreads)*100:.1f}¢")
    print(f"    Max:    {max(tick_spreads)*100:.1f}¢")
    print(f"    <5¢:    {sum(1 for s in tick_spreads if s < 0.05)}/{n} ({sum(1 for s in tick_spreads if s < 0.05)/n*100:.0f}%)")
    print(f"    <10¢:   {sum(1 for s in tick_spreads if s < 0.10)}/{n} ({sum(1 for s in tick_spreads if s < 0.10)/n*100:.0f}%)")
    print(f"    <20¢:   {sum(1 for s in tick_spreads if s < 0.20)}/{n} ({sum(1 for s in tick_spreads if s < 0.20)/n*100:.0f}%)")

    if window_spreads:
        print(f"\n  60s window spread (median per event):")
        print(f"    Mean:   {statistics.mean(window_spreads)*100:.1f}¢")
        print(f"    Median: {statistics.median(window_spreads)*100:.1f}¢")

    # Top/bottom 10
    by_spread = sorted(results, key=lambda r: r["tick_spread_median"])
    print(f"\n  Top 10 TIGHTEST spread events:")
    for r in by_spread[:10]:
        print(f"    {r['tick_spread_median']*100:5.1f}¢  {r['n_trades']:>5} trades  "
              f"{r.get('day_of_week','?'):<10} {r['ticker']}")

    print(f"\n  Top 10 WIDEST spread events:")
    for r in by_spread[-10:]:
        print(f"    {r['tick_spread_median']*100:5.1f}¢  {r['n_trades']:>5} trades  "
              f"{r.get('day_of_week','?'):<10} {r['ticker']}")

    # Step 4: Volume vs spread
    print(f"\n{'='*70}")
    print("VOLUME vs SPREAD")
    print("=" * 70)

    # Bin by trade count
    bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 999999)]
    print(f"\n  {'Trades':<15} {'Events':<8} {'Med Spread':<12} {'Med Window Spread'}")
    for lo, hi in bins:
        in_bin = [r for r in results if lo <= r["n_trades"] < hi]
        if in_bin:
            ms = statistics.median([r["tick_spread_median"] for r in in_bin])
            ws = statistics.median([r["window_spread_median"] for r in in_bin if r["window_spread_median"] > 0]) if any(r["window_spread_median"] > 0 for r in in_bin) else 0
            label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
            print(f"  {label:<15} {len(in_bin):<8} {ms*100:.1f}¢{'':<8} {ws*100:.1f}¢")

    # Step 5: Day of week
    print(f"\n{'='*70}")
    print("DAY OF WEEK")
    print("=" * 70)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        day = r.get("day_of_week", "Unknown")
        by_day[day].append(r)

    print(f"\n  {'Day':<12} {'Events':<8} {'Med Tick Spread':<16} {'Med Trades'}")
    day_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in day_order:
        if day not in by_day:
            continue
        evts = by_day[day]
        ms = statistics.median([r["tick_spread_median"] for r in evts])
        mt = statistics.median([r["n_trades"] for r in evts])
        print(f"  {day:<12} {len(evts):<8} {ms*100:.1f}¢{'':<12} {mt:.0f}")

    weekend = [r for r in results if r.get("is_weekend")]
    midweek = [r for r in results if not r.get("is_weekend")]
    if weekend and midweek:
        we_s = statistics.median([r["tick_spread_median"] for r in weekend])
        mw_s = statistics.median([r["tick_spread_median"] for r in midweek])
        print(f"\n  Weekend median spread: {we_s*100:.1f}¢ ({len(weekend)} events)")
        print(f"  Midweek median spread: {mw_s*100:.1f}¢ ({len(midweek)} events)")

    # Step 6: Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    median_spread = statistics.median(tick_spreads)
    pct_under_5 = sum(1 for s in tick_spreads if s < 0.05) / n * 100

    if median_spread < 0.03:
        print(f"\n  TIGHT SPREADS — median {median_spread*100:.1f}¢ < 3¢")
        print(f"  Kalshi soccer is tradeable. Limit orders can capture 1-2¢ edges.")
    elif median_spread < 0.05:
        print(f"\n  MODERATE SPREADS — median {median_spread*100:.1f}¢")
        print(f"  Tradeable for edges > 5¢. Focus on high-volume matches.")
    elif pct_under_5 > 30:
        print(f"\n  MIXED — median {median_spread*100:.1f}¢ but {pct_under_5:.0f}% of events < 5¢")
        print(f"  Selectively tradeable. Filter for high-volume Saturday matches.")
    elif median_spread > 0.20:
        print(f"\n  UNTRADEABLE — median {median_spread*100:.1f}¢")
        print(f"  Kalshi soccer spreads are too wide for any edge strategy.")
    else:
        print(f"\n  MARGINAL — median {median_spread*100:.1f}¢")
        print(f"  Need limit orders inside the spread. Market-taking is unprofitable.")

    # Save
    report = {
        "league": args.league,
        "events_analyzed": len(results),
        "median_tick_spread": median_spread,
        "mean_tick_spread": statistics.mean(tick_spreads),
        "pct_under_5c": pct_under_5,
        "pct_under_10c": sum(1 for s in tick_spreads if s < 0.10) / n * 100,
        "results": results,
    }
    report_path = OUTPUT_DIR / "spread_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    if client:
        client.close()


if __name__ == "__main__":
    main()
