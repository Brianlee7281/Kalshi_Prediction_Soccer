#!/usr/bin/env python3
"""
Sprint -1: Feasibility Study
=============================
Measures three things:
  Q1: Is there enough Kalshi liquidity to trade?
  Q2: How much do Kalshi prices move on goals?
  Q3: Are there stale trades after goals?

Usage:
  cd ~/Documents/GitHub/FKT-v4
  python scripts/sprint_minus_1.py

Requires: httpx, cryptography (pip install httpx cryptography)
Output:  data/feasibility/report.txt + per-league CSVs
"""

import os
import sys
import time
import json
import base64
import glob
import csv
import statistics
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ─── Config ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
KEYS_DIR = PROJECT_ROOT / "keys"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "feasibility"

KALSHI_BASE = "https://api.elections.kalshi.com"
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "4c74b24e-cf2c-4235-ba54-aacbba60dd79")
KALSHI_KEY_PATH = KEYS_DIR / "kalshi_private.pem"

LEAGUES = {
    "EPL":         {"kalshi_prefix": "KXEPLGAME",         "goalserve_id": "1204"},
    "LaLiga":      {"kalshi_prefix": "KXLALIGAGAME",      "goalserve_id": "1399"},
    "SerieA":      {"kalshi_prefix": "KXSERIEAGAME",      "goalserve_id": "1269"},
    "Bundesliga":  {"kalshi_prefix": "KXBUNDESLIGAGAME",  "goalserve_id": "1229"},
    "Ligue1":      {"kalshi_prefix": "KXLIGUE1GAME",      "goalserve_id": "1221"},
    "MLS":         {"kalshi_prefix": "KXMLSGAME",         "goalserve_id": "1440"},
    "Brasileirao": {"kalshi_prefix": "KXBRASILEIROGAME",  "goalserve_id": "1141"},
    "Argentina":   {"kalshi_prefix": "KXARGPREMDIVGAME",  "goalserve_id": "1081"},
}

# ─── Kalshi Auth ──────────────────────────────────────────────────────────────

def load_private_key():
    with open(KALSHI_KEY_PATH, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

PK = None  # lazy load

def signed_get(path: str, client: httpx.Client) -> httpx.Response:
    global PK
    if PK is None:
        PK = load_private_key()
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(PK.sign(
        (ts + "GET" + path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )).decode()
    return client.get(KALSHI_BASE + path, headers={
        "KALSHI-ACCESS-KEY": KALSHI_API_KEY,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    })

# ─── Step 1: Collect settled Kalshi markets ───────────────────────────────────

def collect_settled_markets(client: httpx.Client) -> dict[str, list[dict]]:
    """Returns {league_name: [market_dicts]} for settled GAME markets."""
    results = {}
    for league, cfg in LEAGUES.items():
        prefix = cfg["kalshi_prefix"]
        print(f"  Fetching settled markets: {prefix}...", end=" ", flush=True)
        markets = []
        cursor = None
        for _ in range(50):  # max 50 pages
            path = f"/trade-api/v2/markets?limit=100&status=settled&series_ticker={prefix}"
            if cursor:
                path += f"&cursor={cursor}"
            r = signed_get(path, client)
            if r.status_code != 200:
                print(f"ERROR {r.status_code}")
                break
            batch = r.json().get("markets", [])
            if not batch:
                break
            markets.extend(batch)
            cursor = r.json().get("cursor")
            if not cursor:
                break
            time.sleep(0.1)  # rate limit courtesy
        print(f"{len(markets)} markets")
        results[league] = markets
    return results

# ─── Step 2: Pull trades for each market ──────────────────────────────────────

def pull_trades(ticker: str, client: httpx.Client) -> list[dict]:
    """Pull all trades for a single ticker. Returns sorted by time."""
    all_trades = []
    cursor = None
    for _ in range(20):  # max 2000 trades per market
        path = f"/trade-api/v2/markets/trades?ticker={ticker}&limit=100"
        if cursor:
            path += f"&cursor={cursor}"
        r = signed_get(path, client)
        if r.status_code != 200:
            break
        trades = r.json().get("trades", [])
        if not trades:
            break
        all_trades.extend(trades)
        cursor = r.json().get("cursor")
        if not cursor:
            break
        time.sleep(0.05)
    all_trades.sort(key=lambda t: t.get("created_time", ""))
    return all_trades

# ─── Step 3: Load Goalserve commentaries (local files) ────────────────────────

def load_goalserve_goals() -> dict[str, list[dict]]:
    """
    Loads goal data from local commentaries files.
    Returns {goalserve_league_id: [{match_date, home, away, goals: [{minute, team}]}]}
    """
    commentaries_dir = DATA_DIR / "commentaries"
    all_goals = defaultdict(list)

    for json_file in sorted(commentaries_dir.glob("**/*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Navigate to tournament/match level
        # Handle both dict and list format
        if isinstance(data, list):
            continue
        tournament = data.get("commentaries", {}).get("tournament", {})
        if not tournament:
            continue

        league_id = tournament.get("@id", "")
        matches = tournament.get("match", [])
        if not isinstance(matches, list):
            matches = [matches]

        for m in matches:
            match_date = m.get("@formatted_date", "")
            home = m.get("localteam", {}).get("@name", "")
            away = m.get("visitorteam", {}).get("@name", "")

            goals = []
            summary = m.get("summary", {})
            if not summary:
                continue

            for team_key in ["localteam", "visitorteam"]:
                team_goals = summary.get(team_key, {}).get("goals", {})
                if not team_goals:
                    continue
                players = team_goals.get("player", [])
                if not isinstance(players, list):
                    players = [players]
                for p in players:
                    minute_str = p.get("@minute", "")
                    try:
                        # Handle "90+5" format
                        if "+" in str(minute_str):
                            parts = str(minute_str).split("+")
                            minute = int(parts[0]) + int(parts[1])
                        else:
                            minute = int(minute_str)
                        goals.append({"minute": minute, "team": team_key})
                    except (ValueError, IndexError):
                        continue

            if goals:
                all_goals[league_id].append({
                    "match_date": match_date,
                    "home": home,
                    "away": away,
                    "goals": sorted(goals, key=lambda g: g["minute"]),
                })

    return dict(all_goals)

# ─── Step 4: Analysis ─────────────────────────────────────────────────────────

def analyze_event(event_ticker: str, event_markets: list[dict],
                  all_trade_data: dict[str, list[dict]]) -> dict:
    """
    Analyze a single Kalshi event (3 outcome markets for one match).
    Returns liquidity and trade density stats.
    """
    total_trades = 0
    total_volume = 0.0
    all_prices = []
    all_timestamps = []

    for market in event_markets:
        ticker = market["ticker"]
        trades = all_trade_data.get(ticker, [])
        total_trades += len(trades)
        for t in trades:
            try:
                price = float(t.get("yes_price_dollars", 0))
                qty = float(t.get("count_fp", 0))
                total_volume += price * qty
                all_prices.append(price)
                all_timestamps.append(t.get("created_time", ""))
            except (ValueError, TypeError):
                continue

    # Compute spread from consecutive trades on the home_win market (first outcome)
    spreads = []
    home_ticker = [m["ticker"] for m in event_markets if m["ticker"].endswith(event_markets[0]["ticker"].split("-")[-1])]
    if home_ticker:
        home_trades = all_trade_data.get(home_ticker[0], [])
        for i in range(1, len(home_trades)):
            try:
                p1 = float(home_trades[i-1].get("yes_price_dollars", 0))
                p2 = float(home_trades[i].get("yes_price_dollars", 0))
                spreads.append(abs(p2 - p1))
            except (ValueError, TypeError):
                continue

    return {
        "event_ticker": event_ticker,
        "total_trades": total_trades,
        "total_volume_dollars": round(total_volume, 2),
        "avg_spread": round(statistics.mean(spreads), 4) if spreads else None,
        "n_markets": len(event_markets),
    }


def analyze_price_impact(trades: list[dict], goal_minute: int,
                         kickoff_utc_approx: str) -> dict | None:
    """
    For a single outcome market + goal event, measure price impact.
    Returns dict with pre/post prices and stale trade info.

    Note: goal_minute has ±2-3min uncertainty due to minute-level precision
    and kickoff/halftime timing uncertainty.
    """
    if len(trades) < 5:
        return None

    # Parse trade timestamps
    parsed = []
    for t in trades:
        try:
            ts_str = t.get("created_time", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            price = float(t.get("yes_price_dollars", 0))
            qty = float(t.get("count_fp", 0))
            parsed.append({"ts": ts, "price": price, "qty": qty, "raw": t})
        except (ValueError, TypeError):
            continue

    if not parsed:
        return None

    # Estimate goal UTC time (very approximate: kickoff + goal_minute + halftime if >45)
    # We can't be precise — this is the limitation documented in architecture.md
    try:
        kickoff_ts = datetime.fromisoformat(kickoff_utc_approx.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    halftime_offset = 15 * 60 if goal_minute > 45 else 0  # ~15min halftime
    goal_ts_approx = kickoff_ts.timestamp() + (goal_minute * 60) + halftime_offset

    # Window: 3 min before goal to 5 min after goal
    pre_window_start = goal_ts_approx - 180  # 3min before
    pre_window_end = goal_ts_approx          # goal time
    post_window_start = goal_ts_approx + 120  # 2min after (skip immediate chaos)
    post_window_end = goal_ts_approx + 300    # 5min after

    pre_trades = [p for p in parsed if pre_window_start <= p["ts"].timestamp() <= pre_window_end]
    post_trades = [p for p in parsed if post_window_start <= p["ts"].timestamp() <= post_window_end]

    # "Stale" trades: trades in the 5min after goal that are within 2¢ of pre-goal price
    stale_window_trades = [p for p in parsed
                           if pre_window_end <= p["ts"].timestamp() <= post_window_end]

    if not pre_trades or not post_trades:
        return None

    pre_price = statistics.mean([p["price"] for p in pre_trades])
    post_price = statistics.mean([p["price"] for p in post_trades])
    price_impact = abs(post_price - pre_price)

    # Count stale trades (within 2¢ of pre-goal price, after goal)
    stale_count = 0
    stale_volume = 0.0
    for p in stale_window_trades:
        if abs(p["price"] - pre_price) <= 0.02:
            stale_count += 1
            stale_volume += p["price"] * p["qty"]

    return {
        "goal_minute": goal_minute,
        "pre_price": round(pre_price, 4),
        "post_price": round(post_price, 4),
        "price_impact": round(price_impact, 4),
        "trades_in_window": len(stale_window_trades),
        "stale_count": stale_count,
        "stale_volume_dollars": round(stale_volume, 2),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Sprint -1: Feasibility Study")
    print("=" * 60)

    # ── Phase A: Collect Kalshi data ──────────────────────────────────────────
    print("\n[Phase A] Collecting settled Kalshi markets...")
    client = httpx.Client(timeout=30)

    markets_by_league = collect_settled_markets(client)

    total_markets = sum(len(v) for v in markets_by_league.values())
    print(f"\nTotal settled markets: {total_markets}")

    if total_markets == 0:
        print("\nNo settled markets found. Kalshi may not have historical soccer data yet.")
        print("Try checking if markets exist with 'active' status instead.")

        # Fallback: check active markets
        print("\n[Fallback] Checking active markets...")
        for league, cfg in LEAGUES.items():
            r = signed_get(f"/trade-api/v2/markets?limit=5&series_ticker={cfg['kalshi_prefix']}", client)
            count = len(r.json().get("markets", []))
            if count:
                print(f"  {league}: {count}+ active markets")

        client.close()
        return

    # ── Phase B: Pull trades per event ────────────────────────────────────────
    print("\n[Phase B] Pulling trade histories...")

    # Group markets by event (3 outcomes per match)
    events_by_league = {}
    for league, markets in markets_by_league.items():
        events = defaultdict(list)
        for m in markets:
            event_ticker = m.get("event_ticker", m["ticker"].rsplit("-", 1)[0])
            events[event_ticker].append(m)
        events_by_league[league] = dict(events)
        print(f"  {league}: {len(events)} events ({len(markets)} markets)")

    # Pull trades for all markets (this is the slow part)
    all_trade_data = {}  # ticker → [trades]
    total_api_calls = sum(len(m) for m in markets_by_league.values())
    call_count = 0

    for league, markets in markets_by_league.items():
        print(f"\n  Pulling trades for {league}...", flush=True)
        for m in markets:
            ticker = m["ticker"]
            call_count += 1
            if call_count % 20 == 0:
                print(f"    [{call_count}/{total_api_calls}] {ticker[:40]}...", flush=True)
            trades = pull_trades(ticker, client)
            all_trade_data[ticker] = trades
            time.sleep(0.05)

    client.close()

    total_trades = sum(len(v) for v in all_trade_data.values())
    print(f"\nTotal trades collected: {total_trades}")

    # ── Phase C: Load Goalserve goals ─────────────────────────────────────────
    print("\n[Phase C] Loading Goalserve commentaries...")
    goalserve_goals = load_goalserve_goals()
    for lid, matches in goalserve_goals.items():
        total_goals = sum(len(m["goals"]) for m in matches)
        print(f"  League {lid}: {len(matches)} matches, {total_goals} goals")

    # ── Phase D: Q1 — Liquidity analysis ──────────────────────────────────────
    print("\n[Phase D] Q1: Liquidity analysis...")

    liquidity_report = []
    for league, events in events_by_league.items():
        league_stats = []
        for event_ticker, event_markets in events.items():
            stats = analyze_event(event_ticker, event_markets, all_trade_data)
            league_stats.append(stats)

        trades_per_event = [s["total_trades"] for s in league_stats]
        volumes = [s["total_volume_dollars"] for s in league_stats]

        if trades_per_event:
            report = {
                "league": league,
                "events": len(league_stats),
                "median_trades": round(statistics.median(trades_per_event)),
                "p25_trades": round(sorted(trades_per_event)[len(trades_per_event)//4]) if len(trades_per_event) >= 4 else 0,
                "p75_trades": round(sorted(trades_per_event)[3*len(trades_per_event)//4]) if len(trades_per_event) >= 4 else 0,
                "median_volume": round(statistics.median(volumes), 2),
                "pct_under_20_trades": round(100 * sum(1 for t in trades_per_event if t < 20) / len(trades_per_event), 1),
            }
        else:
            report = {
                "league": league, "events": 0, "median_trades": 0,
                "p25_trades": 0, "p75_trades": 0, "median_volume": 0,
                "pct_under_20_trades": 100,
            }
        liquidity_report.append(report)

    # ── Phase E: Q2+Q3 — Price impact + stale trades ─────────────────────────
    print("\n[Phase E] Q2+Q3: Price impact and stale trades...")

    # This is a rough analysis due to team name matching limitations
    # We analyze the first outcome market per event for price movement
    impact_results = []

    for league, events in events_by_league.items():
        gs_id = LEAGUES[league]["goalserve_id"]
        gs_matches = goalserve_goals.get(gs_id, [])

        for event_ticker, event_markets in events.items():
            # Get the first outcome market's trades
            if not event_markets:
                continue
            first_market = event_markets[0]
            trades = all_trade_data.get(first_market["ticker"], [])
            if len(trades) < 10:
                continue

            # Try to find matching Goalserve match by title
            title = first_market.get("title", "")
            # Extract team names from title (e.g., "Lazio vs Milan Winner?")
            # and try to match with Goalserve

            # For now, analyze price jumps regardless of Goalserve matching
            # A price jump >5¢ in the trade series likely indicates a goal
            for i in range(1, len(trades)):
                try:
                    p_prev = float(trades[i-1].get("yes_price_dollars", 0))
                    p_curr = float(trades[i].get("yes_price_dollars", 0))
                    jump = abs(p_curr - p_prev)

                    if jump >= 0.05:  # 5¢+ jump = likely event
                        ts_prev = trades[i-1].get("created_time", "")
                        ts_curr = trades[i].get("created_time", "")

                        # Count trades at "old" price after the jump
                        stale_after = 0
                        for j in range(i+1, min(i+20, len(trades))):
                            p_j = float(trades[j].get("yes_price_dollars", 0))
                            if abs(p_j - p_prev) <= 0.02:
                                stale_after += 1

                        impact_results.append({
                            "league": league,
                            "ticker": first_market["ticker"],
                            "jump_cents": round(jump * 100, 1),
                            "price_before": p_prev,
                            "price_after": p_curr,
                            "time_before": ts_prev[:23],
                            "time_after": ts_curr[:23],
                            "stale_trades_after": stale_after,
                        })
                except (ValueError, TypeError):
                    continue

    # ── Phase F: Write reports ────────────────────────────────────────────────
    print("\n[Phase F] Writing reports...")

    # Liquidity CSV
    liquidity_path = OUTPUT_DIR / "liquidity_by_league.csv"
    with open(liquidity_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["league", "events", "median_trades",
                                               "p25_trades", "p75_trades",
                                               "median_volume", "pct_under_20_trades"])
        writer.writeheader()
        writer.writerows(liquidity_report)

    # Price impact CSV
    impact_path = OUTPUT_DIR / "price_jumps.csv"
    if impact_results:
        with open(impact_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=impact_results[0].keys())
            writer.writeheader()
            writer.writerows(impact_results)

    # Summary report
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Sprint -1 Feasibility Study Report\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Q1: LIQUIDITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'League':<14} {'Events':>6} {'Med.Trades':>10} {'P25':>5} {'P75':>5} {'Med.$Vol':>10} {'<20 trades':>10}\n")
        for r in liquidity_report:
            f.write(f"{r['league']:<14} {r['events']:>6} {r['median_trades']:>10} "
                    f"{r['p25_trades']:>5} {r['p75_trades']:>5} "
                    f"${r['median_volume']:>9.2f} {r['pct_under_20_trades']:>9.1f}%\n")

        f.write(f"\nTotal events: {sum(r['events'] for r in liquidity_report)}\n")
        f.write(f"Total trades: {total_trades}\n")

        liquid_leagues = [r["league"] for r in liquidity_report if r["median_trades"] >= 50]
        thin_leagues = [r["league"] for r in liquidity_report if 20 <= r["median_trades"] < 50]
        dead_leagues = [r["league"] for r in liquidity_report if r["median_trades"] < 20]

        f.write(f"\nLiquid (≥50 trades/event): {', '.join(liquid_leagues) or 'none'}\n")
        f.write(f"Thin (20-49 trades/event): {', '.join(thin_leagues) or 'none'}\n")
        f.write(f"Dead (<20 trades/event):    {', '.join(dead_leagues) or 'none'}\n")

        f.write("\n\nQ2+Q3: PRICE IMPACT & STALE TRADES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total price jumps ≥5¢ detected: {len(impact_results)}\n")

        if impact_results:
            jumps = [r["jump_cents"] for r in impact_results]
            stales = [r["stale_trades_after"] for r in impact_results]
            f.write(f"Median jump size: {statistics.median(jumps):.1f}¢\n")
            f.write(f"Mean jump size: {statistics.mean(jumps):.1f}¢\n")
            f.write(f"Max jump size: {max(jumps):.1f}¢\n")
            f.write(f"\nStale trades (within 2¢ of pre-jump price, in next 20 trades):\n")
            f.write(f"  Events with stale trades: {sum(1 for s in stales if s > 0)}/{len(stales)}\n")
            f.write(f"  Mean stale count: {statistics.mean(stales):.1f}\n")
            f.write(f"  Max stale count: {max(stales)}\n")

            # Top 10 biggest jumps
            f.write(f"\nTop 10 price jumps:\n")
            for r in sorted(impact_results, key=lambda x: x["jump_cents"], reverse=True)[:10]:
                f.write(f"  {r['league']:<12} {r['jump_cents']:>5.1f}¢  "
                        f"{r['price_before']:.2f}→{r['price_after']:.2f}  "
                        f"stale={r['stale_trades_after']}  "
                        f"{r['time_before']}\n")

        f.write("\n\nDECISION\n")
        f.write("-" * 40 + "\n")
        if liquid_leagues:
            f.write(f"PROCEED — {len(liquid_leagues)} leagues have sufficient liquidity.\n")
            f.write(f"Target leagues: {', '.join(liquid_leagues)}\n")
        elif thin_leagues:
            f.write(f"PROCEED WITH CAUTION — {len(thin_leagues)} leagues have marginal liquidity.\n")
            f.write(f"Target leagues: {', '.join(thin_leagues)}\n")
        else:
            f.write("PAUSE — No leagues have sufficient liquidity.\n")

        if impact_results and statistics.mean([r["jump_cents"] for r in impact_results]) >= 5:
            f.write("Price impacts are large enough to be interesting (≥5¢ average).\n")
        elif impact_results:
            f.write(f"Price impacts are small ({statistics.mean([r['jump_cents'] for r in impact_results]):.1f}¢ avg) — edge may be thin.\n")

        stale_events = sum(1 for r in impact_results if r["stale_trades_after"] > 0) if impact_results else 0
        if impact_results and stale_events / len(impact_results) > 0.3:
            f.write(f"Stale trades detected in {stale_events}/{len(impact_results)} events — promising signal.\n")
        else:
            f.write("Few stale trades detected — precise lag measurement deferred to Sprint 3 recording.\n")

    # Print summary to console
    print("\n" + "=" * 60)
    with open(report_path) as f:
        print(f.read())

    print(f"\nFiles written:")
    print(f"  {liquidity_path}")
    print(f"  {impact_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
