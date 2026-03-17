#!/usr/bin/env python3
"""
Test script: validate cross-source mapper for today's matches.

Usage:
    PYTHONPATH=. python scripts/test_cross_source_mapper.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

from src.clients.cross_source_mapper import map_all_sources, map_all_leagues, LEAGUES


async def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Cross-Source Mapper Test — {today}")
    print("=" * 120)

    # Check env vars
    for var in ("KALSHI_API_KEY", "ODDS_API_KEY", "GOALSERVE_API_KEY"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set")
            sys.exit(1)

    # Run for all 8 leagues
    all_results = await map_all_leagues()

    # Print results table
    print(f"\n{'League':<12} {'Home':<22} {'Away':<22} {'Kalshi Ticker':<32} {'OddsAPI ID':<14} {'GS ID':<10} {'Status':<20} {'Stale?'}")
    print("-" * 140)

    total = 0
    all_matched = 0
    missing_kalshi = 0
    missing_odds = 0
    missing_gs = 0
    stale_count = 0

    for league_id, results in sorted(all_results.items()):
        for r in results:
            total += 1
            status = r["match_status"]
            if status == "ALL_MATCHED":
                all_matched += 1
            elif status == "MISSING_KALSHI":
                missing_kalshi += 1
            elif status == "MISSING_ODDS_API":
                missing_odds += 1
            elif status == "MISSING_GOALSERVE":
                missing_gs += 1

            stale = "STALE!" if r.get("stale_status") else ""
            if stale:
                stale_count += 1

            k_tick = r["kalshi_event_ticker"] or "—"
            o_id = str(r["odds_api_event_id"] or "—")[:12]
            gs_id = r["goalserve_match_id"] or "—"

            print(
                f"{league_id:<12} {r['home_team']:<22} {r['away_team']:<22} "
                f"{k_tick:<32} {o_id:<14} {gs_id:<10} {status:<20} {stale}"
            )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print(f"  Total fixtures:     {total}")
    print(f"  ALL_MATCHED:        {all_matched}")
    print(f"  MISSING_KALSHI:     {missing_kalshi}")
    print(f"  MISSING_ODDS_API:   {missing_odds}")
    print(f"  MISSING_GOALSERVE:  {missing_gs}")
    print(f"  STALE_STATUS:       {stale_count}")

    # Print raw name details for non-matched entries
    print("\n" + "=" * 120)
    print("RAW NAMES FOR UNMATCHED ENTRIES")
    for league_id, results in sorted(all_results.items()):
        for r in results:
            if r["match_status"] == "ALL_MATCHED":
                continue
            print(f"\n  [{league_id}] {r['home_team']} vs {r['away_team']}  — {r['match_status']}")
            raw = r.get("raw_names", {})
            for key, val in sorted(raw.items()):
                print(f"    {key}: {val}")

    # Print stale entries
    if stale_count > 0:
        print("\n" + "=" * 120)
        print("STALE STATUS ENTRIES")
        for league_id, results in sorted(all_results.items()):
            for r in results:
                if r.get("stale_status"):
                    print(
                        f"  [{league_id}] {r['home_team']} vs {r['away_team']}"
                        f"  OddsAPI={r.get('_odds_api_status', '?')}"
                        f"  Goalserve={r.get('_goalserve_status', '?')}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
