"""Batch-record multiple live matches concurrently.

Usage:
  # Show today's open matches (no recording)
  PYTHONPATH=. python scripts/record_batch.py --discover --league EPL
  PYTHONPATH=. python scripts/record_batch.py --discover --league EPL --league LaLiga

  # Auto-discover and record all open matches for a league
  PYTHONPATH=. python scripts/record_batch.py --league EPL

  # Multiple leagues
  PYTHONPATH=. python scripts/record_batch.py --league EPL --league LaLiga

  # From a JSON config file
  PYTHONPATH=. python scripts/record_batch.py --config matches.json

Config file format:
  {"matches": [
    {"match_id": "KXEPLGAME-26MAR21FULBUR", "league": "EPL"},
    {"match_id": "KXEPLGAME-26MAR21MCIARS", "league": "EPL"}
  ]}

Required environment variables:
  KALSHI_API_KEY
  KALSHI_PRIVATE_KEY_PATH
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import structlog

from src.clients.kalshi import KalshiClient
from src.prematch.phase2_pipeline import LEAGUE_PREFIXES

log = structlog.get_logger("record_batch")

# Import LEAGUE_IDS and run_live from run_phase3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase3 import LEAGUE_IDS, run_live  # noqa: E402


async def _discover_matches(
    kalshi_rest: KalshiClient,
    league: str,
) -> list[dict]:
    """Find all open GAME events for a league on Kalshi.

    Returns list of dicts with event_ticker, title, expected_expiration_time, league.
    """
    league_id = LEAGUE_IDS.get(league)
    if league_id is None:
        log.error("unknown_league", league=league, known=list(LEAGUE_IDS.keys()))
        return []

    prefix = LEAGUE_PREFIXES.get(league_id)
    if prefix is None:
        log.error("no_kalshi_prefix", league=league, league_id=league_id)
        return []

    markets = await kalshi_rest.get_markets(series_ticker=prefix, status="open")

    # Group by event_ticker, keep first market's metadata per event
    seen: dict[str, dict] = {}
    for m in markets:
        et = m["event_ticker"]
        if et not in seen:
            title = m.get("title", "")
            # Extract team names: "Tottenham vs Nottingham Winner?" → "Tottenham vs Nottingham"
            match_name = title
            for suffix in (" Winner?", " Game"):
                if match_name.endswith(suffix):
                    match_name = match_name[: -len(suffix)].strip()
            seen[et] = {
                "event_ticker": et,
                "title": match_name,
                "expected_expiration_time": m.get("expected_expiration_time", ""),
                "league": league,
            }

    result = sorted(seen.values(), key=lambda x: x["expected_expiration_time"])
    log.info("discovered_matches", league=league, count=len(result))
    return result


def _print_discovered(all_matches: list[dict]) -> None:
    """Print a table of discovered matches to stdout."""
    et_tz = ZoneInfo("America/New_York")
    now = datetime.now(et_tz)
    today_str = now.strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append(f"\n{'='*75}")
    lines.append(f"  Open Kalshi Soccer Markets  ({today_str})")
    lines.append(f"{'='*75}")
    lines.append(f"  {'League':<10} {'Kickoff (ET)':<20} {'Match':<25} {'Event Ticker'}")
    lines.append(f"  {'-'*72}")

    today_count = 0
    for m in all_matches:
        exp = m["expected_expiration_time"]
        # Parse and convert to Eastern Time
        try:
            dt = datetime.fromisoformat(exp.replace("Z", "+00:00")).astimezone(et_tz)
            kickoff_str = dt.strftime("%Y-%m-%d %H:%M")
            is_today = dt.strftime("%Y-%m-%d") == today_str
        except (ValueError, AttributeError):
            kickoff_str = exp[:16] if exp else "?"
            is_today = False

        marker = " *" if is_today else ""
        if is_today:
            today_count += 1
        lines.append(
            f"  {m['league']:<10} {kickoff_str:<20} {m['title']:<25} {m['event_ticker']}{marker}"
        )

    lines.append(f"\n  Total: {len(all_matches)} matches  |  Today: {today_count}")
    lines.append(f"  (* = scheduled today)\n")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


async def _run_batch(matches: list[tuple[str, str]]) -> None:
    """Run concurrent recordings for a list of (match_id, league) pairs."""
    if not matches:
        log.warning("no_matches_to_record")
        return

    log.info("batch_start", count=len(matches))
    for match_id, league in matches:
        log.info("batch_match", match_id=match_id, league=league)

    tasks = [
        asyncio.create_task(
            _run_one(match_id, league),
            name=match_id,
        )
        for match_id, league in matches
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (match_id, league), result in zip(matches, results):
        if isinstance(result, Exception):
            log.error("match_failed", match_id=match_id, error=str(result))
        else:
            log.info("match_done", match_id=match_id)

    log.info("batch_done", total=len(matches))


async def _run_one(match_id: str, league: str) -> None:
    """Wrapper around run_live with error logging."""
    try:
        await run_live(match_id, league)
    except Exception:
        log.exception("run_live_error", match_id=match_id)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-record multiple live matches concurrently."
    )
    parser.add_argument(
        "--league", action="append", default=[],
        help="League code (e.g. EPL). Can be repeated. Auto-discovers open matches.",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to JSON config file listing matches.",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="List open matches without recording. Pair with --league.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    # --discover mode: list open matches and exit
    if args.discover:
        if not args.league:
            log.error("discover_needs_league", hint="Use --discover --league EPL")
            sys.exit(1)
        api_key = os.environ.get("KALSHI_API_KEY", "")
        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        kalshi_rest = KalshiClient(
            api_key=api_key, private_key_path=private_key_path,
        )
        all_discovered: list[dict] = []
        try:
            for league in args.league:
                all_discovered.extend(await _discover_matches(kalshi_rest, league))
        finally:
            await kalshi_rest.close()
        _print_discovered(all_discovered)
        return

    # Recording mode: collect matches from --config and/or --league
    matches: list[tuple[str, str]] = []

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            log.error("config_not_found", path=str(config_path))
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        for entry in config.get("matches", []):
            matches.append((entry["match_id"], entry["league"]))

    if args.league:
        api_key = os.environ.get("KALSHI_API_KEY", "")
        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        kalshi_rest = KalshiClient(
            api_key=api_key, private_key_path=private_key_path,
        )
        try:
            for league in args.league:
                discovered = await _discover_matches(kalshi_rest, league)
                for entry in discovered:
                    matches.append((entry["event_ticker"], entry["league"]))
        finally:
            await kalshi_rest.close()

    if not matches:
        log.error("no_matches", hint="Use --league or --config")
        sys.exit(1)

    # Deduplicate by match_id
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for match_id, league in matches:
        if match_id not in seen:
            seen.add(match_id)
            unique.append((match_id, league))

    await _run_batch(unique)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
