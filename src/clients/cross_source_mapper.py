"""Cross-source mapper: Kalshi ↔ Odds-API ↔ Goalserve.

Matches fixtures across all 3 data sources using normalized team names
and date proximity (±1 day for timezone edge cases).

Usage:
    from src.clients.cross_source_mapper import map_all_sources, LEAGUES

    results = await map_all_sources(kalshi_client, odds_api_client, goalserve_client, "EPL")
    for r in results:
        print(r["home_team"], "vs", r["away_team"], "—", r["match_status"])
"""

from __future__ import annotations

import asyncio
import time
import base64
import os
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Any

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.calibration.team_aliases import normalize_team_name

log = structlog.get_logger()

# ─── League Configuration ────────────────────────────────────────────────────

LEAGUES: dict[str, dict[str, Any]] = {
    "EPL": {
        "kalshi_prefix": "KXEPLGAME",
        "odds_api_slug": "england-premier-league",
        "goalserve_id": 1204,
    },
    "La Liga": {
        "kalshi_prefix": "KXLALIGAGAME",
        "odds_api_slug": "spain-laliga",
        "goalserve_id": 1399,
    },
    "Serie A": {
        "kalshi_prefix": "KXSERIEAGAME",
        "odds_api_slug": "italy-serie-a",
        "goalserve_id": 1269,
    },
    "Bundesliga": {
        "kalshi_prefix": "KXBUNDESLIGAGAME",
        "odds_api_slug": "germany-bundesliga",
        "goalserve_id": 1229,
    },
    "Ligue 1": {
        "kalshi_prefix": "KXLIGUE1GAME",
        "odds_api_slug": "france-ligue-1",
        "goalserve_id": 1221,
    },
    "MLS": {
        "kalshi_prefix": "KXMLSGAME",
        "odds_api_slug": "usa-mls",
        "goalserve_id": 1440,
    },
    "Brasileirao": {
        "kalshi_prefix": "KXBRASILEIROGAME",
        "odds_api_slug": "brazil-brasileiro-serie-a",
        "goalserve_id": 1141,
    },
    "Argentina": {
        "kalshi_prefix": "KXARGPREMDIVGAME",
        "odds_api_slug": "argentina-liga-profesional",
        "goalserve_id": 1081,
    },
}


# ─── Kalshi Auth Helpers ──────────────────────────────────────────────────────

_PK = None


def _load_pk() -> Any:
    global _PK
    if _PK is None:
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
        with open(key_path, "rb") as f:
            _PK = serialization.load_pem_private_key(f.read(), password=None)
    return _PK


def _kalshi_sign(method: str, path: str) -> dict[str, str]:
    pk = _load_pk()
    api_key = os.environ.get("KALSHI_API_KEY", "")
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(
        pk.sign(
            (ts + method + path).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    ).decode()
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ─── Internal Helpers ─────────────────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com"
ODDS_API_BASE = "https://api.odds-api.io/v3"
GOALSERVE_BASE = "https://www.goalserve.com/getfeed"


def _extract_date_from_kalshi_ticker(event_ticker: str) -> str | None:
    """Extract ISO date from Kalshi event ticker. e.g. 26MAR16 -> 2026-03-16"""
    parts = event_ticker.split("-", 1)
    if len(parts) < 2 or len(parts[1]) < 7:
        return None

    date_part = parts[1][:7]
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    try:
        year = 2000 + int(date_part[:2])
        month = months.get(date_part[2:5].upper())
        day = int(date_part[5:7])
        if month is None:
            return None
        return f"{year}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def _extract_teams_from_kalshi_title(title: str) -> tuple[str, str]:
    """Extract (home, away) team names from Kalshi market title.

    Handles patterns like:
    - "Arsenal vs Chelsea: Arsenal"
    - "Brentford vs Wolverhampton Winner?"
    """
    if " vs " not in title:
        return ("", "")
    vs_parts = title.split(" vs ", 1)
    home = vs_parts[0].strip()
    away_part = vs_parts[1]
    if ":" in away_part:
        away_part = away_part.split(":")[0]
    away_part = away_part.split("?")[0].strip()
    # Remove trailing "Winner" suffix from Kalshi titles
    if away_part.lower().endswith(" winner"):
        away_part = away_part[: -len(" winner")].strip()
    return (home, away_part)


def _normalize_goalserve_date(date_str: str) -> str:
    """Parse Goalserve date to YYYY-MM-DD."""
    if not date_str or date_str == "?":
        return ""
    if len(date_str) == 10 and date_str[4] == "-":
        return date_str
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _dates_match(d1: str, d2: str) -> bool:
    """Check if dates match within ±1 day (timezone edge cases)."""
    if not d1 or not d2:
        return True
    if d1 == d2:
        return True
    try:
        dt1 = datetime.strptime(d1, "%Y-%m-%d")
        dt2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((dt1 - dt2).days) <= 1
    except ValueError:
        return False


def _teams_match_normalized(
    h1: str, a1: str, h2: str, a2: str,
) -> bool:
    """Check if two (home, away) pairs match after normalization, trying both orderings."""
    n_h1, n_a1 = normalize_team_name(h1), normalize_team_name(a1)
    n_h2, n_a2 = normalize_team_name(h2), normalize_team_name(a2)
    # Exact ordering
    if n_h1 == n_h2 and n_a1 == n_a2:
        return True
    # Swapped ordering
    if n_h1 == n_a2 and n_a1 == n_h2:
        return True
    return False


def _goalserve_status_to_phase(status: str) -> str:
    """Convert Goalserve match status to a standard phase string."""
    if status in ("FT", "AET", "Pen."):
        return "FINISHED"
    if status == "HT":
        return "HALFTIME"
    if status in ("NS", "Postp.", "Canc.", "Susp.", "?", ""):
        return "SCHEDULED"
    try:
        minute = int(status)
        if minute <= 45:
            return "FIRST_HALF"
        return "SECOND_HALF"
    except ValueError:
        return "UNKNOWN"


# ─── Fetch Functions ──────────────────────────────────────────────────────────


async def _fetch_kalshi_markets(
    client: httpx.AsyncClient,
    series_ticker: str,
) -> list[dict]:
    """Fetch ALL open Kalshi markets for a series ticker with full pagination."""
    all_markets: list[dict] = []
    cursor: str | None = None

    while True:
        path = f"/trade-api/v2/markets?limit=200&status=open&series_ticker={series_ticker}"
        if cursor:
            path += f"&cursor={cursor}"

        headers = _kalshi_sign("GET", path)
        try:
            r = await asyncio.wait_for(
                client.get(KALSHI_BASE + path, headers=headers),
                timeout=15,
            )
        except Exception as e:
            log.error("kalshi_fetch_error", series=series_ticker, error=str(e))
            break

        if r.status_code != 200:
            log.error("kalshi_api_error", series=series_ticker, status=r.status_code)
            break

        data = r.json()
        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        await asyncio.sleep(0.15)

    return all_markets


async def _fetch_odds_api_events(
    client: httpx.AsyncClient,
    slug: str,
) -> list[dict]:
    """Fetch ALL pending + live events from Odds-API for a league slug."""
    api_key = os.environ.get("ODDS_API_KEY", "")
    all_events: list[dict] = []

    for status in ("pending", "live"):
        url = (
            f"{ODDS_API_BASE}/events"
            f"?sport=football&league={slug}&status={status}"
            f"&apiKey={api_key}"
        )
        try:
            r = await asyncio.wait_for(client.get(url), timeout=15)
        except Exception as e:
            log.error("odds_api_fetch_error", slug=slug, status=status, error=str(e))
            continue

        if r.status_code != 200:
            log.error("odds_api_error", slug=slug, status_code=r.status_code)
            continue

        events = r.json()
        if isinstance(events, list):
            for e in events:
                e["_status_filter"] = status
                # Normalize field names: API uses "home"/"away", not "home_team"/"away_team"
                if "home" in e and "home_team" not in e:
                    e["home_team"] = e["home"]
                if "away" in e and "away_team" not in e:
                    e["away_team"] = e["away"]
                # Normalize date field: API uses "date", not "commence_time"
                if "date" in e and "commence_time" not in e:
                    e["commence_time"] = e["date"]
            all_events.extend(events)

        await asyncio.sleep(0.3)

    return all_events


async def _fetch_goalserve_matches(
    client: httpx.AsyncClient,
    league_id: int,
) -> list[dict]:
    """Fetch Goalserve matches via commentaries endpoint (today + next 7 days)
    plus live scores for real-time status."""
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    matches: list[dict] = []
    seen_ids: set[str] = set()

    # Commentaries endpoint: has scheduled + live + finished fixtures
    now = datetime.now(timezone.utc)
    dates_to_check = [
        (now + timedelta(days=d)).strftime("%d.%m.%Y") for d in range(8)
    ]

    for date_str in dates_to_check:
        url = (
            f"{GOALSERVE_BASE}/{api_key}/commentaries/{league_id}"
            f"?date={date_str}&json=1"
        )
        try:
            r = await asyncio.wait_for(client.get(url), timeout=15)
        except Exception as e:
            log.warning("goalserve_commentaries_error", date=date_str, error=str(e))
            continue

        if r.status_code != 200:
            continue

        data = r.json()
        comms = data.get("commentaries", data)
        tournament = comms.get("tournament", {})
        day_matches = tournament.get("match", [])
        if isinstance(day_matches, dict):
            day_matches = [day_matches]

        for m in day_matches:
            fix_id = str(m.get("@fix_id") or m.get("@id") or m.get("@static_id") or "")
            if fix_id in seen_ids or not fix_id:
                continue
            seen_ids.add(fix_id)

            home_name = m.get("localteam", {}).get("@name", "")
            away_name = m.get("visitorteam", {}).get("@name", "")
            match_status = str(m.get("@status", m.get("status", "")))
            match_date = m.get("@date", m.get("@formatted_date", ""))

            matches.append({
                "fix_id": fix_id,
                "home_team": home_name,
                "away_team": away_name,
                "status": match_status,
                "date": _normalize_goalserve_date(match_date),
                "raw_date": match_date,
            })

        await asyncio.sleep(0.15)

    # Also check live scores endpoint for real-time status updates
    url = f"{GOALSERVE_BASE}/{api_key}/soccernew/home?json=1"
    try:
        r = await asyncio.wait_for(client.get(url), timeout=15)
        if r.status_code == 200:
            data = r.json()
            scores = data.get("scores", data)
            categories = scores.get("category", [])
            if isinstance(categories, dict):
                categories = [categories]

            target_id = str(league_id)
            for cat in categories:
                if str(cat.get("@id", "")) != target_id:
                    continue

                cat_matches = cat.get("match", [])
                if isinstance(cat_matches, dict):
                    cat_matches = [cat_matches]

                for m in cat_matches:
                    fix_id = str(m.get("@fix_id") or m.get("@id") or m.get("@static_id") or "")
                    live_status = str(m.get("@status", ""))

                    # Update existing match status
                    for existing in matches:
                        if existing["fix_id"] == fix_id:
                            existing["status"] = live_status
                            break
                    else:
                        # New live match not in commentaries
                        if fix_id and fix_id not in seen_ids:
                            seen_ids.add(fix_id)
                            matches.append({
                                "fix_id": fix_id,
                                "home_team": m.get("localteam", {}).get("@name", ""),
                                "away_team": m.get("visitorteam", {}).get("@name", ""),
                                "status": live_status,
                                "date": _normalize_goalserve_date(m.get("@date", "")),
                                "raw_date": m.get("@date", ""),
                            })
    except Exception as e:
        log.warning("goalserve_live_error", error=str(e))

    return matches


# ─── Main Mapping Function ───────────────────────────────────────────────────


async def map_all_sources(
    league_id: str,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """
    Cross-reference all 3 sources for a league.

    Args:
        league_id: League display name (e.g. "EPL", "La Liga").
        client: Optional shared httpx.AsyncClient. Created if not provided.

    Returns:
        List of matched fixtures with:
        - kalshi_event_ticker, kalshi_tickers (dict of market tickers)
        - odds_api_event_id
        - goalserve_match_id (fix_id)
        - home_team, away_team (canonical normalized names)
        - kickoff_utc (from Odds-API commence_time, if available)
        - match_status: ALL_MATCHED | MISSING_KALSHI | MISSING_ODDS_API | MISSING_GOALSERVE
        - stale_status: True if Odds-API says live but Goalserve says FT

    Uses full pagination for Kalshi (not limit=30).
    Uses normalize_team_name for all matching.
    Tries both home/away and away/home orderings.
    """
    league_config = LEAGUES.get(league_id)
    if not league_config:
        log.error("unknown_league", league_id=league_id)
        return []

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20)

    try:
        # Fetch from all 3 sources concurrently
        kalshi_markets, odds_events, gs_matches = await asyncio.gather(
            _fetch_kalshi_markets(client, league_config["kalshi_prefix"]),
            _fetch_odds_api_events(client, league_config["odds_api_slug"]),
            _fetch_goalserve_matches(client, league_config["goalserve_id"]),
        )

        log.info(
            "sources_fetched",
            league=league_id,
            kalshi_markets=len(kalshi_markets),
            odds_events=len(odds_events),
            gs_matches=len(gs_matches),
        )

        # Group Kalshi markets by event ticker
        kalshi_events: dict[str, dict] = {}
        for m in kalshi_markets:
            et = m.get("event_ticker", m["ticker"].rsplit("-", 1)[0])
            if et not in kalshi_events:
                title = m.get("title", "")
                home_raw, away_raw = _extract_teams_from_kalshi_title(title)
                date_iso = _extract_date_from_kalshi_ticker(et)
                kalshi_events[et] = {
                    "event_ticker": et,
                    "home_raw": home_raw,
                    "away_raw": away_raw,
                    "date": date_iso or "",
                    "tickers": {},
                }
            kalshi_events[et]["tickers"][m["ticker"]] = m

        # Build result records
        results: list[dict] = []

        # Track matched indices
        odds_matched: set[int] = set()
        gs_matched: set[int] = set()

        # For each Kalshi event, find matching Odds-API and Goalserve entries
        for et, k_info in kalshi_events.items():
            record: dict = {
                "kalshi_event_ticker": et,
                "kalshi_tickers": {t: m.get("title", "") for t, m in k_info["tickers"].items()},
                "odds_api_event_id": None,
                "goalserve_match_id": None,
                "home_team": normalize_team_name(k_info["home_raw"]),
                "away_team": normalize_team_name(k_info["away_raw"]),
                "kickoff_utc": None,
                "match_status": "MISSING_ODDS_API",
                "stale_status": False,
                "raw_names": {
                    "kalshi_home": k_info["home_raw"],
                    "kalshi_away": k_info["away_raw"],
                },
            }

            # Match with Odds-API
            for oi, oe in enumerate(odds_events):
                if oi in odds_matched:
                    continue
                o_date = oe.get("commence_time", "")[:10]
                if not _dates_match(k_info["date"], o_date):
                    continue
                if _teams_match_normalized(
                    k_info["home_raw"], k_info["away_raw"],
                    oe.get("home_team", ""), oe.get("away_team", ""),
                ):
                    record["odds_api_event_id"] = oe.get("id")
                    record["kickoff_utc"] = oe.get("commence_time")
                    record["raw_names"]["odds_api_home"] = oe.get("home_team", "")
                    record["raw_names"]["odds_api_away"] = oe.get("away_team", "")
                    record["_odds_api_status"] = oe.get("status", "")
                    odds_matched.add(oi)
                    break

            # Match with Goalserve
            for gi, gm in enumerate(gs_matches):
                if gi in gs_matched:
                    continue
                if not _dates_match(k_info["date"], gm["date"]):
                    continue
                if _teams_match_normalized(
                    k_info["home_raw"], k_info["away_raw"],
                    gm["home_team"], gm["away_team"],
                ):
                    record["goalserve_match_id"] = gm["fix_id"]
                    record["raw_names"]["goalserve_home"] = gm["home_team"]
                    record["raw_names"]["goalserve_away"] = gm["away_team"]
                    record["_goalserve_status"] = gm["status"]
                    gs_matched.add(gi)
                    break

            # Determine match_status
            has_odds = record["odds_api_event_id"] is not None
            has_gs = record["goalserve_match_id"] is not None
            if has_odds and has_gs:
                record["match_status"] = "ALL_MATCHED"
            elif has_odds:
                record["match_status"] = "MISSING_GOALSERVE"
            elif has_gs:
                record["match_status"] = "MISSING_ODDS_API"
            else:
                record["match_status"] = "MISSING_ODDS_API"

            # Stale status detection (Step 8)
            odds_st = record.get("_odds_api_status", "")
            gs_st = record.get("_goalserve_status", "")
            if odds_st == "live" and _goalserve_status_to_phase(gs_st) == "FINISHED":
                record["stale_status"] = True

            results.append(record)

        # Add unmatched Odds-API events
        for oi, oe in enumerate(odds_events):
            if oi in odds_matched:
                continue

            record = {
                "kalshi_event_ticker": None,
                "kalshi_tickers": {},
                "odds_api_event_id": oe.get("id"),
                "goalserve_match_id": None,
                "home_team": normalize_team_name(oe.get("home_team", "")),
                "away_team": normalize_team_name(oe.get("away_team", "")),
                "kickoff_utc": oe.get("commence_time"),
                "match_status": "MISSING_KALSHI",
                "stale_status": False,
                "raw_names": {
                    "odds_api_home": oe.get("home_team", ""),
                    "odds_api_away": oe.get("away_team", ""),
                },
                "_odds_api_status": oe.get("status", ""),
            }

            # Try matching with Goalserve
            o_date = oe.get("commence_time", "")[:10]
            for gi, gm in enumerate(gs_matches):
                if gi in gs_matched:
                    continue
                if not _dates_match(o_date, gm["date"]):
                    continue
                if _teams_match_normalized(
                    oe.get("home_team", ""), oe.get("away_team", ""),
                    gm["home_team"], gm["away_team"],
                ):
                    record["goalserve_match_id"] = gm["fix_id"]
                    record["raw_names"]["goalserve_home"] = gm["home_team"]
                    record["raw_names"]["goalserve_away"] = gm["away_team"]
                    record["_goalserve_status"] = gm["status"]
                    gs_matched.add(gi)
                    break

            # Stale status
            gs_st = record.get("_goalserve_status", "")
            if record.get("_odds_api_status") == "live" and _goalserve_status_to_phase(gs_st) == "FINISHED":
                record["stale_status"] = True

            results.append(record)

        # Add unmatched Goalserve entries
        for gi, gm in enumerate(gs_matches):
            if gi in gs_matched:
                continue
            results.append({
                "kalshi_event_ticker": None,
                "kalshi_tickers": {},
                "odds_api_event_id": None,
                "goalserve_match_id": gm["fix_id"],
                "home_team": normalize_team_name(gm["home_team"]),
                "away_team": normalize_team_name(gm["away_team"]),
                "kickoff_utc": None,
                "match_status": "MISSING_KALSHI",
                "stale_status": False,
                "raw_names": {
                    "goalserve_home": gm["home_team"],
                    "goalserve_away": gm["away_team"],
                },
            })

        return results

    finally:
        if own_client:
            await client.aclose()


async def map_all_leagues(
    client: httpx.AsyncClient | None = None,
) -> dict[str, list[dict]]:
    """
    Cross-reference all 3 sources for ALL 8 leagues.

    Returns: {league_id: [matched_fixtures]}
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20)

    try:
        results: dict[str, list[dict]] = {}
        for league_id in LEAGUES:
            results[league_id] = await map_all_sources(league_id, client)
            await asyncio.sleep(0.2)  # Rate limit courtesy
        return results
    finally:
        if own_client:
            await client.aclose()
