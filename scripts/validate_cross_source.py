#!/usr/bin/env python3
"""
Cross-Source Mapping Validator
==============================
Fetches ALL open/pending data from Kalshi, Odds-API, and Goalserve,
then cross-references by team name + date to find mapping gaps.

Usage:
    PYTHONPATH=. python scripts/validate_cross_source.py

Steps:
  1. Fetch ALL open Kalshi soccer markets (full pagination) for 8 league prefixes
  2. Fetch ALL pending Odds-API events for 8 league slugs
  3. Fetch Goalserve live scores for 8 league IDs
  4. Cross-reference all 3 sources by normalized team name + date
  5. Print mismatches with raw names for alias updates
"""

import os
import sys
import json
import time
import base64
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.calibration.team_aliases import normalize_team_name

# ─── Config ───────────────────────────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com"
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "")
KALSHI_KEY_PATH = os.environ.get(
    "KALSHI_PRIVATE_KEY_PATH",
    "keys/kalshi_private.pem",
)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.odds-api.io/v3"

GOALSERVE_API_KEY = os.environ.get("GOALSERVE_API_KEY", "")
GOALSERVE_BASE = "https://www.goalserve.com/getfeed"

# 8-league config: (display_name, kalshi_game_prefix, odds_api_slug, goalserve_league_id)
LEAGUES = [
    ("EPL", "KXEPLGAME", "england-premier-league", 1204),
    ("La Liga", "KXLALIGAGAME", "spain-laliga", 1399),
    ("Serie A", "KXSERIEAGAME", "italy-serie-a", 1269),
    ("Bundesliga", "KXBUNDESLIGAGAME", "germany-bundesliga", 1229),
    ("Ligue 1", "KXLIGUE1GAME", "france-ligue-1", 1221),
    ("MLS", "KXMLSGAME", "usa-mls", 1440),
    ("Brasileirao", "KXBRASILEIROGAME", "brazil-brasileiro-serie-a", 1141),
    ("Argentina", "KXARGPREMDIVGAME", "argentina-liga-profesional", 1081),
]

# All Kalshi series prefixes (GAME + other market types)
KALSHI_ALL_SERIES = {
    "KXEPLGAME", "KXEPL1H", "KXEPLBTTS", "KXEPLTOTAL", "KXEPLSPREAD",
    "KXLALIGAGAME", "KXLALIGA1H", "KXLALIGABTTS", "KXLALIGATOTAL", "KXLALIGASPREAD",
    "KXSERIEAGAME", "KXSERIEA1H", "KXSERIEABTTS", "KXSERIEATOTAL", "KXSERIEASPREAD",
    "KXBUNDESLIGAGAME", "KXBUNDESLIGA1H", "KXBUNDESLIGABTTS", "KXBUNDESLIGATOTAL", "KXBUNDESLIGASPREAD",
    "KXLIGUE1GAME", "KXLIGUE11H", "KXLIGUE1BTTS", "KXLIGUE1TOTAL", "KXLIGUE1SPREAD",
    "KXMLSGAME", "KXMLSSPREAD",
    "KXBRASILEIROGAME", "KXBRASILEIROTOTAL", "KXBRASILEIROSPREAD",
    "KXARGPREMDIVGAME",
}

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TODAY_SHORT = datetime.now(timezone.utc).strftime("%b%d").upper()  # e.g. MAR16

# ─── Kalshi Auth ──────────────────────────────────────────────────────────────

_PK = None


def _load_pk():
    global _PK
    if _PK is None:
        with open(KALSHI_KEY_PATH, "rb") as f:
            _PK = serialization.load_pem_private_key(f.read(), password=None)
    return _PK


def kalshi_sign(method: str, path: str) -> dict[str, str]:
    pk = _load_pk()
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
        "KALSHI-ACCESS-KEY": KALSHI_API_KEY,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def kalshi_get(path: str, client: httpx.Client) -> httpx.Response:
    headers = kalshi_sign("GET", path)
    return client.get(KALSHI_BASE + path, headers=headers)


# ─── Step 1: Fetch ALL Kalshi markets (full pagination) ──────────────────────


def fetch_kalshi_markets(client: httpx.Client) -> dict[str, dict]:
    """
    Fetch ALL open Kalshi soccer markets with full pagination.
    Returns: {event_ticker: {"markets": [...], "league": str, "title": str}}
    """
    print("=" * 70)
    print("STEP 1: Fetching ALL open Kalshi soccer markets (full pagination)")
    print("=" * 70)

    all_events: dict[str, dict] = {}
    total_markets = 0

    for league_name, game_prefix, _, _ in LEAGUES:
        # Fetch ALL series for this league (GAME, 1H, BTTS, Total, Spread)
        league_series = [s for s in KALSHI_ALL_SERIES if s.startswith("KX" + game_prefix[2:].replace("GAME", ""))]
        # Simpler: just use GAME prefix for event discovery, but fetch all series
        cursor = None
        league_markets = []

        while True:
            path = f"/trade-api/v2/markets?limit=200&status=open&series_ticker={game_prefix}"
            if cursor:
                path += f"&cursor={cursor}"

            r = kalshi_get(path, client)
            if r.status_code != 200:
                print(f"  [{league_name}] ERROR {r.status_code}: {r.text[:200]}")
                break

            data = r.json()
            markets = data.get("markets", [])
            league_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

            time.sleep(0.15)

        # Group by event_ticker
        events: dict[str, list] = defaultdict(list)
        for m in league_markets:
            et = m.get("event_ticker", m["ticker"].rsplit("-", 1)[0])
            events[et].append(m)

        for et, ms in events.items():
            title = ms[0].get("title", "?")
            # Extract date from event ticker: e.g. KXEPLGAME-26MAR16ARSCHE -> MAR16
            all_events[et] = {
                "markets": ms,
                "league": league_name,
                "title": title,
                "event_ticker": et,
            }

        total_markets += len(league_markets)
        print(f"  {league_name:<12} {len(events):>3} events, {len(league_markets):>4} markets")
        time.sleep(0.2)

    print(f"\n  TOTAL: {len(all_events)} events, {total_markets} markets")

    # Separate today vs this week
    today_events = {}
    week_events = {}
    for et, info in all_events.items():
        if TODAY_SHORT in et:
            today_events[et] = info
        else:
            week_events[et] = info

    print(f"\n  Today ({TODAY_SHORT}): {len(today_events)} events")
    for et, info in sorted(today_events.items()):
        print(f"    {et} — {info['title']}")

    print(f"\n  This week (other): {len(week_events)} events")
    for et, info in sorted(week_events.items()):
        print(f"    {et} — {info['title']}")

    return all_events


# ─── Step 2: Fetch ALL Odds-API events ───────────────────────────────────────


def fetch_odds_api_events(client: httpx.Client) -> list[dict]:
    """
    Fetch ALL pending + live events from Odds-API for 8 league slugs.
    Returns: list of event dicts with id, home_team, away_team, commence_time, status, league.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Fetching ALL Odds-API events (pending + live)")
    print("=" * 70)

    all_events: list[dict] = []

    for league_name, _, slug, _ in LEAGUES:
        for status in ("pending", "live"):
            url = (
                f"{ODDS_API_BASE}/events"
                f"?sport=football&league={slug}&status={status}"
                f"&apiKey={ODDS_API_KEY}"
            )
            r = client.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  [{league_name}] {status} ERROR {r.status_code}: {r.text[:200]}")
                continue

            events = r.json()
            if not isinstance(events, list):
                print(f"  [{league_name}] {status} unexpected response: {str(events)[:200]}")
                continue

            for e in events:
                e["_league"] = league_name
                e["_slug"] = slug
                # Normalize field names: API uses "home"/"away", not "home_team"/"away_team"
                if "home" in e and "home_team" not in e:
                    e["home_team"] = e["home"]
                if "away" in e and "away_team" not in e:
                    e["away_team"] = e["away"]
                # Normalize date field: API uses "date", not "commence_time"
                if "date" in e and "commence_time" not in e:
                    e["commence_time"] = e["date"]
            all_events.extend(events)

            if events:
                print(f"  {league_name:<12} {status:<8} {len(events):>3} events")
            time.sleep(0.3)

    print(f"\n  TOTAL: {len(all_events)} events")

    # Print details
    for e in sorted(all_events, key=lambda x: x.get("commence_time", "")):
        commence = e.get("commence_time", "?")[:16]
        home = e.get("home_team", "?")
        away = e.get("away_team", "?")
        eid = e.get("id", "?")
        status = e.get("status", "?")
        league = e.get("_league", "?")
        print(f"    [{league:<12}] {commence}  {home:<25} vs {away:<25} id={eid}  status={status}")

    return all_events


# ─── Step 3: Fetch Goalserve live/upcoming ────────────────────────────────────


def fetch_goalserve_matches(client: httpx.Client) -> list[dict]:
    """
    Fetch Goalserve matches for our 8 league IDs.
    Uses commentaries endpoint (has scheduled + live + finished matches)
    plus live scores endpoint (for currently-live games).
    Returns: list of match dicts with fix_id, home, away, status, league_id, date.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Fetching Goalserve fixtures (commentaries + live scores)")
    print("=" * 70)

    league_id_to_name = {str(lid): name for name, _, _, lid in LEAGUES}
    all_matches: list[dict] = []
    seen_ids: set[str] = set()

    # Fetch today's + next 7 days via commentaries endpoint (has all fixtures)
    now = datetime.now(timezone.utc)
    dates_to_check = [(now + timedelta(days=d)).strftime("%d.%m.%Y") for d in range(8)]

    for league_name, _, _, league_id in LEAGUES:
        league_count = 0
        for date_str in dates_to_check:
            url = (
                f"{GOALSERVE_BASE}/{GOALSERVE_API_KEY}/commentaries/{league_id}"
                f"?date={date_str}&json=1"
            )
            r = client.get(url, timeout=15)
            if r.status_code != 200:
                continue

            data = r.json()
            comms = data.get("commentaries", data)
            tournament = comms.get("tournament", {})
            matches = tournament.get("match", [])
            if isinstance(matches, dict):
                matches = [matches]

            for m in matches:
                # Use all 3 ID fields as documented
                fix_id = str(m.get("@fix_id") or m.get("@id") or m.get("@static_id") or "?")
                if fix_id in seen_ids:
                    continue
                seen_ids.add(fix_id)

                home = m.get("localteam", {})
                away = m.get("visitorteam", {})
                home_name = home.get("@name", "?")
                away_name = away.get("@name", "?")
                match_status = m.get("@status", m.get("status", "?"))
                match_date = m.get("@date", m.get("@formatted_date", "?"))
                match_time = m.get("@time", "?")

                match_info = {
                    "fix_id": fix_id,
                    "home_team": home_name,
                    "away_team": away_name,
                    "status": str(match_status),
                    "league_id": str(league_id),
                    "league_name": league_name,
                    "date": match_date,
                    "time": match_time,
                    "home_goals": home.get("@goals", ""),
                    "away_goals": away.get("@goals", ""),
                }
                all_matches.append(match_info)
                league_count += 1

            time.sleep(0.15)

        if league_count:
            print(f"  {league_name:<12} {league_count:>3} matches (next 7 days)")

    # Also check live scores for real-time status
    print("\n  Checking live scores for status updates...")
    url = f"{GOALSERVE_BASE}/{GOALSERVE_API_KEY}/soccernew/home?json=1"
    r = client.get(url, timeout=15)
    if r.status_code == 200:
        data = r.json()
        scores = data.get("scores", data)
        categories = scores.get("category", [])
        if isinstance(categories, dict):
            categories = [categories]

        our_league_ids = {str(lid) for _, _, _, lid in LEAGUES}
        live_count = 0
        for cat in categories:
            cat_league_id = str(cat.get("@id", ""))
            if cat_league_id not in our_league_ids:
                continue

            matches = cat.get("match", [])
            if isinstance(matches, dict):
                matches = [matches]

            for m in matches:
                fix_id = str(m.get("@fix_id") or m.get("@id") or m.get("@static_id") or "?")
                live_status = str(m.get("@status", ""))

                # Update status for matches we already have
                for existing in all_matches:
                    if existing["fix_id"] == fix_id:
                        existing["status"] = live_status
                        live_count += 1
                        break
                else:
                    # New match from live that we didn't have
                    home = m.get("localteam", {})
                    away = m.get("visitorteam", {})
                    match_info = {
                        "fix_id": fix_id,
                        "home_team": home.get("@name", "?"),
                        "away_team": away.get("@name", "?"),
                        "status": live_status,
                        "league_id": cat_league_id,
                        "league_name": league_id_to_name.get(cat_league_id, cat_league_id),
                        "date": m.get("@date", "?"),
                        "time": m.get("@time", "?"),
                        "home_goals": home.get("@goals", ""),
                        "away_goals": away.get("@goals", ""),
                    }
                    all_matches.append(match_info)
                    live_count += 1

        if live_count:
            print(f"    Updated {live_count} matches with live status")
        else:
            print("    No live matches in our leagues right now")

    print(f"\n  TOTAL: {len(all_matches)} matches from Goalserve")
    for m in all_matches:
        print(
            f"    [{m['league_name']:<12}] {m['date']} {m['time']}  "
            f"{m['home_team']:<25} vs {m['away_team']:<25}  "
            f"fix_id={m['fix_id']}  status={m['status']}"
        )

    return all_matches


# ─── Step 4: Cross-reference all 3 sources ────────────────────────────────────


def _parse_kalshi_event_ticker(event_ticker: str) -> tuple[str, str, str]:
    """
    Parse a Kalshi event ticker to extract date and team abbreviations.
    e.g. KXEPLGAME-26MAR16ARSCHE -> ("26MAR16", "ARS", "CHE")

    Returns: (date_str, home_abbrev, away_abbrev)
    """
    parts = event_ticker.split("-", 1)
    if len(parts) < 2:
        return ("", "", "")

    suffix = parts[1]  # e.g. "26MAR16ARSCHE"

    # Extract season + date: first 7 chars = "26MAR16"
    # Then the rest is team abbreviations
    if len(suffix) < 7:
        return ("", "", "")

    date_part = suffix[:7]  # "26MAR16"
    teams_part = suffix[7:]  # "ARSCHE"

    # Team abbrevs are typically 3 chars each, but can vary
    # We'll return the raw teams_part for fuzzy matching later
    return (date_part, teams_part, "")


def _extract_date_from_kalshi(event_ticker: str) -> str | None:
    """Extract ISO date from Kalshi event ticker. e.g. 26MAR16 -> 2026-03-16"""
    date_part, _, _ = _parse_kalshi_event_ticker(event_ticker)
    if not date_part or len(date_part) < 7:
        return None

    try:
        # Format: 26MAR16 = year_suffix + month_abbrev + day
        year = 2000 + int(date_part[:2])
        month_str = date_part[2:5]
        day = int(date_part[5:7])
        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = months.get(month_str.upper())
        if month is None:
            return None
        return f"{year}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def _extract_kalshi_teams_from_title(title: str) -> tuple[str, str]:
    """
    Extract team names from Kalshi market title.
    e.g. "Arsenal vs Chelsea: Arsenal" -> ("Arsenal", "Chelsea")
    e.g. "Brentford vs Wolverhampton Winner?" -> ("Brentford", "Wolverhampton")
    """
    if " vs " not in title:
        return ("", "")

    vs_part = title.split(" vs ", 1)
    home = vs_part[0].strip()
    away_part = vs_part[1]

    # Remove outcome after colon: "Chelsea: Arsenal" -> "Chelsea"
    if ":" in away_part:
        away_part = away_part.split(":")[0]

    # Remove trailing "Winner?" or "?" suffix
    away_part = away_part.split("?")[0].strip()
    if away_part.lower().endswith(" winner"):
        away_part = away_part[: -len(" winner")].strip()

    return (home.strip(), away_part.strip())


def _normalize_date_from_commence(commence_time: str) -> str:
    """Extract YYYY-MM-DD from Odds-API commence_time ISO string."""
    return commence_time[:10] if commence_time else ""


def _normalize_goalserve_date(date_str: str) -> str:
    """
    Parse Goalserve date to YYYY-MM-DD.
    Goalserve uses various formats: DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
    """
    if not date_str or date_str == "?":
        return ""

    # Already ISO
    if len(date_str) == 10 and date_str[4] == "-":
        return date_str

    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_str


def _fuzzy_team_match(name_a: str, name_b: str) -> bool:
    """Check if two team names refer to the same team after normalization."""
    norm_a = normalize_team_name(name_a)
    norm_b = normalize_team_name(name_b)
    return norm_a == norm_b


def cross_reference(
    kalshi_events: dict[str, dict],
    odds_api_events: list[dict],
    goalserve_matches: list[dict],
) -> list[dict]:
    """
    Cross-reference all 3 sources. Returns list of unified match records.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Cross-referencing all 3 sources")
    print("=" * 70)

    # Build lookup structures

    # Kalshi: extract team names from titles, normalize
    kalshi_lookup: list[dict] = []
    for et, info in kalshi_events.items():
        # Use first market's title to get team names
        title = info["title"]
        home_raw, away_raw = _extract_kalshi_teams_from_title(title)
        date_iso = _extract_date_from_kalshi(et)
        tickers = {m["ticker"]: m for m in info["markets"]}

        kalshi_lookup.append({
            "event_ticker": et,
            "league": info["league"],
            "home_raw": home_raw,
            "away_raw": away_raw,
            "home_norm": normalize_team_name(home_raw) if home_raw else "",
            "away_norm": normalize_team_name(away_raw) if away_raw else "",
            "date": date_iso or "",
            "tickers": tickers,
            "title": title,
            "_matched": False,
        })

    # Odds-API: normalize
    odds_lookup: list[dict] = []
    for e in odds_api_events:
        home_raw = e.get("home_team", "")
        away_raw = e.get("away_team", "")
        commence = e.get("commence_time", "")
        date_iso = _normalize_date_from_commence(commence)

        odds_lookup.append({
            "id": e.get("id", ""),
            "league": e.get("_league", ""),
            "home_raw": home_raw,
            "away_raw": away_raw,
            "home_norm": normalize_team_name(home_raw),
            "away_norm": normalize_team_name(away_raw),
            "date": date_iso,
            "commence_time": commence,
            "status": e.get("status", ""),
            "_matched": False,
        })

    # Goalserve: normalize
    gs_lookup: list[dict] = []
    for m in goalserve_matches:
        home_raw = m["home_team"]
        away_raw = m["away_team"]
        date_iso = _normalize_goalserve_date(m["date"])

        gs_lookup.append({
            "fix_id": m["fix_id"],
            "league": m["league_name"],
            "league_id": m["league_id"],
            "home_raw": home_raw,
            "away_raw": away_raw,
            "home_norm": normalize_team_name(home_raw),
            "away_norm": normalize_team_name(away_raw),
            "date": date_iso,
            "status": m["status"],
            "time": m["time"],
            "_matched": False,
        })

    # Match records
    unified: list[dict] = []

    def _teams_match(a_home: str, a_away: str, b_home: str, b_away: str) -> str:
        """
        Check if teams match, trying both orderings.
        Returns: "exact" | "swapped" | "no_match"
        """
        if a_home == b_home and a_away == b_away:
            return "exact"
        if a_home == b_away and a_away == b_home:
            return "swapped"
        return "no_match"

    def _date_match(d1: str, d2: str) -> bool:
        """Check if dates match within ±1 day (timezone edge cases)."""
        if not d1 or not d2:
            return True  # If we don't have a date, don't block on it
        if d1 == d2:
            return True
        try:
            dt1 = datetime.strptime(d1, "%Y-%m-%d")
            dt2 = datetime.strptime(d2, "%Y-%m-%d")
            return abs((dt1 - dt2).days) <= 1
        except ValueError:
            return False

    # Phase 1: Match Kalshi ↔ Odds-API
    kalshi_odds_pairs: list[tuple[int, int, str]] = []  # (k_idx, o_idx, match_type)
    for ki, k in enumerate(kalshi_lookup):
        if not k["home_norm"] or not k["away_norm"]:
            continue
        for oi, o in enumerate(odds_lookup):
            if k["league"] != o["league"]:
                continue
            if not _date_match(k["date"], o["date"]):
                continue
            tm = _teams_match(k["home_norm"], k["away_norm"], o["home_norm"], o["away_norm"])
            if tm != "no_match":
                kalshi_odds_pairs.append((ki, oi, tm))

    # Phase 2: Match Kalshi ↔ Goalserve
    kalshi_gs_pairs: list[tuple[int, int, str]] = []
    for ki, k in enumerate(kalshi_lookup):
        if not k["home_norm"] or not k["away_norm"]:
            continue
        for gi, g in enumerate(gs_lookup):
            if k["league"] != g["league"]:
                continue
            if not _date_match(k["date"], g["date"]):
                continue
            tm = _teams_match(k["home_norm"], k["away_norm"], g["home_norm"], g["away_norm"])
            if tm != "no_match":
                kalshi_gs_pairs.append((ki, gi, tm))

    # Phase 3: Match Odds-API ↔ Goalserve
    odds_gs_pairs: list[tuple[int, int, str]] = []
    for oi, o in enumerate(odds_lookup):
        for gi, g in enumerate(gs_lookup):
            if o["league"] != g["league"]:
                continue
            if not _date_match(o["date"], g["date"]):
                continue
            tm = _teams_match(o["home_norm"], o["away_norm"], g["home_norm"], g["away_norm"])
            if tm != "no_match":
                odds_gs_pairs.append((oi, gi, tm))

    # Build unified records: start from Kalshi events
    kalshi_to_odds = {ki: (oi, tm) for ki, oi, tm in kalshi_odds_pairs}
    kalshi_to_gs = {ki: (gi, tm) for ki, gi, tm in kalshi_gs_pairs}
    odds_to_gs = {oi: (gi, tm) for oi, gi, tm in odds_gs_pairs}

    # Track which Odds-API / Goalserve entries have been matched
    odds_matched: set[int] = set()
    gs_matched: set[int] = set()

    for ki, k in enumerate(kalshi_lookup):
        record: dict = {
            "date": k["date"],
            "league": k["league"],
            "home_norm": k["home_norm"],
            "away_norm": k["away_norm"],
            "kalshi_event_ticker": k["event_ticker"],
            "kalshi_home_raw": k["home_raw"],
            "kalshi_away_raw": k["away_raw"],
            "odds_api_id": None,
            "odds_api_home_raw": None,
            "odds_api_away_raw": None,
            "odds_api_status": None,
            "goalserve_id": None,
            "goalserve_home_raw": None,
            "goalserve_away_raw": None,
            "goalserve_status": None,
            "status": "MISSING_ODDS_API",
        }

        # Check Kalshi ↔ Odds-API
        if ki in kalshi_to_odds:
            oi, tm = kalshi_to_odds[ki]
            o = odds_lookup[oi]
            record["odds_api_id"] = o["id"]
            record["odds_api_home_raw"] = o["home_raw"]
            record["odds_api_away_raw"] = o["away_raw"]
            record["odds_api_status"] = o["status"]
            odds_matched.add(oi)

        # Check Kalshi ↔ Goalserve
        if ki in kalshi_to_gs:
            gi, tm = kalshi_to_gs[ki]
            g = gs_lookup[gi]
            record["goalserve_id"] = g["fix_id"]
            record["goalserve_home_raw"] = g["home_raw"]
            record["goalserve_away_raw"] = g["away_raw"]
            record["goalserve_status"] = g["status"]
            gs_matched.add(gi)

        # Determine overall status
        has_kalshi = True
        has_odds = record["odds_api_id"] is not None
        has_gs = record["goalserve_id"] is not None

        if has_kalshi and has_odds and has_gs:
            record["status"] = "ALL_MATCHED"
        elif has_kalshi and has_odds:
            record["status"] = "MISSING_GOALSERVE"
        elif has_kalshi and has_gs:
            record["status"] = "MISSING_ODDS_API"
        else:
            record["status"] = "MISSING_ODDS_API"  # only Kalshi

        unified.append(record)

    # Add unmatched Odds-API events
    for oi, o in enumerate(odds_lookup):
        if oi in odds_matched:
            continue

        record = {
            "date": o["date"],
            "league": o["league"],
            "home_norm": o["home_norm"],
            "away_norm": o["away_norm"],
            "kalshi_event_ticker": None,
            "kalshi_home_raw": None,
            "kalshi_away_raw": None,
            "odds_api_id": o["id"],
            "odds_api_home_raw": o["home_raw"],
            "odds_api_away_raw": o["away_raw"],
            "odds_api_status": o["status"],
            "goalserve_id": None,
            "goalserve_home_raw": None,
            "goalserve_away_raw": None,
            "goalserve_status": None,
            "status": "MISSING_KALSHI",
        }

        # Check Odds-API ↔ Goalserve
        if oi in odds_to_gs:
            gi, tm = odds_to_gs[oi]
            g = gs_lookup[gi]
            record["goalserve_id"] = g["fix_id"]
            record["goalserve_home_raw"] = g["home_raw"]
            record["goalserve_away_raw"] = g["away_raw"]
            record["goalserve_status"] = g["status"]
            gs_matched.add(gi)

            if record["goalserve_id"]:
                record["status"] = "MISSING_KALSHI"
            else:
                record["status"] = "MISSING_KALSHI"
        else:
            record["status"] = "MISSING_KALSHI"

        unified.append(record)

    # Add unmatched Goalserve entries
    for gi, g in enumerate(gs_lookup):
        if gi in gs_matched:
            continue

        record = {
            "date": g["date"],
            "league": g["league"],
            "home_norm": g["home_norm"],
            "away_norm": g["away_norm"],
            "kalshi_event_ticker": None,
            "kalshi_home_raw": None,
            "kalshi_away_raw": None,
            "odds_api_id": None,
            "odds_api_home_raw": None,
            "odds_api_away_raw": None,
            "odds_api_status": None,
            "goalserve_id": g["fix_id"],
            "goalserve_home_raw": g["home_raw"],
            "goalserve_away_raw": g["away_raw"],
            "goalserve_status": g["status"],
            "status": "MISSING_KALSHI",
        }
        unified.append(record)

    # Check for NAME_MISMATCH: entries where we have 2+ sources but normalization
    # produced different canonical names
    for rec in unified:
        if rec["status"] == "ALL_MATCHED":
            # Verify names actually match across all 3
            names_kalshi = (
                normalize_team_name(rec["kalshi_home_raw"]) if rec["kalshi_home_raw"] else None,
                normalize_team_name(rec["kalshi_away_raw"]) if rec["kalshi_away_raw"] else None,
            )
            names_odds = (
                normalize_team_name(rec["odds_api_home_raw"]) if rec["odds_api_home_raw"] else None,
                normalize_team_name(rec["odds_api_away_raw"]) if rec["odds_api_away_raw"] else None,
            )
            names_gs = (
                normalize_team_name(rec["goalserve_home_raw"]) if rec["goalserve_home_raw"] else None,
                normalize_team_name(rec["goalserve_away_raw"]) if rec["goalserve_away_raw"] else None,
            )
            # All should agree if matched correctly

    # Sort by date, league
    unified.sort(key=lambda x: (x["date"], x["league"], x["home_norm"]))

    # Print table
    print(f"\n  {'Date':<12} {'League':<12} {'Home':<22} {'Away':<22} {'Kalshi':<30} {'OddsAPI':<14} {'Goalserve':<12} {'Status'}")
    print("  " + "-" * 148)
    for rec in unified:
        kalshi_tick = rec["kalshi_event_ticker"] or "—"
        odds_id = str(rec["odds_api_id"] or "—")[:12]
        gs_id = rec["goalserve_id"] or "—"
        home = rec["home_norm"][:20] if rec["home_norm"] else "?"
        away = rec["away_norm"][:20] if rec["away_norm"] else "?"
        print(
            f"  {rec['date']:<12} {rec['league']:<12} {home:<22} {away:<22} "
            f"{kalshi_tick:<30} {odds_id:<14} {gs_id:<12} {rec['status']}"
        )

    # Summary
    statuses = defaultdict(int)
    for rec in unified:
        statuses[rec["status"]] += 1
    print(f"\n  Summary:")
    for s, count in sorted(statuses.items()):
        print(f"    {s}: {count}")

    return unified


# ─── Step 5: Print name mismatches ───────────────────────────────────────────


def print_name_mismatches(unified: list[dict]) -> list[dict]:
    """
    For entries with potential name issues, print raw names from each source.
    Also detect entries that SHOULD match but don't due to missing aliases.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Name mismatch analysis")
    print("=" * 70)

    mismatches: list[dict] = []

    # Check all records that have at least 2 sources
    for rec in unified:
        sources = []
        if rec["kalshi_home_raw"]:
            sources.append(("Kalshi", rec["kalshi_home_raw"], rec["kalshi_away_raw"]))
        if rec["odds_api_home_raw"]:
            sources.append(("OddsAPI", rec["odds_api_home_raw"], rec["odds_api_away_raw"]))
        if rec["goalserve_home_raw"]:
            sources.append(("Goalserve", rec["goalserve_home_raw"], rec["goalserve_away_raw"]))

        if len(sources) < 2:
            continue

        # Check if normalized names agree across all present sources
        norms = set()
        for src_name, h, a in sources:
            norms.add((normalize_team_name(h), normalize_team_name(a)))

        if len(norms) > 1:
            mismatches.append(rec)
            print(f"\n  NAME_MISMATCH: {rec['date']} {rec['league']}")
            for src_name, h, a in sources:
                h_norm = normalize_team_name(h)
                a_norm = normalize_team_name(a)
                print(f"    {src_name:<10} raw: {h:<30} vs {a:<30}")
                print(f"    {'':<10} norm: {h_norm:<30} vs {a_norm:<30}")

    # Also look for unmatched entries that MIGHT match with better aliases
    # (same league, same date, but different normalized names)
    print(f"\n  Potential unmatched pairs (same league + date, different names):")
    unmatched_kalshi = [r for r in unified if r["status"] in ("MISSING_ODDS_API",) and r["kalshi_event_ticker"]]
    unmatched_odds = [r for r in unified if r["status"] == "MISSING_KALSHI" and r["odds_api_id"]]

    for k in unmatched_kalshi:
        for o in unmatched_odds:
            if k["league"] != o["league"]:
                continue
            if k["date"] and o["date"] and k["date"] != o["date"]:
                continue
            # These are in the same league on the same date but didn't match
            print(f"\n    Could be same match? [{k['league']}] {k['date']}")
            print(f"      Kalshi:  {k.get('kalshi_home_raw', '?'):<30} vs {k.get('kalshi_away_raw', '?'):<30}  ({k['kalshi_event_ticker']})")
            print(f"      OddsAPI: {o.get('odds_api_home_raw', '?'):<30} vs {o.get('odds_api_away_raw', '?'):<30}  ({o['odds_api_id']})")
            # Check normalized
            if k["home_norm"] and o["home_norm"]:
                print(f"      norm:    {k['home_norm']:<30} vs {o['home_norm']:<30} (home)")
                print(f"      norm:    {k['away_norm']:<30} vs {o['away_norm']:<30} (away)")

    if not mismatches:
        print("\n  No name mismatches found across matched entries.")

    return mismatches


# ─── Step 8: Stale status detection ──────────────────────────────────────────


def check_stale_status(unified: list[dict]) -> None:
    """
    For any Odds-API event marked 'live', cross-check with Goalserve.
    If Goalserve says FT but Odds-API says live, flag as STALE_STATUS.
    """
    print("\n" + "=" * 70)
    print("STEP 8: Stale status detection (Odds-API live vs Goalserve FT)")
    print("=" * 70)

    stale_count = 0
    for rec in unified:
        odds_status = rec.get("odds_api_status", "")
        gs_status = rec.get("goalserve_status", "")

        if odds_status == "live" and gs_status in ("FT", "AET", "Pen."):
            stale_count += 1
            print(
                f"  STALE_STATUS: [{rec['league']}] {rec['home_norm']} vs {rec['away_norm']}"
                f"  OddsAPI={odds_status}  Goalserve={gs_status}"
            )

        elif odds_status == "pending" and gs_status and gs_status not in ("?", "NS", "Postp.", "FT", "AET"):
            # Odds-API says pending but Goalserve says it's in progress
            try:
                # Goalserve status is often a minute number for live games
                minute = int(gs_status)
                if minute > 0:
                    stale_count += 1
                    print(
                        f"  STALE_STATUS: [{rec['league']}] {rec['home_norm']} vs {rec['away_norm']}"
                        f"  OddsAPI={odds_status}  Goalserve=minute {minute} (LIVE)"
                    )
            except ValueError:
                if gs_status == "HT":
                    stale_count += 1
                    print(
                        f"  STALE_STATUS: [{rec['league']}] {rec['home_norm']} vs {rec['away_norm']}"
                        f"  OddsAPI={odds_status}  Goalserve={gs_status} (LIVE)"
                    )

    if stale_count == 0:
        print("  No stale status detected.")
    else:
        print(f"\n  Total stale entries: {stale_count}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not KALSHI_API_KEY:
        print("ERROR: KALSHI_API_KEY not set")
        sys.exit(1)
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        sys.exit(1)
    if not GOALSERVE_API_KEY:
        print("ERROR: GOALSERVE_API_KEY not set")
        sys.exit(1)

    print(f"Cross-Source Mapping Validator — {TODAY}")
    print(f"Today's date marker: {TODAY_SHORT}")
    print()

    client = httpx.Client(timeout=20)

    try:
        # Step 1
        kalshi_events = fetch_kalshi_markets(client)

        # Step 2
        odds_api_events = fetch_odds_api_events(client)

        # Step 3
        goalserve_matches = fetch_goalserve_matches(client)

        # Step 4
        unified = cross_reference(kalshi_events, odds_api_events, goalserve_matches)

        # Step 5
        mismatches = print_name_mismatches(unified)

        # Step 8
        check_stale_status(unified)

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"  Kalshi events:    {len(kalshi_events)}")
        print(f"  Odds-API events:  {len(odds_api_events)}")
        print(f"  Goalserve matches: {len(goalserve_matches)}")
        print(f"  Unified records:  {len(unified)}")
        print(f"  Name mismatches:  {len(mismatches)}")

        all_matched = sum(1 for r in unified if r["status"] == "ALL_MATCHED")
        print(f"  ALL_MATCHED:      {all_matched}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
