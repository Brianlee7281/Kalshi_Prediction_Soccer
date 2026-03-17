#!/usr/bin/env python3
"""
Latency Measurement Tool — Cross-Market Lag (architecture.md §3.7.5 Metric 1)
==============================================================================
Simultaneously connects to Odds-API WS, Goalserve polling (1s), and Kalshi WS
for a specific live match. Detects significant price movements from each source,
logs timestamps, and computes cross-market lag after the match.

This is the MOST IMPORTANT measurement for the project: does Kalshi react
slower than Betfair/bookmakers after in-game events (goals, red cards)?

Usage:
  PYTHONPATH=. python scripts/measure_latency.py --match-id 4190023 --league EPL
  PYTHONPATH=. python scripts/measure_latency.py --analyze data/latency/4190023

Output: data/latency/{match_id}/
  odds_api.jsonl      — raw Odds-API WS messages
  goalserve.jsonl     — raw Goalserve poll responses (every 1s)
  kalshi.jsonl        — raw Kalshi WS orderbook messages
  events.jsonl        — detected events with timestamps from all sources
  latency_report.json — computed cross-market lag per event
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ─── Config ──────────────────────────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com"
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

GOALSERVE_BASE = "https://www.goalserve.com/getfeed"

LEAGUE_IDS = {
    "EPL": "1204", "LaLiga": "1399", "SerieA": "1269", "Bundesliga": "1229",
    "Ligue1": "1221", "MLS": "1440", "Brasileirao": "1141", "Argentina": "1081",
}

LEAGUE_PREFIXES = {
    "EPL": "KXEPLGAME", "LaLiga": "KXLALIGAGAME", "SerieA": "KXSERIEAGAME",
    "Bundesliga": "KXBUNDESLIGAGAME", "Ligue1": "KXLIGUE1GAME",
    "MLS": "KXMLSGAME", "Brasileirao": "KXBRASILEIROGAME", "Argentina": "KXARGPREMDIVGAME",
}

# Movement thresholds
ODDS_MOVE_THRESHOLD = 0.03   # 3% implied probability change
KALSHI_MOVE_THRESHOLD = 0.03  # 3¢ best-ask change

OUTPUT_DIR = Path("data/latency")


def _iter_json_lines(raw: str) -> list[str]:
    """Split a WS frame that may contain multiple JSON objects."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        json.loads(raw)
        return [raw]
    except (json.JSONDecodeError, TypeError):
        pass
    parts = [line.strip() for line in raw.split("\n") if line.strip()]
    return parts if parts else [raw]


# ─── Kalshi Auth ─────────────────────────────────────────────────────────────

_PK = None


def _load_private_key() -> object:
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
    # Fallback to local path if Docker path doesn't exist
    if not Path(key_path).exists():
        key_path = "keys/kalshi_private.pem"
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def kalshi_sign(method: str, path: str) -> dict[str, str]:
    global _PK
    if _PK is None:
        _PK = _load_private_key()
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(_PK.sign(
        (ts + method + path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )).decode()
    return {
        "KALSHI-ACCESS-KEY": os.environ.get("KALSHI_API_KEY", ""),
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ─── Shared State ────────────────────────────────────────────────────────────

class LatencyTracker:
    """Collects timestamped events from all three sources."""

    def __init__(self, match_id: str, match_dir: Path) -> None:
        self.match_id = match_id
        self.match_dir = match_dir
        self.match_dir.mkdir(parents=True, exist_ok=True)

        self._start_mono = time.monotonic()
        self._start_wall = time.time()

        # File handles (opened lazily)
        self._files: dict[str, object] = {}

        # Price state for movement detection
        self.odds_state: dict[str, float] = {}      # bookie → last home_win implied
        self.kalshi_best_ask: dict[str, float] = {}  # ticker → last best_ask
        self.goalserve_score: tuple[int, int] = (0, 0)
        self.goalserve_status: str = ""

        # Detected events: [{source, ts_mono, ts_wall, type, data}, ...]
        self.events: list[dict] = []

        # Running flag
        self.running = True

    def _get_file(self, name: str):
        if name not in self._files:
            path = self.match_dir / f"{name}.jsonl"
            self._files[name] = open(path, "a", encoding="utf-8")
        return self._files[name]

    def _ts(self) -> tuple[float, float, str]:
        """Return (monotonic, wall_time, iso_utc)."""
        mono = time.monotonic()
        wall = time.time()
        utc = datetime.fromtimestamp(wall, tz=timezone.utc).isoformat()
        return mono, wall, utc

    def record_odds_api(self, message: dict) -> None:
        """Record raw Odds-API WS message and check for movement."""
        mono, wall, utc = self._ts()
        record = {"_ts_mono": mono - self._start_mono, "_ts_wall": wall, "_utc": utc, **message}
        f = self._get_file("odds_api")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        # Check for significant movement
        if message.get("type") != "updated":
            return
        bookie = message.get("bookie", "")
        for mkt in message.get("markets", []):
            if mkt.get("name") != "ML":
                continue
            odds_list = mkt.get("odds", [])
            home_odds = None
            for o in odds_list:
                if o.get("name", "").lower() == "home" and o.get("price"):
                    home_odds = float(o["price"])
                    break
            if home_odds is None or home_odds <= 0:
                continue

            implied = 1.0 / home_odds
            prev = self.odds_state.get(bookie)
            self.odds_state[bookie] = implied

            if prev is not None and abs(implied - prev) > ODDS_MOVE_THRESHOLD:
                direction = "up" if implied > prev else "down"
                evt = {
                    "source": "odds_api",
                    "ts_mono": mono,
                    "ts_wall": wall,
                    "utc": utc,
                    "type": "odds_move",
                    "bookie": bookie,
                    "prev": round(prev, 4),
                    "curr": round(implied, 4),
                    "move": round(implied - prev, 4),
                    "direction": direction,
                }
                self.events.append(evt)
                ef = self._get_file("events")
                ef.write(json.dumps(evt) + "\n")
                ef.flush()
                _print_event(f"[ODDS-API] {bookie} home_win {prev:.3f}→{implied:.3f} "
                             f"({direction} {abs(implied-prev)*100:.1f}%)")

    def record_goalserve(self, match_data: dict | None) -> None:
        """Record Goalserve poll and check for score/status change."""
        mono, wall, utc = self._ts()
        record = {"_ts_mono": mono - self._start_mono, "_ts_wall": wall, "_utc": utc}
        if match_data:
            record["data"] = match_data
        else:
            record["data"] = None
        f = self._get_file("goalserve")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        if match_data is None:
            return

        # Check score change
        try:
            home_goals = int(match_data.get("localteam", {}).get("@goals", "0"))
            away_goals = int(match_data.get("visitorteam", {}).get("@goals", "0"))
        except (ValueError, TypeError):
            return

        new_score = (home_goals, away_goals)
        if new_score != self.goalserve_score and self.goalserve_score != (0, 0) or (
            new_score != (0, 0) and self.goalserve_score == (0, 0)
            and (home_goals > 0 or away_goals > 0)
        ):
            old = self.goalserve_score
            if new_score != old:
                team = "home" if home_goals > old[0] else "away"
                evt = {
                    "source": "goalserve",
                    "ts_mono": mono,
                    "ts_wall": wall,
                    "utc": utc,
                    "type": "goal",
                    "prev_score": list(old),
                    "new_score": list(new_score),
                    "team": team,
                }
                self.events.append(evt)
                ef = self._get_file("events")
                ef.write(json.dumps(evt) + "\n")
                ef.flush()
                _print_event(f"[GOALSERVE] GOAL! {old[0]}-{old[1]} → {home_goals}-{away_goals}")

        self.goalserve_score = new_score

        # Check status change
        status = match_data.get("@status", "")
        if status != self.goalserve_status and self.goalserve_status:
            evt = {
                "source": "goalserve",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "status_change",
                "prev_status": self.goalserve_status,
                "new_status": status,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            if status == "FT":
                _print_event(f"[GOALSERVE] FULL TIME")
                self.running = False
        self.goalserve_status = status

    def record_kalshi(self, message: dict) -> None:
        """Record Kalshi WS message and check for price movement."""
        mono, wall, utc = self._ts()
        record = {"_ts_mono": mono - self._start_mono, "_ts_wall": wall, "_utc": utc, **message}
        f = self._get_file("kalshi")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        # Extract best ask from snapshot or delta
        msg = message.get("msg", message)
        msg_type = msg.get("type", message.get("type", ""))
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        # Parse best ask from yes side of orderbook
        best_ask = None
        if msg_type == "orderbook_snapshot":
            yes_book = msg.get("yes", [])
            if yes_book:
                # yes book: [[price, qty], ...] — lowest ask
                prices = [float(entry[0]) for entry in yes_book if len(entry) >= 2]
                if prices:
                    best_ask = min(prices) / 100.0  # cents → dollars
        elif msg_type == "orderbook_delta":
            # Delta updates — track price from the delta
            yes_delta = msg.get("yes", [])
            if yes_delta:
                prices = [float(entry[0]) for entry in yes_delta if len(entry) >= 2 and float(entry[1]) > 0]
                if prices:
                    best_ask = min(prices) / 100.0

        if best_ask is None:
            return

        prev = self.kalshi_best_ask.get(ticker)
        self.kalshi_best_ask[ticker] = best_ask

        if prev is not None and abs(best_ask - prev) > KALSHI_MOVE_THRESHOLD:
            direction = "up" if best_ask > prev else "down"
            evt = {
                "source": "kalshi",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "price_move",
                "ticker": ticker,
                "prev": round(prev, 4),
                "curr": round(best_ask, 4),
                "move": round(best_ask - prev, 4),
                "direction": direction,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            short_ticker = ticker[-15:]
            _print_event(f"[KALSHI] {short_ticker} {prev:.2f}→{best_ask:.2f} "
                         f"({direction} {abs(best_ask-prev)*100:.1f}¢)")

    def finalize(self) -> None:
        """Close file handles."""
        for f in self._files.values():
            f.flush()
            f.close()
        self._files.clear()


def _print_event(msg: str) -> None:
    utc = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"  ** {utc} {msg}")


# ─── Coroutines ──────────────────────────────────────────────────────────────

async def odds_api_ws(tracker: LatencyTracker) -> None:
    """Connect to Odds-API WS and feed updates to tracker."""
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("  [Odds-API] WARNING: ODDS_API_KEY not set, skipping")
        return

    ws_url = (
        f"wss://api.odds-api.io/v3/ws?apiKey={api_key}"
        f"&markets=ML,Spread,Totals&sport=football&status=live"
    )
    reconnect_delay = 1.0
    msg_count = 0

    while tracker.running:
        try:
            async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10, max_size=10_000_000) as ws:
                print("  [Odds-API] Connected")
                reconnect_delay = 1.0
                async for raw in ws:
                    if not tracker.running:
                        break
                    # Odds-API may send multiple JSON objects per WS frame
                    for json_str in _iter_json_lines(raw):
                        msg_count += 1
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
                        tracker.record_odds_api(data)
                    if msg_count % 50 == 0:
                        print(f"  [Odds-API] {msg_count} messages received")
        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if not tracker.running:
                break
            print(f"  [Odds-API] Disconnected: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
        except Exception as exc:
            if not tracker.running:
                break
            print(f"  [Odds-API] Error: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

    print(f"  [Odds-API] Stopped. {msg_count} total messages.")


async def goalserve_poller(tracker: LatencyTracker, match_id: str) -> None:
    """Poll Goalserve every 1s and feed to tracker."""
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("  [Goalserve] WARNING: GOALSERVE_API_KEY not set, skipping")
        return

    url = f"{GOALSERVE_BASE}/{api_key}/soccernew/home"
    poll_count = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        while tracker.running:
            try:
                resp = await asyncio.wait_for(
                    client.get(url, params={"json": "1"}),
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    match_data = _find_match(data, match_id)
                    tracker.record_goalserve(match_data)
                    poll_count += 1
                    if poll_count % 60 == 0:
                        status = match_data.get("@status", "?") if match_data else "not_found"
                        print(f"  [Goalserve] {poll_count} polls, status={status}")
                else:
                    tracker.record_goalserve(None)
            except (asyncio.TimeoutError, httpx.HTTPError, Exception) as exc:
                if poll_count % 30 == 0:
                    print(f"  [Goalserve] Poll error: {exc}")
                tracker.record_goalserve(None)

            await asyncio.sleep(1.0)

    print(f"  [Goalserve] Stopped. {poll_count} total polls.")


def _find_match(live_data: dict, match_id: str) -> dict | None:
    """Search live scores for a match by @id, @fix_id, or @static_id."""
    categories = live_data.get("scores", {}).get("category", [])
    if isinstance(categories, dict):
        categories = [categories]
    for cat in categories:
        matches = cat.get("matches", {}).get("match", [])
        if isinstance(matches, dict):
            matches = [matches]
        for m in matches:
            if match_id in (m.get("@id"), m.get("@fix_id"), m.get("@static_id")):
                return m
    return None


async def kalshi_ws(tracker: LatencyTracker, tickers: list[str]) -> None:
    """Connect to Kalshi WS orderbook and feed updates to tracker."""
    api_key = os.environ.get("KALSHI_API_KEY", "")
    if not api_key or not tickers:
        print(f"  [Kalshi] WARNING: {'no API key' if not api_key else 'no tickers'}, skipping")
        return

    reconnect_delay = 1.0
    msg_count = 0

    while tracker.running:
        try:
            headers = kalshi_sign("GET", "/trade-api/ws/v2")
            async with websockets.connect(
                KALSHI_WS_URL, additional_headers=headers,
                ping_interval=30, ping_timeout=10,
            ) as ws:
                print(f"  [Kalshi] Connected")
                reconnect_delay = 1.0

                # Subscribe to orderbook
                sub = {
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": tickers,
                    },
                }
                await ws.send(json.dumps(sub))
                print(f"  [Kalshi] Subscribed to {len(tickers)} tickers")

                async for raw in ws:
                    if not tracker.running:
                        break
                    msg_count += 1
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    tracker.record_kalshi(data)
                    if msg_count % 50 == 0:
                        print(f"  [Kalshi] {msg_count} messages received")

        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if not tracker.running:
                break
            print(f"  [Kalshi] Disconnected: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
        except Exception as exc:
            if not tracker.running:
                break
            print(f"  [Kalshi] Error: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

    print(f"  [Kalshi] Stopped. {msg_count} total messages.")


async def heartbeat(tracker: LatencyTracker) -> None:
    """Print periodic status summary."""
    while tracker.running:
        await asyncio.sleep(60)
        n_events = len(tracker.events)
        n_odds = len(tracker.odds_state)
        n_kalshi = len(tracker.kalshi_best_ask)
        score = tracker.goalserve_score
        status = tracker.goalserve_status
        print(f"\n  [Heartbeat] score={score[0]}-{score[1]} status={status} "
              f"bookmakers={n_odds} kalshi_tickers={n_kalshi} events={n_events}\n")


# ─── Kalshi Ticker Resolution ────────────────────────────────────────────────

def resolve_kalshi_tickers(league: str, home_team: str, away_team: str) -> list[str]:
    """Find Kalshi tickers for this match via REST API."""
    api_key = os.environ.get("KALSHI_API_KEY", "")
    if not api_key:
        return []

    prefix = LEAGUE_PREFIXES.get(league, "")
    if not prefix:
        return []

    client = httpx.Client(timeout=15)
    path = f"/trade-api/v2/markets?limit=100&status=open&series_ticker={prefix}"
    headers = kalshi_sign("GET", path)
    try:
        resp = client.get(KALSHI_BASE + path, headers=headers)
        if resp.status_code != 200:
            print(f"  [Kalshi] Markets API returned {resp.status_code}")
            return []
        markets = resp.json().get("markets", [])
    except Exception as exc:
        print(f"  [Kalshi] Markets fetch error: {exc}")
        return []
    finally:
        client.close()

    # Match by team names in title
    from src.calibration.team_aliases import normalize_team_name
    home_norm = normalize_team_name(home_team).lower()
    away_norm = normalize_team_name(away_team).lower()

    matched = []
    for m in markets:
        title = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
        ticker = m.get("ticker", "")
        # Check if both team names appear in the title
        if home_norm in title and away_norm in title:
            matched.append(ticker)
        # Also try the event_ticker approach
        elif home_norm[:4] in ticker.lower() or away_norm[:4] in ticker.lower():
            # Looser match on ticker string
            event_ticker = m.get("event_ticker", ticker.rsplit("-", 1)[0])
            if home_norm[:3] in event_ticker.lower() and away_norm[:3] in event_ticker.lower():
                matched.append(ticker)

    return matched


# ─── Match Info from Goalserve ───────────────────────────────────────────────

async def get_match_info(match_id: str, league_id: str) -> dict | None:
    """Fetch match info from Goalserve to get team names."""
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        return None

    url = f"{GOALSERVE_BASE}/{api_key}/soccernew/home"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params={"json": "1"})
        if resp.status_code != 200:
            return None
        data = resp.json()

    match_data = _find_match(data, match_id)
    if match_data:
        import html
        return {
            "home_team": html.unescape(match_data.get("localteam", {}).get("@name", "")),
            "away_team": html.unescape(match_data.get("visitorteam", {}).get("@name", "")),
            "status": match_data.get("@status", ""),
        }

    # Fallback: search all matches in the league
    categories = data.get("scores", {}).get("category", [])
    if isinstance(categories, dict):
        categories = [categories]
    for cat in categories:
        if cat.get("@gid") != league_id and cat.get("@id") != league_id:
            continue
        matches = cat.get("matches", {}).get("match", [])
        if isinstance(matches, dict):
            matches = [matches]
        for m in matches:
            import html
            return {
                "home_team": html.unescape(m.get("localteam", {}).get("@name", "")),
                "away_team": html.unescape(m.get("visitorteam", {}).get("@name", "")),
                "status": m.get("@status", ""),
                "fix_id": m.get("@fix_id", ""),
            }
    return None


# ─── Recording Mode ──────────────────────────────────────────────────────────

async def run_recording(match_id: str, league: str) -> None:
    """Record all three sources simultaneously for a live match."""
    league_id = LEAGUE_IDS.get(league)
    if not league_id:
        print(f"ERROR: Unknown league '{league}'. Options: {list(LEAGUE_IDS.keys())}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"LATENCY MEASUREMENT — match_id={match_id} league={league}")
    print(f"{'='*70}")

    # Get match info
    print("\n[Setup] Fetching match info from Goalserve...")
    info = await get_match_info(match_id, league_id)
    if info:
        print(f"  Match: {info['home_team']} vs {info['away_team']}")
        print(f"  Status: {info['status']}")
        home_team = info["home_team"]
        away_team = info["away_team"]
    else:
        print("  WARNING: Match not found in Goalserve. Will record anyway.")
        home_team = ""
        away_team = ""

    # Resolve Kalshi tickers
    print("\n[Setup] Resolving Kalshi tickers...")
    kalshi_tickers = []
    if home_team and away_team:
        kalshi_tickers = resolve_kalshi_tickers(league, home_team, away_team)
    if kalshi_tickers:
        for t in kalshi_tickers:
            print(f"  {t}")
    else:
        print("  No Kalshi tickers found (will record Odds-API + Goalserve only)")

    # Setup tracker
    match_dir = OUTPUT_DIR / match_id
    tracker = LatencyTracker(match_id, match_dir)

    # Save metadata
    meta = {
        "match_id": match_id,
        "league": league,
        "league_id": league_id,
        "home_team": home_team,
        "away_team": away_team,
        "kalshi_tickers": kalshi_tickers,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(match_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Recording to: {match_dir}")
    print(f"  Press Ctrl+C to stop\n")
    print(f"{'='*70}")

    # Launch all three coroutines
    try:
        await asyncio.gather(
            odds_api_ws(tracker),
            goalserve_poller(tracker, match_id),
            kalshi_ws(tracker, kalshi_tickers),
            heartbeat(tracker),
        )
    except KeyboardInterrupt:
        pass
    finally:
        tracker.running = False
        tracker.finalize()

        # Update metadata
        meta["stopped_utc"] = datetime.now(timezone.utc).isoformat()
        meta["events_detected"] = len(tracker.events)
        meta["final_score"] = list(tracker.goalserve_score)
        with open(match_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Recording stopped. {len(tracker.events)} events detected.")
        print(f"Files saved to: {match_dir}")
        print(f"\nTo analyze: PYTHONPATH=. python scripts/measure_latency.py --analyze {match_dir}")
        print(f"{'='*70}")


# ─── Analysis Mode ───────────────────────────────────────────────────────────

def analyze(match_dir: Path) -> None:
    """Analyze recorded data and compute cross-market lag."""
    print(f"{'='*70}")
    print(f"LATENCY ANALYSIS — {match_dir.name}")
    print(f"{'='*70}")

    # Load metadata
    meta_path = match_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Match: {meta.get('home_team', '?')} vs {meta.get('away_team', '?')}")
        print(f"  League: {meta.get('league', '?')}")
        print(f"  Final score: {meta.get('final_score', '?')}")
    else:
        meta = {}

    # Load events
    events_path = match_dir / "events.jsonl"
    events: list[dict] = []
    if events_path.exists():
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    # Count raw records per source
    for name in ("odds_api", "goalserve", "kalshi"):
        path = match_dir / f"{name}.jsonl"
        if path.exists():
            count = sum(1 for _ in open(path))
            print(f"  {name}: {count} raw records")
        else:
            print(f"  {name}: no data")

    print(f"\n  Total detected events: {len(events)}")

    if not events:
        print("\n  No events detected. Possible reasons:")
        print("  - No goals/events during recording")
        print("  - Match hadn't started or already finished")
        print("  - Data sources not delivering for this match")
        return

    # Categorize events by source
    odds_events = [e for e in events if e["source"] == "odds_api"]
    gs_events = [e for e in events if e["source"] == "goalserve"]
    kalshi_events = [e for e in events if e["source"] == "kalshi"]

    print(f"\n  Events by source:")
    print(f"    Odds-API moves:    {len(odds_events)}")
    print(f"    Goalserve events:  {len(gs_events)}")
    print(f"    Kalshi moves:      {len(kalshi_events)}")

    # ─── Cluster events into match incidents ─────────────────────────────
    # Group events that occur within 60s of each other into "incidents"
    all_events_sorted = sorted(events, key=lambda e: e["ts_wall"])
    incidents: list[list[dict]] = []
    current_cluster: list[dict] = []

    for evt in all_events_sorted:
        if not current_cluster or (evt["ts_wall"] - current_cluster[0]["ts_wall"]) < 60:
            current_cluster.append(evt)
        else:
            incidents.append(current_cluster)
            current_cluster = [evt]
    if current_cluster:
        incidents.append(current_cluster)

    # Filter to incidents that have events from multiple sources (likely real events)
    multi_source = [inc for inc in incidents if len({e["source"] for e in inc}) >= 2]

    print(f"\n  Event clusters (within 60s window): {len(incidents)}")
    print(f"  Multi-source clusters: {len(multi_source)}")

    # ─── Compute cross-market lag ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-MARKET LAG MEASUREMENTS")
    print(f"{'='*70}")

    lag_measurements = []

    for i, incident in enumerate(incidents):
        sources = {e["source"] for e in incident}
        t_odds = None
        t_goalserve = None
        t_kalshi = None
        odds_bookie = None

        for e in sorted(incident, key=lambda x: x["ts_wall"]):
            if e["source"] == "odds_api" and t_odds is None:
                t_odds = e["ts_wall"]
                odds_bookie = e.get("bookie", "?")
            elif e["source"] == "goalserve" and t_goalserve is None:
                t_goalserve = e["ts_wall"]
            elif e["source"] == "kalshi" and t_kalshi is None:
                t_kalshi = e["ts_wall"]

        # Print incident
        first_ts = incident[0]["ts_wall"]
        first_utc = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%H:%M:%S")
        desc = _describe_incident(incident)
        print(f"\n  Incident #{i+1} at {first_utc} — {desc}")

        for e in sorted(incident, key=lambda x: x["ts_wall"]):
            ts_str = datetime.fromtimestamp(e["ts_wall"], tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            if e["source"] == "odds_api":
                print(f"    {ts_str} [Odds-API] {e.get('bookie','?')} moved {e.get('move',0)*100:+.1f}%")
            elif e["source"] == "goalserve":
                print(f"    {ts_str} [Goalserve] {e.get('type','?')}: {e.get('prev_score','?')}→{e.get('new_score','?')}")
            elif e["source"] == "kalshi":
                print(f"    {ts_str} [Kalshi] {e.get('ticker','?')[-15:]} moved {e.get('move',0)*100:+.1f}¢")

        # Compute lags
        measurement = {"incident": i + 1, "utc": first_utc, "description": desc}
        if t_odds is not None and t_kalshi is not None:
            lag = t_kalshi - t_odds
            measurement["odds_to_kalshi_s"] = round(lag, 2)
            measurement["odds_bookie"] = odds_bookie
            print(f"    → CROSS-MARKET LAG (Odds-API → Kalshi): {lag:.2f}s")
        if t_odds is not None and t_goalserve is not None:
            lag = t_goalserve - t_odds
            measurement["odds_to_goalserve_s"] = round(lag, 2)
            print(f"    → Odds-API → Goalserve: {lag:.2f}s")
        if t_goalserve is not None and t_kalshi is not None:
            lag = t_kalshi - t_goalserve
            measurement["goalserve_to_kalshi_s"] = round(lag, 2)
            print(f"    → Goalserve → Kalshi: {lag:.2f}s")

        lag_measurements.append(measurement)

    # ─── Summary ─────────────────────────────────────────────────────────
    cross_lags = [m["odds_to_kalshi_s"] for m in lag_measurements if "odds_to_kalshi_s" in m]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if cross_lags:
        cross_lags_sorted = sorted(cross_lags)
        median = cross_lags_sorted[len(cross_lags_sorted) // 2]
        mean = sum(cross_lags) / len(cross_lags)
        print(f"\n  Cross-market lag (Odds-API → Kalshi):")
        print(f"    Measurements: {len(cross_lags)}")
        print(f"    Median:       {median:.2f}s")
        print(f"    Mean:         {mean:.2f}s")
        print(f"    Min:          {min(cross_lags):.2f}s")
        print(f"    Max:          {max(cross_lags):.2f}s")

        print(f"\n  VERDICT (§3.7.5 Metric 1):")
        if median > 10:
            print(f"    ✓ STRONG EDGE — median lag {median:.1f}s > 10s. Large tradeable window.")
        elif median > 3:
            print(f"    ~ CAUTIOUS — median lag {median:.1f}s. Edge exists but thin.")
        else:
            print(f"    ✗ NO EDGE — median lag {median:.1f}s < 3s. Kalshi reacts too fast.")
    else:
        print("\n  No cross-market lag measurements available.")
        print("  Need events detected by both Odds-API AND Kalshi.")

    # Save report
    report = {
        "match_id": meta.get("match_id", match_dir.name),
        "home_team": meta.get("home_team", "?"),
        "away_team": meta.get("away_team", "?"),
        "total_events": len(events),
        "incidents": len(incidents),
        "multi_source_incidents": len(multi_source),
        "lag_measurements": lag_measurements,
        "summary": {
            "cross_market_lags": cross_lags,
            "median_lag_s": cross_lags_sorted[len(cross_lags_sorted) // 2] if cross_lags else None,
            "mean_lag_s": sum(cross_lags) / len(cross_lags) if cross_lags else None,
            "n_measurements": len(cross_lags),
        },
    }
    report_path = match_dir / "latency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")


def _describe_incident(events: list[dict]) -> str:
    """Generate a one-line description of an incident cluster."""
    types = {e.get("type") for e in events}
    if "goal" in types:
        goal = next(e for e in events if e.get("type") == "goal")
        return f"Goal ({goal.get('team','?')}) {goal.get('prev_score','?')}→{goal.get('new_score','?')}"
    if "status_change" in types:
        sc = next(e for e in events if e.get("type") == "status_change")
        return f"Status: {sc.get('prev_status','?')}→{sc.get('new_status','?')}"
    n_odds = sum(1 for e in events if e["source"] == "odds_api")
    n_kalshi = sum(1 for e in events if e["source"] == "kalshi")
    return f"Price movement (odds×{n_odds}, kalshi×{n_kalshi})"


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cross-market lag: Odds-API vs Kalshi (§3.7.5 Metric 1)",
    )
    parser.add_argument("--match-id", type=str, help="Goalserve fix_id for the match")
    parser.add_argument("--league", type=str, help="League code (EPL, LaLiga, etc.)")
    parser.add_argument("--analyze", type=str, metavar="DIR", help="Analyze recorded data from DIR")
    args = parser.parse_args()

    if args.analyze:
        analyze(Path(args.analyze))
    elif args.match_id and args.league:
        try:
            asyncio.run(run_recording(args.match_id, args.league))
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        parser.error("Provide --match-id and --league to record, or --analyze DIR to analyze")


if __name__ == "__main__":
    main()
