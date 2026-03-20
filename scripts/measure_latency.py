#!/usr/bin/env python3
"""
Latency Measurement Tool — Cross-Market Lag (architecture.md §3.7.5 Metric 1)
==============================================================================
Simultaneously connects to Odds-API WS, Kalshi live data (1s poll), and Kalshi WS
for a specific live match. Detects significant price movements and events from each
source, logs timestamps, and computes cross-market lag after the match.

Key question: how fast does the Kalshi orderbook react after the Kalshi live event
feed reports a goal/event?

Usage:
  PYTHONPATH=. python scripts/measure_latency.py --event-ticker KXSERIEAGAME-25MAR23-JUVROM --league SerieA
  PYTHONPATH=. python scripts/measure_latency.py --analyze data/latency/KXSERIEAGAME-25MAR23-JUVROM

Output: data/latency/{event_ticker}/
  odds_api.jsonl       — raw Odds-API WS messages
  kalshi_live.jsonl    — Kalshi live data poll responses (every 1s)
  kalshi.jsonl         — raw Kalshi WS orderbook messages
  events.jsonl         — detected events with timestamps from all sources
  latency_report.json  — computed cross-market lag per event
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import httpx
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.clients.kalshi_live_data import KalshiLiveDataClient, MatchState

# ─── Config ──────────────────────────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com"
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

LEAGUE_PREFIXES = {
    "EPL": "KXEPLGAME", "LaLiga": "KXLALIGAGAME", "SerieA": "KXSERIEAGAME",
    "Bundesliga": "KXBUNDESLIGAGAME", "Ligue1": "KXLIGUE1GAME",
    "MLS": "KXMLSGAME", "Brasileirao": "KXBRASILEIROGAME", "Argentina": "KXARGPREMDIVGAME",
}

# Movement thresholds
ODDS_MOVE_THRESHOLD = 0.03   # 3% implied probability change
KALSHI_MOVE_THRESHOLD = 0.03  # 3¢ best-ask change

OUTPUT_DIR = Path("data/latency")

_POLL_INTERVAL_S = 1.0


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
    """Collects timestamped events from all sources."""

    def __init__(self, event_ticker: str, match_dir: Path) -> None:
        self.event_ticker = event_ticker
        self.match_dir = match_dir
        self.match_dir.mkdir(parents=True, exist_ok=True)

        self._start_mono = time.monotonic()
        self._start_wall = time.time()

        # File handles (opened lazily)
        self._files: dict[str, object] = {}

        # Price state for movement detection
        self.odds_state: dict[str, float] = {}       # bookie → last home_win implied
        self.kalshi_best_ask: dict[str, float] = {}  # ticker → last best_ask

        # Kalshi live data state (replaces goalserve)
        self.kalshi_live_score: tuple[int, int] = (0, 0)
        self.kalshi_live_half: str = ""
        self._processed_red_cards: set[str] = set()

        # Detected events
        self.events: list[dict] = []

        # Running flag
        self.running = True

    def _get_file(self, name: str):
        if name not in self._files:
            path = self.match_dir / f"{name}.jsonl"
            self._files[name] = open(path, "a", encoding="utf-8")
        return self._files[name]

    def _ts(self) -> tuple[float, float, str]:
        mono = time.monotonic()
        wall = time.time()
        utc = datetime.fromtimestamp(wall, tz=timezone.utc).isoformat()
        return mono, wall, utc

    def record_odds_api(self, message: dict) -> None:
        mono, wall, utc = self._ts()
        record = {"_ts_mono": mono - self._start_mono, "_ts_wall": wall, "_utc": utc, **message}
        f = self._get_file("odds_api")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        if message.get("type") != "updated":
            return
        bookie = message.get("bookie", "")
        for mkt in message.get("markets", []):
            if mkt.get("name") != "ML":
                continue
            home_odds = None
            for o in mkt.get("odds", []):
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

    def record_kalshi_live(self, state: MatchState) -> None:
        """Record Kalshi live data poll and detect goals/period changes/red cards."""
        mono, wall, utc = self._ts()
        record = {
            "_ts_mono": mono - self._start_mono,
            "_ts_wall": wall,
            "_utc": utc,
            **state.model_dump(),
        }
        f = self._get_file("kalshi_live")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        # ── Goal detection ──────────────────────────────────────────────
        prev_home, prev_away = self.kalshi_live_score
        home_diff = state.home_score - prev_home
        away_diff = state.away_score - prev_away

        for _ in range(home_diff):
            evt = {
                "source": "kalshi_live",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "goal",
                "team": "home",
                "minute": state.minute,
                "prev_score": [prev_home, prev_away],
                "new_score": [state.home_score, state.away_score],
                "occurence_ts": state.last_play_ts,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            _print_event(
                f"[KALSHI-LIVE] GOAL home! {prev_home}-{prev_away} → {state.home_score}-{state.away_score}"
            )

        for _ in range(away_diff):
            evt = {
                "source": "kalshi_live",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "goal",
                "team": "away",
                "minute": state.minute,
                "prev_score": [prev_home, prev_away],
                "new_score": [state.home_score, state.away_score],
                "occurence_ts": state.last_play_ts,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            _print_event(
                f"[KALSHI-LIVE] GOAL away! {prev_home}-{prev_away} → {state.home_score}-{state.away_score}"
            )

        self.kalshi_live_score = (state.home_score, state.away_score)

        # ── Period change ───────────────────────────────────────────────
        if state.half and state.half != self.kalshi_live_half and self.kalshi_live_half:
            evt = {
                "source": "kalshi_live",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "period_change",
                "prev_half": self.kalshi_live_half,
                "new_half": state.half,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            _print_event(f"[KALSHI-LIVE] Period: {self.kalshi_live_half} → {state.half}")
            if state.half == "FT":
                self.running = False
        self.kalshi_live_half = state.half

        # ── Red cards from significant_events ──────────────────────────
        for ev in state.significant_events:
            if ev.get("event_type") != "red_card":
                continue
            team = ev.get("team", "")
            player = ev.get("player", "")
            ev_time = ev.get("time", "")
            dedup_key = f"{team}_{player}_{ev_time}"
            if dedup_key in self._processed_red_cards:
                continue
            self._processed_red_cards.add(dedup_key)
            try:
                minute = int(str(ev_time).rstrip("'"))
            except (ValueError, TypeError):
                minute = state.minute
            evt = {
                "source": "kalshi_live",
                "ts_mono": mono,
                "ts_wall": wall,
                "utc": utc,
                "type": "red_card",
                "team": team,
                "player": player,
                "minute": minute,
            }
            self.events.append(evt)
            ef = self._get_file("events")
            ef.write(json.dumps(evt) + "\n")
            ef.flush()
            _print_event(f"[KALSHI-LIVE] RED CARD {team} {player} {minute}'")

    def record_kalshi(self, message: dict) -> None:
        """Record Kalshi WS message and check for orderbook price movement."""
        mono, wall, utc = self._ts()
        record = {"_ts_mono": mono - self._start_mono, "_ts_wall": wall, "_utc": utc, **message}
        f = self._get_file("kalshi")
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

        msg = message.get("msg", message)
        msg_type = msg.get("type", message.get("type", ""))
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        best_ask = None
        if msg_type == "orderbook_snapshot":
            yes_book = msg.get("yes", [])
            if yes_book:
                prices = [float(entry[0]) for entry in yes_book if len(entry) >= 2]
                if prices:
                    best_ask = min(prices) / 100.0
        elif msg_type == "orderbook_delta":
            yes_delta = msg.get("yes", [])
            if yes_delta:
                prices = [
                    float(entry[0]) for entry in yes_delta
                    if len(entry) >= 2 and float(entry[1]) > 0
                ]
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
            _print_event(f"[KALSHI-OB] {short_ticker} {prev:.2f}→{best_ask:.2f} "
                         f"({direction} {abs(best_ask-prev)*100:.1f}¢)")

    def finalize(self) -> None:
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
            async with websockets.connect(
                ws_url, ping_interval=30, ping_timeout=10, max_size=10_000_000,
            ) as ws:
                print("  [Odds-API] Connected")
                reconnect_delay = 1.0
                async for raw in ws:
                    if not tracker.running:
                        break
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


async def kalshi_live_poller(tracker: LatencyTracker, event_ticker: str) -> None:
    """Poll Kalshi live data every 1s, detect goals/events, feed to tracker."""
    api_key = os.environ.get("KALSHI_API_KEY", "")
    private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")

    client = KalshiLiveDataClient(api_key=api_key, private_key_path=private_key_path)
    poll_count = 0

    try:
        print(f"  [Kalshi-Live] Resolving milestone for {event_ticker}...")
        try:
            milestone_uuid = await client.resolve_milestone_uuid(event_ticker)
        except Exception as exc:
            print(f"  [Kalshi-Live] ERROR: failed to resolve milestone: {exc}")
            return
        print(f"  [Kalshi-Live] milestone_uuid={milestone_uuid}")

        next_tick = time.monotonic()
        error_count = 0

        while tracker.running:
            next_tick += _POLL_INTERVAL_S

            try:
                state = await asyncio.wait_for(
                    client.get_live_data(milestone_uuid),
                    timeout=10.0,
                )
                error_count = 0
                tracker.record_kalshi_live(state)
                poll_count += 1
                if poll_count % 60 == 0:
                    score = tracker.kalshi_live_score
                    print(f"  [Kalshi-Live] {poll_count} polls, "
                          f"score={score[0]}-{score[1]} half={tracker.kalshi_live_half}")
            except (asyncio.TimeoutError, Exception) as exc:
                error_count += 1
                if error_count == 1 or error_count % 30 == 0:
                    print(f"  [Kalshi-Live] Waiting for match to go live... ({exc})")

            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        await client.close()

    print(f"  [Kalshi-Live] Stopped. {poll_count} total polls.")


async def kalshi_ws(tracker: LatencyTracker, tickers: list[str]) -> None:
    """Connect to Kalshi WS orderbook and feed updates to tracker."""
    api_key = os.environ.get("KALSHI_API_KEY", "")
    if not api_key or not tickers:
        print(f"  [Kalshi-OB] WARNING: {'no API key' if not api_key else 'no tickers'}, skipping")
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
                print(f"  [Kalshi-OB] Connected")
                reconnect_delay = 1.0

                sub = {
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": tickers,
                    },
                }
                await ws.send(json.dumps(sub))
                print(f"  [Kalshi-OB] Subscribed to {len(tickers)} tickers")

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
                        print(f"  [Kalshi-OB] {msg_count} messages received")

        except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
            if not tracker.running:
                break
            print(f"  [Kalshi-OB] Disconnected: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
        except Exception as exc:
            if not tracker.running:
                break
            print(f"  [Kalshi-OB] Error: {exc}. Reconnecting in {reconnect_delay:.0f}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

    print(f"  [Kalshi-OB] Stopped. {msg_count} total messages.")


async def heartbeat(tracker: LatencyTracker) -> None:
    """Print periodic status summary."""
    while tracker.running:
        await asyncio.sleep(60)
        score = tracker.kalshi_live_score
        half = tracker.kalshi_live_half
        n_events = len(tracker.events)
        n_odds = len(tracker.odds_state)
        n_kalshi = len(tracker.kalshi_best_ask)
        print(f"\n  [Heartbeat] score={score[0]}-{score[1]} half={half} "
              f"bookmakers={n_odds} kalshi_tickers={n_kalshi} events={n_events}\n")


# ─── Kalshi Ticker + Team Name Resolution ─────────────────────────────────────

def resolve_kalshi_tickers(event_ticker: str) -> tuple[list[str], str, str]:
    """Find market tickers and team names for this event via REST API.

    Returns (market_tickers, home_team, away_team).
    """
    api_key = os.environ.get("KALSHI_API_KEY", "")
    if not api_key:
        return [], "", ""

    client = httpx.Client(timeout=15)
    path = f"/trade-api/v2/markets?limit=100&status=open&event_ticker={event_ticker}"
    headers = kalshi_sign("GET", path)
    try:
        resp = client.get(KALSHI_BASE + path, headers=headers)
        if resp.status_code != 200:
            print(f"  [Kalshi] Markets API returned {resp.status_code}")
            return [], "", ""
        markets = resp.json().get("markets", [])
    except Exception as exc:
        print(f"  [Kalshi] Markets fetch error: {exc}")
        return [], "", ""
    finally:
        client.close()

    if not markets:
        return [], "", ""

    tickers = [m["ticker"] for m in markets if m.get("ticker")]

    # Extract team names from the first market title
    # Kalshi titles look like "Arsenal vs Chelsea - Winner"
    home_team = ""
    away_team = ""
    title = markets[0].get("title", "") + " " + markets[0].get("subtitle", "")
    if " vs " in title:
        parts = title.split(" vs ")
        home_team = parts[0].strip()
        away_team = parts[1].split(" - ")[0].strip() if " - " in parts[1] else parts[1].strip()

    return tickers, home_team, away_team


# ─── Recording Mode ──────────────────────────────────────────────────────────

async def run_recording(event_ticker: str, league: str) -> None:
    """Record all sources simultaneously for a live match."""
    print(f"{'='*70}")
    print(f"LATENCY MEASUREMENT — event_ticker={event_ticker} league={league}")
    print(f"{'='*70}")

    # Resolve Kalshi market tickers and team names
    print("\n[Setup] Resolving Kalshi market tickers...")
    kalshi_tickers, home_team, away_team = resolve_kalshi_tickers(event_ticker)
    if home_team and away_team:
        print(f"  Match: {home_team} vs {away_team}")
    if kalshi_tickers:
        for t in kalshi_tickers:
            print(f"  {t}")
    else:
        print("  No market tickers found (will record Odds-API + Kalshi-Live only)")

    # Setup tracker
    safe_ticker = event_ticker.replace("/", "_")
    match_dir = OUTPUT_DIR / safe_ticker
    tracker = LatencyTracker(event_ticker, match_dir)

    # Save metadata
    meta = {
        "event_ticker": event_ticker,
        "league": league,
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

    try:
        await asyncio.gather(
            odds_api_ws(tracker),
            kalshi_live_poller(tracker, event_ticker),
            kalshi_ws(tracker, kalshi_tickers),
            heartbeat(tracker),
        )
    except KeyboardInterrupt:
        pass
    finally:
        tracker.running = False
        tracker.finalize()

        meta["stopped_utc"] = datetime.now(timezone.utc).isoformat()
        meta["events_detected"] = len(tracker.events)
        meta["final_score"] = list(tracker.kalshi_live_score)
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

    meta_path = match_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Match: {meta.get('home_team', '?')} vs {meta.get('away_team', '?')}")
        print(f"  League: {meta.get('league', '?')}")
        print(f"  Final score: {meta.get('final_score', '?')}")
    else:
        meta = {}

    events_path = match_dir / "events.jsonl"
    events: list[dict] = []
    if events_path.exists():
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    for name in ("odds_api", "kalshi_live", "kalshi"):
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

    odds_events = [e for e in events if e["source"] == "odds_api"]
    live_events = [e for e in events if e["source"] == "kalshi_live"]
    kalshi_events = [e for e in events if e["source"] == "kalshi"]

    print(f"\n  Events by source:")
    print(f"    Odds-API moves:         {len(odds_events)}")
    print(f"    Kalshi-Live events:     {len(live_events)}")
    print(f"    Kalshi orderbook moves: {len(kalshi_events)}")

    # ─── Cluster events into match incidents ──────────────────────────
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

    multi_source = [inc for inc in incidents if len({e["source"] for e in inc}) >= 2]

    print(f"\n  Event clusters (within 60s window): {len(incidents)}")
    print(f"  Multi-source clusters: {len(multi_source)}")

    # ─── Compute cross-market lag ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-MARKET LAG MEASUREMENTS")
    print(f"{'='*70}")

    lag_measurements = []

    for i, incident in enumerate(incidents):
        t_odds = None
        t_kalshi_live = None
        t_kalshi = None
        odds_bookie = None

        for e in sorted(incident, key=lambda x: x["ts_wall"]):
            if e["source"] == "odds_api" and t_odds is None:
                t_odds = e["ts_wall"]
                odds_bookie = e.get("bookie", "?")
            elif e["source"] == "kalshi_live" and t_kalshi_live is None:
                t_kalshi_live = e["ts_wall"]
            elif e["source"] == "kalshi" and t_kalshi is None:
                t_kalshi = e["ts_wall"]

        first_ts = incident[0]["ts_wall"]
        first_utc = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%H:%M:%S")
        desc = _describe_incident(incident)
        print(f"\n  Incident #{i+1} at {first_utc} — {desc}")

        for e in sorted(incident, key=lambda x: x["ts_wall"]):
            ts_str = datetime.fromtimestamp(e["ts_wall"], tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            if e["source"] == "odds_api":
                print(f"    {ts_str} [Odds-API]     {e.get('bookie','?')} moved {e.get('move',0)*100:+.1f}%")
            elif e["source"] == "kalshi_live":
                etype = e.get("type", "?")
                if etype == "goal":
                    print(f"    {ts_str} [Kalshi-Live] GOAL {e.get('team','?')}: "
                          f"{e.get('prev_score','?')}→{e.get('new_score','?')}")
                elif etype == "red_card":
                    print(f"    {ts_str} [Kalshi-Live] RED CARD {e.get('team','?')} {e.get('player','?')}")
                elif etype == "period_change":
                    print(f"    {ts_str} [Kalshi-Live] Period {e.get('prev_half','?')}→{e.get('new_half','?')}")
                else:
                    print(f"    {ts_str} [Kalshi-Live] {etype}")
            elif e["source"] == "kalshi":
                print(f"    {ts_str} [Kalshi-OB]   {e.get('ticker','?')[-15:]} moved {e.get('move',0)*100:+.1f}¢")

        measurement = {"incident": i + 1, "utc": first_utc, "description": desc}

        if t_kalshi_live is not None and t_kalshi is not None:
            lag = t_kalshi - t_kalshi_live
            measurement["live_to_orderbook_s"] = round(lag, 2)
            print(f"    → KEY: Kalshi-Live event → orderbook move: {lag:.2f}s")

        if t_odds is not None and t_kalshi_live is not None:
            lag = t_kalshi_live - t_odds
            measurement["odds_to_live_s"] = round(lag, 2)
            measurement["odds_bookie"] = odds_bookie
            print(f"    → Odds-API → Kalshi-Live: {lag:.2f}s")

        if t_odds is not None and t_kalshi is not None:
            lag = t_kalshi - t_odds
            measurement["odds_to_kalshi_s"] = round(lag, 2)
            print(f"    → Odds-API → Kalshi orderbook: {lag:.2f}s")

        lag_measurements.append(measurement)

    # ─── Summary ──────────────────────────────────────────────────────
    live_to_ob_lags = [m["live_to_orderbook_s"] for m in lag_measurements if "live_to_orderbook_s" in m]
    cross_lags = [m["odds_to_kalshi_s"] for m in lag_measurements if "odds_to_kalshi_s" in m]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    lags_sorted: list[float] = []
    if live_to_ob_lags:
        lags_sorted = sorted(live_to_ob_lags)
        median = lags_sorted[len(lags_sorted) // 2]
        mean = sum(live_to_ob_lags) / len(live_to_ob_lags)
        print(f"\n  Kalshi-Live event → Kalshi orderbook lag (KEY METRIC):")
        print(f"    Measurements: {len(live_to_ob_lags)}")
        print(f"    Median:       {median:.2f}s")
        print(f"    Mean:         {mean:.2f}s")
        print(f"    Min:          {min(live_to_ob_lags):.2f}s")
        print(f"    Max:          {max(live_to_ob_lags):.2f}s")
        print(f"\n  VERDICT (§3.7.5 Metric 1):")
        if median > 10:
            print(f"    ✓ STRONG EDGE — median lag {median:.1f}s > 10s. Large tradeable window.")
        elif median > 3:
            print(f"    ~ CAUTIOUS — median lag {median:.1f}s. Edge exists but thin.")
        else:
            print(f"    ✗ NO EDGE — median lag {median:.1f}s < 3s. Orderbook reacts too fast.")
    else:
        print("\n  No Kalshi-Live → orderbook lag measurements available.")
        print("  Need events detected by both kalshi_live AND kalshi sources.")

    cross_sorted: list[float] = []
    if cross_lags:
        cross_sorted = sorted(cross_lags)
        median_cross = cross_sorted[len(cross_sorted) // 2]
        print(f"\n  Odds-API → Kalshi orderbook lag:")
        print(f"    Median: {median_cross:.2f}s  (n={len(cross_lags)})")

    # Save report
    report = {
        "event_ticker": meta.get("event_ticker", match_dir.name),
        "home_team": meta.get("home_team", "?"),
        "away_team": meta.get("away_team", "?"),
        "total_events": len(events),
        "incidents": len(incidents),
        "multi_source_incidents": len(multi_source),
        "lag_measurements": lag_measurements,
        "summary": {
            "live_to_orderbook_lags": live_to_ob_lags,
            "median_live_to_ob_s": lags_sorted[len(lags_sorted) // 2] if lags_sorted else None,
            "mean_live_to_ob_s": sum(live_to_ob_lags) / len(live_to_ob_lags) if live_to_ob_lags else None,
            "cross_market_lags": cross_lags,
            "median_cross_lag_s": cross_sorted[len(cross_sorted) // 2] if cross_sorted else None,
            "n_measurements": len(live_to_ob_lags),
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
    if "red_card" in types:
        rc = next(e for e in events if e.get("type") == "red_card")
        return f"Red card ({rc.get('team','?')} {rc.get('player','')})"
    if "period_change" in types:
        pc = next(e for e in events if e.get("type") == "period_change")
        return f"Period: {pc.get('prev_half','?')}→{pc.get('new_half','?')}"
    n_odds = sum(1 for e in events if e["source"] == "odds_api")
    n_kalshi = sum(1 for e in events if e["source"] == "kalshi")
    return f"Price movement (odds×{n_odds}, kalshi×{n_kalshi})"


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cross-market lag: Kalshi-Live event feed vs orderbook (§3.7.5 Metric 1)",
    )
    parser.add_argument(
        "--event-ticker", type=str,
        help="Kalshi event ticker (e.g. KXSERIEAGAME-25MAR23-JUVROM)",
    )
    parser.add_argument("--league", type=str, help="League code (EPL, LaLiga, SerieA, etc.)")
    parser.add_argument("--analyze", type=str, metavar="DIR", help="Analyze recorded data from DIR")
    args = parser.parse_args()

    if args.analyze:
        analyze(Path(args.analyze))
    elif args.event_ticker and args.league:
        try:
            asyncio.run(run_recording(args.event_ticker, args.league))
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        parser.error("Provide --event-ticker and --league to record, or --analyze DIR to analyze")


if __name__ == "__main__":
    main()
