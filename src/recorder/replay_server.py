"""ReplayServer — replays recorded JSONL data as mock HTTP/WS endpoints.

Reads JSONL files from a latency recording directory and serves them at
configurable speed (1x, 10x, 100x real-time) for offline development
and testing.

Data sources (from data/latency/{match}/):
    kalshi_live.jsonl   → HTTP endpoint (match state: goals, periods, scores)
    kalshi.jsonl        → WS endpoint (orderbook snapshots + deltas)
    odds_api.jsonl      → WS endpoint (bookmaker odds updates)

Timestamp field: _ts_mono (monotonic seconds from recording start).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web

from src.common.logging import get_logger

logger = get_logger("recorder.replay_server")

_TS_FIELDS = {"_ts_mono", "_ts_wall", "_utc"}


def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file, sorted by _ts_mono."""
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("_ts_mono", r.get("ts_mono", 0.0)))
    return records


def _strip_ts(record: dict) -> dict:
    """Remove internal timestamp fields before serving to clients."""
    return {k: v for k, v in record.items() if k not in _TS_FIELDS}


def _get_ts(record: dict) -> float:
    """Extract monotonic timestamp from a record."""
    return record.get("_ts_mono", record.get("ts_mono", 0.0))


def _matchstate_to_api_response(record: dict) -> dict:
    """Wrap a recorded MatchState dict into the Kalshi API response format.

    The recorded data has parsed fields (minute, stoppage, home_score, etc.).
    KalshiLiveDataClient.get_live_data() expects the raw API shape:
        {"live_data": {"details": {status, half, time, home_same_game_score, ...}}}
    """
    minute = record.get("minute", 0)
    stoppage = record.get("stoppage", 0)
    time_str = f"{minute}'" if stoppage == 0 else f"{minute}+{stoppage}'"

    last_play: dict | None = None
    if record.get("last_play_ts") is not None or record.get("last_play_desc") is not None:
        last_play = {
            "occurence_ts": record.get("last_play_ts"),
            "description": record.get("last_play_desc"),
        }

    # significant_events were merged during recording; put them all in home
    significant = record.get("significant_events", [])

    return {
        "live_data": {
            "details": {
                "status": record.get("status", "live"),
                "half": record.get("half", ""),
                "time": time_str,
                "home_same_game_score": record.get("home_score", 0),
                "away_same_game_score": record.get("away_score", 0),
                "last_play": last_play,
                "home_significant_events": significant,
                "away_significant_events": [],
            }
        }
    }


class ReplayServer:
    """Replays recorded match data as mock endpoints.

    Provides:
    - Mock Kalshi live data HTTP endpoint (match state — goals, periods, scores)
    - Mock Kalshi orderbook WS endpoint (orderbook snapshots + deltas)
    - Mock Odds-API WS endpoint (bookmaker odds updates)
    """

    def __init__(self, recording_dir: Path | str, speed: float = 1.0) -> None:
        self.recording_dir = Path(recording_dir)
        self.speed = speed

        # Load recorded data
        self.kalshi_live_records = _load_jsonl(self.recording_dir / "kalshi_live.jsonl")
        self.kalshi_ob_records = _load_jsonl(self.recording_dir / "kalshi.jsonl")
        self.odds_api_records = _load_jsonl(self.recording_dir / "odds_api.jsonl")

        # Kalshi live poll index (advances on each HTTP request)
        self._live_index = 0

        # Server state
        self._live_app: web.Application | None = None
        self._live_runner: web.AppRunner | None = None
        self._odds_ws_app: web.Application | None = None
        self._odds_ws_runner: web.AppRunner | None = None
        self._kalshi_ws_app: web.Application | None = None
        self._kalshi_ws_runner: web.AppRunner | None = None

        self.kalshi_live_port: int = 0
        self.odds_ws_port: int = 0
        self.kalshi_ws_port: int = 0

        logger.info(
            "replay_server_loaded",
            dir=str(self.recording_dir),
            kalshi_live=len(self.kalshi_live_records),
            kalshi_ob=len(self.kalshi_ob_records),
            odds_api=len(self.odds_api_records),
            speed=speed,
        )

    async def start(
        self,
        kalshi_live_port: int = 8555,
        odds_ws_port: int = 8556,
        kalshi_ws_port: int = 8557,
    ) -> None:
        """Start mock HTTP + WS servers."""
        # Kalshi live data HTTP mock
        self._live_app = web.Application()
        self._live_app.router.add_get(
            "/trade-api/v2/milestones", self._serve_milestones,
        )
        self._live_app.router.add_get(
            "/trade-api/v2/live_data/soccer/milestone/{uuid}", self._serve_live_data,
        )
        self._live_runner = web.AppRunner(self._live_app)
        await self._live_runner.setup()
        live_site = web.TCPSite(self._live_runner, "127.0.0.1", kalshi_live_port)
        await live_site.start()
        self.kalshi_live_port = kalshi_live_port

        # Odds-API WS mock
        self._odds_ws_app = web.Application()
        self._odds_ws_app.router.add_get("/ws", self._serve_odds_ws)
        self._odds_ws_runner = web.AppRunner(self._odds_ws_app)
        await self._odds_ws_runner.setup()
        odds_site = web.TCPSite(self._odds_ws_runner, "127.0.0.1", odds_ws_port)
        await odds_site.start()
        self.odds_ws_port = odds_ws_port

        # Kalshi orderbook WS mock
        self._kalshi_ws_app = web.Application()
        self._kalshi_ws_app.router.add_get("/ws", self._serve_kalshi_ws)
        self._kalshi_ws_runner = web.AppRunner(self._kalshi_ws_app)
        await self._kalshi_ws_runner.setup()
        kalshi_site = web.TCPSite(self._kalshi_ws_runner, "127.0.0.1", kalshi_ws_port)
        await kalshi_site.start()
        self.kalshi_ws_port = kalshi_ws_port

        logger.info(
            "replay_server_started",
            kalshi_live_port=kalshi_live_port,
            odds_ws_port=odds_ws_port,
            kalshi_ws_port=kalshi_ws_port,
        )

    async def stop(self) -> None:
        """Stop all mock servers."""
        for runner in (self._live_runner, self._odds_ws_runner, self._kalshi_ws_runner):
            if runner is not None:
                await runner.cleanup()
        self._live_runner = None
        self._odds_ws_runner = None
        self._kalshi_ws_runner = None
        logger.info("replay_server_stopped")

    # ── Kalshi live data (HTTP) ──────────────────────────────────

    async def _serve_milestones(self, request: web.Request) -> web.Response:
        """Return a dummy milestone UUID for resolve_milestone_uuid()."""
        return web.json_response({
            "milestones": [{"id": "replay-milestone-uuid"}],
        })

    async def _serve_live_data(self, request: web.Request) -> web.Response:
        """Return next recorded MatchState wrapped in Kalshi API format.

        Sequential: each poll advances the index by 1. When exhausted,
        returns the last record (match finished state).
        """
        if self._live_index >= len(self.kalshi_live_records):
            if self.kalshi_live_records:
                record = self.kalshi_live_records[-1]
            else:
                return web.json_response({"live_data": {"details": {}}})
        else:
            record = self.kalshi_live_records[self._live_index]
            self._live_index += 1

        response = _matchstate_to_api_response(_strip_ts(record))
        return web.json_response(response)

    # ── Odds-API (WS) ───────────────────────────────────────────

    async def _serve_odds_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Send odds updates via WS with timing from _ts_mono and speed."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "welcome"})

        prev_ts = 0.0
        for record in self.odds_api_records:
            ts = _get_ts(record)
            delay = (ts - prev_ts) / self.speed
            if delay > 0:
                await asyncio.sleep(delay)
            prev_ts = ts

            try:
                await ws.send_json(_strip_ts(record))
            except (ConnectionResetError, ConnectionError):
                break

        await ws.close()
        return ws

    # ── Kalshi orderbook (WS) ────────────────────────────────────

    async def _serve_kalshi_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Send Kalshi orderbook updates via WS with timing from _ts_mono and speed."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "auth_response", "status": "ok"})

        prev_ts = 0.0
        for record in self.kalshi_ob_records:
            ts = _get_ts(record)
            if prev_ts > 0:
                delay = (ts - prev_ts) / self.speed
                if delay > 0:
                    await asyncio.sleep(delay)
            prev_ts = ts

            try:
                await ws.send_json(_strip_ts(record))
            except ConnectionError:
                break

        await ws.close()
        return ws

    def reset(self) -> None:
        """Reset replay state for re-running."""
        self._live_index = 0
