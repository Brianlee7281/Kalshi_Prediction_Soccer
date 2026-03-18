"""ReplayServer — replays recorded JSONL data as mock HTTP/WS endpoints.

Reads JSONL files from a recording directory and serves them at
configurable speed (1x, 10x, 100x real-time) for offline development
and testing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web

from src.common.logging import get_logger

logger = get_logger("recorder.replay_server")


def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file, sorted by _ts."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("_ts", 0.0))
    return records


class ReplayServer:
    """Replays recorded match data as mock endpoints.

    Provides:
    - Mock Goalserve HTTP endpoint (returns poll responses in sequence)
    - Mock Odds-API WS endpoint (sends odds updates with timing)
    - Mock Kalshi WS endpoint (sends orderbook snapshots/deltas)
    """

    def __init__(self, recording_dir: Path | str, speed: float = 1.0) -> None:
        self.recording_dir = Path(recording_dir)
        self.speed = speed

        # Load recorded data
        self.goalserve_records = _load_jsonl(self.recording_dir / "goalserve.jsonl")
        self.odds_api_records = _load_jsonl(self.recording_dir / "odds_api.jsonl")
        self.kalshi_ob_records = _load_jsonl(self.recording_dir / "kalshi_ob.jsonl")

        # Goalserve poll index (advances on each request)
        self._gs_index = 0

        # Server state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._ws_app: web.Application | None = None
        self._ws_runner: web.AppRunner | None = None
        self._kalshi_ws_app: web.Application | None = None
        self._kalshi_ws_runner: web.AppRunner | None = None

        self.goalserve_port: int = 0
        self.odds_ws_port: int = 0
        self.kalshi_ws_port: int = 0

        logger.info(
            "replay_server_loaded",
            dir=str(self.recording_dir),
            goalserve=len(self.goalserve_records),
            odds_api=len(self.odds_api_records),
            kalshi_ob=len(self.kalshi_ob_records),
            speed=speed,
        )

    async def start(
        self, goalserve_port: int = 8555, odds_ws_port: int = 8556,
        kalshi_ws_port: int = 8557,
    ) -> None:
        """Start mock HTTP + WS servers."""
        # Goalserve HTTP mock
        self._app = web.Application()
        self._app.router.add_get("/soccernew/home", self._serve_goalserve)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        gs_site = web.TCPSite(self._runner, "127.0.0.1", goalserve_port)
        await gs_site.start()
        self.goalserve_port = goalserve_port

        # Odds-API WS mock
        self._ws_app = web.Application()
        self._ws_app.router.add_get("/ws", self._serve_odds_ws)
        self._ws_runner = web.AppRunner(self._ws_app)
        await self._ws_runner.setup()
        ws_site = web.TCPSite(self._ws_runner, "127.0.0.1", odds_ws_port)
        await ws_site.start()
        self.odds_ws_port = odds_ws_port

        # Kalshi WS mock
        self._kalshi_ws_app = web.Application()
        self._kalshi_ws_app.router.add_get("/ws", self._serve_kalshi_ws)
        self._kalshi_ws_runner = web.AppRunner(self._kalshi_ws_app)
        await self._kalshi_ws_runner.setup()
        kalshi_site = web.TCPSite(self._kalshi_ws_runner, "127.0.0.1", kalshi_ws_port)
        await kalshi_site.start()
        self.kalshi_ws_port = kalshi_ws_port

        logger.info(
            "replay_server_started",
            goalserve_port=goalserve_port,
            odds_ws_port=odds_ws_port,
            kalshi_ws_port=kalshi_ws_port,
        )

    async def stop(self) -> None:
        """Stop all mock servers."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._ws_runner is not None:
            await self._ws_runner.cleanup()
            self._ws_runner = None
        if self._kalshi_ws_runner is not None:
            await self._kalshi_ws_runner.cleanup()
            self._kalshi_ws_runner = None
        logger.info("replay_server_stopped")

    async def _serve_goalserve(self, request: web.Request) -> web.Response:
        """Return next Goalserve poll response based on sequential index."""
        if self._gs_index >= len(self.goalserve_records):
            # Replay exhausted — return last record or empty
            if self.goalserve_records:
                data = self.goalserve_records[-1]
            else:
                data = {}
        else:
            data = self.goalserve_records[self._gs_index]
            self._gs_index += 1

        # Strip internal _ts before serving
        response = {k: v for k, v in data.items() if k != "_ts"}
        return web.json_response(response)

    async def _serve_odds_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Send odds updates via WS with timing based on _ts and speed."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Send welcome message
        await ws.send_json({"type": "welcome"})

        prev_ts = 0.0
        for record in self.odds_api_records:
            ts = record.get("_ts", 0.0)
            delay = (ts - prev_ts) / self.speed
            if delay > 0:
                await asyncio.sleep(delay)
            prev_ts = ts

            msg = {k: v for k, v in record.items() if k != "_ts"}
            try:
                await ws.send_json(msg)
            except (ConnectionResetError, ConnectionError):
                break

        await ws.close()
        return ws

    async def _serve_kalshi_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Send Kalshi orderbook updates via WS with timing based on _ts and speed."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "auth_response", "status": "ok"})
        prev_ts = 0.0
        for record in self.kalshi_ob_records:
            ts = record.get("_ts", 0.0)
            if prev_ts > 0:
                delay = (ts - prev_ts) / self.speed
                if delay > 0:
                    await asyncio.sleep(delay)
            prev_ts = ts
            clean = {k: v for k, v in record.items() if k != "_ts"}
            try:
                await ws.send_json(clean)
            except ConnectionError:
                break
        await ws.close()
        return ws

    def reset(self) -> None:
        """Reset replay state for re-running."""
        self._gs_index = 0
