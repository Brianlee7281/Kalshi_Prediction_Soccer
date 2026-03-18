"""Odds-API.io REST + WebSocket client."""

import asyncio
from collections.abc import Awaitable, Callable

import websockets

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

log = get_logger(__name__)

ODDS_API_BASE_URL = "https://api.odds-api.io/v3"
ODDS_API_WS_URL = "wss://api.odds-api.io/v3/ws"

# League slug mapping (from architecture.md §4.3)
LEAGUE_SLUGS: dict[str, str] = {
    "1204": "england-premier-league",
    "1399": "spain-laliga",
    "1269": "italy-serie-a",
    "1229": "germany-bundesliga",
    "1221": "france-ligue-1",
    "1440": "usa-mls",
    "1141": "brazil-brasileiro-serie-a",
    "1081": "argentina-liga-profesional",
}


class OddsApiClient:
    """Odds-API.io REST + WebSocket client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base = BaseClient(
            base_url=ODDS_API_BASE_URL,
            timeout=15.0,
            rate_limit_delay=0.1,
        )
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._ws_stop = asyncio.Event()

    # ─── REST ─────────────────────────────────────────────

    async def get_events(
        self, league_slug: str, status: str = "pending",
    ) -> list[dict]:
        """GET /events?sport=football&league={slug}&status={status}
        Returns list of event dicts with id, teams, commence_time.
        """
        data = await self._base.get(
            "/events",
            params={
                "apiKey": self._api_key,
                "sport": "football",
                "league": league_slug,
                "status": status,
            },
        )
        # API returns a list directly
        if isinstance(data, list):
            return data
        return data.get("events", [])

    async def get_odds(
        self, event_id: str, bookmakers: str = "Bet365,Betfair Exchange",
    ) -> dict:
        """GET /odds?eventId={id}&bookmakers={names}
        bookmakers param is REQUIRED (architecture.md §4.3).
        Returns odds for requested bookmakers.
        """
        return await self._base.get(
            "/odds",
            params={
                "apiKey": self._api_key,
                "eventId": str(event_id),
                "bookmakers": bookmakers,
            },
        )

    async def get_historical_odds(
        self, event_id: str, bookmakers: str = "Bet365",
    ) -> dict:
        """GET /historical/odds?eventId={id}&bookmakers={names}
        Returns settled event with closing odds.
        """
        return await self._base.get(
            "/historical/odds",
            params={
                "apiKey": self._api_key,
                "eventId": str(event_id),
                "bookmakers": bookmakers,
            },
        )

    # ─── WebSocket ────────────────────────────────────────

    async def connect_live_ws(
        self,
        on_message: Callable[[dict], Awaitable[None]],
        markets: str = "ML,Spread,Totals",
        sport: str = "football",
    ) -> None:
        """Connect to wss://api.odds-api.io/v3/ws

        NOTE (v5 migration): In v5 architecture, this live feed is used for
        RECORDING ONLY, not for live trading decisions. P_model from the MMPP
        mathematical model is the sole trading authority. OddsConsensus is
        removed in Sprint 3 migration (Task 3.14).

        Receives: welcome message, then live odds updates.
        Update format: {type: "updated", bookie: "Bet365", markets: [{name: "ML", odds: [...]}]}

        Calls on_message callback for each update.
        Auto-reconnect with exponential backoff (1s base, 30s max, 10 retries).
        """
        self._ws_stop.clear()
        base_delay = 1.0
        max_delay = 30.0
        max_retries = 10
        attempt = 0

        ws_url = (
            f"{ODDS_API_WS_URL}"
            f"?apiKey={self._api_key}"
            f"&markets={markets}"
            f"&sport={sport}"
            f"&status=live"
        )

        while attempt < max_retries and not self._ws_stop.is_set():
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    attempt = 0  # reset on successful connection
                    log.info("odds_api_ws_connected")

                    async for raw_msg in ws:
                        if self._ws_stop.is_set():
                            break
                        import json
                        try:
                            msg = json.loads(raw_msg)
                        except (json.JSONDecodeError, TypeError):
                            log.warning("odds_api_ws_bad_message", raw=str(raw_msg)[:200])
                            continue
                        await on_message(msg)

            except websockets.ConnectionClosed as exc:
                log.warning("odds_api_ws_closed", code=exc.code, reason=exc.reason)
            except (OSError, asyncio.TimeoutError) as exc:
                log.warning("odds_api_ws_error", error=str(exc))

            if self._ws_stop.is_set():
                break

            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            log.info("odds_api_ws_reconnecting", attempt=attempt, delay_s=delay)
            await asyncio.sleep(delay)

        self._ws = None
        if attempt >= max_retries:
            log.error("odds_api_ws_max_retries", max_retries=max_retries)

    def stop_ws(self) -> None:
        """Signal the WebSocket loop to stop."""
        self._ws_stop.set()

    async def close(self) -> None:
        """Close HTTP client and WebSocket."""
        self._ws_stop.set()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        await self._base.close()
