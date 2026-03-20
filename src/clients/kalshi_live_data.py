"""Kalshi live data client for soccer match state."""

import asyncio
import base64
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic import BaseModel

from src.common.logging import get_logger

log = get_logger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com"


class MatchState(BaseModel):
    status: str                      # "live" | "finished" | "halftime"
    half: str                        # "1st" | "2nd" | "HT" | "FT"
    minute: int                      # 0–90
    stoppage: int                    # 0 normally; 3 from "90+3'"
    home_score: int
    away_score: int
    last_play_ts: int | None         # occurence_ts (Sportradar real event time)
    last_play_desc: str | None
    significant_events: list[dict]   # raw list, preserved for future use


class KalshiLiveDataClient:
    """Kalshi live data client for soccer match state via signed REST calls."""

    def __init__(self, api_key: str, private_key_path: str) -> None:
        self._api_key = api_key
        self._private_key = self._load_private_key(private_key_path)
        self._client = httpx.AsyncClient(base_url=KALSHI_BASE_URL)

    @staticmethod
    def _load_private_key(path: str):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-PSS SHA-256 auth headers.

        Returns dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP,
        KALSHI-ACCESS-SIGNATURE.
        Padding: PSS(mgf=MGF1(SHA256), salt_length=MAX_LENGTH).
        Signature = base64(sign(timestamp_ms + METHOD + path))
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode()
        signature = base64.b64encode(
            self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
        ).decode()
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    async def resolve_milestone_uuid(self, event_ticker: str) -> str:
        """GET /trade-api/v2/milestones?related_event_ticker={event_ticker}&limit=5

        Returns the id of the first milestone for the given event ticker.
        Raises ValueError if no milestones are found.
        """
        path = "/trade-api/v2/milestones"
        headers = self._sign_request("GET", path)
        params = {"related_event_ticker": event_ticker, "limit": 5}
        response = await asyncio.wait_for(
            self._client.get(path, headers=headers, params=params),
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        milestones = data.get("milestones", [])
        if not milestones:
            raise ValueError(f"No milestones found for event_ticker={event_ticker!r}")
        milestone_id: str = milestones[0]["id"]
        log.info(
            "kalshi_resolve_milestone",
            event_ticker=event_ticker,
            milestone_id=milestone_id,
        )
        return milestone_id

    async def get_live_data(self, milestone_uuid: str) -> MatchState:
        """GET /trade-api/v2/live_data/soccer/{milestone_uuid}

        Parses the response into a MatchState.
        """
        path = f"/trade-api/v2/live_data/soccer/milestone/{milestone_uuid}"
        headers = self._sign_request("GET", path)
        response = await asyncio.wait_for(
            self._client.get(path, headers=headers),
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        details = data["live_data"]["details"]

        minute, stoppage = self._parse_time_field(details.get("time", ""))

        raw_half: str = details.get("half", "")
        raw_status: str = details.get("status", "")

        if raw_status == "finished" or details.get("winner", "") != "":
            half = "FT"
        elif raw_half == "1st":
            half = "1st"
        elif raw_half == "2nd":
            half = "2nd"
        elif raw_status == "live" and raw_half == "HT":
            half = "HT"
        else:
            half = raw_half

        last_play: dict = details.get("last_play") or {}
        last_play_ts: int | None = last_play.get("occurence_ts")
        last_play_desc: str | None = last_play.get("description")

        significant_events: list[dict] = (
            (details.get("home_significant_events") or [])
            + (details.get("away_significant_events") or [])
        )

        state = MatchState(
            status=raw_status,
            half=half,
            minute=minute,
            stoppage=stoppage,
            home_score=details.get("home_same_game_score", 0),
            away_score=details.get("away_same_game_score", 0),
            last_play_ts=last_play_ts,
            last_play_desc=last_play_desc,
            significant_events=significant_events,
        )
        log.debug(
            "kalshi_live_data",
            milestone_uuid=milestone_uuid,
            half=half,
            minute=minute,
            stoppage=stoppage,
        )
        return state

    def _parse_time_field(self, time_str: str) -> tuple[int, int]:
        """Parse Kalshi time strings into (minute, stoppage).

        Examples:
            "62'"   → (62, 0)
            "90+3'" → (90, 3)
            "45+1'" → (45, 1)
            ""      → (0, 0)
            "0'"    → (0, 0)
        """
        stripped = time_str.rstrip("'").strip()
        if not stripped:
            return (0, 0)
        parts = stripped.split("+")
        minute = int(parts[0])
        stoppage = int(parts[1]) if len(parts) > 1 else 0
        return (minute, stoppage)

    async def close(self) -> None:
        """Close underlying httpx.AsyncClient."""
        await self._client.aclose()
