"""Odds-API WebSocket listener — feeds live bookmaker odds into OddsConsensus.

DEPRECATED (v5 migration): In v5, OddsConsensus is removed. This module
will be demoted to a recording-only logger in Sprint 3 (Task 3.13).
P_model is the sole trading authority — bookmaker odds are recorded for
post-match analysis only, not used for live trading decisions.

Current behavior (v4): Connects to the Odds-API live WS endpoint, parses
ML (moneyline) odds updates, converts to implied probabilities with vig
removal, and updates the model's OddsConsensus on each message.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import websockets

from src.common.logging import get_logger
from src.common.types import MarketProbs

if TYPE_CHECKING:
    from src.engine.model import LiveMatchModel

logger = get_logger("engine.odds_api_listener")

_WS_URL = "wss://api.odds-api.io/v3/ws"
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


async def odds_api_listener(model: LiveMatchModel) -> None:
    """Coroutine: connect to Odds-API WS, update OddsConsensus on each message.

    Runs until model.engine_phase == "FINISHED".
    Auto-reconnect with exponential backoff (1s base, 30s max).
    Records all raw WS messages to JSONL if recorder is attached.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        logger.error("odds_api_listener_no_key")
        return

    ws_url = (
        f"{_WS_URL}"
        f"?apiKey={api_key}"
        f"&markets=ML,Spread,Totals"
        f"&sport=football"
        f"&status=live"
    )

    attempt = 0

    while model.engine_phase != "FINISHED":
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as ws:
                attempt = 0
                logger.info("odds_api_ws_connected")

                async for raw_msg in ws:
                    if model.engine_phase == "FINISHED":
                        break

                    # Odds-API may send multiple JSON objects per WS frame
                    # (newline-delimited) or a single object
                    for json_str in _iter_json(raw_msg):
                        try:
                            msg = json.loads(json_str)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("odds_api_ws_bad_message", raw=json_str[:200])
                            continue

                        # Record raw message if recorder is attached
                        recorder = getattr(model, "recorder", None)
                        if recorder is not None:
                            recorder.record_odds_api(msg)

                        # Skip non-update messages (e.g. welcome)
                        parsed = _parse_odds_update(msg)
                        if parsed is None:
                            continue

                        bookie, implied = parsed

                        logger.debug(
                            "odds_recorded",
                            bookmaker=bookie,
                            home_win=round(implied.home_win, 4),
                        )

        except websockets.ConnectionClosed as exc:
            logger.warning("odds_api_ws_closed", code=exc.code, reason=exc.reason)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("odds_api_ws_error", error=str(exc))

        if model.engine_phase == "FINISHED":
            break

        attempt += 1
        delay = min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)
        logger.info("odds_api_ws_reconnecting", attempt=attempt, delay_s=delay)
        await asyncio.sleep(delay)


def _parse_odds_update(message: dict) -> tuple[str, MarketProbs] | None:
    """Parse an Odds-API WS 'updated' message into (bookmaker_name, MarketProbs).

    Returns None if message is not relevant (wrong type, missing fields).
    """
    if message.get("type") != "updated":
        return None

    bookie = message.get("bookie")
    if not bookie:
        return None

    markets = message.get("markets")
    if not markets:
        return None

    # Find ML (moneyline) market
    ml_market = None
    for mkt in markets:
        if mkt.get("name") == "ML":
            ml_market = mkt
            break

    if ml_market is None:
        return None

    odds_list = ml_market.get("odds")
    if not odds_list:
        return None

    # Extract home/draw/away decimal odds
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None

    for entry in odds_list:
        name = entry.get("name", "").lower()
        price = entry.get("price")
        if price is None or price <= 0:
            continue
        if name == "home":
            home_odds = float(price)
        elif name == "draw":
            draw_odds = float(price)
        elif name == "away":
            away_odds = float(price)

    if home_odds is None or draw_odds is None or away_odds is None:
        return None

    implied = _odds_to_implied(home_odds, draw_odds, away_odds)
    return bookie, implied


def _iter_json(raw: str) -> list[str]:
    """Split a WS frame that may contain multiple JSON objects.

    Odds-API sometimes delivers multiple JSON objects in a single WS
    text frame (newline-separated or concatenated). This function
    extracts each complete JSON object.
    """
    # Fast path: single JSON object
    raw = raw.strip()
    if not raw:
        return []

    # Try parsing as a single object first
    try:
        json.loads(raw)
        return [raw]
    except (json.JSONDecodeError, TypeError):
        pass

    # Split by newlines and try each line
    parts = []
    for line in raw.split("\n"):
        line = line.strip()
        if line:
            parts.append(line)

    if parts:
        return parts

    # Last resort: return the whole thing for the caller to handle
    return [raw]


def _odds_to_implied(home: float, draw: float, away: float) -> MarketProbs:
    """Convert decimal odds to implied probabilities with vig removal.

    p_raw = 1/odds for each outcome, then normalize so sum = 1.0.
    """
    p_home = 1.0 / home
    p_draw = 1.0 / draw
    p_away = 1.0 / away

    total = p_home + p_draw + p_away

    return MarketProbs(
        home_win=p_home / total,
        draw=p_draw / total,
        away_win=p_away / total,
    )
