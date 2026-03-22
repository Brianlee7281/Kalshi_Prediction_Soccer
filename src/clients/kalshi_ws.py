"""Kalshi WebSocket client for live orderbook and trade feed.

Connects to Kalshi's streaming API, subscribes to orderbook channels
for specified tickers, and calls callbacks on each update.

Used by src/engine/kalshi_ob_sync.py (Sprint 3) to maintain live P_kalshi.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.common.logging import get_logger

log = get_logger(__name__)

KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class KalshiWSClient:
    """Kalshi WebSocket client for live orderbook + trade feed.

    Connects to Kalshi streaming API, authenticates via RSA-PSS,
    subscribes to orderbook channels for specified tickers.

    For replay mode, pass ws_url to point at a local ReplayServer
    and omit api_key/private_key_path (auth is skipped).

    Usage:
        client = KalshiWSClient(api_key="...", private_key_path="keys/kalshi_private.pem")
        await client.connect(
            tickers=["KXEPLGAME-..."],
            on_orderbook=my_ob_handler,
            on_trade=my_trade_handler,
        )
        # ... runs until disconnect() is called
        await client.disconnect()
    """

    def __init__(
        self,
        api_key: str = "",
        private_key_path: str = "",
        ws_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._ws_url = ws_url or KALSHI_WS_URL
        self._private_key = (
            self._load_private_key(private_key_path)
            if private_key_path
            else None
        )
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._stop = asyncio.Event()
        self._connected = False

    @staticmethod
    def _load_private_key(path: str):
        """Load RSA private key from PEM file."""
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_ws_auth(self) -> dict[str, str] | None:
        """Generate RSA-PSS SHA-256 auth message for WS handshake.

        Returns None in replay mode (no private key loaded).
        """
        if self._private_key is None:
            return None
        timestamp_ms = str(int(time.time() * 1000))
        # For WS auth, sign: timestamp + GET + /trade-api/ws/v2
        message = (timestamp_ms + "GET" + "/trade-api/ws/v2").encode()
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
            "id": 1,
            "cmd": "login",
            "params": {
                "api_key": self._api_key,
                "timestamp": int(timestamp_ms),
                "signature": signature,
            },
        }

    async def connect(
        self,
        tickers: list[str],
        on_orderbook: Callable[[str, dict], Awaitable[None]] | None = None,
        on_trade: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> None:
        """Connect to Kalshi WS, authenticate, subscribe, and process messages.

        Auto-reconnects with exponential backoff (1s base, 30s max).
        Runs until disconnect() is called.

        Args:
            tickers: List of Kalshi ticker strings to subscribe to.
            on_orderbook: Async callback(ticker, orderbook_data) for orderbook updates.
            on_trade: Async callback(ticker, trade_data) for trade updates.
        """
        self._stop.clear()
        base_delay = 1.0
        max_delay = 30.0
        attempt = 0

        while not self._stop.is_set():
            try:
                # Include RSA-PSS auth headers in HTTP upgrade request
                extra_headers = {}
                if self._private_key is not None:
                    timestamp_ms = str(int(time.time() * 1000))
                    message = (timestamp_ms + "GET" + "/trade-api/ws/v2").encode()
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
                    extra_headers = {
                        "KALSHI-ACCESS-KEY": self._api_key,
                        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                        "KALSHI-ACCESS-SIGNATURE": signature,
                    }

                async with websockets.connect(
                    self._ws_url,
                    max_size=10_000_000,
                    additional_headers=extra_headers,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    attempt = 0
                    log.info("kalshi_ws_connected", url=self._ws_url)

                    # Auth is handled via HTTP headers on connect.
                    # In replay mode (no private key), no auth needed.

                    # Subscribe to orderbook channels
                    for ticker in tickers:
                        sub_msg = {
                            "id": 2,
                            "cmd": "subscribe",
                            "params": {
                                "channels": ["orderbook_delta", "trade"],
                                "market_tickers": [ticker],
                            },
                        }
                        await ws.send(json.dumps(sub_msg))
                        log.info("kalshi_ws_subscribed", ticker=ticker)

                    # Process messages
                    async for raw_msg in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw_msg)
                        except (json.JSONDecodeError, TypeError):
                            continue

                        msg_type = msg.get("type", "")
                        sid = msg.get("sid")  # sequence ID

                        if msg_type == "orderbook_snapshot" and on_orderbook:
                            ticker = msg.get("msg", {}).get("market_ticker", "")
                            await on_orderbook(ticker, msg)
                        elif msg_type == "orderbook_delta" and on_orderbook:
                            ticker = msg.get("msg", {}).get("market_ticker", "")
                            await on_orderbook(ticker, msg)
                        elif msg_type == "trade" and on_trade:
                            ticker = msg.get("msg", {}).get("market_ticker", "")
                            await on_trade(ticker, msg.get("msg", {}))

            except websockets.ConnectionClosed as exc:
                log.warning("kalshi_ws_closed", code=exc.code, reason=exc.reason)
            except (OSError, asyncio.TimeoutError) as exc:
                log.warning("kalshi_ws_error", error=str(exc))

            self._connected = False
            if self._stop.is_set():
                break

            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            log.info("kalshi_ws_reconnecting", attempt=attempt, delay_s=delay)
            await asyncio.sleep(delay)

        self._ws = None
        self._connected = False

    async def disconnect(self) -> None:
        """Signal the WebSocket loop to stop and close connection."""
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
