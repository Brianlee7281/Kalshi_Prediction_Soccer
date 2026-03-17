"""Kalshi REST client with RSA-PSS SHA-256 authentication."""

import asyncio
import base64
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

log = get_logger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com"


class KalshiClient:
    """Kalshi REST client with RSA-PSS authentication."""

    def __init__(self, api_key: str, private_key_path: str) -> None:
        self._api_key = api_key
        self._private_key = self._load_private_key(private_key_path)
        self._client = httpx.AsyncClient(base_url=KALSHI_BASE_URL)
        self._rate_limit_delay = 0.1

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

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Signed GET with timeout and rate limiting."""
        headers = self._sign_request("GET", path)
        t0 = time.monotonic()
        response = await asyncio.wait_for(
            self._client.get(path, headers=headers, params=params),
            timeout=15.0,
        )
        duration_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info(
            "kalshi_request",
            method="GET",
            path=path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        response.raise_for_status()
        await asyncio.sleep(self._rate_limit_delay)
        return response.json()

    async def _post(self, path: str, json_body: dict) -> dict:
        """Signed POST with timeout and rate limiting."""
        headers = self._sign_request("POST", path)
        t0 = time.monotonic()
        response = await asyncio.wait_for(
            self._client.post(path, headers=headers, json=json_body),
            timeout=15.0,
        )
        duration_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info(
            "kalshi_request",
            method="POST",
            path=path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        response.raise_for_status()
        await asyncio.sleep(self._rate_limit_delay)
        return response.json()

    async def _delete(self, path: str) -> dict:
        """Signed DELETE with timeout and rate limiting."""
        headers = self._sign_request("DELETE", path)
        t0 = time.monotonic()
        response = await asyncio.wait_for(
            self._client.delete(path, headers=headers),
            timeout=15.0,
        )
        duration_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info(
            "kalshi_request",
            method="DELETE",
            path=path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        response.raise_for_status()
        await asyncio.sleep(self._rate_limit_delay)
        return response.json()

    # ─── Markets ─────────────────────────────────────────────

    async def get_markets(
        self,
        series_ticker: str,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict]:
        """GET /trade-api/v2/markets?series_ticker={prefix}&status={status}

        Handles pagination via cursor. Returns all markets.
        IMPORTANT: List endpoint returns yes_ask=None. Use get_market() for prices.
        """
        all_markets: list[dict] = []
        cursor: str | None = None

        for _ in range(50):  # safety cap
            path = "/trade-api/v2/markets"
            params: dict = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": str(limit),
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._get(path, params=params)
            batch = data.get("markets", [])
            if not batch:
                break
            all_markets.extend(batch)

            cursor = data.get("cursor")
            if not cursor:
                break

        log.info(
            "kalshi_get_markets",
            series_ticker=series_ticker,
            status=status,
            total=len(all_markets),
        )
        return all_markets

    async def get_market(self, ticker: str) -> dict:
        """GET /trade-api/v2/markets/{ticker}

        Returns single market with actual prices (last_price_dollars, yes_ask, etc).
        """
        data = await self._get(f"/trade-api/v2/markets/{ticker}")
        return data.get("market", data)

    async def get_orderbook(self, ticker: str) -> dict:
        """GET /trade-api/v2/markets/{ticker}/orderbook

        Returns {yes: [[price, qty], ...], no: [[price, qty], ...]}.
        """
        return await self._get(f"/trade-api/v2/markets/{ticker}/orderbook")

    async def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """GET /trade-api/v2/markets/trades?ticker={ticker}

        Returns trade list with yes_price_dollars, count_fp, taker_side, created_time.
        """
        data = await self._get(
            "/trade-api/v2/markets/trades",
            params={"ticker": ticker, "limit": str(limit)},
        )
        return data.get("trades", [])

    # ─── Orders ──────────────────────────────────────────────

    async def submit_order(self, order: dict) -> dict:
        """POST /trade-api/v2/orders

        order = {ticker, action, side, type, count, yes_price}
        Returns order response with id, status.
        """
        return await self._post("/trade-api/v2/orders", json_body=order)

    async def cancel_order(self, order_id: str) -> dict:
        """DELETE /trade-api/v2/orders/{order_id}"""
        return await self._delete(f"/trade-api/v2/orders/{order_id}")

    # ─── Portfolio ───────────────────────────────────────────

    async def get_balance(self) -> float:
        """GET /trade-api/v2/portfolio/balance

        Returns balance in dollars.
        """
        data = await self._get("/trade-api/v2/portfolio/balance")
        return data.get("balance", 0.0) / 100  # cents → dollars

    async def get_positions(self) -> list[dict]:
        """GET /trade-api/v2/portfolio/positions"""
        data = await self._get("/trade-api/v2/portfolio/positions")
        return data.get("market_positions", [])

    # ─── Lifecycle ───────────────────────────────────────────

    async def close(self) -> None:
        """Close underlying httpx.AsyncClient."""
        await self._client.aclose()
