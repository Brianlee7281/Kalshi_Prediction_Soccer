"""Tests for KalshiClient — RSA-PSS auth, markets, orderbook, trades."""

import json
import os
from pathlib import Path

import pytest

from src.clients.kalshi import KalshiClient

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _have_kalshi_creds() -> bool:
    return bool(
        os.environ.get("KALSHI_API_KEY")
        and Path(os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")).exists()
    )


# ── Unit tests (fixtures, no API) ──────────────────────────


def test_sign_request_headers():
    """Verify _sign_request returns required header keys."""
    if not Path("keys/kalshi_private.pem").exists():
        pytest.skip("kalshi_private.pem not found")
    client = KalshiClient(api_key="test-key", private_key_path="keys/kalshi_private.pem")
    headers = client._sign_request("GET", "/trade-api/v2/markets")
    assert "KALSHI-ACCESS-KEY" in headers
    assert headers["KALSHI-ACCESS-KEY"] == "test-key"
    assert "KALSHI-ACCESS-TIMESTAMP" in headers
    assert "KALSHI-ACCESS-SIGNATURE" in headers
    # Timestamp should be numeric milliseconds
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
    # Signature should be valid base64
    import base64
    base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])


def test_fixture_market_has_ticker():
    """Verify saved market fixture has expected fields."""
    with open(FIXTURE_DIR / "kalshi_market.json") as f:
        market = json.load(f)
    assert "ticker" in market
    assert market["ticker"].startswith("KXEPLGAME")
    assert "last_price_dollars" in market


def test_fixture_orderbook_structure():
    """Verify saved orderbook fixture has expected structure."""
    with open(FIXTURE_DIR / "kalshi_orderbook.json") as f:
        ob = json.load(f)
    assert "orderbook_fp" in ob


def test_fixture_trades_structure():
    """Verify saved trades fixture has expected fields."""
    with open(FIXTURE_DIR / "kalshi_trades.json") as f:
        trades = json.load(f)
    assert isinstance(trades, list)
    if trades:
        assert "yes_price_dollars" in trades[0]
        assert "taker_side" in trades[0]
        assert "created_time" in trades[0]


# ── Live API tests (require KALSHI_API_KEY + private key) ──


@pytest.mark.asyncio
async def test_kalshi_auth_and_markets():
    """Verify RSA-PSS auth works and can fetch EPL markets."""
    if not _have_kalshi_creds():
        pytest.skip("Kalshi credentials not available")

    from src.common.config import Config
    config = Config.from_env()
    client = KalshiClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )
    try:
        markets = await client.get_markets("KXEPLGAME", limit=5)
        assert len(markets) > 0
        assert "ticker" in markets[0]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kalshi_orderbook():
    """Fetch orderbook for a real ticker."""
    if not _have_kalshi_creds():
        pytest.skip("Kalshi credentials not available")

    from src.common.config import Config
    config = Config.from_env()
    client = KalshiClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )
    try:
        markets = await client.get_markets("KXEPLGAME", status="open", limit=1)
        if markets:
            ob = await client.get_orderbook(markets[0]["ticker"])
            assert "orderbook_fp" in ob
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kalshi_balance():
    """Verify balance endpoint returns a float."""
    if not _have_kalshi_creds():
        pytest.skip("Kalshi credentials not available")

    from src.common.config import Config
    config = Config.from_env()
    client = KalshiClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )
    try:
        balance = await client.get_balance()
        assert isinstance(balance, float)
        assert balance >= 0.0
    finally:
        await client.close()
