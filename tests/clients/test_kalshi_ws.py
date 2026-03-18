"""Tests for KalshiWSClient — auth signing, message parsing, reconnect logic."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.kalshi_ws import KalshiWSClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> KalshiWSClient:
    """Create client with real key (skip if absent)."""
    if not Path("keys/kalshi_private.pem").exists():
        pytest.skip("kalshi_private.pem not found")
    return KalshiWSClient(api_key="test-key", private_key_path="keys/kalshi_private.pem")


class FakeWS:
    """Minimal async-iterable fake WebSocket for testing message dispatch.

    Accepts an optional ``stop_client`` reference. When all messages have been
    consumed, it calls ``stop_client._stop.set()`` so that ``connect()`` exits
    cleanly instead of trying to reconnect.
    """

    def __init__(
        self,
        messages: list[dict],
        auth_response: dict | None = None,
        stop_client: KalshiWSClient | None = None,
    ):
        self._auth_response = auth_response or {"type": "login", "msg": {}}
        self._messages = [json.dumps(m) for m in messages]
        self._sent: list[str] = []
        self._closed = False
        self._stop_client = stop_client

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def recv(self) -> str:
        return json.dumps(self._auth_response)

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration:
            # Signal stop so connect() doesn't try to reconnect
            if self._stop_client is not None:
                self._stop_client._stop.set()
            raise StopAsyncIteration

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Auth signing tests
# ---------------------------------------------------------------------------

def test_kalshi_ws_sign_auth():
    """Verify _sign_ws_auth produces valid auth message structure."""
    client = _make_client()
    auth = client._sign_ws_auth()

    assert auth["cmd"] == "login"
    assert auth["params"]["api_key"] == "test-key"
    assert isinstance(auth["params"]["timestamp"], int)
    assert isinstance(auth["params"]["signature"], str)

    import base64
    base64.b64decode(auth["params"]["signature"])


def test_kalshi_ws_sign_consistency():
    """Two consecutive signatures should differ (different timestamps)."""
    client = _make_client()
    auth1 = client._sign_ws_auth()
    time.sleep(0.01)
    auth2 = client._sign_ws_auth()

    import base64
    base64.b64decode(auth1["params"]["signature"])
    base64.b64decode(auth2["params"]["signature"])


def test_kalshi_ws_initial_state():
    """Verify client starts disconnected."""
    client = _make_client()
    assert client.is_connected is False
    assert client._ws is None


# ---------------------------------------------------------------------------
# Connect / subscribe / message dispatch tests (mocked WS)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_sends_auth_then_subscribes():
    """connect() should send auth message first, then subscribe for each ticker."""
    client = _make_client()

    fake_ws = FakeWS(messages=[], stop_client=client)

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICKER-A", "TICKER-B"]),
            timeout=2.0,
        )

    # Auth message should be first sent
    assert len(fake_ws._sent) >= 1
    auth_sent = json.loads(fake_ws._sent[0])
    assert auth_sent["cmd"] == "login"

    # Then one subscribe per ticker
    assert len(fake_ws._sent) == 3  # 1 auth + 2 subscribes
    sub1 = json.loads(fake_ws._sent[1])
    assert sub1["cmd"] == "subscribe"
    assert sub1["params"]["market_tickers"] == ["TICKER-A"]
    sub2 = json.loads(fake_ws._sent[2])
    assert sub2["cmd"] == "subscribe"
    assert sub2["params"]["market_tickers"] == ["TICKER-B"]


@pytest.mark.asyncio
async def test_orderbook_snapshot_dispatched():
    """on_orderbook callback should fire for orderbook_snapshot messages."""
    client = _make_client()

    ob_calls: list[tuple[str, dict]] = []

    async def on_ob(ticker: str, data: dict) -> None:
        ob_calls.append((ticker, data))
        client._stop.set()  # stop after first message

    snapshot_msg = {
        "type": "orderbook_snapshot",
        "msg": {
            "market_ticker": "TICK-1",
            "yes": [[50, 100], [51, 200]],
            "no": [[49, 150]],
        },
    }

    fake_ws = FakeWS(messages=[snapshot_msg])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICK-1"], on_orderbook=on_ob),
            timeout=2.0,
        )

    assert len(ob_calls) == 1
    assert ob_calls[0][0] == "TICK-1"
    assert ob_calls[0][1]["yes"] == [[50, 100], [51, 200]]


@pytest.mark.asyncio
async def test_orderbook_delta_dispatched():
    """on_orderbook callback should fire for orderbook_delta messages."""
    client = _make_client()

    ob_calls: list[tuple[str, dict]] = []

    async def on_ob(ticker: str, data: dict) -> None:
        ob_calls.append((ticker, data))
        client._stop.set()

    delta_msg = {
        "type": "orderbook_delta",
        "msg": {
            "market_ticker": "TICK-1",
            "price": 55,
            "delta": 50,
            "side": "yes",
        },
    }

    fake_ws = FakeWS(messages=[delta_msg])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICK-1"], on_orderbook=on_ob),
            timeout=2.0,
        )

    assert len(ob_calls) == 1
    assert ob_calls[0][1]["side"] == "yes"


@pytest.mark.asyncio
async def test_trade_message_dispatched():
    """on_trade callback should fire for trade messages."""
    client = _make_client()

    trade_calls: list[tuple[str, dict]] = []

    async def on_trade(ticker: str, data: dict) -> None:
        trade_calls.append((ticker, data))
        client._stop.set()

    trade_msg = {
        "type": "trade",
        "msg": {
            "market_ticker": "TICK-2",
            "count": 10,
            "yes_price": 55,
            "no_price": 45,
        },
    }

    fake_ws = FakeWS(messages=[trade_msg])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICK-2"], on_trade=on_trade),
            timeout=2.0,
        )

    assert len(trade_calls) == 1
    assert trade_calls[0][0] == "TICK-2"
    assert trade_calls[0][1]["yes_price"] == 55


@pytest.mark.asyncio
async def test_unknown_message_type_ignored():
    """Messages with unrecognized types should not cause errors."""
    client = _make_client()

    ob_calls: list[tuple[str, dict]] = []

    async def on_ob(ticker: str, data: dict) -> None:
        ob_calls.append((ticker, data))

    messages = [
        {"type": "heartbeat", "msg": {}},
        {"type": "unknown_type", "msg": {"market_ticker": "X"}},
        {
            "type": "orderbook_snapshot",
            "msg": {"market_ticker": "TICK-1", "yes": [], "no": []},
        },
    ]

    fake_ws = FakeWS(messages=messages, stop_client=client)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICK-1"], on_orderbook=on_ob),
            timeout=2.0,
        )

    # Only the snapshot should have been dispatched
    assert len(ob_calls) == 1


@pytest.mark.asyncio
async def test_auth_error_breaks_loop():
    """If auth response is an error, connect() should stop (not reconnect)."""
    client = _make_client()

    fake_ws = FakeWS(messages=[], auth_response={"type": "error", "msg": "bad auth"})
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["TICK-1"]),
            timeout=2.0,
        )

    assert client.is_connected is False


@pytest.mark.asyncio
async def test_disconnect_sets_stop_and_clears_state():
    """disconnect() should set stop event and clear connection state."""
    client = _make_client()

    # Simulate connected state
    fake_ws = MagicMock()
    fake_ws.close = AsyncMock()
    client._ws = fake_ws
    client._connected = True

    await client.disconnect()

    assert client._stop.is_set()
    assert client._ws is None
    assert client.is_connected is False
    fake_ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_callback_no_crash():
    """If no callbacks are provided, messages should be silently ignored."""
    client = _make_client()

    messages = [
        {"type": "orderbook_snapshot", "msg": {"market_ticker": "T", "yes": []}},
        {"type": "trade", "msg": {"market_ticker": "T", "count": 1}},
    ]

    fake_ws = FakeWS(messages=messages, stop_client=client)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=fake_ws)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.clients.kalshi_ws.websockets.connect", return_value=cm):
        await asyncio.wait_for(
            client.connect(tickers=["T"]),  # no callbacks
            timeout=2.0,
        )
    # Should complete without error
