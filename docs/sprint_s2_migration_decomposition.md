# Sprint S2 Migration Decomposition — Phase 2 + Clients (v4 → v5)

Branch: `v5-migration`
Precondition: Sprint S1 migration complete, all tests green.
Postcondition: `make test` green after every task.

Dependency order: **2.1 → 2.2 → 2.3 → 2.5** and **2.4 → 2.5**, then **2.6** (independent).

---

## Task 2.1 — Add Shin vig removal function

### Prompt

```
You are working on the v5-migration branch. All 102+ existing tests pass. Your job is to add a single new function to src/prematch/phase2_pipeline.py. Do NOT modify any existing function. Only ADD.

Add a function `_shin_vig_removal` after the existing `_extract_implied_probs` function (after line 580). This implements the Shin (1992) method for vig removal, which corrects for favourite-longshot bias.

Here is the exact function to add:

```python
def _shin_vig_removal(
    odds_h: float, odds_d: float, odds_a: float,
) -> tuple[float, float, float]:
    """Shin method: recover true probabilities from bookmaker odds.

    Unlike naive normalization (1/odds / sum), Shin models informed
    insider trading to correctly handle favourite-longshot bias.
    Favorites get slightly higher probability; longshots get lower.

    Reference: Shin (1992, 1993).

    Solves for z in: Σ sqrt(z² + 4(1-z)·q_i/O) = 2 + z
    Then: p_i = (sqrt(z² + 4(1-z)·q_i/O) - z) / (2(1-z))
    """
    import math

    q = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    O = sum(q)

    # Bisection to find z ∈ (0, 1)
    lo, hi = 0.0, 0.99
    for _ in range(64):
        z = (lo + hi) / 2.0
        lhs = sum(math.sqrt(z * z + 4.0 * (1.0 - z) * qi / O) for qi in q)
        if lhs > 2.0 + z:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2.0

    denom = 2.0 * (1.0 - z)
    probs = [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * qi / O) - z) / denom
        for qi in q
    ]
    return probs[0], probs[1], probs[2]
```

Also add `import math` to the file's imports if not already present (it currently imports from __future__, json, pickle, datetime, pathlib, asyncpg, numpy, structlog, scipy.optimize — math is not there yet). Add `import math` in the stdlib imports section after `import json`.

Do NOT:
- Modify any existing function
- Change any existing vig removal logic (that happens in Task 2.3)
- Touch any test files

After you're done, run `make test` to verify all existing tests still pass.
```

---

## Task 2.2 — Add ekf_P0 field to Phase2Result

### Prompt

```
You are working on the v5-migration branch. Your job is to add ONE new field to the Phase2Result Pydantic model in src/common/types.py. This is a purely additive change — the field has a default value so no existing code breaks.

In src/common/types.py, find the Phase2Result class (starts at line 99). After the last field `prediction_method: str` (line 127), add:

    # v5: initial EKF uncertainty — larger P0 = more aggressive early EKF updates
    ekf_P0: float = 0.25  # default = sigma_a² = 0.5² = 0.25

This field represents the initial uncertainty for the Extended Kalman Filter in Phase 3. The default 0.25 corresponds to σ_a²=0.5², which is the regularization strength from Phase 1. Phase 2 will set this to a tier-dependent value (lower for better odds sources, higher for fallbacks) in Task 2.3.

Do NOT:
- Modify any other class or field
- Change any existing defaults
- Touch any other file

After you're done, run `make test` to verify all existing tests still pass. Existing code that constructs Phase2Result without ekf_P0 will use the default 0.25.
```

---

## Task 2.3 — Wire Shin method + ekf_P0 into phase2_pipeline

### Prompt

```
You are working on the v5-migration branch. Tasks 2.1 and 2.2 are complete — `_shin_vig_removal` exists in phase2_pipeline.py and `ekf_P0` exists on Phase2Result with default 0.25. Now wire them in.

Make these changes to src/prematch/phase2_pipeline.py:

**Change 1: Replace naive vig removal in `_extract_implied_probs` (around line 570-577)**

Currently the function does:
```python
raw_h, raw_d, raw_a = 1.0 / h, 1.0 / d, 1.0 / a
total = raw_h + raw_d + raw_a
return MarketProbs(
    home_win=raw_h / total,
    draw=raw_d / total,
    away_win=raw_a / total,
)
```

Replace with:
```python
p_h, p_d, p_a = _shin_vig_removal(h, d, a)
return MarketProbs(
    home_win=p_h,
    draw=p_d,
    away_win=p_a,
)
```

**Change 2: Replace naive vig removal in `_fetch_pinnacle_odds` (around line 532-544)**

Currently the function does:
```python
raw_h, raw_d, raw_a = 1.0 / ph, 1.0 / pd, 1.0 / pa
total = raw_h + raw_d + raw_a
...
return MarketProbs(
    home_win=raw_h / total,
    draw=raw_d / total,
    away_win=raw_a / total,
)
```

Replace with:
```python
p_h, p_d, p_a = _shin_vig_removal(ph, pd, pa)
log.info(
    "phase2_pinnacle_found",
    home=home_team,
    away=away_team,
    csv=csv_path.name,
)
return MarketProbs(
    home_win=p_h,
    draw=p_d,
    away_win=p_a,
)
```

**Change 3: Compute ekf_P0 in `run_phase2` (around line 167, before Phase2Result construction)**

Add this block just before `return Phase2Result(` (around line 167):
```python
    # v5: EKF initial uncertainty — tier-dependent
    ekf_P0_map = {
        "backsolve_odds_api": 0.15,   # Tier 1: Betfair/Bet365, high confidence
        "backsolve_pinnacle": 0.20,   # Tier 2: Pinnacle CSV, medium confidence
        "xgboost": 0.25,              # Tier 3: ML prior, medium-high uncertainty
        "form_mle": 0.35,             # Form-based: lower confidence
        "league_mle": 0.50,           # Tier 4: league average, lowest confidence
    }
    ekf_P0 = ekf_P0_map.get(prediction_method, 0.25)
```

**Change 4: Add ekf_P0 to the Phase2Result construction (around line 168-185)**

Add `ekf_P0=ekf_P0,` to the Phase2Result constructor call, after `prediction_method=prediction_method,`.

Do NOT:
- Change backsolve_intensities or sanity_check
- Change the tier selection logic (Tier 1/2/3/4 order)
- Touch _skip_result (it doesn't need ekf_P0 since it's a SKIP result and defaults to 0.25)

After you're done, run `make test`. All existing tests should pass. The Shin method produces slightly different probabilities than naive normalization, but the backsolve tests use wide tolerance ranges (-6.0 < a_H < -2.0) and reproduced-probability tolerances of 0.02-0.03, which should accommodate the small shift. If a backsolve test fails with a probability mismatch slightly over 0.03, widen that specific tolerance to 0.04.
```

---

## Task 2.4 — Create Kalshi WebSocket client

### Prompt

```
You are working on the v5-migration branch. Your job is to create a NEW file src/clients/kalshi_ws.py — a WebSocket client for Kalshi's live orderbook and trade feed. This is purely additive — no existing code uses it until Sprint 3.

Study the existing REST client at src/clients/kalshi.py first. It uses RSA-PSS SHA-256 signing via `_sign_request(method, path)` which returns auth headers. The WS client needs the same authentication mechanism.

Create src/clients/kalshi_ws.py with:

```python
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

    def __init__(self, api_key: str, private_key_path: str) -> None:
        self._api_key = api_key
        self._private_key = self._load_private_key(private_key_path)
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._stop = asyncio.Event()
        self._connected = False

    @staticmethod
    def _load_private_key(path: str):
        """Load RSA private key from PEM file."""
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_ws_auth(self) -> dict[str, str]:
        """Generate RSA-PSS SHA-256 auth message for WS handshake.

        Same signing algorithm as REST client (src/clients/kalshi.py).
        Returns dict with api_key, timestamp, and signature.
        """
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
                async with websockets.connect(
                    KALSHI_WS_URL, max_size=10_000_000
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    attempt = 0
                    log.info("kalshi_ws_connected")

                    # Authenticate
                    auth_msg = self._sign_ws_auth()
                    await ws.send(json.dumps(auth_msg))

                    # Wait for auth response
                    auth_resp = json.loads(await ws.recv())
                    if auth_resp.get("type") == "error":
                        log.error("kalshi_ws_auth_failed", response=auth_resp)
                        break

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
                            await on_orderbook(ticker, msg.get("msg", {}))
                        elif msg_type == "orderbook_delta" and on_orderbook:
                            ticker = msg.get("msg", {}).get("market_ticker", "")
                            await on_orderbook(ticker, msg.get("msg", {}))
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
```

Also create tests/clients/test_kalshi_ws.py:

```python
"""Tests for KalshiWSClient — auth signing and message parsing."""

import json
from pathlib import Path

import pytest

from src.clients.kalshi_ws import KalshiWSClient


def test_kalshi_ws_sign_auth():
    """Verify _sign_ws_auth produces valid auth message structure."""
    if not Path("keys/kalshi_private.pem").exists():
        pytest.skip("kalshi_private.pem not found")

    client = KalshiWSClient(
        api_key="test-key",
        private_key_path="keys/kalshi_private.pem",
    )
    auth = client._sign_ws_auth()

    assert auth["cmd"] == "login"
    assert auth["params"]["api_key"] == "test-key"
    assert isinstance(auth["params"]["timestamp"], int)
    assert isinstance(auth["params"]["signature"], str)

    # Signature should be valid base64
    import base64
    base64.b64decode(auth["params"]["signature"])


def test_kalshi_ws_sign_consistency():
    """Two consecutive signatures should differ (different timestamps)."""
    if not Path("keys/kalshi_private.pem").exists():
        pytest.skip("kalshi_private.pem not found")

    import time
    client = KalshiWSClient(
        api_key="test-key",
        private_key_path="keys/kalshi_private.pem",
    )
    auth1 = client._sign_ws_auth()
    time.sleep(0.01)
    auth2 = client._sign_ws_auth()

    # Timestamps might be the same at ms resolution, but signatures
    # should still be valid base64
    import base64
    base64.b64decode(auth1["params"]["signature"])
    base64.b64decode(auth2["params"]["signature"])


def test_kalshi_ws_initial_state():
    """Verify client starts disconnected."""
    if not Path("keys/kalshi_private.pem").exists():
        pytest.skip("kalshi_private.pem not found")

    client = KalshiWSClient(
        api_key="test-key",
        private_key_path="keys/kalshi_private.pem",
    )
    assert client.is_connected is False
    assert client._ws is None
```

Do NOT modify any existing file. This is a purely additive new module.

After you're done, run `make test` to verify all tests pass (existing + 3 new).
```

---

## Task 2.5 — Add Sprint 2 tests

### Prompt

```
You are working on the v5-migration branch. Tasks 2.1-2.4 are complete. Now add tests for the Shin method and ekf_P0.

**File 1: Create tests/prematch/test_shin_vig.py**

```python
"""Tests for Shin vig removal method."""

import math

import pytest

from src.prematch.phase2_pipeline import _shin_vig_removal


def test_shin_probabilities_sum_to_one():
    """Shin-corrected probabilities must sum to ~1.0."""
    # Typical EPL odds: home 2.10, draw 3.40, away 3.20
    p_h, p_d, p_a = _shin_vig_removal(2.10, 3.40, 3.20)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6, (
        f"Sum = {p_h + p_d + p_a:.8f}, expected 1.0"
    )


def test_shin_probabilities_positive():
    """All probabilities must be positive."""
    p_h, p_d, p_a = _shin_vig_removal(1.50, 4.00, 6.00)
    assert p_h > 0.0
    assert p_d > 0.0
    assert p_a > 0.0


def test_shin_favourite_higher_than_naive():
    """For the favourite (lowest odds), Shin gives HIGHER prob than naive."""
    odds_h, odds_d, odds_a = 1.50, 4.00, 6.00  # strong home favourite

    # Naive normalization
    raw = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    total = sum(raw)
    naive_h = raw[0] / total

    # Shin
    shin_h, _, _ = _shin_vig_removal(odds_h, odds_d, odds_a)

    assert shin_h > naive_h, (
        f"Shin({shin_h:.6f}) should be > naive({naive_h:.6f}) for favourite"
    )


def test_shin_longshot_lower_than_naive():
    """For the longshot (highest odds), Shin gives LOWER prob than naive."""
    odds_h, odds_d, odds_a = 1.50, 4.00, 6.00

    raw = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
    total = sum(raw)
    naive_a = raw[2] / total

    _, _, shin_a = _shin_vig_removal(odds_h, odds_d, odds_a)

    assert shin_a < naive_a, (
        f"Shin({shin_a:.6f}) should be < naive({naive_a:.6f}) for longshot"
    )


def test_shin_fair_odds_identity():
    """When odds are already fair (sum of implied = 1.0), Shin ≈ naive."""
    # Fair odds: no vig
    p_h, p_d, p_a = _shin_vig_removal(2.0, 4.0, 4.0)
    # With fair odds: 1/2 + 1/4 + 1/4 = 1.0, so z ≈ 0
    # Shin should return approximately the naive values
    assert abs(p_h - 0.50) < 0.01
    assert abs(p_d - 0.25) < 0.01
    assert abs(p_a - 0.25) < 0.01


def test_shin_extreme_favourite():
    """Shin handles very short-priced favourite without error."""
    p_h, p_d, p_a = _shin_vig_removal(1.10, 8.00, 15.00)
    assert p_h > 0.85  # very strong favourite
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6


def test_shin_even_match():
    """Shin handles near-even odds correctly."""
    p_h, p_d, p_a = _shin_vig_removal(2.80, 3.20, 2.60)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-6
    # All probs should be in reasonable range
    assert 0.20 < p_h < 0.45
    assert 0.20 < p_d < 0.40
    assert 0.25 < p_a < 0.45
```

**File 2: Add one test to the BOTTOM of tests/prematch/test_phase2_pipeline.py**

Add this test function at the end of the file (after the last existing test):

```python
def test_ekf_P0_by_prediction_method():
    """Phase2Result.ekf_P0 defaults to 0.25 and can be set explicitly."""
    from src.common.types import Phase2Result

    # Default (not specified)
    r = Phase2Result(
        match_id="m1", league_id=1, a_H=-4.0, a_A=-4.2,
        mu_H=1.5, mu_A=1.1, C_time=90.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="league_mle",
    )
    assert r.ekf_P0 == 0.25  # default

    # Explicit Tier 1 value
    r2 = Phase2Result(
        match_id="m2", league_id=1, a_H=-4.0, a_A=-4.2,
        mu_H=1.5, mu_A=1.1, C_time=90.0, verdict="GO",
        skip_reason=None, param_version=1, home_team="A",
        away_team="B", kickoff_utc="2026-03-15T15:00:00Z",
        kalshi_tickers={}, market_implied=None,
        prediction_method="backsolve_odds_api",
        ekf_P0=0.15,
    )
    assert r2.ekf_P0 == 0.15
```

Do NOT modify any existing tests. Only ADD new test files and append to existing test files.

After you're done, run `make test`. All existing + new tests should pass.
```

---

## Task 2.6 — Annotate Odds-API WS as recording-only

### Prompt

```
You are working on the v5-migration branch. Your job is to update docstrings in two files to mark the Odds-API WebSocket as recording-only under v5. This is purely documentation — NO behavior changes.

**File 1: src/clients/odds_api.py**

Change the docstring of `connect_live_ws` method (line 103-109). Replace:
```python
        """Connect to wss://api.odds-api.io/v3/ws

        Receives: welcome message, then live odds updates.
        Update format: {type: "updated", bookie: "Bet365", markets: [{name: "ML", odds: [...]}]}

        Calls on_message callback for each update.
        Auto-reconnect with exponential backoff (1s base, 30s max, 10 retries).
        """
```

With:
```python
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
```

**File 2: src/engine/odds_api_listener.py**

Change the module docstring at the top (lines 1-6). Replace:
```python
"""Odds-API WebSocket listener — feeds live bookmaker odds into OddsConsensus.

Connects to the Odds-API live WS endpoint, parses ML (moneyline) odds
updates, converts to implied probabilities with vig removal, and updates
the model's OddsConsensus on each message.
"""
```

With:
```python
"""Odds-API WebSocket listener — feeds live bookmaker odds into OddsConsensus.

DEPRECATED (v5 migration): In v5, OddsConsensus is removed. This module
will be demoted to a recording-only logger in Sprint 3 (Task 3.13).
P_model is the sole trading authority — bookmaker odds are recorded for
post-match analysis only, not used for live trading decisions.

Current behavior (v4): Connects to the Odds-API live WS endpoint, parses
ML (moneyline) odds updates, converts to implied probabilities with vig
removal, and updates the model's OddsConsensus on each message.
"""
```

Do NOT change any code behavior. Only docstrings.

After you're done, run `make test` to verify nothing broke.
```

---

## Execution Checklist

```
[ ] Task 2.1 — _shin_vig_removal function added
    Run: make test → all pass
    Commit: v5-migration-s2: add Shin vig removal function

[ ] Task 2.2 — ekf_P0 field added to Phase2Result
    Run: make test → all pass
    Commit: v5-migration-s2: add ekf_P0 to Phase2Result

[ ] Task 2.3 — Shin method + ekf_P0 wired into pipeline
    Run: make test → all pass (check backsolve tests carefully)
    Commit: v5-migration-s2: wire Shin vig removal and ekf_P0 into phase2

[ ] Task 2.4 — Kalshi WS client created
    Run: make test → all pass (3 new tests)
    Commit: v5-migration-s2: add Kalshi WebSocket client

[ ] Task 2.5 — Sprint 2 tests added
    Run: make test → all pass (8 new tests)
    Commit: v5-migration-s2: add Shin and ekf_P0 tests

[ ] Task 2.6 — Odds-API WS annotated as recording-only
    Run: make test → all pass
    Commit: v5-migration-s2: annotate Odds-API WS as recording-only
```
