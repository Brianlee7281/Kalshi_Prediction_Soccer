# Sprint 2: Phase 2 + External Clients — Decomposition

Reference: `docs/architecture.md` §3.2 (Phase 2), §4.1 (Goalserve), §4.2 (Kalshi), §4.3 (Odds-API)

## Overview

Sprint 2 builds three API clients + the Phase 2 pre-match pipeline.

The clients are shared across phases:
- Goalserve: Phase 1 (historical), Phase 2 (upcoming), Phase 3 (live)
- Odds-API: Phase 2 (pre-match odds), Phase 3 (live odds WS)
- Kalshi: Phase 2 (ticker matching), Phase 4 (execution)

Phase 2 runs 65 minutes before kickoff and produces `Phase2Result`.

---

## Task 2.1: Base HTTP Client

Shared base client with retry, timeout, rate limit, and structured logging.

**File:** `src/clients/base_client.py`

```python
class BaseClient:
    """Async HTTP client with retry + rate limiting."""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
    ):
        ...

    async def get(self, path: str, params: dict | None = None, headers: dict | None = None) -> dict:
        """GET with retry on 429/5xx. Returns parsed JSON."""
        ...

    async def post(self, path: str, json_body: dict, headers: dict | None = None) -> dict:
        """POST with retry."""
        ...

    async def close(self) -> None:
        """Close underlying httpx.AsyncClient."""
        ...
```

Requirements:
- Use `httpx.AsyncClient` (not `requests`)
- Retry on 429 with exponential backoff (1s, 2s, 4s)
- Retry on 5xx with same backoff
- Log every request/response with structlog: method, path, status, duration_ms
- `asyncio.wait_for(coro, timeout=N)` for all external calls
- Rate limit: `asyncio.sleep(rate_limit_delay)` after every request

**Test:** `tests/clients/test_base_client.py`

```python
@pytest.mark.asyncio
async def test_base_client_get():
    """Test basic GET against a known endpoint."""
    from src.clients.base_client import BaseClient
    client = BaseClient(base_url="https://httpbin.org")
    result = await client.get("/get", params={"test": "1"})
    assert result["args"]["test"] == "1"
    await client.close()
```

**Done:** Base client with retry + logging works.

---

## Task 2.2: Goalserve Client

**File:** `src/clients/goalserve.py`

```python
class GoalserveClient:
    """Goalserve REST client for live scores and commentaries."""
    
    def __init__(self, api_key: str):
        # Base URL: https://www.goalserve.com/getfeed/{api_key}/
        ...

    async def get_live_scores(self) -> dict:
        """GET /soccernew/home?json=1
        Returns raw JSON response with all live matches.
        """
        ...

    async def get_commentaries(self, league_id: str, date: str) -> dict:
        """GET /commentaries/{league_id}?date={date}&json=1
        date format: DD.MM.YYYY
        Returns raw JSON response.
        """
        ...

    async def find_match_in_live(
        self, match_id: str, live_data: dict
    ) -> dict | None:
        """Search live scores for a match by @id, @fix_id, or @static_id.
        Returns the match dict or None if not found.
        Must search ALL three ID fields (post-mortem anti-pattern).
        """
        ...

    async def get_upcoming_fixtures(self, league_id: str) -> list[dict]:
        """Get upcoming fixtures for a league from live scores endpoint.
        Filters matches with status = time string (e.g., "15:00", not numeric/FT/HT).
        Returns list of fixture dicts with id, teams, kickoff time.
        """
        ...

    async def close(self) -> None:
        ...
```

Requirements:
- `find_match_in_live` MUST search `@id`, `@fix_id`, AND `@static_id` (anti-pattern from post-mortem)
- Handle nested JSON: `scores.category[].matches.match[]` — match can be dict (single) or list (multiple)
- Use `html.unescape()` for team names with `&apos;` entities
- Save sample responses to `tests/fixtures/goalserve_live.json` and `tests/fixtures/goalserve_commentaries.json`

**Test:** `tests/clients/test_goalserve.py`

```python
@pytest.mark.asyncio
async def test_goalserve_live_scores():
    """Fetch live scores — should return valid JSON with 'scores' key."""
    from src.clients.goalserve import GoalserveClient
    from src.common.config import Config
    config = Config.from_env()
    client = GoalserveClient(api_key=config.goalserve_api_key)
    data = await client.get_live_scores()
    assert "scores" in data
    assert "category" in data["scores"]
    await client.close()

def test_find_match_by_fix_id():
    """Verify find_match searches @fix_id, not just @id."""
    from src.clients.goalserve import GoalserveClient
    # Load saved fixture
    import json
    with open("tests/fixtures/goalserve_live.json") as f:
        live_data = json.load(f)
    client = GoalserveClient.__new__(GoalserveClient)
    # Use a known @fix_id from the saved data
    # This test verifies the anti-pattern fix
```

**Done:** Goalserve client fetches live scores and commentaries.

---

## Task 2.3: Kalshi Client (REST + Auth)

**File:** `src/clients/kalshi.py`

```python
class KalshiClient:
    """Kalshi REST client with RSA-PSS authentication."""
    
    def __init__(self, api_key: str, private_key_path: str):
        # Base URL: https://api.elections.kalshi.com
        # Load RSA private key from PEM file
        ...

    def _sign_request(self, method: str, path: str) -> dict:
        """Generate RSA-PSS SHA-256 auth headers.
        Returns dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE.
        Padding: PSS(mgf=MGF1(SHA256), salt_length=MAX_LENGTH).
        Signature = base64(sign(timestamp_ms + METHOD + path))
        """
        ...

    async def get_markets(
        self, series_ticker: str, status: str = "open", limit: int = 100
    ) -> list[dict]:
        """GET /trade-api/v2/markets?series_ticker={prefix}&status={status}
        Handles pagination via cursor. Returns all markets.
        IMPORTANT: List endpoint returns yes_ask=None. Use get_market() for prices.
        """
        ...

    async def get_market(self, ticker: str) -> dict:
        """GET /trade-api/v2/markets/{ticker}
        Returns single market with actual prices (last_price_dollars, yes_ask, etc).
        """
        ...

    async def get_orderbook(self, ticker: str) -> dict:
        """GET /trade-api/v2/markets/{ticker}/orderbook
        Returns {yes_dollars_fp: [[price, qty], ...], no_dollars_fp: [[price, qty], ...]}.
        """
        ...

    async def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """GET /trade-api/v2/markets/trades?ticker={ticker}
        Returns trade list with yes_price_dollars, count_fp, taker_side, created_time.
        """
        ...

    async def submit_order(self, order: dict) -> dict:
        """POST /trade-api/v2/orders
        order = {ticker, action, side, type, count, yes_price}
        Returns order response with id, status.
        """
        ...

    async def cancel_order(self, order_id: str) -> dict:
        """DELETE /trade-api/v2/orders/{order_id}"""
        ...

    async def get_balance(self) -> float:
        """GET /trade-api/v2/portfolio/balance
        Returns balance in dollars.
        """
        ...

    async def get_positions(self) -> list[dict]:
        """GET /trade-api/v2/portfolio/positions"""
        ...

    async def close(self) -> None:
        ...
```

Requirements:
- RSA-PSS SHA-256 authentication (NOT PKCS1v15 — this was a previous bug)
- Use `cryptography` library for signing, same as `scripts/sprint_minus_1.py`
- Pagination: follow `cursor` field until no more pages
- Rate limit: 0.1s delay between requests, exponential backoff on 429
- Save sample responses to `tests/fixtures/kalshi_market.json`, `tests/fixtures/kalshi_orderbook.json`

**Test:** `tests/clients/test_kalshi.py`

```python
@pytest.mark.asyncio
async def test_kalshi_auth_and_markets():
    """Verify RSA-PSS auth works and can fetch EPL markets."""
    from src.clients.kalshi import KalshiClient
    from src.common.config import Config
    config = Config.from_env()
    client = KalshiClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )
    markets = await client.get_markets("KXEPLGAME", limit=5)
    assert len(markets) > 0
    assert "ticker" in markets[0]
    await client.close()

@pytest.mark.asyncio
async def test_kalshi_orderbook():
    """Fetch orderbook for a real ticker."""
    from src.clients.kalshi import KalshiClient
    from src.common.config import Config
    config = Config.from_env()
    client = KalshiClient(
        api_key=config.kalshi_api_key,
        private_key_path=config.kalshi_private_key_path,
    )
    markets = await client.get_markets("KXEPLGAME", status="open", limit=1)
    if markets:
        ob = await client.get_orderbook(markets[0]["ticker"])
        assert "yes" in ob or "orderbook" in ob
    await client.close()
```

**Done:** Kalshi client authenticates and fetches markets/orderbooks.

---

## Task 2.4: Odds-API Client (REST + WebSocket)

**File:** `src/clients/odds_api.py`

```python
class OddsApiClient:
    """Odds-API.io REST + WebSocket client."""
    
    def __init__(self, api_key: str):
        # Base URL: https://api.odds-api.io/v3
        ...

    # ─── REST ─────────────────────────────────────────────

    async def get_events(
        self, league_slug: str, status: str = "pending"
    ) -> list[dict]:
        """GET /events?sport=football&league={slug}&status={status}
        Returns list of event dicts with id, teams, commence_time.
        """
        ...

    async def get_odds(
        self, event_id: str, bookmakers: str = "Bet365,Betfair Exchange"
    ) -> dict:
        """GET /odds?eventId={id}&bookmakers={names}
        bookmakers param is REQUIRED (architecture.md §4.3).
        Returns odds for requested bookmakers.
        """
        ...

    async def get_historical_odds(
        self, league_slug: str, date_from: str, date_to: str
    ) -> list[dict]:
        """GET /historical/odds?sport=football&league={slug}&dateFrom={}&dateTo={}
        Available from December 2025 onwards. Max 31-day span.
        Returns list of settled events with closing odds.
        """
        ...

    # ─── WebSocket ────────────────────────────────────────

    async def connect_live_ws(
        self,
        on_message: Callable[[dict], Awaitable[None]],
        markets: str = "ML,Spread,Totals",
        sport: str = "football",
    ) -> None:
        """Connect to wss://api.odds-api.io/v3/ws
        
        Receives: welcome message, then live odds updates.
        Update format: {type: "updated", bookie: "Bet365", markets: [{name: "ML", odds: [...]}]}
        
        Calls on_message callback for each update.
        Auto-reconnect with exponential backoff (1s base, 30s max).
        """
        ...

    async def close(self) -> None:
        ...


# League slug mapping (from architecture.md §4.3)
LEAGUE_SLUGS = {
    "1204": "england-premier-league",
    "1399": "spain-laliga",
    "1269": "italy-serie-a",
    "1229": "germany-bundesliga",
    "1221": "france-ligue-1",
    "1440": "usa-mls",
    "1141": "brazil-brasileiro-serie-a",
    "1081": "argentina-liga-profesional",
}
```

Requirements:
- REST: API key as query parameter `?apiKey={key}`
- WebSocket: `wss://api.odds-api.io/v3/ws?apiKey={key}&markets=ML,Spread,Totals&sport=football&status=live`
- WS reconnection: exponential backoff (1s base, 30s max, 10 retries)
- WS receives `welcome` message first, then `updated` messages
- Use `websockets` library for WS connection
- Save sample responses to `tests/fixtures/odds_api_events.json`, `tests/fixtures/odds_api_odds.json`

**Test:** `tests/clients/test_odds_api.py`

```python
@pytest.mark.asyncio
async def test_odds_api_events():
    """Fetch upcoming EPL events."""
    from src.clients.odds_api import OddsApiClient
    from src.common.config import Config
    config = Config.from_env()
    client = OddsApiClient(api_key=config.odds_api_key)
    events = await client.get_events("england-premier-league")
    assert isinstance(events, list)
    # May be empty if no upcoming matches, but should not crash
    await client.close()

@pytest.mark.asyncio
async def test_odds_api_odds():
    """Fetch odds for an EPL event (if available)."""
    from src.clients.odds_api import OddsApiClient
    from src.common.config import Config
    config = Config.from_env()
    client = OddsApiClient(api_key=config.odds_api_key)
    events = await client.get_events("england-premier-league")
    if events:
        odds = await client.get_odds(events[0]["id"], bookmakers="Bet365,Betfair Exchange")
        assert isinstance(odds, dict)
    await client.close()
```

**Done:** Odds-API fetches events and odds. WS connection method exists (tested manually in Sprint 3).

---

## Task 2.5: Kalshi Ticker Matching

Matches Goalserve fixtures to Kalshi market tickers.

**File:** `src/clients/kalshi_ticker_matcher.py`

```python
async def match_fixtures_to_tickers(
    fixtures: list[dict],
    kalshi_client: KalshiClient,
    league_prefix: str,
) -> dict[str, dict[str, str]]:
    """
    Match Goalserve fixtures to Kalshi tickers.
    
    Args:
        fixtures: [{id, home_team, away_team, kickoff_utc}, ...]
        kalshi_client: authenticated Kalshi client
        league_prefix: e.g. "KXEPLGAME"
    
    Returns:
        {match_id: {"home_win": "KXEPLGAME-...", "draw": "...", "away_win": "..."}}
    
    Matching logic:
    1. Fetch open markets for the series prefix
    2. For each fixture, find matching event by:
       - Team name alias matching (using normalize_team_name)
       - Accent stripping + per-word matching
       - Time window: market close_time >= fixture kickoff_utc
    3. Each match has 3 outcome markets: HOME, TIE, AWAY
    """
    ...

def _extract_teams_from_ticker(ticker: str) -> tuple[str, str] | None:
    """Parse ticker pattern: KXEPLGAME-26MAR15LAZACM-LAZ → ('LAZ', 'ACM')
    Returns (home_code, away_code) or None if unparseable.
    """
    ...
```

Requirements:
- Use `normalize_team_name` from `src/calibration/team_aliases.py`
- Handle ticker format: `KX{LEAGUE}{TYPE}-{season}{date}{teams}-{outcome}`
- Time window: `close_time >= kickoff_utc` (Kalshi close_time is weeks out, not same-day)
- Log unmatched fixtures as warnings

**Test:** `tests/clients/test_kalshi_ticker_matcher.py`

```python
def test_extract_teams_from_ticker():
    from src.clients.kalshi_ticker_matcher import _extract_teams_from_ticker
    result = _extract_teams_from_ticker("KXSERIEAGAME-26MAR15LAZACM-LAZ")
    assert result is not None
    # Should extract the two 3-letter team codes

def test_extract_teams_epl():
    from src.clients.kalshi_ticker_matcher import _extract_teams_from_ticker
    result = _extract_teams_from_ticker("KXEPLGAME-26MAR14BURBOU-BUR")
    assert result is not None
```

**Done:** Ticker matcher links Goalserve fixtures to Kalshi market tickers.

---

## Task 2.6: Phase 2 Pipeline

The full pre-match pipeline: load params → get odds → predict → backsolve → sanity check → Phase2Result.

**File:** `src/prematch/phase2_pipeline.py`

```python
async def run_phase2(
    match_id: str,
    league_id: int,
    home_team: str,
    away_team: str,
    kickoff_utc: datetime,
    config: Config,
) -> Phase2Result:
    """
    Full Phase 2 pipeline. Runs at kickoff -65 minutes.
    
    Steps:
    1. Load active production_params for this league from DB
    2. Collect pre-match odds from Odds-API.io (Bet365 + Betfair Exchange)
    3. Build features (same as Phase 1 XGBoost, but for this specific match)
    4. XGBoost predict a_H, a_A (load model from DB BYTEA blob)
       Fallback: team form MLE (last 5 matches) → league MLE
    5. Compute mu_H, mu_A = exp(a_H) * C_time, exp(a_A) * C_time
    6. Sanity check: compare model P(1x2) vs Bet365/Betfair implied probs
       If max deviation > 0.15 → verdict = "SKIP"
    7. Match Kalshi tickers (using ticker matcher from Task 2.5)
    8. Build and return Phase2Result
    """
    ...

async def load_production_params(config: Config, league_id: int) -> dict:
    """Load active production_params row for league from DB.
    Returns dict with Q, b, gamma_H, gamma_A, delta_H, delta_A, sigma_a,
    xgb_model_blob, feature_mask.
    """
    ...

def backsolve_intensities(
    odds_implied: MarketProbs,
    b: np.ndarray,
    Q: np.ndarray,
) -> tuple[float, float]:
    """Backsolve a_H, a_A from market-implied probabilities.
    Given P(home_win), find a_H, a_A that produce those probabilities
    via the MMPP model. Uses scipy.optimize.minimize.
    """
    ...

def sanity_check(
    model_probs: MarketProbs,
    market_probs: MarketProbs | None,
    threshold: float = 0.15,
) -> tuple[str, str | None]:
    """Compare model vs market probabilities.
    Returns ("GO", None) or ("SKIP", "reason string").
    """
    ...
```

Requirements:
- Load XGBoost model from DB BYTEA blob (`pickle.loads()`)
- Pre-match odds: Odds-API.io `get_odds(event_id, bookmakers="Bet365,Betfair Exchange")`
- Convert odds to implied probs, remove vig
- Sanity check: if `max(|model - market|)` for any of home/draw/away > 0.15 → SKIP
- Phase2Result must include param_version for version pinning
- Use `normalize_team_name` for team matching

**Test:** `tests/prematch/test_phase2_pipeline.py`

```python
def test_sanity_check_pass():
    from src.prematch.phase2_pipeline import sanity_check
    from src.common.types import MarketProbs
    model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    market = MarketProbs(home_win=0.48, draw=0.28, away_win=0.24)
    verdict, reason = sanity_check(model, market)
    assert verdict == "GO"

def test_sanity_check_fail():
    from src.prematch.phase2_pipeline import sanity_check
    from src.common.types import MarketProbs
    model = MarketProbs(home_win=0.70, draw=0.15, away_win=0.15)
    market = MarketProbs(home_win=0.40, draw=0.30, away_win=0.30)
    verdict, reason = sanity_check(model, market)
    assert verdict == "SKIP"  # max deviation = 0.30 > 0.15

def test_sanity_check_no_market():
    from src.prematch.phase2_pipeline import sanity_check
    from src.common.types import MarketProbs
    model = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    verdict, reason = sanity_check(model, None)
    assert verdict == "GO"  # no market data = proceed with model only

@pytest.mark.asyncio
async def test_load_production_params():
    """Verify we can load EPL params from DB (saved in Sprint 1)."""
    from src.prematch.phase2_pipeline import load_production_params
    from src.common.config import Config
    config = Config.from_env()
    params = await load_production_params(config, 1204)
    assert params is not None
    assert "Q" in params
    assert "b" in params

def test_backsolve_basic():
    """Verify backsolve produces reasonable a_H, a_A."""
    from src.prematch.phase2_pipeline import backsolve_intensities
    from src.common.types import MarketProbs
    import numpy as np
    odds = MarketProbs(home_win=0.45, draw=0.30, away_win=0.25)
    b = np.zeros(6)
    Q = np.zeros((4, 4))
    a_H, a_A = backsolve_intensities(odds, b, Q)
    # a_H, a_A should be in reasonable range
    assert -6.0 < a_H < -2.0
    assert -6.0 < a_A < -2.0
```

**Done:** Phase 2 pipeline produces Phase2Result for a specific match.

---

## Task 2.7: Phase 2 CLI + Integration Test

**File:** `scripts/run_phase2.py`

```python
"""
Usage: python scripts/run_phase2.py --league EPL
       Finds next upcoming match and runs Phase 2 for it.
"""
import asyncio
import argparse

from src.prematch.phase2_pipeline import run_phase2
from src.clients.goalserve import GoalserveClient
from src.clients.odds_api import OddsApiClient, LEAGUE_SLUGS
from src.common.config import Config

LEAGUE_IDS = {
    "EPL": 1204, "LaLiga": 1399, "SerieA": 1269, "Bundesliga": 1229,
    "Ligue1": 1221, "MLS": 1440, "Brasileirao": 1141, "Argentina": 1081,
}

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, required=True)
    args = parser.parse_args()
    
    config = Config.from_env()
    lid = LEAGUE_IDS[args.league]
    
    # Find next upcoming fixture from Goalserve
    gs = GoalserveClient(api_key=config.goalserve_api_key)
    fixtures = await gs.get_upcoming_fixtures(str(lid))
    await gs.close()
    
    if not fixtures:
        print(f"No upcoming fixtures for {args.league}")
        return
    
    fixture = fixtures[0]
    print(f"Next match: {fixture['home_team']} vs {fixture['away_team']}")
    print(f"Kickoff: {fixture['kickoff_utc']}")
    
    result = await run_phase2(
        match_id=fixture["match_id"],
        league_id=lid,
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
        kickoff_utc=fixture["kickoff_utc"],
        config=config,
    )
    
    print(f"\nPhase2Result:")
    print(f"  verdict: {result.verdict}")
    print(f"  a_H: {result.a_H:.4f}, a_A: {result.a_A:.4f}")
    print(f"  mu_H: {result.mu_H:.4f}, mu_A: {result.mu_A:.4f}")
    print(f"  method: {result.prediction_method}")
    print(f"  param_version: {result.param_version}")
    if result.kalshi_tickers:
        print(f"  kalshi_tickers: {result.kalshi_tickers}")
    if result.market_implied:
        print(f"  market_implied: H={result.market_implied.home_win:.3f} "
              f"D={result.market_implied.draw:.3f} A={result.market_implied.away_win:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Test:** Run with a real upcoming match:

```bash
make up  # ensure postgres running
PYTHONPATH=. python scripts/run_phase2.py --league EPL
```

Expected output:
```
Next match: Arsenal vs Chelsea
Kickoff: 2026-03-22T15:00:00Z
Phase2Result:
  verdict: GO
  a_H: -3.85, a_A: -4.12
  mu_H: 1.65, mu_A: 1.23
  method: xgboost
  param_version: 2
  kalshi_tickers: {"home_win": "KXEPLGAME-...", "draw": "...", "away_win": "..."}
  market_implied: H=0.520 D=0.270 A=0.210
```

**Done:** Phase2Result produced for a real upcoming match. Sprint 2 complete.

---

## Execution Order

Task 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7

After each task, run `make test` and fix any issues before proceeding.
Git commit after each task with message `sprint2: {brief description}`.

Tasks 2.2, 2.3, 2.4 make real API calls — ensure `.env` has valid API keys.
Task 2.6 requires production_params in DB (from Sprint 1).

Sprint 2 is DONE when:
- All client tests pass
- `python scripts/run_phase2.py --league EPL` produces a Phase2Result with verdict GO
- Phase2Result has populated kalshi_tickers and market_implied
