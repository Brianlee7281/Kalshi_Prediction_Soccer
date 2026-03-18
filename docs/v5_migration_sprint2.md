# Sprint 2 Migration — Phase 2 + Clients

Reference: `docs/MMPP_v5_Complete.md` §5 (Phase 2), §6.3 (Kalshi WS)

---

## Task 2.1: Implement Shin vig removal method

**What:** Add the Shin (1992) vig removal function that corrects for favourite-longshot bias, replacing naive proportional normalization.

**Files touched:**
- `src/prematch/phase2_pipeline.py` — add `_shin_vig_removal` function

**Detailed steps:**
1. Add function after existing `_extract_implied_probs` (~line 519):
```python
def _shin_vig_removal(
    odds_h: float, odds_d: float, odds_a: float,
) -> tuple[float, float, float]:
    """Shin method: recover true probabilities from bookmaker odds.

    Solves for z in: Σ sqrt(z² + 4(1-z)·q_i/O) = 2 + z
    Then: p_i = (sqrt(z² + 4(1-z)·q_i/O) - z) / (2(1-z))
    """
    q = [1.0/odds_h, 1.0/odds_d, 1.0/odds_a]
    O = sum(q)
    # Bisection to find z ∈ (0, 1)
    lo, hi = 0.0, 0.99
    for _ in range(64):  # converges in ~50 iterations
        z = (lo + hi) / 2.0
        lhs = sum(math.sqrt(z*z + 4*(1-z)*qi/O) for qi in q)
        if lhs > 2.0 + z:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2.0
    probs = [(math.sqrt(z*z + 4*(1-z)*qi/O) - z) / (2*(1-z)) for qi in q]
    return probs[0], probs[1], probs[2]
```
2. Keep old `_odds_to_implied_naive` (rename current proportional method) as fallback.

**Breaking changes:** None — new function, existing code unchanged.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/prematch/test_shin_vig.py` — verify probabilities sum to 1.0; verify favourite gets slightly higher prob than naive normalization; verify on real odds examples.

**Verify:** `make test`

---

## Task 2.2: Add ekf_P0 to Phase2Result

**What:** Add initial EKF uncertainty field to Phase2Result with default so existing code doesn't break.

**Files touched:**
- `src/common/types.py` — add field to `Phase2Result`

**Detailed steps:**
1. After `prediction_method` field (~line 108), add:
```python
ekf_P0: float = 0.25  # initial EKF uncertainty; default = sigma_a² = 0.5² = 0.25
```

**New/changed types:**
- `Phase2Result.ekf_P0`: `float = 0.25`

**Breaking changes:** None — has default.

**Test impact:**
- Existing tests that break: None — all Phase2Result constructions work without specifying `ekf_P0`.
- New tests to add: None (tested in Task 2.5)

**Verify:** `make test`

---

## Task 2.3: Update phase2_pipeline to use Shin method and compute ekf_P0

**What:** Replace naive vig removal with Shin method in `_extract_implied_probs` and `_fetch_pinnacle_odds`. Compute `ekf_P0` based on prediction tier.

**Files touched:**
- `src/prematch/phase2_pipeline.py` — update `_extract_implied_probs`, `_fetch_pinnacle_odds`, `run_phase2`

**Detailed steps:**
1. In `_extract_implied_probs` (~line 491-519), replace:
```python
# Old: raw_h/total, raw_d/total, raw_a/total
# New:
p_h, p_d, p_a = _shin_vig_removal(odds_h, odds_d, odds_a)
```
2. Same change in `_fetch_pinnacle_odds` (~line 415-488) where vig removal happens.
3. In `run_phase2` Phase2Result construction (~line 168-185), compute `ekf_P0`:
```python
# Tier-based initial EKF uncertainty
ekf_P0_map = {
    "backsolve_odds_api": 0.15,   # Tier 1: high confidence
    "backsolve_pinnacle": 0.20,   # Tier 2: medium
    "xgboost": 0.25,              # ML prior: medium-high
    "form_mle": 0.35,             # Form-based: lower confidence
    "league_mle": 0.50,           # League average: lowest confidence
}
ekf_P0 = ekf_P0_map.get(prediction_method, 0.25)
```
4. Add `ekf_P0=ekf_P0` to Phase2Result construction.

**Breaking changes:** Vig removal method changes, which slightly shifts backsolve results. Fix: backsolve tests use `abs(a_H - expected) < tolerance` style assertions with wide enough margins.

**Test impact:**
- Existing tests that break: Possibly `test_backsolve_basic` if Shin method shifts implied probs enough to move `a_H` outside `-6.0 < a_H < -2.0` range. Unlikely — the range is wide. Verify.
- If a test breaks: Adjust the tolerance in the assertion, or add a comment explaining the Shin method shift.
- New tests to add: in Task 2.5

**Verify:** `make test`

---

## Task 2.4: Create Kalshi WebSocket client

**What:** Add WebSocket client for Kalshi live orderbook and trade feed. Purely additive — no existing code uses it until Sprint 3.

**Files touched:**
- `src/clients/kalshi_ws.py` — new file

**Detailed steps:**
1. Create `KalshiWSClient` class:
```python
class KalshiWSClient:
    """Kalshi WebSocket client for live orderbook + trade feed.

    Connects to Kalshi streaming API, subscribes to orderbook
    channels for specified tickers. Calls on_orderbook/on_trade
    callbacks for each update.
    """
    def __init__(self, api_key: str, private_key_path: str): ...
    async def connect(self, tickers: list[str]): ...
    async def disconnect(self): ...
    def _sign_ws_auth(self) -> dict: ...
```
2. Implement WS connection with RSA-PSS authentication (same signing as REST client).
3. Auto-reconnect with exponential backoff (1s base, 30s max).
4. Callback pattern: `on_orderbook(ticker: str, orderbook: dict)`, `on_trade(ticker: str, trade: dict)`.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/clients/test_kalshi_ws.py` — test auth signing matches REST; test message parsing for mock orderbook/trade messages.

**Verify:** `make test`

---

## Task 2.5: Add Sprint 2 tests

**What:** Tests for Shin vig removal, ekf_P0 computation, and Kalshi WS client.

**Files touched:**
- `tests/prematch/test_shin_vig.py` — new
- `tests/prematch/test_phase2_pipeline.py` — add ekf_P0 test
- `tests/clients/test_kalshi_ws.py` — new

**Detailed steps:**
1. `test_shin_vig.py`:
   - `test_shin_probabilities_sum_to_one` — verify p_h + p_d + p_a ≈ 1.0
   - `test_shin_favourite_higher_than_naive` — for favourite (low odds), Shin prob > naive prob
   - `test_shin_longshot_lower_than_naive` — for longshot (high odds), Shin prob < naive prob
   - `test_shin_no_vig_identity` — when odds sum to 1.0, Shin returns same probs

2. `test_phase2_pipeline.py` addition:
   - `test_ekf_P0_by_prediction_method` — verify ekf_P0 varies by tier

3. `test_kalshi_ws.py`:
   - `test_ws_auth_signature` — signing produces valid KALSHI-ACCESS-SIGNATURE
   - `test_parse_orderbook_message` — mock orderbook message parsed correctly

**Breaking changes:** None

**Test impact:** ~7 new tests added

**Verify:** `make test`

---

## Task 2.6: Annotate Odds-API WS as recording-only

**What:** Add docstring and code comments clarifying that Odds-API WebSocket data is for recording/logging only in v5, NOT for live trading decisions. No code behavior change.

**Files touched:**
- `src/clients/odds_api.py` — update docstrings
- `src/engine/odds_api_listener.py` — update module docstring

**Detailed steps:**
1. In `odds_api.py::connect_live_ws`, update docstring:
```python
"""Connect to Odds-API live WebSocket.

NOTE (v5): This feed is used for RECORDING only, not for live
trading decisions. P_model is the sole trading authority.
OddsConsensus will be removed in Sprint 3 migration.
"""
```
2. In `odds_api_listener.py`, add module-level deprecation note:
```python
"""Odds-API WebSocket listener.

DEPRECATED (v5 migration): This module feeds OddsConsensus, which is
removed in v5. After Sprint 3 migration, this module becomes a
recording-only logger. See Task 3.14.
"""
```

**Breaking changes:** None — documentation only.

**Test impact:**
- Existing tests that break: None
- New tests to add: None

**Verify:** `make test`

---

## Sprint 2 Dependency Graph

```
2.1 ──→ 2.3 ──→ 2.5
2.2 ──→ 2.3
2.4 ──→ 2.5
2.6 (independent)
```
