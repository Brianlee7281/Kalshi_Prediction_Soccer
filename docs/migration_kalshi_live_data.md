# Migration: Goalserve → Kalshi Live Data (Phase 3 Engine)

## What changes and why

Kalshi exposes a live data endpoint (`/trade-api/v2/live_data/{type}/{milestone_uuid}`)
that returns real-time score, match minute, half, and significant events. The upstream
source is Sportradar (confirmed by `source_id: sr:sport_event:...`), which is the same
feed driving Kalshi's own market. This closes the 17–60s Goalserve latency gap.

**What changes:**
- `src/engine/goalserve_poller.py` → replaced by `src/engine/kalshi_live_poller.py`
- `src/clients/kalshi_live_data.py` → new client
- `src/recorder/recorder.py` → add `kalshi_live_data` stream

**What does NOT change:**
- `src/engine/event_handlers.py` (handle_goal, handle_period_change, handle_red_card)
- `src/engine/tick_loop.py`
- Phase 1 training (Goalserve historical commentaries)
- Phase 2 fixture lookup (Goalserve)
- All math: EKF, HMM, DomIndex, MC

**Known limitation:** Kalshi live_data does not provide shots_on_target, corners, or
possession. `_extract_live_stats` for Layer 2 HMM will return None. HMM will run on
prior-only until we find an alternative stats source.

---

## Sprint KLD-1 — `KalshiLiveDataClient` + `MatchState`

**Files:** `src/clients/kalshi_live_data.py` (new), `tests/clients/test_kalshi_live_data.py` (new)

---

### Prompt KLD-1

```
Read these files before writing any code:
- src/clients/kalshi.py  (RSA-PSS auth pattern — replicate exactly)
- src/clients/base_client.py
- src/common/logging.py (or wherever get_logger is defined)

Create src/clients/kalshi_live_data.py with the following.

──────────────────────────────────────────────────
Pydantic model MatchState:
  status: str                   # "live" | "finished" | "halftime"
  half: str                     # "1st" | "2nd" | "HT" | "FT"
  minute: int                   # 0-90
  stoppage: int                 # 0 normally; 3 from "90+3'"
  home_score: int
  away_score: int
  last_play_ts: int | None      # occurence_ts (Sportradar real event time)
  last_play_desc: str | None
  significant_events: list[dict]  # raw list, preserved for future use

──────────────────────────────────────────────────
Class KalshiLiveDataClient:

  __init__(self, api_key: str, private_key_path: str) -> None
    Use the same RSA-PSS auth pattern as KalshiClient in src/clients/kalshi.py.
    Store api_key, load private key, create httpx.AsyncClient with base_url
    "https://api.elections.kalshi.com".

  _sign_request(self, method: str, path: str) -> dict[str, str]
    Identical to KalshiClient._sign_request. Returns KALSHI-ACCESS-KEY,
    KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE headers.

  async resolve_milestone_uuid(self, event_ticker: str) -> str
    GET /trade-api/v2/milestones?related_event_ticker={event_ticker}&limit=5
    Signed GET. Parse response["milestones"][0]["id"] and return it.
    Raise ValueError if milestones list is empty.
    Use asyncio.wait_for(..., timeout=10.0).

  async get_live_data(self, milestone_uuid: str) -> MatchState
    GET /trade-api/v2/live_data/soccer/{milestone_uuid}
    Signed GET. Parse response["live_data"]["details"] into MatchState.
    Use _parse_time_field for the "time" field.
    Use asyncio.wait_for(..., timeout=10.0).

  _parse_time_field(self, time_str: str) -> tuple[int, int]
    Parse Kalshi time strings into (minute, stoppage).
    Examples:
      "62'"  → (62, 0)
      "90+3'" → (90, 3)
      "45+1'" → (45, 1)
      ""     → (0, 0)
    Strip trailing apostrophe, split on "+".

  async close(self) -> None
    Close httpx.AsyncClient.

──────────────────────────────────────────────────
half/status mapping when parsing get_live_data response:
  details["half"] == "1st"  → half="1st"
  details["half"] == "2nd"  → half="2nd"
  details["status"] == "live" + half == "HT" → half="HT"  (halftime)
  details["status"] == "finished" or details.get("winner") != "" → half="FT"

status field on MatchState:
  details["status"] as-is ("live", "finished", etc.)

──────────────────────────────────────────────────
Use structlog (get_logger) for logging. Never print().
All type hints required.
Follow src/clients/kalshi.py import style exactly.

──────────────────────────────────────────────────
Create tests/clients/test_kalshi_live_data.py.

Test _parse_time_field with: "62'", "90+3'", "45+1'", "", "0'".
Test get_live_data parsing with a fixture dict that matches the real API
response shape shown below — mock the HTTP call with httpx.MockTransport
or monkeypatch, do NOT make real HTTP calls in tests.

Real API response shape for get_live_data:
{
  "live_data": {
    "details": {
      "status": "live",
      "status_text": "2nd - 62'",
      "half": "2nd",
      "time": "62'",
      "home_same_game_score": 0,
      "away_same_game_score": 0,
      "home_significant_events": [...],
      "away_significant_events": [...],
      "last_play": {
        "description": "Free kick Corinthians.",
        "occurence_ts": 1773971711
      },
      "winner": ""
    },
    "milestone_id": "0855ee5b-...",
    "type": "soccer_tournament_multi_leg"
  }
}
```

---

## Sprint KLD-2 — Recorder integration

**Files:** `src/recorder/recorder.py`

---

### Prompt KLD-2

```
Read src/recorder/recorder.py in full before editing.

Make the following minimal changes to MatchRecorder:

1. Add "kalshi_live_data" to the _STREAM_NAMES tuple.

2. Add method:
   def record_kalshi_live_data(self, state_dict: dict) -> None:
       """Append raw Kalshi live_data poll response with _ts."""
       self._write("kalshi_live_data", state_dict)

That is all. Do not change anything else in recorder.py.
The _write and _get_handle machinery already handles the new stream automatically
once the name is in _STREAM_NAMES.
```

---

## Sprint KLD-3 — `kalshi_live_poller.py`

**Files:** `src/engine/kalshi_live_poller.py` (new), `tests/engine/test_kalshi_live_poller.py` (new)

---

### Prompt KLD-3

```
Read these files in full before writing any code:
- src/engine/goalserve_poller.py  (structural template)
- src/engine/event_handlers.py    (handle_goal, handle_period_change,
                                   handle_red_card — do NOT modify)
- src/engine/model.py             (LiveMatchModel fields)
- src/clients/kalshi_live_data.py (KalshiLiveDataClient, MatchState)

Create src/engine/kalshi_live_poller.py. This coroutine replaces
goalserve_poller. It must produce identical downstream effects on
LiveMatchModel — the EKF, HMM, tick_loop do not know or care which
poller is running.

──────────────────────────────────────────────────
Module-level constant:
  _POLL_INTERVAL_S = 1.0   # 1s — well within Basic tier 20 reads/s

──────────────────────────────────────────────────
async def kalshi_live_poller(model: LiveMatchModel) -> None

  Reads KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH from os.environ.
  Calls KalshiLiveDataClient.resolve_milestone_uuid(model.kalshi_event_ticker)
  once at startup. Log the resolved uuid. Raise RuntimeError if resolution fails.

  Poll loop (runs until model.engine_phase == "FINISHED"):
    - Call get_live_data(milestone_uuid) inside asyncio.wait_for(..., timeout=10.0)
    - On timeout or exception: log warning, sleep _POLL_INTERVAL_S, continue

    Recording:
      recorder = getattr(model, "recorder", None)
      if recorder: recorder.record_kalshi_live_data(state.model_dump())

    Late join time sync (first poll with live status):
      if first_poll and state.status == "live":
        now = time.monotonic()
        model.kickoff_wall_clock = now - ((state.minute + state.stoppage) * 60.0)
        model.t = float(state.minute + state.stoppage)
        log "late_join_time_sync", minute=state.minute, stoppage=state.stoppage
        first_poll = False
      if first_poll and state.half in ("HT", "FT"):
        first_poll = False

    Event detection — call _detect_events_from_state(model, state):
      Returns list of event dicts. Dispatch exactly as goalserve_poller does:
        "goal"         → handle_goal(model, event["team"], event["minute"])
        "red_card"     → handle_red_card(model, event["team"], event["minute"])
        "period_change"→ handle_period_change(model, event["new_phase"])

    Stoppage time update (mirrors goalserve_poller logic):
      if state.stoppage > 0 and model.engine_phase == "SECOND_HALF" and model.t >= 85.0:
        model.update_T_exp(state.stoppage)

    HMM/DomIndex: Kalshi live_data does not provide shots/corners/possession.
      Skip _extract_live_stats. Log a debug once per match that live_stats
      are unavailable from this source.

    Sleep _POLL_INTERVAL_S (absolute time scheduling per Pattern 3 anti-pattern).

  On loop exit: await client.close(). Log "kalshi_live_poller_finished".

──────────────────────────────────────────────────
def _detect_events_from_state(
    model: LiveMatchModel,
    state: MatchState,
) -> list[dict]:

  Score diff (same logic as detect_events_from_poll in event_handlers.py):
    prev_home, prev_away = model._last_score
    home_diff = state.home_score - prev_home
    away_diff = state.away_score - prev_away

    For each home goal: append {"type": "goal", "team": "home",
      "minute": state.minute, "t": model.t + i*0.1,
      "occurence_ts": state.last_play_ts}
    For each away goal: similarly.

    After processing: model._last_score = (state.home_score, state.away_score)

  Period change (same spam guard as handle_period_change):
    new_phase = _kalshi_half_to_phase(state.half)
    if new_phase is not None and new_phase != model._last_period:
      append {"type": "period_change", "new_phase": new_phase}

  Red cards from significant_events:
    Iterate state.significant_events (combined home + away).
    If event_type == "red_card", check if already processed (use
    model._processed_red_cards: set[str] — add this field to model if absent
    via getattr with default set()).
    Dedup key: f"{team}_{event['player']}_{event['time']}".
    If not already processed: append {"type": "red_card", "team": ..., "minute": ...}
    and add to processed set.

  Return events list.

──────────────────────────────────────────────────
def _kalshi_half_to_phase(half: str) -> str | None:
  "1st" → "FIRST_HALF"
  "2nd" → "SECOND_HALF"
  "HT"  → "HALFTIME"
  "FT"  → "FINISHED"
  else  → None

──────────────────────────────────────────────────
Create tests/engine/test_kalshi_live_poller.py.

Test _detect_events_from_state with:
  - Score 0→1 home: should emit one goal event
  - Score jumps 0→2 home: should emit two goal events
  - Half "1st"→"2nd": should emit period_change SECOND_HALF
  - Duplicate period_change call: should NOT emit again
  - Red card in significant_events: should emit once, not twice on re-poll

Use a minimal LiveMatchModel stub (dataclass or SimpleNamespace) for tests.
Do NOT test the async poller loop itself — test the pure functions only.
```

---

## Sprint KLD-4 — Parallel validation (run both pollers)

**Files:** `src/engine/tick_loop.py` or wherever the engine coroutines are gathered

---

### Prompt KLD-4

```
Read these files before editing:
- src/engine/tick_loop.py
- src/engine/goalserve_poller.py
- src/engine/kalshi_live_poller.py
- src/engine/model.py

Find where goalserve_poller is launched alongside tick_loop (likely an
asyncio.gather or task creation call in tick_loop.py or the orchestrator).

Add kalshi_live_poller as an ADDITIONAL coroutine running in parallel with
goalserve_poller — do NOT remove goalserve_poller yet. Both will poll and
both will call the same event handlers. To prevent double-firing of goal
events, wrap each _detect_events_from_state and detect_events_from_poll
call with a short asyncio.Lock stored on the model:
  model._event_lock = asyncio.Lock()  (add via getattr default)

In goalserve_poller, acquire model._event_lock before calling
detect_events_from_poll and releasing after dispatching events.
In kalshi_live_poller, acquire model._event_lock before calling
_detect_events_from_state and releasing after dispatching events.

This way whichever poller detects the goal first will update model._last_score,
and the second poller's diff will be 0 (no duplicate events).

Log which source detected each goal first:
  logger.info("goal_source", source="kalshi_live"|"goalserve", ...)

The purpose of this sprint is one match of parallel recording to verify
kalshi_live_poller fires goals earlier than goalserve_poller. After
validation, proceed to KLD-5.
```

---

## Sprint KLD-5 — Cutover

**Files:** wherever goalserve_poller is launched, `src/engine/goalserve_poller.py`

---

### Prompt KLD-5

```
Read the location where goalserve_poller and kalshi_live_poller are both
launched (found in KLD-4).

Remove goalserve_poller from the engine coroutine group. Keep
kalshi_live_poller only. Remove the asyncio.Lock wrapper added in KLD-4
from kalshi_live_poller (no longer needed with single poller).

Remove the asyncio.Lock acquisition from kalshi_live_poller._detect_events_from_state
dispatch block.

Do NOT delete src/engine/goalserve_poller.py — rename it to
goalserve_poller.py.disabled or add a module docstring:
  "# DISABLED Sprint KLD-5: replaced by kalshi_live_poller. Kept for reference."

Do NOT remove GoalserveClient from src/clients/ — it is still used by
Phase 2 calibration and Phase 1 training pipeline.

After the change, run: make test && make lint
```

---

## Checklist

- [ ] KLD-1: `KalshiLiveDataClient` + `MatchState` + tests
- [ ] KLD-2: `MatchRecorder.record_kalshi_live_data` + `_STREAM_NAMES`
- [ ] KLD-3: `kalshi_live_poller` + `_detect_events_from_state` + tests
- [ ] KLD-4: Parallel run — verify goal detected earlier from Kalshi source
- [ ] KLD-5: Remove goalserve_poller from engine, disable file
