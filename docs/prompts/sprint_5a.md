Implement Sprint 5a: Match Discovery + Lifecycle State Machine for the Phase 5 orchestrator.

This sprint builds the orchestrator's match discovery (finding upcoming matches across 8 leagues) and lifecycle management (tracking each match through SCHEDULED → PHASE2_RUNNING → PHASE2_DONE → PHASE3_RUNNING → FINISHED → ARCHIVED). No Docker containers yet — just database state management.

Read these files before writing any code:
- `src/clients/cross_source_mapper.py` — `map_all_leagues()` function that queries Kalshi + Odds-API + Goalserve and returns matched fixtures. This makes LIVE API calls — all tests must mock it.
- `src/common/types.py` — Phase2Result, TradingMode, and types you'll add
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 5a section — this is your spec
- `docs/architecture.md` §5.1 — match_schedule table schema

Do NOT modify any files in `src/execution/`, `src/engine/`, or `src/clients/`.

## Step 1: Add types to `src/common/types.py`

After the `ExposureStatus` class (or after `SettlementResult`/`MatchPnL` if Sprint 4d is done), add:

```python
class MatchStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    PHASE2_RUNNING = "PHASE2_RUNNING"
    PHASE2_DONE = "PHASE2_DONE"
    PHASE2_SKIPPED = "PHASE2_SKIPPED"
    PHASE3_RUNNING = "PHASE3_RUNNING"
    FINISHED = "FINISHED"
    ARCHIVED = "ARCHIVED"

class MatchScheduleRecord(BaseModel):
    match_id: str
    league_id: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: MatchStatus
    trading_mode: TradingMode
    param_version: int | None = None
    kalshi_tickers: dict[str, str] | None = None
    goalserve_fix_id: str | None = None
```

## Step 2: Create `src/orchestrator/__init__.py`

Empty file (if it doesn't already exist).

## Step 3: Create `src/orchestrator/match_discovery.py`

**Function:** `async def discover_matches(db_pool: asyncpg.Pool) -> list[MatchScheduleRecord]`
- Call `map_all_leagues()` from `src.clients.cross_source_mapper`
- For each league's results, filter: only entries with `match_status == "ALL_MATCHED"` and `kalshi_event_ticker is not None`
- For each matched fixture: check if already in `match_schedule` table (SELECT by match_id). If not: INSERT with status='SCHEDULED', trading_mode='paper'.
- CRITICAL: `kalshi_tickers` is a dict — store it as JSONB using `json.dumps()`. NEVER use `str()` on a dict (produces single-quoted Python repr, not valid JSON).
- Return list of newly inserted MatchScheduleRecord objects.

**Function:** `async def get_actionable_matches(db_pool: asyncpg.Pool) -> dict[str, list[MatchScheduleRecord]]`
- Query match_schedule for 3 categories:
  - `needs_phase2`: `status = 'SCHEDULED' AND kickoff_utc - NOW() <= INTERVAL '65 minutes'`
  - `needs_container`: `status = 'PHASE2_DONE' AND kickoff_utc - NOW() <= INTERVAL '2 minutes'`
  - `needs_cleanup`: `status = 'FINISHED' AND updated_at < NOW() - INTERVAL '1 hour'`
- Return dict with these 3 keys, each mapping to a list of MatchScheduleRecord.

## Step 4: Create `src/orchestrator/lifecycle.py`

### Class: `MatchLifecycle`

**Constructor:** `__init__(self, db_pool: asyncpg.Pool)`

**Method: `async def transition(self, match_id: str, from_status: MatchStatus, to_status: MatchStatus, **extra_fields) -> bool`**

Legal transitions (reject all others):
```
SCHEDULED → PHASE2_RUNNING
PHASE2_RUNNING → PHASE2_DONE
PHASE2_RUNNING → PHASE2_SKIPPED
PHASE2_DONE → PHASE3_RUNNING
PHASE3_RUNNING → FINISHED
FINISHED → ARCHIVED
```

Implementation:
- Validate the transition is in the allowed set. If not, log error and return False.
- Execute: `UPDATE match_schedule SET status=$1, updated_at=NOW() WHERE match_id=$2 AND status=$3`
- If extra_fields provided (like param_version, kalshi_tickers), include them in the UPDATE
- If row count = 0: log warning "transition_failed" (concurrent modification or wrong from_status), return False
- Log: `structlog.info("match_transition", match_id=match_id, from_status=from_status.value, to_status=to_status.value)`
- Return True

**Method: `async def get_status(self, match_id: str) -> MatchStatus | None`**
- `SELECT status FROM match_schedule WHERE match_id = $1`
- Return MatchStatus enum value or None if not found

**Method: `async def run_phase2(self, match_id: str) -> Phase2Result | None`**
- Transition SCHEDULED → PHASE2_RUNNING (if fails, return None)
- Call the Phase 2 pipeline (this is a placeholder — for now, log that Phase 2 would run here)
- If verdict == "GO": transition PHASE2_RUNNING → PHASE2_DONE, return Phase2Result
- If verdict == "SKIP": transition PHASE2_RUNNING → PHASE2_SKIPPED, return None
- On exception: transition to PHASE2_SKIPPED, log error, return None

NOTE: The actual Phase 2 pipeline integration depends on `src/prematch/` code. For this sprint, `run_phase2` should accept an optional `phase2_runner` callable for dependency injection, defaulting to a stub that logs a warning. Tests will mock this.

## Step 5: Create tests

`tests/orchestrator/__init__.py` — empty file

`tests/orchestrator/test_match_discovery.py`:
- `test_discover_new_match`: mock `map_all_leagues` returning 1 ALL_MATCHED fixture → inserted into DB, returned in list
- `test_discover_existing_match`: fixture already in DB → not duplicated, not in returned list
- `test_discover_missing_kalshi`: fixture without kalshi_event_ticker → skipped
- `test_actionable_phase2`: match SCHEDULED with kickoff in 60 min → appears in needs_phase2
- `test_actionable_container`: match PHASE2_DONE with kickoff in 1 min → appears in needs_container
- `test_not_actionable_too_early`: match SCHEDULED with kickoff in 3 hours → not in any list

`tests/orchestrator/test_lifecycle.py`:
- `test_valid_transition`: SCHEDULED → PHASE2_RUNNING → returns True, status updated
- `test_invalid_transition`: SCHEDULED → FINISHED → returns False, status unchanged
- `test_concurrent_transition`: two simultaneous SCHEDULED → PHASE2_RUNNING on same match → only one succeeds
- `test_transition_with_extra_fields`: pass param_version=42 → stored in DB
- `test_phase2_go`: mock phase2_runner returns GO verdict → PHASE2_DONE
- `test_phase2_skip`: mock phase2_runner returns SKIP → PHASE2_SKIPPED
- `test_phase2_exception`: mock phase2_runner raises → PHASE2_SKIPPED, error logged

All tests require a test PostgreSQL database. Create fixtures that run the match_schedule CREATE TABLE (from architecture.md §5.1) before each test.

Integration test:
```python
async def test_sprint_5a_integration():
    # Insert a SCHEDULED match 60 min from kickoff
    # Verify it appears in get_actionable_matches needs_phase2
    # Run run_phase2 with mocked GO result
    # Verify status is now PHASE2_DONE
    # Verify it no longer appears in needs_phase2
```

## Step 6: Verify

1. `python -m pytest tests/orchestrator/ -v` — all pass
2. `python -c "from src.common.types import MatchStatus; assert len(MatchStatus) == 7"`
3. Existing tests unaffected

Do NOT modify any `src/execution/` files.
