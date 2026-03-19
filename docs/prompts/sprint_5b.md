Implement Sprint 5b: Container Management + Scheduler Loop + Recovery for the Phase 5 orchestrator.

This sprint adds Docker container lifecycle management, the main orchestrator loop, and crash recovery. It depends on Sprint 5a (match discovery + lifecycle) being complete.

Read these files before writing any code:
- `src/orchestrator/match_discovery.py` — `discover_matches()`, `get_actionable_matches()`
- `src/orchestrator/lifecycle.py` — `MatchLifecycle` class
- `src/common/types.py` — MatchStatus, MatchScheduleRecord, TradingMode, Phase2Result
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 5b section — this is your spec

Do NOT modify Sprint 5a files or any `src/execution/` files.

## Step 1: Create `src/orchestrator/container_manager.py`

### Class: `ContainerManager`

**Constructor:** `__init__(self, docker_client=None)`
- If None: `self.client = docker.from_env()` (import docker)
- `self.network_name = "mmpp-net"`

**Method: `async def launch_match_container(self, match_id: str, phase2_result: Phase2Result, trading_mode: TradingMode) -> str`**
- Container name: `f"match-{match_id}"`
- Check if container with this name already exists → stop and remove it (stale cleanup)
- Serialize Phase2Result fields as environment variables:
  - `MATCH_ID`, `LEAGUE_ID`, `A_H`, `A_A`, `PARAM_VERSION`, `EKF_P0`, `TRADING_MODE`
  - CRITICAL: `KALSHI_TICKERS=json.dumps(phase2_result.kalshi_tickers)` — use `json.dumps()`, NEVER `str()`. `str()` produces `{'key': 'val'}` with single quotes which is not valid JSON.
- Launch with `self.client.containers.run(...)` on network `self.network_name`
- Return container id

**Method: `async def stop_container(self, match_id: str) -> bool`**
- Find by name `f"match-{match_id}"`, stop(timeout=30), remove
- Return True if successful, False if not found

**Method: `async def check_heartbeat(self, match_id: str) -> bool`**
- Check container is running via Docker API
- Check Redis for `tick:{match_id}` — if no message in >60 seconds, unhealthy
- Return True if healthy

**Method: `async def list_running_containers(self) -> list[str]`**
- List all containers with name prefix `match-` on the mmpp-net network
- Return list of match_ids (strip "match-" prefix)

## Step 2: Create `src/orchestrator/scheduler.py`

**Function: `async def orchestrator_main_loop(db_pool: asyncpg.Pool, redis_client: object) -> None`**

This runs forever. Every 60 seconds it checks for actionable matches. Every 6 hours it runs full discovery.

```python
lifecycle = MatchLifecycle(db_pool)
containers = ContainerManager()
discovery_interval = 6 * 3600  # 6 hours
check_interval = 60  # 1 minute
last_discovery = 0.0

while True:
    now = time.time()

    # Periodic discovery
    if now - last_discovery >= discovery_interval:
        try:
            await discover_matches(db_pool)
        except Exception as e:
            log.error("discovery_failed", error=str(e))
        last_discovery = now

    # Get actionable matches
    actionable = await get_actionable_matches(db_pool)

    # Trigger Phase 2 for matches approaching kickoff
    for match in actionable["needs_phase2"]:
        await lifecycle.run_phase2(match.match_id)

    # Launch containers for matches ready to go live
    for match in actionable["needs_container"]:
        phase2 = await _load_phase2_result(db_pool, match.match_id)
        if phase2:
            await containers.launch_match_container(match.match_id, phase2, match.trading_mode)
            await lifecycle.transition(match.match_id, MatchStatus.PHASE2_DONE, MatchStatus.PHASE3_RUNNING)

    # Cleanup finished matches
    for match in actionable["needs_cleanup"]:
        await containers.stop_container(match.match_id)
        await lifecycle.transition(match.match_id, MatchStatus.FINISHED, MatchStatus.ARCHIVED)

    # Heartbeat check
    running = await containers.list_running_containers()
    for match_id in running:
        if not await containers.check_heartbeat(match_id):
            await _publish_alert(redis_client, match_id, "heartbeat_missing")

    await asyncio.sleep(check_interval)
```

**Function: `async def recover_orchestrator_state(db_pool: asyncpg.Pool, containers: ContainerManager) -> None`**

Called once on orchestrator startup. Handles 3 cases:
1. `PHASE2_RUNNING` with `updated_at < NOW() - 10 minutes` → re-run Phase 2
2. `PHASE2_DONE` with kickoff past and no running container → launch container
3. `PHASE3_RUNNING` with no running container → log alert, mark for investigation

## Step 3: Create tests

ALL tests must mock Docker. Never call real Docker in tests.

`tests/orchestrator/test_container_manager.py`:
- `test_launch_creates_container`: mock docker.containers.run → verify called with correct env vars including json.dumps for KALSHI_TICKERS
- `test_launch_removes_stale`: mock existing container → verify stop+remove called before new launch
- `test_stop_container`: mock container.stop and remove → verify called
- `test_heartbeat_healthy`: mock running container + recent Redis tick → True
- `test_heartbeat_stale`: mock running container + no recent tick → False
- `test_list_running_containers`: mock 2 containers with "match-" prefix → returns 2 match_ids

`tests/orchestrator/test_scheduler.py`:
- `test_discovery_triggers_at_interval`: first iteration → discovery runs; second within 6h → skipped
- `test_phase2_triggered_at_65min`: SCHEDULED match 60 min before kickoff → run_phase2 called
- `test_container_launched_at_2min`: PHASE2_DONE match 1 min before kickoff → launch called
- `test_cleanup_after_1hour`: FINISHED match 2 hours old → stop+archive called

`tests/orchestrator/test_recovery.py`:
- `test_recover_stale_phase2`: PHASE2_RUNNING for 15 min → re-triggered
- `test_recover_missed_launch`: PHASE2_DONE past kickoff, no container → launched
- `test_recover_dead_container`: PHASE3_RUNNING, no container → alert

Integration test:
```python
async def test_sprint_5b_integration():
    # Insert SCHEDULED match 60 min from kickoff
    # Mock Phase 2 to return GO
    # Run run_phase2 → verify PHASE2_DONE
    # Mock container launch
    # Transition to PHASE3_RUNNING
    # Verify full lifecycle: SCHEDULED → PHASE2_RUNNING → PHASE2_DONE → PHASE3_RUNNING
```

## Step 4: Verify

1. `python -m pytest tests/orchestrator/ -v` — all Sprint 5a + 5b tests pass
2. Existing tests unaffected
